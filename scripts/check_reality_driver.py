#!/usr/bin/env python3
"""Replay real camera inputs through the base commit's driver and the opt-in driver on CPU FP32."""
import argparse
import functools
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keipilot.agent import KeiPilotAgent
from keireality.dataset import file_hash, load_field, load_manifest, write_json
from keireality.encoder import state_hash
from keireality.snapshot import capture_env, state_digest


def check(args):
    torch.set_num_threads(2)
    root = Path(args.data)
    collection = load_manifest(root)
    plan = json.loads((root / "plan.json").read_text())
    if args.town is None:
        ep = plan["episodes"][0]
    else:
        matches = [e for e in plan["episodes"] if e["town_seed"] == args.town]
        if len(matches) != 1:
            raise ValueError("select a town with exactly one recorded episode")
        ep = matches[0]
    base = KeiPilotAgent(args.ckpt, "cpu", yield_hold=True)
    opt = KeiPilotAgent(args.ckpt, "cpu", yield_hold=True)
    # Import the unmodified model definition recorded in the collection, using the same runtime and weights.
    with tempfile.TemporaryDirectory() as temp:
        file = Path(temp) / "base_model.py"
        file.write_bytes(subprocess.check_output(["git", "show", plan["commit"] + ":keipilot/model.py"], cwd=ROOT))
        spec = importlib.util.spec_from_file_location("base_keipilot_model", file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ck = torch.load(args.ckpt, map_location="cpu")
        plain = module.KeiPilot(pretrained=False, **ck["model_cfg"])
        plain.load_state_dict(ck["model"])
        base.model = plain.eval().to(memory_format=torch.channels_last)
    opt.model.forward = functools.partial(opt.model.forward, with_reality_features=True, with_reality_pyramid=args.pyramid)
    before = state_hash(opt.model)
    base.reset()
    opt.reset()
    env = capture_env()
    env.cfg.renderer = "keisim"
    env.reset(town_seed=ep["town_seed"], episode_seed=ep["episode_seed"], render=False)
    for _ in range(ep.get("start_step", 0)):
        env.step(env.expert_action())
    maxima, n, trace = {}, 0, sha256()
    try:
        for row in collection["shards"]:
            if row["town_seed"] != ep["town_seed"] or row["episode_seed"] != ep["episode_seed"]:
                continue
            pixels = load_field(root, row, "rgb_in")
            expected_states = load_field(root, row, "state_sha256")
            for i, rgbs in enumerate(pixels):
                digest = state_digest(env.world.ego.pose, env.world.t, env.world.scene_state())
                if digest != expected_states[i].decode():
                    raise RuntimeError("reference route and captured images differ")
                inputs = ([im[..., ::-1] for im in rgbs], env.command(), env.target_point(), env.world.ego.v, .1)
                a, p = base.act(*inputs, with_seg=True)
                b, q = opt.act(*inputs, with_seg=True)
                if set(p) != set(q):
                    raise RuntimeError("driver result keys changed")
                for k, (left, right) in {"control": (a, b), **{k: (p[k], q[k]) for k in p}}.items():
                    diff = np.max(np.abs(np.asarray(left, np.float64) - np.asarray(right, np.float64)))
                    maxima[k] = max(maxima.get(k, 0.), float(diff))
                    trace.update(np.ascontiguousarray(left).tobytes())
                n += 1
                if n >= args.frames:
                    break
                env.step(env.expert_action())
            if n >= args.frames:
                break
    finally:
        env.close()
    after = state_hash(opt.model)
    passed = n == min(args.frames, ep["frames"]) and n > 0 and all(d == 0 for d in maxima.values()) and state_hash(base.model) == before == after
    result = dict(passed=passed, cpu_fp32=True, frames=n, base_commit=plan["commit"], checkpoint_sha256=file_hash(args.ckpt),
                  town_seed=ep["town_seed"], episode_seed=ep["episode_seed"], model_before=before, model_after=after,
                  maxima=maxima, reference_trace_sha256=trace.hexdigest())
    write_json(Path(args.out), result)
    print(json.dumps(result))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--town", type=int, help="select a town in a multi-episode collection")
    parser.add_argument("--pyramid", action="store_true", help="also read the c2/c3 features during driving replay")
    check(parser.parse_args())
