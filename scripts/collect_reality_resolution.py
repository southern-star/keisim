#!/usr/bin/env python3
"""Capture native resolution teachers and two driving cameras on the exact same simulator state."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collect_reality_sweep import sweep
from keireality.dataset import file_hash, write_json, writer_lock
from keireality.geometry import orbit_camera
from keireality.snapshot import state_digest
from keisim.config import EnvConfig, set_camera_rig
from keisim.env import KeiEnv

OUT = ROOT / "runs/reality/p2_resolution"
CFG = json.loads((ROOT / "configs/reality/p2_resolution.json").read_text())
H, W = CFG["height"], CFG["width"]
TRACKS = ("chase", "sweep")
EPISODES = {e["town_seed"]: e for e in json.loads((ROOT / "configs/reality/p2_episodes.json").read_text())}
ANGLES = sorted({sweep(t) for t in range(48)})


def camera(a):
    return replace(orbit_camera(a), width=W, height=H)


def open_fields(path):
    return {p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in path.glob("*.npy")}


def collect(env, town):
    ep = EPISODES[town]
    val = town in CFG["validation_towns"]
    n = CFG["validation_ticks" if val else "training_ticks"]
    start = 0 if val else ep["start_step"]
    tape = ROOT / f"runs/reality/closed_loop/town{town}-stable.json"
    spec = dict(town=town, seed=ep["episode_seed"], count=n, start=start, size=[W,H],
                capture=file_hash(Path(__file__)), action_tape=file_hash(tape) if val else "expert")
    dest = OUT / "capture" / f"town{town}"
    if dest.is_dir():
        if json.loads((dest / "capture.json").read_text())["spec"] != spec:
            raise ValueError(f"Different capture for town {town}")
        return
    trace = json.loads(tape.read_text()) if val else None
    if val and len(trace["control"]) != n:
        raise ValueError("Incomplete saved driving controls")
    env.reset(town, ep["episode_seed"], render=False)
    for _ in range(start):
        env.step(env.expert_action())
    kv = env.keiview()
    if kv.gl != "hw" or kv.W < W or kv.H < H:
        raise ValueError("KeiView canvas is smaller than the native teacher")
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        folder = Path(tmp) / "capture"
        folder.mkdir()
        def array(key, shape, dtype):
            return np.lib.format.open_memmap(folder / (key+".npy"), "w+", dtype, (n,*shape))
        inputs = array("cameras", (2,160,320,3), np.uint8)
        rgb = array("rgb", (2,H,W,3), np.uint8)
        labels = array("labels", (2,H,W), np.uint8)
        depth = array("depth", (2,H,W), np.uint16)
        source_depth = array("source_depth", (2,160,320), np.uint16)
        poses, nav = array("pose", (3,), np.float64), array("nav", (5,), np.float64)
        states, same_original = sha256(), 0
        for t in range(n):
            if env.done:
                raise ValueError(f"Early end on town {town} at {t}")
            world = env.world
            pose, speed = tuple(world.ego.pose), world.ego.v
            scene = world.scene_state(include_ego=False)
            states.update(state_digest(pose, world.t, scene).encode())
            if val:
                row = trace["control"][t]
                if row["tick"] != t or not np.allclose(pose, row["pose"], atol=1e-8, rtol=0) or abs(row["speed"]-speed)>1e-8:
                    raise ValueError(f"Different closed-loop state at town {town}, tick {t}")
            a = sweep(t)
            which = [0, 1, 2+ANGLES.index(0)]
            if a:
                which.append(2+ANGLES.index(a))
            views, depths = kv.render_rig_depth(pose, scene, world.t, which=which)
            # Both candidates later consume this one pair of camera images, byte for byte.
            inputs[t] = np.stack([bgr for bgr,_ in views[:2]])
            source_depth[t] = np.stack(depths[:2])
            for slot,index in enumerate((2, len(views)-1)):
                image, seg = views[index]
                if image.shape != (H,W,3) or seg.shape != (H,W) or depths[index].shape != (H,W):
                    raise ValueError("KeiView returned an incorrectly sized framebuffer")
                rgb[t,slot], labels[t,slot], depth[t,slot] = image[:,:,::-1], seg, depths[index]
            poses[t] = pose
            nav[t] = (env.elapsed, speed, env.command(), *env.target_point())
            if val:
                same_original += sha256(inputs[t]).hexdigest() == row["camera_hash"]
            env.step(row["action"] if val else env.expert_action())
        if val and not np.allclose(env.world.ego.pose,trace["summary"]["result"]["final_pose"],rtol=0,atol=1e-8):
            raise ValueError("Different final replay state")
        for data in (inputs, source_depth, rgb, labels, depth, poses, nav):
            data.flush()
        write_json(folder/"capture.json", dict(spec=spec, state_sequence=states.hexdigest(), canvas=[kv.W,kv.H],
                   original_matching_camera_pairs=same_original if val else None, renderer_gpu=kv.gpu,
                   files={p.name:file_hash(p) for p in sorted(folder.glob("*.npy"))}))
        folder.rename(dest)
    print("native capture",town,n,"frames × 2 paths; original camera matches",same_original if val else "-",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all","fit","selection","validation"), default="all")
    args = parser.parse_args()
    groups = ("fit","selection","validation") if args.stage == "all" else (args.stage,)
    cfg = set_camera_rig(EnvConfig(renderer="keiview", weather="noon", keiview_gl="hw"), "wide_narrow")
    cfg.extra_cameras = (*cfg.extra_cameras, *(camera(a) for a in ANGLES))
    env = KeiEnv(cfg)
    try:
        with writer_lock(OUT/"capture"):
            for group in groups:
                for town in CFG[group+"_towns"]:
                    collect(env,town)
    finally:
        env.close()
