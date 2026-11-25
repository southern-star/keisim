#!/usr/bin/env python3
"""Render a rotating display from live encoder outputs on recorded test inputs and measure batch-one latency."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, load_field, load_manifest, write_json
from keireality.encoder import extract_mem, frozen_pilot, state_hash
from keireality.geometry import ORBIT_ANGLES, orbit_camera, view_pose
from keireality.multiview import MultiViewFrames
from keireality.stream import TemporalDisplay
from keireality.video import quad, write_video
from eval_reality_temporal import as_uint8, load_renderer


@torch.inference_mode()
def replay(args):
    data = load_manifest(args.data)
    pilot_hash = file_hash(args.pilot)
    model = load_renderer(args.temporal, pilot_hash, args.device)
    display = TemporalDisplay(model)
    offline = MultiViewFrames(args.data, args.features, "test", temporal=True) if args.features else None
    pilot = frozen_pilot(args.pilot, args.device)
    before = state_hash(pilot)
    timings = {k: [] for k in ("encoder", "history_update", "renderer", "download_rgb_support")}
    max_repeat, resets = 0., 0
    cache_checks = []
    args.out.mkdir(parents=True, exist_ok=True)
    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()
    def images():
        nonlocal max_repeat, resets
        old_episode, local, global_index = None, 0, 0
        for row in data["shards"]:
            if row["split"] != "test":
                continue
            episode = row["episode_id"]
            a = {k: load_field(args.data, row, k) for k in ("rgb_in", "rgb_target", "step", "delta_pose", "t_sim")}
            for i, t in enumerate(a["t_sim"]):
                if episode != old_episode:
                    local = 0
                    old_episode = episode
                # Smooth orbit followed by two abrupt camera jumps at fixed driving ticks.
                angle = 36*math.sin(2*math.pi*local/64) if local < 64 else (-36 if local < 80 else 36)
                requested = torch.tensor(view_pose(orbit_camera(angle)), device=args.device)[None]
                rgb = torch.from_numpy(a["rgb_in"][i].copy()).permute(0, 3, 1, 2)[None].to(args.device)
                sync()
                start = time.perf_counter()
                f = {k: v.half() for k, v in extract_mem(pilot, rgb, pyramid=True).items()}
                sync()
                encoded = time.perf_counter()
                display.observe(f, episode, t, a["delta_pose"][i])
                sync()
                observed = time.perf_counter()
                pred, _, support = display.render(requested)
                sync()
                rendered = time.perf_counter()
                generated = pred[0].permute(1, 2, 0).cpu().numpy()
                prob = support[0].softmax(0)[1:3].sum(0).cpu().numpy()
                copied = time.perf_counter()
                if local >= 8:
                    for k, elapsed in zip(timings, np.diff([start, encoded, observed, rendered, copied]), strict=True):
                        timings[k].append(1000*elapsed)
                if local == 0:
                    if display.valid.any():
                        raise RuntimeError("cross-episode history leak")
                    resets += 1
                if local in (32, 64, 80):
                    display.render(torch.tensor(view_pose(orbit_camera(-angle)), device=args.device)[None])
                    again = display.render(requested)[0]
                    max_repeat = max(max_repeat, float((again-pred).abs().max()))
                if offline is not None and local in (0, 32, 80):
                    row_input = offline[global_index]
                    keys = ("mem", "c2", "c3", "past", "motion", "valid")
                    x = {k: row_input[k][None].to(args.device) for k in keys}
                    cached = model(view_pose=requested, **x)[0]
                    diff = (cached-pred).abs()
                    cache_checks.append({"episode": episode, "step": int(a["step"][i]),
                                         "mean_rgb_difference": float(diff.mean()), "max_rgb_difference": float(diff.max())})
                nearest = min(range(len(ORBIT_ANGLES)), key=lambda j: abs(ORBIT_ANGLES[j]-angle))
                dimmed = generated * (.16+.84*prob[..., None])
                yield quad([(a["rgb_in"][i, 0], f'広角入力　{episode}　時点 {a["step"][i]}'),
                            (a["rgb_target"][i, nearest], f'最も近い収録視点の教師　{ORBIT_ANGLES[nearest]:+d}°'),
                            (as_uint8(generated), f'要求視点での時系列生成　{angle:+.1f}°'),
                            (as_uint8(dimmed), "推定した観測支持度に応じて暗く表示")])
                local += 1
                global_index += 1
    count = write_video(args.out / "rotation.mp4", images())
    after = state_hash(pilot)
    if max_repeat != 0 or before != after:
        raise RuntimeError("view repeatability or frozen encoder isolation failed")
    result = dict(frames=count, resets=resets, view_return_max_rgb_diff=max_repeat, live_vs_cached=cache_checks,
                  pilot_sha256=pilot_hash, model_sha256_before=before, model_sha256_after=after,
                  samples=len(timings["renderer"]), precision="FP32 renderer; FP16 feature cache convention; batch 1",
                  latency_ms={k: {f"p{p}": float(np.percentile(v, p)) for p in (50, 95, 99)} for k, v in timings.items()},
                  peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20 if args.device == "cuda" else None)
    write_json(args.out / "stream.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("pilot", "data", "temporal", "out"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--device", choices=("cuda","cpu"), default="cuda")
    p.add_argument("--features", type=Path, help="independently compare the live and offline model inputs")
    replay(p.parse_args())
