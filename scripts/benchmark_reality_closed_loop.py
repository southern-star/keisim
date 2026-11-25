#!/usr/bin/env python3
"""Measure paired real-time closed-loop drives from rest on the two validation towns."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from keipilot.agent import KeiPilotAgent
from keireality.dataset import file_hash, write_json, writer_lock
from keireality.drive_metrics import summarize
from keireality.encoder import state_hash
from reality_closed_loop import Drive, OUTPUT, Runtime, TOWNS, YIELD_HOLD, ROOT, WEIGHTS


def compare(base, other, base_result, other_result):
    n = min(len(base), len(other))
    def values(rows, key):
        return np.asarray([r[key] for r in rows[:n]])
    actions = np.abs(values(base, "action") - values(other, "action"))
    poses = values(base, "pose") - values(other, "pose")
    def first(mask):
        indices = np.flatnonzero(mask)
        return int(indices[0]) if len(indices) else None
    return dict(paired_ticks=n, same_tick_count=len(base)==len(other),
                first_camera_difference=first(values(base,"camera_hash") != values(other,"camera_hash")),
                first_action_difference=first(actions.any(axis=1)), max_action_delta=actions.max(axis=0).tolist(),
                max_xy_delta_m=float(np.linalg.norm(poses[:,:2],axis=1).max()),
                max_yaw_delta_rad=float(np.abs(np.arctan2(np.sin(poses[:,2]),np.cos(poses[:,2]))).max()),
                max_speed_delta_mps=float(np.abs(values(base,"speed")-values(other,"speed")).max()),
                max_target_delta_mps=float(np.abs(values(base,"target")-values(other,"target")).max()),
                final_xy_delta_m=float(np.linalg.norm(np.asarray(base_result["final_pose"][:2])-other_result["final_pose"][:2])),
                identical_outcome=base_result==other_result)


def benchmark(dest, steps, warmup):
    runtime = Runtime()
    before = state_hash(runtime.agent.model)
    results, comparisons = {}, {}
    order = {40: ("pilot", "sharp", "stable"), 41: ("stable", "sharp", "pilot")}
    try:
        for town, modes in order.items():
            drives, results[town] = {}, {}
            for mode in modes:
                drive = Drive(runtime, town, mode, capture_trace=True)
                drive.run(steps)
                drives[mode] = drive
                report = summarize(drive, warmup)
                results[town][mode] = report
                write_json(dest / f"town{town}-{mode}.json", dict(summary=report, control=drive.steps,
                           generated=drive.rendered, generated_columns=["observation_tick", "completion_wall_s",
                                                                              "generation_ms", "observation_to_rgb_ms"]))
                print(town, mode, "distance", drive.result["distance"], "Hz", round(report["wall_control_hz"],2),
                      "control p95", round(report["control_work_ms"]["p95"],2), "generated fps", report["generated_fps"],
                      "dropped", report["skipped_frames"], flush=True)
            base = drives["pilot"]
            comparisons[town] = {m:compare(base.steps, drives[m].steps, base.result, drives[m].result)
                                 for m in ("sharp", "stable")}
        if state_hash(runtime.agent.model) != before:
            raise RuntimeError("Driving model state changed")
        write_json(dest / "metrics.json", dict(towns=TOWNS, methods=runtime.methods, ticks_requested=steps,
                   warmup_ticks=warmup, order=order, control_dt=runtime.dt, results=results, paired=comparisons,
                   gpu=torch.cuda.get_device_name(), camera_gpu=runtime.env.keiview().gpu, torch=torch.__version__,
                   checkpoint=file_hash(ROOT / WEIGHTS[0]), driver_state_before_and_after=before,
                   holds=dict(brake=True, red=True, yield_hold=YIELD_HOLD)))
    finally:
        runtime.env.close()


def repeat_pilot(dest):
    report = json.loads((dest / "metrics.json").read_text())
    runtime = Runtime()
    try:
        if state_hash(runtime.agent.model) != report["driver_state_before_and_after"]:
            raise ValueError("Driving model differs from first run")
        repeats = {}
        for town in (41, 40):
            Drive(runtime, town, "stable").run(32)
            drive = Drive(runtime, town, "pilot", capture_trace=True)
            drive.run(report["ticks_requested"])
            summary = summarize(drive, report["warmup_ticks"])
            original = json.loads((dest / f"town{town}-pilot.json").read_text())
            paired = compare(original["control"], drive.steps, original["summary"]["result"], drive.result)
            repeats[town] = dict(summary=summary, paired_to_first=paired)
            write_json(dest / f"town{town}-pilot-repeat.json", dict(**repeats[town], control=drive.steps))
            print("repeat", town, round(summary["wall_control_hz"],2), summary["control_work_ms"], flush=True)
        report["pilot_repeat"] = dict(order=[41,40], preceding_stable_ticks=32, towns=repeats)
        write_json(dest / "metrics.json", report)
    finally:
        runtime.env.close()


@torch.inference_mode()
def shared_input_check(dest):
    """Give two independent driver/controller states the exact same observation before each action."""
    report = json.loads((dest / "metrics.json").read_text())
    runtime = Runtime()
    other = KeiPilotAgent(ROOT / WEIGHTS[0], runtime.device, brake_hold=True, red_hold=True, yield_hold=YIELD_HOLD)
    other.model.requires_grad_(False)
    checks = {}
    try:
        expected = report["driver_state_before_and_after"]
        if state_hash(runtime.agent.model) != expected or state_hash(other.model) != expected:
            raise ValueError("Driver state differs from the timed runs")
        torch.cuda.synchronize()
        for town, seed in TOWNS.items():
            env = runtime.env
            runtime.agent.reset()
            other.reset()
            env.reset(town, seed, render=False)
            maxima, count = {}, 0
            with torch.cuda.stream(runtime.control_stream):
                while count < report["ticks_requested"] and not env.done:
                    cameras = [image for image, _ in env.render_cameras(want_rgb=True, want_seg=False)]
                    args = (cameras, env.command(), env.target_point(), env.world.ego.v, runtime.dt)
                    action, plain = runtime.agent.act(*args)
                    enabled, with_features = other.act(*args, with_reality=True)
                    with_features.pop("reality")
                    if plain.keys() != with_features.keys():
                        raise ValueError("Inconsistent driving outputs")
                    for key, a, b in [("action", action, enabled), *((k, plain[k], with_features[k]) for k in plain)]:
                        delta = np.max(np.abs(np.asarray(a, np.float64)-np.asarray(b, np.float64)))
                        if not np.isfinite(delta):
                            raise ValueError(f"Nonfinite driver comparison for {key}")
                        maxima[key] = max(maxima.get(key, 0.), float(delta))
                    env.step(action)
                    count += 1
            checks[town] = dict(ticks=count, max_absolute_differences=maxima)
            print("same driving input", town, count, maxima, flush=True)
        report["shared_input"] = checks
        write_json(dest / "metrics.json", report)
    finally:
        runtime.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    extra = parser.add_mutually_exclusive_group()
    extra.add_argument("--repeat-pilot", action="store_true", help="repeat the saved baseline after a display run on each town")
    extra.add_argument("--check-shared-input", action="store_true", help="compare controls with identical current inputs")
    args = parser.parse_args()
    if not 0 <= args.warmup <= args.steps - 2:
        parser.error("at least two measured ticks are required")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    with writer_lock(args.out):
        if args.repeat_pilot:
            repeat_pilot(args.out)
        elif args.check_shared_input:
            shared_input_check(args.out)
        else:
            benchmark(args.out, args.steps, args.warmup)
