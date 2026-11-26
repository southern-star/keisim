#!/usr/bin/env python3
"""Time the released display and the selected background repair in real closed-loop drives."""
import argparse
import json

import cv2
import numpy as np
import torch

from benchmark_reality_closed_loop import compare
from keireality.dataset import file_hash, write_json, writer_lock
from keireality.drive_metrics import summarize
from keireality.encoder import state_hash
from reality_closed_loop import Drive, ROOT, Runtime, TOWNS


OUT = ROOT / "runs/reality/disocclusion384"
EPISODES = ROOT / "runs/reality/release384_gate/fresh/episodes.json"
STEPS, WARMUP = 320, 16


def benchmark(fresh=False):
    source = EPISODES
    modes = ("stable384", "repair384")
    dest = OUT / "timing"
    seeds = dict(TOWNS)
    if fresh:
        source = OUT / "fresh/episodes.json"
        quality = json.loads((OUT / "fresh/summary.json").read_text())
        if quality["selection"] != file_hash(OUT / "fit/selection.json") or quality["episodes"] != file_hash(source):
            raise ValueError("The four new episodes or the selected repair differ from the quality evaluation")
        modes = ("repair384",)
        seeds = {}
        dest = OUT / "fresh/closed_loop"
    seeds.update((row["town_seed"], row["episode_seed"]) for row in json.loads(source.read_text()))
    runtime = Runtime(background_repair=True)
    try:
        with writer_lock(dest):
            driver_sha = state_hash(runtime.agent.model)
            towns = {}
            for i, (town, seed) in enumerate(seeds.items()):
                order = modes[::-1] if i % 2 else modes
                drives, reports = {}, {}
                for mode in order:
                    drive = Drive(runtime, town, mode, capture_trace=True, episode_seed=seed)
                    drive.run(STEPS)
                    report = summarize(drive, WARMUP)
                    frame, rgb = drive.latest
                    if (rgb is None or rgb.shape != (192, 384, 3) or rgb.dtype != np.uint8
                            or np.ptp(rgb) == 0):
                        raise ValueError(f"Invalid native RGB from town {town}, mode {mode}")
                    report["last_rgb_tick"] = frame.tick
                    report["timing_pass"] = (report["ticks"] == STEPS and report["wall_control_hz"] >= 9.9
                                             and report["control_work_ms"]["p95"] <= 100.)
                    drives[mode], reports[mode] = drive, report
                    write_json(dest / f"town{town}-{mode}.json", dict(
                        summary=report, control=drive.steps, rendered=drive.rendered,
                        rendered_columns=["tick", "wall_s", "generation_ms", "observation_to_rgb_ms"]))
                    print(town, mode, f'{report["wall_control_hz"]:.3f} Hz',
                          f'p95 {report["control_work_ms"]["p95"]:.2f} ms',
                          f'{report["generated_frames"]}/{report["ticks"]} RGB',
                          len(drive.result["infractions"]), "infractions", flush=True)
                towns[town] = dict(episode_seed=seed, order=order, methods=reports)
                if len(modes) == 2:
                    a, b = (drives[mode] for mode in modes)
                    towns[town]["paired"] = compare(a.steps, b.steps, a.result, b.result)
            if driver_sha != state_hash(runtime.agent.model):
                raise ValueError("Driving weights changed across runs")
            write_json(dest / "summary.json", dict(towns=towns, steps=STEPS, warmup=WARMUP,
                       dt=runtime.dt, gpu=torch.cuda.get_device_name(), camera_gpu=runtime.env.keiview().gpu,
                       driver_sha256=driver_sha, selection=file_hash(OUT / "fit/selection.json")))
    finally:
        runtime.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true", help="Time the selected repair in the four newly recorded towns")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    benchmark(fresh=parser.parse_args().fresh)
