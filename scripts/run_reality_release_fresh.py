#!/usr/bin/env python3
"""Score the four reserved drives; separately time the released pilot driving from rest."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import torch

from collect_reality_sweep import FRESH, sweep
from eval_reality_release_gate import MANIFEST, METHODS, OUT, ROOT, evaluate_town, load_release
from eval_reality_depth_refiner import gate
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.drive_metrics import summarize
from keireality.geometry import orbit_camera
from keisim.config import EnvConfig, set_camera_rig
from keisim.env import KeiEnv
from reality_closed_loop import Drive, Runtime
from run_reality_blur_fresh import moving_teacher


def validated():
    folder = OUT / "val"
    path = folder / "result.json"
    result = json.loads(path.read_text())
    if (result["status"] != "PASS" or set(result["towns"]) != {"40", "41"}
            or result["release"] != file_hash(MANIFEST)
            or result["evaluator"] != file_hash(Path(__file__).with_name("eval_reality_release_gate.py"))):
        raise ValueError("The validation does not match this release and evaluator")
    for name, sha in result["protocols"].items():
        if file_hash(Path(__file__).with_name(name)) != sha:
            raise ValueError(f"Scoring code changed: {name}")
    for town, summary in result["towns"].items():
        report_path = folder / f"town{town}.json"
        row = json.loads(report_path.read_text())
        if (file_hash(report_path) != summary["sha256"] or gate(row["tracks"], METHODS[-1])["status"] != "PASS"
                or not all(track["passed"] for track in row["white"].values())):
            raise ValueError(f"Missing validation evidence for town {town}")
    return file_hash(path)


def run():
    validation = validated()
    config = ROOT / "configs/reality/p2_surface_history.json"
    split = json.loads(config.read_text())
    episodes = [dict(split="test", town_seed=town, episode_seed=seed, start_step=start, frames=320)
                for town, seed, start in zip(split["fresh_towns"], split["fresh_seeds"], split["fresh_starts"], strict=True)]
    dest = OUT / "fresh"
    with writer_lock(dest):
        plan = dest / "episodes.json"
        write_json(plan, episodes)
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts/collect_reality.py"), "--gl", "hw",
                        "--orbit", "--audit", "--plan", str(plan), "--out", str(FRESH)], cwd=ROOT, check=True)
        manifest = load_manifest(FRESH)
        if json.loads((FRESH / "plan.json").read_text())["episodes"] != episodes:
            raise ValueError("Recorded episodes differ from the reserved drives")
        models = load_release()
        angles = sorted({sweep(t) for t in range(48)})
        cfg = set_camera_rig(EnvConfig(renderer="keiview", weather="noon", keiview_gl="hw"), "wide_narrow")
        cfg.extra_cameras = (*cfg.extra_cameras, *(orbit_camera(a) for a in angles))
        env = KeiEnv(cfg)
        summaries = {}
        try:
            for ep in episodes:
                town = ep["town_seed"]
                rows = [r for r in manifest["shards"] if r["town_seed"] == town and r["split"] == "test"]
                n = sum(r["count"] for r in rows)
                report_path = dest / f"town{town}.json"
                if n != ep["frames"]:
                    report = dict(status="INCOMPLETE", town=town, ticks=n, requested_ticks=ep["frames"],
                                  collection_status=manifest["episodes"][f't{town}-e{ep["episode_seed"]}']["status"])
                    write_json(report_path, report)
                else:
                    teacher = dest / f"town{town}-teacher.npz"
                    moving_teacher(env, angles, ep, rows, teacher)
                    report = evaluate_town(FRESH, rows, models, teacher, dest)
                summaries[town] = dict(status=report["status"], ticks=n, sha256=file_hash(report_path))
        finally:
            env.close()
        status = "PASS" if all(s["status"] == "PASS" for s in summaries.values()) else "HOLD"
        write_json(dest / "result.json", dict(status=status, release=file_hash(MANIFEST), validation=validation,
                   reserved_split=file_hash(config), plan=file_hash(plan), collection=manifest["contract_sha256"],
                   evaluator=file_hash(Path(__file__)), gpu=torch.cuda.get_device_name(), towns=summaries))
        print("reserved drives", status, flush=True)


def time_drives():
    validation = validated()
    dest = OUT / "fresh"
    quality = json.loads((dest / "result.json").read_text())
    plan = dest / "episodes.json"
    if quality["validation"] != validation or quality["plan"] != file_hash(plan):
        raise ValueError("Run the paired quality evaluation on these episodes first")
    with writer_lock(dest / "closed_loop"):
        runtime = Runtime()
        summaries = {}
        try:
            for ep in json.loads(plan.read_text()):
                town = ep["town_seed"]
                drive = Drive(runtime, town, "stable384", episode_seed=ep["episode_seed"], capture_trace=True)
                drive.run(ep["frames"])
                if not drive.rendered or drive.latest[1].shape != (192, 384, 3):
                    raise ValueError(f"Missing 384×192 output in town {town}")
                timing = summarize(drive, 16) if len(drive.steps) >= 18 else None
                passed = (timing is not None and len(drive.steps) == ep["frames"]
                          and timing["wall_control_hz"] >= 9.9 and timing["control_work_ms"]["p95"] <= 100)
                report = dict(town=town, seed=ep["episode_seed"], start_step=0, status="PASS" if passed else "HOLD",
                              timing=timing, full=summarize(drive) if len(drive.steps) >= 2 else None, result=drive.result)
                write_json(dest / "closed_loop" / f"town{town}.json",
                           dict(**report, steps=drive.steps, generated=drive.rendered))
                summaries[town] = report
                print("closed loop", town, report["status"], len(drive.steps), "control ticks", flush=True)
        finally:
            runtime.env.close()
        write_json(dest / "closed_loop/result.json", dict(release=file_hash(MANIFEST), quality=file_hash(dest / "result.json"),
                   evaluator=file_hash(Path(__file__)), runtime=file_hash(Path(__file__).with_name("reality_closed_loop.py")),
                   gpu=torch.cuda.get_device_name(), mode="stable384", warmup=16, towns=summaries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", action="store_true", help="Measure real-time autonomous drives after image scoring.")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if parser.parse_args().timing:
        time_drives()
    else:
        run()
