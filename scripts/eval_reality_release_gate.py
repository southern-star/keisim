#!/usr/bin/env python3
"""Replay the frozen 384 release on the original A/D validation drives and scoring grid."""
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from collect_reality_sweep import split_info, sweep
from diagnose_reality_blur import frozen, inputs
from download_reality_release import MANIFEST, ROOT, verify
from eval_reality_depth_refiner import NAMES, TRACKS, contract, gate, read_refiner, score
from eval_reality_projection import bytes_rgb
from fit_reality_resolution import FIT, read_head
from keireality.background_history import BackgroundHistoryDisplay
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.geometry import orbit_camera, view_pose
from keireality.refined_display import RefinedDrive
from keireality.resolution import ResolutionDrive, SIZE
from keireality.video import quad, write_video
from prepare_reality_projection import checked_fields
from score_reality_resolution import Scores


OUT = ROOT / "runs/reality/release384_gate"
PRIOR = ROOT / "runs/reality/p2_blur"
METHODS = ("A", "D", "release384")


def load_release():
    manifest = json.loads(MANIFEST.read_text())
    for item in manifest["files"]:
        verify(ROOT / item["path"], item)
    pair = json.loads((PRIOR / "development/paired-selection.json").read_text())
    selection = json.loads((FIT / "selection.json").read_text())
    if (pair["weights_and_software"] != contract()
            or selection["model"] != file_hash(ROOT / "keireality/resolution.py")
            or selection["checkpoint"] != file_hash(FIT / "head.pt")):
        raise ValueError("The release model or its rendering code differs from the frozen version")
    return (*frozen("cuda"), read_refiner("cuda"), read_head(), pair["secondary"], selection["strength"])


@torch.inference_mode()
def generate(data, rows, models, folder):
    pilot, depth, renderer, detail, refiner, head, variant, strength = models
    old = BackgroundHistoryDisplay(renderer, detail)
    current = ResolutionDrive(RefinedDrive(renderer, detail, refiner, variant), head, strength)
    views = {a: torch.tensor(view_pose(orbit_camera(a)), device="cuda")[None] for a in range(-36, 37, 3)}
    n = sum(r["count"] for r in rows)
    frames = np.lib.format.open_memmap(folder / "scored.npy", "w+", np.float32, (n, 2, 3, 128, 256, 3))
    native = np.lib.format.open_memmap(folder / "native-sweep.npy", "w+", np.uint8, (n, *SIZE, 3))
    tick = 0
    for row in rows:
        raw = checked_fields(data, row, ("step", "t_sim", "delta_pose", "rgb_in"))
        for j, t in enumerate(raw["t_sim"]):
            if raw["step"][j] != rows[0]["first"] + tick:
                raise ValueError("Nonconsecutive recorded observations")
            _, features = inputs(pilot, depth, raw["rgb_in"][j], "cuda", tick == 0)
            for display in (old, current):
                display.observe(features, row["episode_id"], t, raw["delta_pose"][j])
            for slot, angle in enumerate((0, sweep(tick))):
                view = views[angle]
                a, _, d = old.render(view, slot, comparisons=True)
                high = current.render(view, slot)
                if high.shape != (1, 3, *SIZE) or not torch.isfinite(high).all():
                    raise ValueError("Invalid native release frame")
                # The historical thresholds, masks and teacher are defined on the 256 grid.
                low = F.interpolate(high.float(), size=a.shape[-2:], mode="bicubic",
                                    align_corners=False, antialias=True).clamp(0, 1)
                frames[tick, slot] = torch.cat((a, d, low)).permute(0, 2, 3, 1).cpu().numpy()
                if slot == 1:
                    native[tick] = bytes_rgb(high[0].permute(1, 2, 0).cpu().numpy())
            tick += 1
    if tick != n or not np.isfinite(frames).all():
        raise ValueError("Incomplete or nonfinite predictions")
    return frames, native


def white_lines_and_video(data, rows, frames, native, teacher, dest):
    # Use the published native scorer's white-only mask and signed contrast. Motion is scored by score().
    metrics = [Scores(METHODS) for _ in TRACKS]
    no_warp = np.full(frames.shape[3:5], -1, dtype=np.int32)
    town = rows[0]["town_seed"]
    with np.load(teacher) as archive:
        moving = {name: archive[name] for name in ("rgb", "seg", "seen")}

    def panels():
        tick = 0
        for row in rows:
            truth = checked_fields(data, row, ("rgb_target", "seg_target", "visibility_now"))
            for j in range(row["count"]):
                for slot, (gt, seg, seen) in enumerate((
                    (truth["rgb_target"][j, 0], truth["seg_target"][j, 0], truth["visibility_now"][j, 0]),
                    (moving["rgb"][tick], moving["seg"][tick], moving["seen"][tick]),
                )):
                    metrics[slot].frame(bytes_rgb(frames[tick, slot]), gt, seg, seen == 1, no_warp)
                a, d = bytes_rgb(frames[tick, 1, :2])
                panel = quad(((moving["rgb"][tick], f"正解256　町{town}　{tick / 10:.1f}秒　{sweep(tick):+d}°"),
                              (native[tick], "公開384版　ネイティブ表示"), (a, "A　元のPhase 2"),
                              (d, "D　256版の輪郭基準")))
                if tick == len(frames) // 2:
                    Image.fromarray(panel).save(dest / f"town{town}-sweep.png")
                yield panel
                tick += 1

    if write_video(dest / f"town{town}-sweep.mp4", panels()) != len(frames):
        raise ValueError("Incomplete comparison video")
    result = {}
    for track, metric in zip(TRACKS, metrics, strict=True):
        row = metric.results()
        values = {name: row["methods"][name]["white_line_contrast"] for name in METHODS}
        result[track] = dict(count=row["counts"]["white_line_contrast"], values=values,
                             passed=values["D"] is not None and values["release384"] >= values["D"] - 1e-5)
    return result


def verify_controls(tracks, prior):
    for track in TRACKS:
        observed, expected = tracks[track], prior["tracks"][track]
        if observed["counts"] != expected["counts"]:
            raise ValueError(f"Different scoring samples for {track}")
        for name in METHODS[:2]:
            for key in NAMES:
                a, b = observed[name][key], expected[name][key]
                if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-5):
                    raise ValueError(f"Different reference score for {track}/{name}/{key}")
            if abs(observed["classes"]["marking"]["edge"][name] - expected["classes"]["marking"]["edge"][name]) > 1e-5:
                raise ValueError(f"Different marking control for {track}/{name}")


def evaluate_town(data, rows, models, teacher, dest, prior=None):
    town = rows[0]["town_seed"]
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        frames, native = generate(data, rows, models, Path(tmp))
        tracks, angles, tail = score(data, rows, frames, METHODS, teacher, None, METHODS[-1])
        if prior is not None:
            verify_controls(tracks, prior)
        white = white_lines_and_video(data, rows, frames, native, teacher, dest)
    original = gate(tracks, METHODS[-1])
    passed = original["status"] == "PASS" and all(w["passed"] for w in white.values())
    report = dict(town=town, ticks=len(frames), episode=rows[0]["episode_id"], start_step=rows[0]["first"],
                  status="PASS" if passed else "HOLD", original_gate=original, white=white, tracks=tracks,
                  sweep_by_angle=angles, chase_last_80=tail, teacher_sha256=file_hash(teacher))
    write_json(dest / f"town{town}.json", report)
    print(f"town {town}: {report['status']}", original, {k: v["passed"] for k, v in white.items()}, flush=True)
    return report


def validation():
    data, split, towns, ticks = split_info("val")
    manifest = load_manifest(data)
    models = load_release()
    dest = OUT / "val"
    with writer_lock(dest):
        reports = {}
        for town in towns:
            rows = [r for r in manifest["shards"] if r["town_seed"] == town and r["split"] == split]
            if sum(r["count"] for r in rows) != ticks or len({r["episode_id"] for r in rows}) != 1:
                raise ValueError(f"Incomplete episode for town {town}")
            teacher = PRIOR / f"val/sweep_truth/town{town}.npz"
            prior = json.loads((PRIOR / f"val-stability/town{town}.json").read_text())
            if file_hash(teacher) != prior["sweep_teacher_sha256"]:
                raise ValueError("Mismatched moving-camera teacher")
            reports[town] = evaluate_town(data, rows, models, teacher, dest, prior)
        status = "PASS" if all(r["status"] == "PASS" for r in reports.values()) else "HOLD"
        write_json(dest / "result.json", dict(status=status, release=file_hash(MANIFEST),
                   evaluator=file_hash(__file__), collection=manifest["contract_sha256"], gpu=torch.cuda.get_device_name(),
                   protocols={name: file_hash(Path(__file__).with_name(name)) for name in
                              ("eval_reality_depth_refiner.py", "eval_reality_scene_detail.py", "score_reality_resolution.py")},
                   scoring_grid=[256, 128], reduction="bicubic, align_corners=False, antialias=True, clipped to [0,1]",
                   towns={t: {"status": r["status"], "sha256": file_hash(dest / f"town{t}.json")} for t, r in reports.items()}))
        print("validation", status, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    validation()
