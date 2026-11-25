#!/usr/bin/env python3
"""Make synchronized videos and representative errors from the evaluated, cached frames."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import load_field, load_manifest, write_json
from keireality.geometry import UNKNOWN, VISIBLE
from keireality.metrics import train_baseline
from keireality.video import quad, write_video
from keisim.config import N_SEM, SEM_NAMES


def audit_rgb_classes(root, manifest, rows, evaluated):
    """Use teacher labels to measure displayed RGB error independently of the predicted semantics."""
    mean, _ = train_baseline(root, [row for row in manifest["shards"] if row["split"] == "train"])
    absolute = np.zeros((2, N_SEM), dtype=np.float64)
    counts = np.zeros(N_SEM, dtype=np.int64)
    for row in rows:
        target = load_field(root, row, "rgb_target")[:, 0].astype(np.float32) / 255
        seen = load_field(root, row, "visibility_now")[:, 0] == VISIBLE
        label = load_field(root, row, "seg_target")[:, 0][seen]
        generated = np.load(evaluated / "predictions" / row["path"] / "rgb.npy", mmap_mode="r")
        counts += np.bincount(label, minlength=N_SEM)
        for i, rgb in enumerate((generated.astype(np.float32) / 255, mean)):
            difference = np.abs(rgb - target).mean(-1)[seen]
            absolute[i] += np.bincount(label, weights=difference, minlength=N_SEM)
    return {name: dict(visible_pixels=int(n), generated_l1=float(absolute[0, i]/n) if n else None,
                       mean_l1=float(absolute[1, i]/n) if n else None)
            for i, (name, n) in enumerate(zip(SEM_NAMES, counts, strict=True))}


def inputs(rgb):
    tile = np.full((128, 256, 3), (24, 24, 27), np.uint8)
    for k in range(2):
        x = k * 128
        tile[24:88, x:x + 128] = cv2.resize(rgb[k], (128, 64), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, ("WIDE", "NARROW")[k], (x + 4, 104), cv2.FONT_HERSHEY_SIMPLEX, .35,
                    (230, 230, 235), 1, cv2.LINE_AA)
    return tile


def demo(args):
    root, recon, evaluated, out = map(Path, (args.data, args.recon, args.eval, args.out))
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((evaluated / "metrics.json").read_text())
    manifest = load_manifest(root)
    if (manifest["contract_sha256"] != metrics["collection_sha256"] or
            json.loads((recon / "manifest.json").read_text())["data_sha256"] != metrics["collection_sha256"]):
        raise ValueError("visualization sources belong to different collections")
    rows = [r for r in manifest["shards"] if r["split"] == metrics["split"]]
    write_json(out / "rgb-class-audit.json", audit_rgb_classes(root, manifest, rows, evaluated))
    frames = json.loads((evaluated / "frames.json").read_text())
    clips = json.loads(Path(args.clips).read_text())
    selected, videos = [], []
    for clip in clips:
        records = [r for r in frames if r["episode"] == clip["episode"]
                   and clip["first"] <= r["step"] < clip["first"] + clip["frames"]]
        if [r["step"] for r in records] != list(range(clip["first"], clip["first"] + clip["frames"])):
            raise ValueError(f'incomplete clip {clip["name"]}')
        tiles, cached = [], None
        for rec in records:
            row, i = rows[rec["row"]], rec["offset"]
            if cached != row["path"]:
                rgb = load_field(root, row, "rgb_in")
                gt = load_field(root, row, "rgb_target")
                times = load_field(root, row, "t_sim")
                generated = np.load(evaluated / "predictions" / row["path"] / "rgb.npy", mmap_mode="r")
                with np.load(recon / row["path"] / "recon.npz") as native:
                    baseline = native["model_accum"]
                    if not np.array_equal(native["step"], load_field(root, row, "step")):
                        raise ValueError("replay time mismatch")
                cached = row["path"]
            rr = baseline[i, ..., :3].copy()
            rr[baseline[i, ..., 3] == 0] = (18, 19, 28)
            tiles.append(quad([(inputs(rgb[i]), f'{clip["label"]} | {clip["episode"]} | {times[i]:.1f}s'),
                               (gt[i, 0], "KeiView | same-time teacher"), (rr, "ReconView | model + accumulation"),
                               (generated[i], "KeiReality | frozen driving features")]))
        name = clip["name"] + ".mp4"
        write_video(out / name, tiles)
        cv2.imwrite(str(out / (clip["name"] + ".png")), tiles[len(tiles)//2][..., ::-1])
        videos.extend(tiles)
        selected.append(dict(**clip, file=name))
    write_video(out / "compare.mp4", videos)

    failures = out / "failures"
    failures.mkdir(exist_ok=True)
    samples = []
    for ep in sorted({rec["episode"] for rec in frames}):
        group = [rec for rec in frames if rec["episode"] == ep and rec["visible_dynamic_pixels"] >= 32]
        if not group:
            continue
        rec = max(group, key=lambda x: x["dynamic_rgb_l1"])
        row, i = rows[rec["row"]], rec["offset"]
        target = load_field(root, row, "rgb_target")[i, 0]
        pred = np.load(evaluated / "predictions" / row["path"] / "rgb.npy", mmap_mode="r")[i]
        label = load_field(root, row, "seg_target")[i, 0]
        mask = load_field(root, row, "visibility_now")[i, 0]
        highlight = (target * .22).astype(np.uint8)
        observed_actor = (mask == VISIBLE) & np.isin(label, [11, 12])
        highlight[observed_actor] = (255, 181, 70)
        highlight[mask == UNKNOWN] //= 2
        diff = np.abs(target.astype(np.float32) - pred).mean(-1)
        heatmap = cv2.applyColorMap(np.clip(diff * 2, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
        sheet = quad([(target, f'KeiView | {ep} | step {rec["step"]}'), (pred, "KeiReality | RGB"),
                      (highlight, "Gold | visible teacher vehicle / pedestrian"),
                      (heatmap, "RGB error | brighter = larger")])
        file = f'{ep}-{rec["step"]}.png'
        cv2.imwrite(str(failures / file), sheet[..., ::-1])
        samples.append(dict(**rec, file=file))
    write_json(out / "clips.json", dict(clips=selected, total_video_frames=len(videos), failures=samples))
    print(f"{len(selected)} clips, {len(videos)} frames, {len(samples)} error sheets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "recon", "eval", "clips", "out"):
        parser.add_argument("--" + name, required=True)
    demo(parser.parse_args())
