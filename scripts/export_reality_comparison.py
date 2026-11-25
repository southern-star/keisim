#!/usr/bin/env python3
"""Export synchronized native RGB frames from the two recorded validation experiments."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image
import torch

from collect_reality_sweep import OUT, split_info, sweep
from eval_reality_depth_refiner import (
    TRACKS, bytes_rgb, contract, file_hash, frozen, generate, load_manifest, read_refiner,
)


PAGE = Path(__file__).with_name("reality_comparison.html")
DEST = OUT / "large"
STAGES = ("val", "val-stability")  # left (contours), right (flicker)
PER_SHEET = 12
FPS = 10


def validation_reports(collection):
    pair_file = OUT / "development/paired-selection.json"
    pair = json.loads(pair_file.read_text())
    common = contract()
    selection = file_hash(OUT / "development/selection.json")
    contracts = (
        dict(common, selection=selection),
        dict(common, selection=file_hash(pair_file),
             comparison_runner=file_hash(Path(__file__).with_name("eval_reality_blur_pair.py"))),
    )
    reports = []
    for stage, method, expected in zip(STAGES, (pair["primary"], pair["secondary"]), contracts, strict=True):
        report = json.loads((OUT / stage / "metrics.json").read_text())
        if report["method"] != method or report["contract"] != expected or report["collection"] != collection:
            raise ValueError(f"Validation inputs or method changed for {stage}")
        reports.append(report)
    return reports


def check_midpoint(rgb, stage, town, report):
    for i, track in enumerate(TRACKS):
        path = OUT / stage / f"town{town}-{track}.png"
        if file_hash(path) != report["media"][path.name]:
            raise ValueError(f"Invalid reference still {path}")
        with Image.open(path) as panel:
            reference = np.array(panel.crop((384, 242, 768, 434)), dtype=np.int16)
        rebuilt = cv2.resize(rgb[len(rgb) // 2, i], (384, 192), interpolation=cv2.INTER_NEAREST)
        # GPU scatter reductions may differ at rounding boundaries.
        diff = np.abs(reference - rebuilt)
        if diff.mean() > 0.05 or np.mean(diff > 2) > 0.001:
            raise ValueError(f"Rendered pixels differ from {path.name} in {stage}")
        print(f"town{town}/{track} {stage} midpoint mean |ΔRGB8| {diff.mean():.6f}", flush=True)


def preview_video(path, pair):
    # Each half retains the coordinates of the same full-width frame.
    width = pair.shape[2] // 2
    cut = width // 2
    wipe = np.concatenate((pair[:, :, :cut], pair[:, :, width + cut:]), axis=2)
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{pair.shape[1]}", "-framerate", str(FPS), "-i", "pipe:0",
        "-vf", "scale=1280:640:flags=neighbor,drawbox=x=640:y=0:w=2:h=ih:color=white@0.65:t=fill",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "15", "-movflags", "+faststart", str(path),
    ], input=wipe.tobytes(), check=True)


def export():
    data, split, towns, count = split_info("val")
    manifest = load_manifest(data)
    reports = validation_reports(manifest["contract_sha256"])
    modules = frozen("cuda")
    refiner = read_refiner("cuda")
    config = dict(width=256, height=128, fps=FPS, perSheet=PER_SHEET,
                  towns=towns, angles=[sweep(t) for t in range(count)], clips={})
    DEST.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT) as work:
        work = Path(work)
        for town in towns:
            rows = [r for r in manifest["shards"] if r["split"] == split and r["town_seed"] == town]
            if sum(r["count"] for r in rows) != count or len({r["episode_id"] for r in rows}) != 1:
                raise ValueError(f"Incomplete validation drive for {town}")
            candidates = []
            for stage, report in zip(STAGES, reports, strict=True):
                method = report["method"]
                with tempfile.TemporaryDirectory(dir=work) as tmp:
                    frames, _ = generate(data, rows, modules, refiner, method, ("A", "D", method),
                                         "cuda", Path(tmp) / "outputs.npy")
                    rgb = bytes_rgb(frames[:, :, 2])
                    del frames
                check_midpoint(rgb, stage, town, report)
                candidates.append(rgb)
            for view, track in enumerate(TRACKS):
                pair = np.concatenate([rgb[:, view] for rgb in candidates], axis=2)
                name = f"town{town}-{track}"
                sheets = []
                for start in range(0, count, PER_SHEET):
                    block = np.concatenate(pair[start:start + PER_SHEET], axis=0)
                    path = work / f"{name}-{start:02d}.png"
                    Image.fromarray(block).save(path, compress_level=3)
                    sheets.append(path.name)
                config["clips"][name] = sheets
                if town == 41 and track == "sweep":
                    preview_video(work / "town41-sweep.mp4", pair)
        (work / "clips.js").write_text("const clips = " + json.dumps(config) + ";\n", encoding="utf-8")
        shutil.copyfile(PAGE, work / "index.html")
        for path in sorted(work.iterdir(), key=lambda p: p.name == "index.html"):
            path.replace(DEST / path.name)
    print(DEST / "index.html", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    export()
