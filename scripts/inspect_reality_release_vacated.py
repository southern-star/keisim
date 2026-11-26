#!/usr/bin/env python3
"""Locate the one-second interval that contributes most to a measured vacated-background regression."""
import argparse
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from collect_reality_sweep import FRESH
from eval_reality_release_gate import MANIFEST, METHODS, OUT, generate, load_release
from eval_reality_projection import bytes_rgb
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.geometry import orbit_camera
from keireality.temporal_metrics import correspondence
from keireality.video import title_font
from keisim.render import rays
from prepare_reality_projection import checked_fields


def inspect(town):
    folder = OUT / "fresh"
    quality = json.loads((folder / "result.json").read_text())
    report_path = folder / f"town{town}.json"
    row = json.loads(report_path.read_text())
    if (quality["release"] != file_hash(MANIFEST) or quality["towns"][str(town)]["sha256"] != file_hash(report_path)
            or row["original_gate"]["six_vs_A"]["checks"]["vacated_l1"]):
        raise ValueError("Expected a measured regression for the published display")
    rows = [r for r in load_manifest(FRESH)["shards"] if r["town_seed"] == town]
    keys = ("rgb_target", "depth_target", "seg_target", "ego_pose_world")
    shards = [checked_fields(FRESH, r, keys) for r in rows]
    rgb, depth, seg = (np.concatenate([s[k][:, 0] for s in shards]) for k in keys[:3])
    poses = np.concatenate([s["ego_pose_world"] for s in shards])
    camera = orbit_camera(0)
    directions = rays.camera_rays(camera)
    dest = folder / "diagnostics"
    with writer_lock(dest), tempfile.TemporaryDirectory(dir=dest) as tmp:
        frames, _ = generate(FRESH, rows, load_release(), Path(tmp))
        sums = np.zeros((len(rgb), len(METHODS)))
        indices = [np.empty(0, dtype=np.intp)]
        for t in range(1, len(rgb)):
            _, (idx, _) = correspondence(camera, directions, (seg[t], depth[t], poses[t]),
                                         (seg[t-1], depth[t-1], poses[t-1]))
            indices.append(idx)
            gt = rgb[t].reshape(-1, 3)[idx].astype(np.float32) / 255.
            pred = frames[t, 0].reshape(3, -1, 3)[:, idx]
            sums[t] = np.abs(pred - gt).sum((1, 2), dtype=np.float64) / 3
        count = np.array([len(idx) for idx in indices])
        metric = row["tracks"]["chase"]
        recorded = [metric[m]["vacated_l1"] for m in METHODS]
        if count.sum() != metric["counts"]["vacated_l1"] or not np.allclose(sums.sum(0)/count.sum(), recorded, rtol=0, atol=1e-6):
            raise ValueError("The diagnostic does not reproduce the full-drive score")
        extra = sums[:, 2] - sums[:, 0]
        start = int(np.convolve(extra, np.ones(10), mode="valid").argmax())
        t = start + int(extra[start:start+10].argmax())
        mask = np.zeros(seg.shape[1:], dtype=np.uint8)
        mask.ravel()[indices[t]] = 1
        pred = frames[t, 0]
        error = np.abs(pred - rgb[t].astype(np.float32)/255.).mean(-1)
        # Find the 96×48 crop with most excess error on the measured pixels.
        integral = cv2.integral(mask * (error[2] - error[0]), sdepth=cv2.CV_64F)
        h, w = 48, 96
        boxes = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
        y, x = (int(i) for i in np.unravel_index(boxes.argmax(), boxes.shape))
        overlay = rgb[t].copy()
        overlay[mask.astype(bool)] = (255, 222, 0)
        tiles = [(rgb[t-1], "0.1秒前の正解"), (rgb[t], "現在の正解"), (overlay, "黄色　今回採点した背景"),
                 (bytes_rgb(pred[0]), "A　元のPhase 2"), (bytes_rgb(pred[1]), "D　256版の輪郭基準"),
                 (bytes_rgb(pred[2]), "公開384版　採点用に縮小")]
        width, height = w*3, h*3
        board = Image.new("RGB", (3*width, 31+2*(height+25)), (24, 27, 29))
        draw = ImageDraw.Draw(board)
        draw.text((7, 4), f"町{town}　記録開始から{t/10:.1f}秒　固定視点・同じ範囲を3倍に拡大", font=title_font())
        for i, (tile, label) in enumerate(tiles):
            left, top = i%3*width, 31+i//3*(height+25)
            draw.text((left+7, top), label, font=title_font())
            crop = Image.fromarray(tile[y:y+h, x:x+w]).resize((width, height), Image.Resampling.NEAREST)
            board.paste(crop, (left, top+25))
        board.save(dest / f"town{town}.png")
        window = slice(start, start+10)
        summary = dict(town=town, quality=file_hash(report_path), tick=t, window_ticks=[start, start+9],
                       crop_xywh=[x, y, w, h], crop_vacated_pixels=int(mask[y:y+h, x:x+w].sum()),
                       window_pixels=int(count[window].sum()), drive_pixels=int(count.sum()),
                       window_l1=dict(zip(METHODS, (sums[window].sum(0)/count[window].sum()).tolist())),
                       drive_l1=dict(zip(METHODS, recorded)), evaluator=file_hash(__file__))
        write_json(dest / f"town{town}.json", summary)
        print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("town", type=int)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    inspect(parser.parse_args().town)
