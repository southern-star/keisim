#!/usr/bin/env python3
"""Generate offline static-surface history labels using synchronized teacher depth and semantics."""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, load_field, load_manifest, write_json, writer_lock
from keisim.config import CameraConfig
from keisim.render import rays


DEPTH = rays.decode_depth(np.arange(65536, dtype=np.uint16))


def historical_labels(now, seg, depth, pose, ray_tables, past, cameras):
    """Confirm a past-only label by a same-class, radial-depth match. Previous moving objects never supply support."""
    labels = now.copy()
    if not past:
        return labels
    idx = np.flatnonzero((now == 0) & (seg >= 1) & (seg <= 7) & (depth > 0))
    if not len(idx):
        return labels
    vi, y, x = np.unravel_index(idx, now.shape)
    local = ray_tables[0][vi] + ray_tables[1][vi, y, x] * DEPTH[depth.ravel()[idx], None]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    world = np.stack((pose[0] + c*local[:, 0] - s*local[:, 1],
                      pose[1] + s*local[:, 0] + c*local[:, 1], local[:, 2]), 1)
    static = seg.ravel()[idx]
    confirmed, ambiguous = np.zeros(len(idx), bool), np.zeros(len(idx), bool)
    for old_pose, old_dep, old_seg in reversed(past):
        for ic, cam in enumerate(cameras):
            remain = np.flatnonzero(~confirmed)
            if not remain.size:
                break
            uv, dist, ok = rays.project(cam, world[remain], *rays.level_frame(cam, *old_pose))
            inside = remain[ok]
            if not inside.size:
                continue
            u, v = np.floor(uv[ok] + .5).astype(int).T
            expected = dist[ok]
            measured = DEPTH[old_dep[ic, v, u]]
            same = old_seg[ic, v, u] == static[inside]
            match = (measured > 0) & (np.abs(measured - expected) <= .03 * expected)
            confirmed[inside[match & same]] = True
            ambiguous[inside[(measured == 0) | (measured > 1.03*expected) | (match & ~same)]] = True
    labels.ravel()[idx[ambiguous]] = 255
    labels.ravel()[idx[confirmed]] = 2
    return labels


def make_labels(root, out):
    collection = load_manifest(root)
    plan = json.loads((root / "plan.json").read_text())
    cal = plan["calibration"]
    cameras = [CameraConfig(**c) for c in cal["camera_in"]]
    targets = [CameraConfig(**c) for c in cal.get("targets", [cal["target"]])]
    frames = [rays.level_frame(c, 0, 0, 0) for c in targets]
    tables = (np.stack([c for c, _ in frames]), np.stack([rays.camera_rays(cam) @ r
                                                        for cam, (_, r) in zip(targets, frames, strict=True)]))
    past = deque(maxlen=7)
    episode = previous_time = None
    record = dict(collection_sha256=collection["contract_sha256"], version=1, previous_frames=7,
                  tolerance=.03, classes={"0": "no confirmed support", "1": "current", "2": "past static", "255": "unknown"},
                  shards={})
    tick = time.perf_counter()
    with writer_lock(out):
        if (out / "manifest.json").exists():
            saved = json.loads((out / "manifest.json").read_text())
            if (saved["collection_sha256"] != collection["contract_sha256"] or
                    set(saved["shards"]) != {r["path"] for r in collection["shards"]} or
                    any(file_hash(out / p / "visibility.npy") != m["sha256"] for p, m in saved["shards"].items())):
                raise ValueError("incompatible or corrupt history labels")
            print("Verified previously generated history labels")
            return
        for row in collection["shards"]:
            name = row["path"]
            directory = out / name
            directory.mkdir(parents=True, exist_ok=True)
            keys = ("t_sim", "ego_pose_world", "depth_in", "seg_in", "seg_target", "depth_target", "visibility_now")
            for key in keys:
                if file_hash(root / name / (key + ".npy")) != row["fields"][key]["sha256"]:
                    raise ValueError(f"invalid teacher field {key} in {name}")
            a = {k: load_field(root, row, k) for k in keys}
            all_labels = []
            for i, t in enumerate(a["t_sim"]):
                if episode != row["episode_id"] or previous_time is None or not math.isclose(t-previous_time, .1, abs_tol=1e-6):
                    past.clear()
                pose = a["ego_pose_world"][i]
                all_labels.append(historical_labels(a["visibility_now"][i], a["seg_target"][i], a["depth_target"][i],
                                                    pose, tables, past, cameras))
                past.append((pose, a["depth_in"][i], a["seg_in"][i]))
                episode, previous_time = row["episode_id"], t
            labels = np.stack(all_labels)
            path = directory / "visibility.npy"
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as tmp:
                np.save(tmp, labels, allow_pickle=False)
            os.replace(tmp.name, path)
            counts = np.bincount(labels.ravel(), minlength=256)
            record["shards"][name] = dict(sha256=file_hash(path), counts={k: int(counts[int(k)]) for k in record["classes"]})
            print(f'{name}: {counts[2]} past-only static pixels', flush=True)
        record["seconds"] = time.perf_counter() - tick
        write_json(out / "manifest.json", record)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    make_labels(args.data, args.out)
