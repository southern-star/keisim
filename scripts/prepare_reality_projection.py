#!/usr/bin/env python3
"""Cache the unmodified causal Phase 2 RGB and frozen monocular depths for the requested split."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, load_field, load_manifest, write_json, writer_lock
from keireality.encoder import frozen_pilot, state_hash
from keireality.geometry import ORBIT_ANGLES, orbit_camera
from keireality.multiview import MultiViewFrames
from keireality.temporal_metrics import correspondence
from keisim.render import rays
from eval_reality_temporal import load_renderer


def checked_fields(data, row, names):
    for name in names:
        if file_hash(data / row["path"] / (name + ".npy")) != row["fields"][name]["sha256"]:
            raise ValueError(f"corrupt {name} in {row['path']}")
    return {k: load_field(data, row, k) for k in names}


def training_warps(data, rows, views, dest):
    """Signed previous pixel plus one. These teacher-derived maps are only used by the training loss."""
    n = sum(row["count"] for row in rows)
    result = np.lib.format.open_memmap(dest / "training_warp.npy", "w+", np.int32, (n, len(views), 128, 256))
    cameras = [orbit_camera(ORBIT_ANGLES[v]) for v in views]
    directions = [rays.camera_rays(c) for c in cameras]
    previous, old_id, t = None, None, 0
    for row in rows:
        a = checked_fields(data, row, ("seg_target", "depth_target", "ego_pose_world", "step"))
        for j, step in enumerate(a["step"]):
            current = [(a["seg_target"][j, v], a["depth_target"][j, v], a["ego_pose_world"][j]) for v in views]
            result[t] = 0
            if old_id == (row["episode_id"], step - 1):
                for k, cam in enumerate(cameras):
                    (idx, prev), (vacated, old) = correspondence(cam, directions[k], current[k], previous[k])
                    flat = result[t, k].reshape(-1)
                    flat[idx] = prev + 1
                    flat[vacated] = -(old + 1)
            previous, old_id, t = current, (row["episode_id"], step), t + 1
    result.flush()


@torch.inference_mode()
def prepare(a):
    cfg = json.loads(a.config.read_text())
    raw = load_manifest(a.data)
    angles = cfg["train_angles"] if a.split == "train" else cfg["evaluation_angles"]
    views = tuple(ORBIT_ANGLES.index(x) for x in angles)
    rows = [r for r in raw["shards"] if r["split"] == a.split]
    towns = list(dict.fromkeys(r["town_seed"] for r in rows))
    expected = cfg[{"train":"train_towns", "val":"validation_towns", "test":"fresh_towns"}[a.split]]
    if towns != expected:
        raise ValueError(f"expected {expected}, found {towns}")
    fm = json.loads((a.features / "manifest.json").read_text())
    contract = dict(collection=raw["contract_sha256"], features=file_hash(a.features / "manifest.json"),
                    pilot=fm["checkpoint_sha256"], phase2=file_hash(a.phase2), depth_head=file_hash(a.depth_head),
                    split=a.split, angles=angles, rows=[r["path"] for r in rows], protocol=file_hash(a.config))
    out = a.cache / a.split
    torch.set_num_threads(2)
    with writer_lock(a.cache):
        if out.exists():
            meta = json.loads((out / "manifest.json").read_text())
            if meta["contract"] != contract or any(file_hash(out / name) != sha for name, sha in meta["files"].items()):
                raise ValueError("incompatible or corrupted projection cache")
            return
        with tempfile.TemporaryDirectory(dir=a.cache) as tmp:
            stage = Path(tmp) / "split"
            stage.mkdir()
            ds = MultiViewFrames(a.data, a.features, a.split, views, temporal=True, teachers=False)
            n = len(ds.index)
            baseline = np.lib.format.open_memmap(stage / "base.npy", "w+", np.float32, (n, len(views), 3, 128, 256))
            pixels = baseline.reshape(-1, 3, 128, 256)
            model = load_renderer(a.phase2, fm["checkpoint_sha256"], a.device).requires_grad_(False)
            pos = 0
            for batch in DataLoader(ds, batch_size=8):
                args = [batch[k].to(a.device) for k in ("mem", "view_pose", "c2", "c3", "past", "motion", "valid")]
                rgb = model(*args)[0]
                pixels[pos:pos+len(rgb)] = rgb.cpu().numpy()
                pos += len(rgb)
            if pos != len(ds):
                raise ValueError("incomplete Phase 2 replay")
            baseline.flush()
            depth = np.lib.format.open_memmap(stage / "estimated.npy", "w+", np.float32, (n, 2, 80, 160))
            head = frozen_pilot(a.depth_head, a.device)
            if not head.aux_depth:
                raise ValueError("missing frozen radial-depth head")
            before = state_hash(head)
            pos = 0
            for row in rows:
                inputs = checked_fields(a.data, row, ("rgb_in",))["rgb_in"]
                for start in range(0, len(inputs), 8):
                    rgb = torch.from_numpy(inputs[start:start+8].copy()).permute(0,1,4,2,3).to(a.device)
                    predicted = head.depth_metres(head.forward_depth(rgb)).reshape(-1, 2, 80, 160)
                    if not predicted.isfinite().all():
                        raise ValueError("invalid estimated depth")
                    depth[pos:pos+len(rgb)] = predicted.cpu().numpy()
                    pos += len(rgb)
            if pos != n or state_hash(head) != before:
                raise ValueError("depth cache failed parity or length check")
            depth.flush()
            if a.split == "train":
                training_warps(a.data, rows, views, stage)
            write_json(stage / "manifest.json", dict(contract=contract, ticks=n, depth_state=before,
                       files={f.name:file_hash(f) for f in sorted(stage.glob("*.npy"))}))
            stage.rename(out)
    print(f"{a.split}: {n} ticks, {angles}, frozen reference and depth verified", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key, default in dict(config="configs/reality/p2_depth_projection.json", data="data/reality/p2",
                             features="data/reality/p2_spatial", phase2="runs/reality/p2/temporal/best.pt",
                             depth_head="runs/reference/mc14-depth.pt", cache="data/reality/p2_projection").items():
        p.add_argument("--"+key.replace("_", "-"), type=Path, default=Path(default))
    p.add_argument("--split", choices=("train","val","test"), required=True)
    p.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    prepare(p.parse_args())
