"""Paired flow data. The inference path consumes current features, odometry, camera poses and timestamps only."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .dataset import file_hash, load_field, load_manifest
from .geometry import ORBIT_ANGLES


def phase2_readout(model, b):
    feat = model.decode_features(b["mem"], b["view_pose"], b["c2"], b["c3"], b["past"], b["motion"], b["valid"])
    rgb = model.rgb(feat).sigmoid()
    support = model.support(feat).softmax(1)[:, 1:3].sum(1, keepdim=True)
    return feat,rgb,support


def phase2_condition(model, b):
    feat,rgb,support=phase2_readout(model,b)
    return F.avg_pool2d(torch.cat((rgb*2-1, support, feat), 1), 2), rgb


def teacher_shard(root, row, views, device):
    rgb = torch.from_numpy(np.ascontiguousarray(load_field(root, row, "rgb_target")[:, views])).to(device)
    rgb = rgb.permute(0,1,4,2,3).flatten(0,1).float()
    low = (F.avg_pool2d(rgb,2)/127.5-1).unflatten(0,(row["count"],len(views)))
    seg = torch.from_numpy(load_field(root,row,"seg_target")[:,views,1::2,1::2].copy()).to(device)
    return low,seg


@dataclass(frozen=True)
class Episode:
    name: str
    town: int
    seed: int
    start: int
    length: int


class FlowSplit:
    def __init__(self, data, cache, split, device="cpu", teachers=True):
        self.root = Path(data)
        self.cache = Path(cache) / split
        self.meta = json.loads((self.cache / "manifest.json").read_text())
        collection = load_manifest(self.root)
        if self.meta["collection"] != collection["contract_sha256"] or self.meta["split"] != split:
            raise ValueError("flow conditions do not match the data split")
        self.angles = self.meta["angles"]
        vi = [ORBIT_ANGLES.index(a) for a in self.angles]
        self.rows = [row for row in collection["shards"] if row["split"] == split]
        if self.meta["shards"] != [r["path"] for r in self.rows]:
            raise ValueError("flow condition rows or their order differ")
        for key, digest in self.meta["arrays"].items():
            if file_hash(self.cache / (key + ".npy")) != digest:
                raise ValueError(f"corrupt condition cache {key}")
        self.current = torch.from_numpy(np.load(self.cache / "current.npy", mmap_mode="c")).to(device)
        n = sum(r["count"] for r in self.rows)
        if self.current.shape != (n, len(vi), 28, 64, 128):
            raise ValueError("unexpected flow condition shape")
        self.episodes, self.offsets = [], []
        self.times = np.empty(n, np.float64)
        self.steps = np.empty(n, np.int64)
        poses, deltas = [], []
        if teachers:
            self.target = torch.empty(n, len(vi), 3, 64, 128, dtype=torch.float16, device=device)
            self.seg = torch.empty(n, len(vi), 64, 128, dtype=torch.uint8, device=device)
        start = 0
        for row in self.rows:
            self.offsets.append(start)
            end = start+row["count"]
            sl = slice(start, end)
            keys = ["step", "t_sim", "view_pose", "delta_pose"] + (["rgb_target", "seg_target"] if teachers else [])
            for key in keys:
                if file_hash(self.root / row["path"] / (key+".npy")) != row["fields"][key]["sha256"]:
                    raise ValueError(f"invalid {row['path']}/{key}")
            self.times[sl] = load_field(data, row, "t_sim")
            self.steps[sl] = load_field(data, row, "step")
            poses.append(load_field(data, row, "view_pose")[:, vi])
            deltas.append(load_field(data, row, "delta_pose"))
            if teachers:
                self.target[sl],self.seg[sl] = teacher_shard(data,row,vi,device)
            if not self.episodes or self.episodes[-1].name != row["episode_id"]:
                self.episodes.append(Episode(row["episode_id"], row["town_seed"], row["episode_seed"], start, row["count"]))
            else:
                old = self.episodes[-1]
                self.episodes[-1] = Episode(old.name, old.town, old.seed, old.start, old.length+row["count"])
            start = end
        self.poses = torch.from_numpy(np.concatenate(poses)).to(device)
        self.delta = torch.from_numpy(np.concatenate(deltas)).to(device)
        for ep in self.episodes:
            sl = slice(ep.start, ep.start+ep.length)
            if not np.allclose(np.diff(self.times[sl]), .1, rtol=0, atol=1e-6) or not (np.diff(self.steps[sl])==1).all():
                raise ValueError(f"discontinuous episode {ep.name}")
