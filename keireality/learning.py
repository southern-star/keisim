"""Paired feature/teacher shards for supervised learning. Only mem and view_pose reach the renderer."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .dataset import file_hash, load_field, load_manifest


class RealityFrames(Dataset):
    def __init__(self, data, features, split, limit=None, *, teachers=True):
        self.root, self.features = Path(data), Path(features)
        self.teachers = teachers
        self.fields = ("view_pose", "rgb_target", "seg_target", "visibility_now") if teachers else ("view_pose",)
        self.collection = load_manifest(self.root)
        self.cache = json.loads((self.features / "manifest.json").read_text())
        if self.cache["collection_sha256"] != self.collection["contract_sha256"]:
            raise ValueError("feature cache and dataset collection differ")
        self.rows = [r for r in self.collection["shards"] if r["split"] == split]
        self.index = []
        for j, row in enumerate(self.rows):
            meta = self.cache["shards"][row["path"]]
            if (meta["rgb_sha256"] != row["fields"]["rgb_in"]["sha256"] or meta["count"] != row["count"]
                    or file_hash(self.features / row["path"] / "mem.npy") != meta["sha256"]):
                raise ValueError(f'invalid feature/teacher pairing: {row["path"]}')
            for key in self.fields:
                if file_hash(self.root / row["path"] / (key + ".npy")) != row["fields"][key]["sha256"]:
                    raise ValueError(f'corrupt {key} in {row["path"]}')
            self.index.extend((j, i) for i in range(row["count"]))
        if not self.index:
            raise ValueError(f"empty split {split}")
        if limit is not None:
            if limit < 1:
                raise ValueError("frame limit must be positive")
            self.index = [self.index[i] for i in np.linspace(0, len(self.index) - 1,
                                                            min(limit, len(self.index)), dtype=int)]

    @lru_cache(maxsize=24)
    def arrays(self, j):
        row = self.rows[j]
        fields = {key: load_field(self.root, row, key) for key in self.fields}
        fields["mem"] = np.load(self.features / row["path"] / "mem.npy", mmap_mode="r", allow_pickle=False)
        return fields

    def __len__(self):
        return len(self.index)

    def mem_at(self, index):
        j, i = self.index[index]
        return torch.from_numpy(self.arrays(j)["mem"][i].astype(np.float32))

    def __getitem__(self, index):
        j, i = self.index[index]
        f = self.arrays(j)
        item = dict(mem=self.mem_at(index), view_pose=torch.from_numpy(f["view_pose"][i, 0].copy()), index=index)
        if self.teachers:
            item.update(rgb=torch.from_numpy(f["rgb_target"][i, 0].copy()).permute(2, 0, 1).float() / 255,
                        semantic=torch.from_numpy(f["seg_target"][i, 0].astype(np.int64)),
                        visibility=torch.from_numpy(f["visibility_now"][i, 0].copy()))
        return item


def ssim_map(pred, target):
    """Local RGB SSIM, 7x7 box windows, unit dynamic range; average only over in-image window pixels."""
    a, b = pred.float(), target.float()
    pooled = F.avg_pool2d(torch.cat([a, b, a*a, b*b, a*b], 1), 7, 1, 3, count_include_pad=False)
    ma, mb, aa, bb, ab = pooled.chunk(5, 1)
    va, vb = (aa - ma.square()).clamp_min(0), (bb - mb.square()).clamp_min(0)
    return ((2 * ma * mb + .01**2) * (2 * (ab - ma * mb) + .03**2)
            / ((ma.square() + mb.square() + .01**2) * (va + vb + .03**2))).mean(1)


def objective(rgb, logits, batch, weights, past_rgb_weight=1.0):
    # Extra supervision on small objects and markings; these weights are never inference inputs.
    w = weights[batch["semantic"]]
    if past_rgb_weight != 1.0:
        w = w * torch.where(batch["support"] == 2, past_rgb_weight, 1.)
    l1 = ((rgb.float() - batch["rgb"]).abs().mean(1) * w).sum() / w.sum()
    ssim = ssim_map(rgb, batch["rgb"]).mean()
    ce = F.cross_entropy(logits.float(), batch["semantic"], weight=weights)
    return l1 + .2 * (1 - ssim) + .3 * ce
