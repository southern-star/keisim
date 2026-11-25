"""Join time-aligned spatial features, optional causal history, and the chosen teacher-camera index."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import file_hash, load_field
from .history import HistoryWindow, compact_mem
from .learning import RealityFrames


class MultiViewFrames(RealityFrames):
    def __init__(self, data, features, split, views=(0,), labels=None, temporal=False, limit=None, *, teachers=True):
        if labels and not teachers:
            raise ValueError("support labels require teacher loading")
        super().__init__(data, features, split, limit, teachers=teachers)
        self.views, self.temporal = tuple(views), temporal
        self.labels = Path(labels) if labels else None
        meta = json.loads((self.labels / "manifest.json").read_text()) if labels else {}
        if labels and meta["collection_sha256"] != self.collection["contract_sha256"]:
            raise ValueError("history labels and collection differ")
        self.history_index = {}
        window = HistoryWindow()
        for j, row in enumerate(self.rows):
            name = row["path"]
            if min(self.views) < 0 or max(self.views) >= row["fields"]["view_pose"]["shape"][1]:
                raise ValueError("requested teacher camera is absent")
            for field in ("c2", "c3"):
                if file_hash(self.features / name / (field + ".npy")) != self.cache["shards"][name]["pyramid"][field]:
                    raise ValueError(f"invalid {field} in {name}")
            if labels and file_hash(self.labels / name / "visibility.npy") != meta["shards"][name]["sha256"]:
                raise ValueError(f"invalid history supervision in {name}")
            if temporal:
                for key in ("t_sim", "delta_pose"):
                    if file_hash(self.root / name / (key + ".npy")) != row["fields"][key]["sha256"]:
                        raise ValueError(f"invalid history field {key} in {name}")
                times, delta = [load_field(self.root, row, key) for key in ("t_sim", "delta_pose")]
                for i, (t, move) in enumerate(zip(times, delta, strict=True)):
                    self.history_index[j, i] = window.push((j, i), row["episode_id"], t, move)

    @lru_cache(maxsize=32)
    def pyramid(self, j):
        path = self.features / self.rows[j]["path"]
        return tuple(np.load(path / (key + ".npy"), mmap_mode="r") for key in ("c2", "c3"))

    @lru_cache(maxsize=96)
    def compact(self, j):
        return compact_mem(self.arrays(j)["mem"])

    @lru_cache(maxsize=32)
    def history_labels(self, j):
        return np.load(self.labels / self.rows[j]["path"] / "visibility.npy", mmap_mode="r")

    def __len__(self):
        return len(self.index) * len(self.views)

    def frame_view(self, index):
        f, v = divmod(index, len(self.views))
        return (*self.index[f], self.views[v])

    def __getitem__(self, index):
        j, i, view = self.frame_view(index)
        f = self.arrays(j)
        c2, c3 = self.pyramid(j)
        b = dict(index=index, mem=torch.from_numpy(f["mem"][i].copy()),
                 c2=torch.from_numpy(c2[i].copy()), c3=torch.from_numpy(c3[i].copy()),
                 view_pose=torch.from_numpy(f["view_pose"][i, view].copy()))
        if self.teachers:
            b.update(rgb=torch.from_numpy(f["rgb_target"][i, view].copy()).permute(2, 0, 1).float() / 255,
                     semantic=torch.from_numpy(f["seg_target"][i, view].astype(np.int64)),
                     visibility=torch.from_numpy(f["visibility_now"][i, view].copy()))
        if self.labels:
            b["support"] = torch.from_numpy(self.history_labels(j)[i, view].copy())
        if self.temporal:
            refs, motion = self.history_index[j, i]
            past = np.zeros((7, 2, 256, 5, 10), np.float16)
            for k, (sj, si) in enumerate(refs):
                past[k] = self.compact(sj)[si]
            b.update(past=torch.from_numpy(past), motion=torch.from_numpy(motion.copy()),
                     valid=torch.arange(7) < len(refs))
        return b
