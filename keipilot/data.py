"""Dataset shards: JPEG/PNG bytes packed into flat numpy buffers (fork-friendly)."""
from __future__ import annotations

import glob
import json
import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from keisim.expert import target_for_speed

LABEL_KEYS = ("cmd", "tp", "path", "speed", "tl", "v", "reason", "town", "episode", "step", "weather", "virtual")
# inputs of the expert's traffic-light decision (keisim.expert.target_for_speed); shards recorded before
# these existed load with cf_ok = False and are never relabelled
CF_KEYS = ("target_nolight", "lt_over", "lt_d", "lt_st", "lt_trem", "lt_blocked")
CF_DEFAULTS = {"target_nolight": np.float32(np.nan), "lt_over": False, "lt_d": np.float32(np.nan), "lt_st": np.int8(3),
               "lt_trem": np.float32(np.nan), "lt_blocked": False}


class ShardWriter:
    """Accumulates frames and writes compressed shards."""

    def __init__(self, out_dir, prefix, frames_per_shard=2000, jpeg_quality=92):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.prefix = out_dir, prefix
        self.fps, self.q = frames_per_shard, jpeg_quality
        self.k = 0
        self.total = 0
        self._reset()

    def _reset(self):
        self.jpg, self.seg, self.prev = [], [], []
        self.lab = {k: [] for k in LABEL_KEYS}

    def add(self, rgb, seg, prev=None, **labels):
        """prev: the same camera's frame from `history` seconds earlier (multi-frame models), None if unknown."""
        ok, j = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, self.q])
        seg_small = cv2.resize(seg, (seg.shape[1] // 2, seg.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
        ok2, s = cv2.imencode(".png", seg_small, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        self.jpg.append(j.reshape(-1))
        self.seg.append(s.reshape(-1))
        if prev is not None:
            prev = cv2.imencode(".jpg", prev, [cv2.IMWRITE_JPEG_QUALITY, self.q])[1].reshape(-1)
        self.prev.append(np.zeros(0, np.uint8) if prev is None else prev)
        for k in LABEL_KEYS:
            self.lab[k].append(labels[k])
        for k in CF_KEYS:
            if k in labels:
                self.lab.setdefault(k, []).append(labels[k])
        if len(self.jpg) >= self.fps:
            self.flush()

    def flush(self):
        if not self.jpg:
            return
        jo = np.concatenate([[0], np.cumsum([len(x) for x in self.jpg])]).astype(np.int64)
        so = np.concatenate([[0], np.cumsum([len(x) for x in self.seg])]).astype(np.int64)
        arrs = {k: np.asarray(v) for k, v in self.lab.items()}
        if any(len(x) for x in self.prev):
            arrs["jpg_prev"] = np.concatenate(self.prev)
            arrs["jpg_prev_off"] = np.concatenate([[0], np.cumsum([len(x) for x in self.prev])]).astype(np.int64)
        path = os.path.join(self.out_dir, f"{self.prefix}_{self.k:04d}.npz")
        np.savez(path, jpg=np.concatenate(self.jpg), jpg_off=jo, seg=np.concatenate(self.seg), seg_off=so, **arrs)
        self.total += len(self.jpg)
        self.k += 1
        self._reset()


def write_meta(out_dir, **meta):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)


def _stop_profile(x, b=2.5):
    return 0.0 if x <= 0.25 else min(math.sqrt(2 * b * x), 0.8 * x + 0.2)


def relabel_v1(speed, reason):
    """Map label-version-1 target speeds (pure sqrt stopping profile, 2.0 m stop-line
    margin) onto the version-2 profile (sqrt + linear creep, 2.5 m margin).
    Exact whenever the stopping constraint was the binding one (reason code)."""
    out = speed.astype(np.float32).copy()
    for i in np.nonzero((reason == 2) | (reason == 3) | (reason == 4))[0]:
        x = float(speed[i]) ** 2 / 5.0
        if reason[i] == 4:
            x -= 0.5
        out[i] = _stop_profile(x) if speed[i] > 0 else 0.0
    return out


def load_shards(dirs):
    files = []
    dir_version, dir_renderer = {}, {}
    for d in dirs if isinstance(dirs, (list, tuple)) else [dirs]:
        fs = sorted(glob.glob(os.path.join(d, "*.npz")))
        files += fs
        meta = os.path.join(d, "meta.json")
        m = json.load(open(meta)) if os.path.exists(meta) else {}
        for f in fs:
            dir_version[f] = m.get("label_version", 1)
            dir_renderer[f] = m.get("renderer", "keisim")
    if not files:
        raise FileNotFoundError(f"no shards in {dirs}")
    jpg, jo, seg, so, src = [], [], [], [], []
    pj, po, pbase = [], [], 0
    lab = {k: [] for k in LABEL_KEYS + CF_KEYS}
    cf_ok = []
    jbase = sbase = 0
    for fi, f in enumerate(files):
        z = np.load(f)
        src.append(np.full(len(z["jpg_off"]) - 1, fi, np.int32))
        jpg.append(z["jpg"])
        seg.append(z["seg"])
        jo.append(z["jpg_off"][:-1] + jbase)
        so.append(z["seg_off"][:-1] + sbase)
        jbase += len(z["jpg"])
        sbase += len(z["seg"])
        n_frames = len(z["jpg_off"]) - 1
        if "jpg_prev" in z:
            pj.append(z["jpg_prev"])
            po.append(z["jpg_prev_off"][:-1] + pbase)
            pbase += len(z["jpg_prev"])
        else:                                        # no history recorded: zero-length entries
            po.append(np.full(n_frames, pbase, np.int64))
        for k in LABEL_KEYS:
            lab[k].append(z[k])
        n_f = len(z["jpg_off"]) - 1
        has_cf = all(k in z for k in CF_KEYS)
        for k in CF_KEYS:
            lab[k].append(z[k] if has_cf else np.full(n_f, CF_DEFAULTS[k]))
        cf_ok.append(np.full(n_f, has_cf))
    data = {
        "jpg": np.concatenate(jpg), "jpg_off": np.concatenate(jo + [np.array([jbase])]),
        "seg": np.concatenate(seg), "seg_off": np.concatenate(so + [np.array([sbase])]),
        "jpg_prev": np.concatenate(pj) if pj else np.zeros(0, np.uint8),
        "jpg_prev_off": np.concatenate(po + [np.array([pbase])]),
    }
    for k in LABEL_KEYS + CF_KEYS:
        data[k] = np.concatenate(lab[k])
    data["cf_ok"] = np.concatenate(cf_ok)
    data["src"] = np.concatenate(src)
    version = np.array([dir_version[f] for f in files])[data["src"]]
    v1 = version < 2
    if v1.any():
        data["speed"] = data["speed"].astype(np.float32)
        data["speed"][v1] = relabel_v1(data["speed"][v1], data["reason"][v1])
    data["label_version"] = version
    data["keiview"] = np.array([dir_renderer[f] == "keiview" for f in files])[data["src"]]
    data["on_policy"] = np.array([os.path.basename(files[i]).startswith("dagger") for i in range(len(files))])[data["src"]]
    return data, files


class DrivingDataset(Dataset):
    """cf_prob: chance that a training frame with recorded light-decision inputs gets a random ego speed and the
    expert's label for that speed (counterfactual). speed_drop: chance that the ego speed is hidden ("v_known"
    False) from any other training frame, against the inertia problem. A counterfactual speed is always shown,
    since its label depends on it."""

    def __init__(self, data, indices, train=True, cf_prob=0.0, speed_drop=0.0, cf_vmax=11.5, history_drop=0.0):
        self.d = data
        self.idx = np.asarray(indices)
        self.train = train
        self.cf_prob, self.speed_drop, self.cf_vmax = cf_prob, speed_drop, cf_vmax
        self.history_drop = history_drop

    def __len__(self):
        return len(self.idx)

    @staticmethod
    def _aug_params(rng):
        return {"a": rng.uniform(0.75, 1.25), "b": rng.uniform(-22, 22), "gain": rng.uniform(0.9, 1.1, 3).astype(np.float32),
                "sat": rng.uniform(0.6, 1.3) if rng.random() < 0.5 else None,
                "blur": rng.uniform(0.4, 1.0) if rng.random() < 0.25 else None,
                "noise": rng.uniform(1, 6) if rng.random() < 0.5 else None}

    def _augment(self, img, rng, p=None):
        """Photometric augmentation; pass the same `p` to give two frames of one sample the same look."""
        p = p or self._aug_params(rng)
        img = img.astype(np.float32) * p["a"] * p["gain"] + p["b"]
        if p["sat"] is not None:                         # saturation jitter
            g = img.mean(2, keepdims=True)
            img = g + (img - g) * p["sat"]
        if p["blur"] is not None:
            img = cv2.GaussianBlur(img, (0, 0), p["blur"])
        if p["noise"] is not None:
            img = img + rng.normal(0, p["noise"], img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)

    def __getitem__(self, k):
        i = int(self.idx[k])
        d = self.d
        img = cv2.imdecode(d["jpg"][d["jpg_off"][i]:d["jpg_off"][i + 1]], cv2.IMREAD_COLOR)
        seg = cv2.imdecode(d["seg"][d["seg_off"][i]:d["seg_off"][i + 1]], cv2.IMREAD_UNCHANGED)
        v, speed, known = float(d["v"][i]), float(d["speed"][i]), True
        cf = bool(d["cf_ok"][i])
        a, b = d["jpg_prev_off"][i], d["jpg_prev_off"][i + 1]
        prev = cv2.imdecode(d["jpg_prev"][a:b], cv2.IMREAD_COLOR) if b > a else None
        if self.train:
            rng = np.random.default_rng()
            p = self._aug_params(rng)
            img = self._augment(img, rng, p)
            if prev is not None:
                prev = None if rng.random() < self.history_drop else self._augment(prev, rng, p)
            if cf and self.cf_prob > 0 and rng.random() < self.cf_prob:
                v = float(rng.uniform(0.0, self.cf_vmax))
            elif rng.random() < self.speed_drop:
                known = False
        if cf:          # label from the recorded decision inputs with the current rule (and the chosen speed)
            speed = target_for_speed(v, {k: d[k][i] for k in CF_KEYS})
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR -> RGB, CHW, uint8
        has_prev = prev is not None
        prev = prev[:, :, ::-1].transpose(2, 0, 1).copy() if has_prev else np.zeros_like(img)
        return {
            "img": torch.from_numpy(img),
            "seg": torch.from_numpy(seg.astype(np.int64)),
            "cmd": torch.tensor(int(d["cmd"][i])),
            "tp": torch.from_numpy(d["tp"][i].astype(np.float32)),
            "path": torch.from_numpy(d["path"][i].astype(np.float32)),
            "speed": torch.tensor(speed, dtype=torch.float32),
            "tl": torch.tensor(int(d["tl"][i])),
            "v": torch.tensor(v, dtype=torch.float32),                     # ego speed (input of speed models)
            "v_known": torch.tensor(known),
            "img_prev": torch.from_numpy(prev),                            # frame `history` s earlier (or zeros)
            "has_prev": torch.tensor(has_prev),
        }
