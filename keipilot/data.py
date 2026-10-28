"""Dataset shards: JPEG/PNG bytes packed into flat numpy buffers (fork-friendly)."""
from __future__ import annotations

import glob
import json
import math
import os
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from keisim.expert import CROSSING_V, STOP_SIGN_MARGIN, STOP_SIGN_V, STOP_SIGN_ZONE, target_for_speed

LABEL_KEYS = ("cmd", "tp", "path", "speed", "tl", "v", "reason", "town", "episode", "step", "weather", "virtual")
# inputs of the expert's traffic-light decision (keisim.expert.target_for_speed); shards recorded before
# these existed load with cf_ok = False and are never relabelled
CF_KEYS = ("target_nolight", "lt_over", "lt_d", "lt_st", "lt_trem", "lt_blocked")
# auxiliary labels (recorded since 2026-10, NaN where missing or not applicable)
AUX_KEYS = ("yield_tta",)        # earliest arrival [s] of a car with priority at the next right turn that gives way
AUX_DEFAULTS = {"yield_tta": np.float32(np.nan)}
CF_DEFAULTS = {"target_nolight": np.float32(np.nan), "lt_over": False, "lt_d": np.float32(np.nan), "lt_st": np.int8(3),
               "lt_trem": np.float32(np.nan), "lt_blocked": False}
# 止まれ (keisim.expert.target_for_speed): the pending stop line ahead [m] and the target without it; recorded since
# 2026-10-24, NaN where missing (older shards keep their light counterfactuals: these keys are optional)
SS_KEYS = ("ss_d", "ss_go")
SS_DEFAULTS = {"ss_d": np.float32(np.nan), "ss_go": np.float32(np.nan)}
SS_STAND_PROB = 0.5     # counterfactual frames before a pending stop line: chance of the speed 0 (standing at it)
# the ego's one piece of memory (memory-input models): how far it has driven since it last stood (v < STAND_V), as
# the expert remembers a stop made at a 止まれ line. Without it a single frame cannot tell a car that is still to stop
# from one moving off after its stop, a few metres on. Recorded since 2026-10-24 (stand_m); for older shards it is
# rebuilt as 0 (standing) / STAND_CAP (moving) where that cannot change a label (towns without stop lines), else
# unknown (NaN)
MEM_KEYS = ("stand_m",)
STAND_V = STOP_SIGN_V
STAND_CAP = 10.0        # [m]: "not stood within the last 10 m"; the distance saturates here


def stand_update(m, v, dt):
    """stand_m after a control step at speed v [m/s] (dt s): 0 while standing, else the distance since then."""
    return 0.0 if v < STAND_V else min(m + v * dt, STAND_CAP)


class ShardWriter:
    """Accumulates frames and writes compressed shards."""

    def __init__(self, out_dir, prefix, frames_per_shard=2000, jpeg_quality=92, start=0):
        """start: index of the first shard written (a resumed collection continues after the shards it has)."""
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.prefix = out_dir, prefix
        self.fps, self.q = frames_per_shard, jpeg_quality
        self.k = start
        self.total = 0
        self._reset()

    def _reset(self):
        self.jpg, self.seg, self.prev = [], [], []
        self.extra = []                  # per extra camera of the rig: ([jpg], [seg])
        self.lab = {k: [] for k in LABEL_KEYS}

    def _jpg(self, rgb):
        return cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, self.q])[1].reshape(-1)

    @staticmethod
    def _seg(seg):
        seg_small = cv2.resize(seg, (seg.shape[1] // 2, seg.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
        return cv2.imencode(".png", seg_small, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].reshape(-1)

    def add(self, rgb, seg, prev=None, extra=(), **labels):
        """prev: the same camera's frame from `history` seconds earlier (multi-frame models), None if unknown.
        extra: (rgb, seg) of the rig's other cameras (EnvConfig.extra_cameras), stored as jpg_c1, seg_c1, ..."""
        self.jpg.append(self._jpg(rgb))
        self.seg.append(self._seg(seg))
        if not self.extra:
            self.extra = [([], []) for _ in extra]
        assert len(extra) == len(self.extra), "every frame of a shard has the same cameras"
        for (jl, sl), (r, s) in zip(self.extra, extra):
            jl.append(self._jpg(r))
            sl.append(self._seg(s))
        if prev is not None:
            prev = self._jpg(prev)
        self.prev.append(np.zeros(0, np.uint8) if prev is None else prev)
        for k in LABEL_KEYS:
            self.lab[k].append(labels[k])
        for k in CF_KEYS + AUX_KEYS + SS_KEYS + MEM_KEYS:
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
        for c, (jl, sl) in enumerate(self.extra, 1):
            for key, xs in ((f"jpg_c{c}", jl), (f"seg_c{c}", sl)):
                arrs[key] = np.concatenate(xs)
                arrs[key + "_off"] = np.concatenate([[0], np.cumsum([len(x) for x in xs])]).astype(np.int64)
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


def _npy_len(zf, key):
    """Length of array `key` of an open .npz (zipfile), from its .npy header without reading the data; 0 if absent."""
    try:
        info = zf.getinfo(key + ".npy")
    except KeyError:
        return 0
    with zf.open(info) as fp:
        version = np.lib.format.read_magic(fp)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        return read(fp)[0][0]


def _n_cams(zf):
    """Cameras of an open shard (zipfile): the first one plus its extra cameras jpg_c1, jpg_c2, ..."""
    names, c = set(zf.namelist()), 1
    while f"jpg_c{c}.npy" in names:
        c += 1
    return c


def load_shards(dirs, with_prev=True):
    """All shards of `dirs` in memory. The byte buffers (jpg, seg, jpg_prev; ~45 GB for v0.7's data) are allocated
    once from the shard headers and filled in place, so loading peaks at about the data size instead of twice that.
    with_prev=False skips the earlier frames (a quarter of the bytes) for single-frame models.
    Shards of a camera rig also hold jpg_c1 / seg_c1, ... (data["n_cams"] cameras); all shards need the same rig."""
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
    sizes, cams = [], []
    for f in files:
        with zipfile.ZipFile(f) as zf:
            cams.append(_n_cams(zf))
            sizes.append((_npy_len(zf, "jpg"), _npy_len(zf, "seg"), _npy_len(zf, "jpg_prev") if with_prev else 0)
                         + tuple(_npy_len(zf, f"{k}_c{c}") for c in range(1, cams[-1]) for k in ("jpg", "seg")))
    n_cams = cams[0]
    if any(c != n_cams for c in cams):
        raise ValueError(f"shards of different camera rigs ({sorted(set(cams))} cameras) in {dirs}")
    extra_keys = tuple(f"{k}_c{c}" for c in range(1, n_cams) for k in ("jpg", "seg"))
    total = np.sum(np.array(sizes, np.int64), axis=0)
    buf = {k: np.empty(int(n), np.uint8) for k, n in zip(("jpg", "seg", "jpg_prev") + extra_keys, total)}
    xo, xbase = {k: [] for k in extra_keys}, dict.fromkeys(extra_keys, 0)
    jo, so, src = [], [], []
    po, pbase = [], 0
    lab = {k: [] for k in LABEL_KEYS + CF_KEYS + AUX_KEYS + SS_KEYS + MEM_KEYS}
    cf_ok = []
    jbase = sbase = 0
    for fi, f in enumerate(files):
        z = np.load(f)
        nj, ns, npv = sizes[fi][:3]
        src.append(np.full(len(z["jpg_off"]) - 1, fi, np.int32))
        buf["jpg"][jbase:jbase + nj] = z["jpg"]
        buf["seg"][sbase:sbase + ns] = z["seg"]
        jo.append(z["jpg_off"][:-1] + jbase)
        so.append(z["seg_off"][:-1] + sbase)
        jbase += nj
        sbase += ns
        for key, nx in zip(extra_keys, sizes[fi][3:]):
            buf[key][xbase[key]:xbase[key] + nx] = z[key]
            xo[key].append(z[key + "_off"][:-1] + xbase[key])
            xbase[key] += nx
        n_frames = len(z["jpg_off"]) - 1
        if npv:
            buf["jpg_prev"][pbase:pbase + npv] = z["jpg_prev"]
            po.append(z["jpg_prev_off"][:-1] + pbase)
            pbase += npv
        else:                                        # no history recorded (or not loaded): zero-length entries
            po.append(np.full(n_frames, pbase, np.int64))
        for k in LABEL_KEYS:
            lab[k].append(z[k])
        n_f = len(z["jpg_off"]) - 1
        has_cf = all(k in z for k in CF_KEYS)
        for k in CF_KEYS:
            lab[k].append(z[k] if has_cf else np.full(n_f, CF_DEFAULTS[k]))
        for k in AUX_KEYS:
            lab[k].append(z[k] if k in z else np.full(n_f, AUX_DEFAULTS[k]))
        for k in SS_KEYS:
            lab[k].append(z[k] if k in z else np.full(n_f, SS_DEFAULTS[k]))
        lab["stand_m"].append(z["stand_m"].astype(np.float32) if "stand_m" in z else np.full(n_f, np.nan, np.float32))
        cf_ok.append(np.full(n_f, has_cf))
    data = {
        "jpg": buf["jpg"], "jpg_off": np.concatenate(jo + [np.array([jbase])]),
        "seg": buf["seg"], "seg_off": np.concatenate(so + [np.array([sbase])]),
        "jpg_prev": buf["jpg_prev"], "jpg_prev_off": np.concatenate(po + [np.array([pbase])]),
        "n_cams": n_cams,
    }
    for key in extra_keys:
        data[key], data[key + "_off"] = buf[key], np.concatenate(xo[key] + [np.array([xbase[key]])])
    for k in LABEL_KEYS + CF_KEYS + AUX_KEYS + SS_KEYS + MEM_KEYS:
        data[k] = np.concatenate(lab[k])
    data["cf_ok"] = np.concatenate(cf_ok)
    data["src"] = np.concatenate(src)
    # stand_m of shards recorded without it: where no label depends on it (no 止まれ line in the whole directory),
    # 0 when standing, else "not stood lately"; before stop lines it stays unknown
    file_dir = np.array([os.path.dirname(os.path.abspath(f)) for f in files])
    stop_dirs = set(file_dir[np.unique(data["src"][data["reason"] == 8])])
    rebuild = np.isnan(data["stand_m"]) & ~np.isin(data["src"], [i for i, d in enumerate(file_dir) if d in stop_dirs])
    data["stand_m"][rebuild] = np.where(data["v"][rebuild] < STAND_V, 0.0, STAND_CAP)
    version = np.array([dir_version[f] for f in files])[data["src"]]
    v1 = version < 2
    if v1.any():
        data["speed"] = data["speed"].astype(np.float32)
        data["speed"][v1] = relabel_v1(data["speed"][v1], data["reason"][v1])
    data["label_version"] = version
    # label version 6 (keisim.expert): 徐行 (CROSSING_V) from a 止まれ line's stop zone on, up to the wait point.
    # Older frames in the zone (a pending line within it) only differ in the counterfactual "standing at the line"
    # (ss_go), which takes the cap exactly. Past the stop the recorded inputs cannot place a frame before or after
    # the wait point: in directories with 止まれ lines, a faster label after a stand within the last STAND_CAP m,
    # with a crossing ahead that gives way and no line pending, is stale (stale_creep; train.py drops it)
    old = version < 6
    zone = old & np.isfinite(data["ss_d"]) & (data["ss_d"] <= STOP_SIGN_MARGIN + STOP_SIGN_ZONE)
    data["ss_go"][zone] = np.minimum(data["ss_go"][zone], CROSSING_V)
    in_stop_dir = np.isin(data["src"], [i for i, d in enumerate(file_dir) if d in stop_dirs])
    data["stale_creep"] = (old & in_stop_dir & np.isnan(data["ss_d"]) & np.isfinite(data["yield_tta"]) &
                           (data["stand_m"] < STAND_CAP) & (data["target_nolight"] > CROSSING_V + 1e-3))
    data["keiview"] =np.array([dir_renderer[f] == "keiview" for f in files])[data["src"]]
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
        n_cams = d.get("n_cams", 1)
        extra = [(cv2.imdecode(d[f"jpg_c{c}"][d[f"jpg_c{c}_off"][i]:d[f"jpg_c{c}_off"][i + 1]], cv2.IMREAD_COLOR),
                  cv2.imdecode(d[f"seg_c{c}"][d[f"seg_c{c}_off"][i]:d[f"seg_c{c}_off"][i + 1]], cv2.IMREAD_UNCHANGED))
                 for c in range(1, n_cams)]
        v, speed, known = float(d["v"][i]), float(d["speed"][i]), True
        stand = float(d["stand_m"][i])
        cf = bool(d["cf_ok"][i])
        a, b = d["jpg_prev_off"][i], d["jpg_prev_off"][i + 1]
        prev = cv2.imdecode(d["jpg_prev"][a:b], cv2.IMREAD_COLOR) if b > a else None
        if self.train:
            rng = np.random.default_rng()
            p = self._aug_params(rng)
            img = self._augment(img, rng, p)
            if prev is not None:
                prev = None if rng.random() < self.history_drop else self._augment(prev, rng, p)
            extra = [(self._augment(e, rng), s) for e, s in extra]     # each camera its own look
            if cf and self.cf_prob > 0 and rng.random() < self.cf_prob:
                ss_d = float(d["ss_d"][i])
                if math.isfinite(ss_d) and ss_d <= STOP_SIGN_MARGIN + STOP_SIGN_ZONE and rng.random() < SS_STAND_PROB:
                    v = 0.0                     # standing at the 止まれ line: the stop is made, the label says go on
                else:
                    v = float(rng.uniform(0.0, self.cf_vmax))
                if v < STAND_V:                 # standing now; moving, the frame keeps its recorded past
                    stand = 0.0
            elif rng.random() < self.speed_drop:
                known = False
        if cf:          # label from the recorded decision inputs with the current rule (and the chosen speed)
            speed = target_for_speed(v, {k: d[k][i] for k in CF_KEYS + SS_KEYS})
        stand_known = known and math.isfinite(stand)    # a hidden speed hides the memory too (it would give v = 0 away)
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR -> RGB, CHW, uint8
        has_prev = prev is not None
        prev = prev[:, :, ::-1].transpose(2, 0, 1).copy() if has_prev else np.zeros_like(img)
        if extra:       # camera rigs: (K, 3, H, W) and (K, H/2, W/2); no earlier frames
            img = np.stack([img] + [e[:, :, ::-1].transpose(2, 0, 1) for e, _ in extra])
            seg = np.stack([seg] + [s for _, s in extra])
            prev = np.zeros(1, np.uint8)
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
            "stand_m": torch.tensor(stand if stand_known else 0.0, dtype=torch.float32),   # memory-input models
            "stand_known": torch.tensor(stand_known),
            "img_prev": torch.from_numpy(prev),                            # frame `history` s earlier (or zeros)
            "has_prev": torch.tensor(has_prev),
            "yield_tta": torch.tensor(float(d["yield_tta"][i]), dtype=torch.float32),   # NaN: no label
        }
