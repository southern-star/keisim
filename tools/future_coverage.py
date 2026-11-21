"""Coverage of the future head's labels (docs/future_aux.md, phase 0): the share of the frames that have a
real-camera frame of the same episode h control steps later (keipilot.data.future_index), from the labels only (no
image is decoded), per data directory, expert reason and look-ahead.

    uv run python tools/future_coverage.py [--ckpt runs/keipilot_mc14/last.pt] [--data DIR ...]
        [--horizons 4 10 16 20] [--out runs/future_aux/coverage.json]

Without --data the checkpoint's training data (its args) is counted, and also the share of the training samples
("weighted"): scripts/train.py's sampling weights (sample_weights, --cf_weight, --dir_weight, the stale frames it
drops) over its training episodes (--seed, --val_frac). A frame is "ok" when any of the look-aheads finds a frame;
the share is of all frames (a virtual frame is never ok, so ~70 % is the ceiling with collect.py's --virtual_prob
0.3). "focus" = the expert reasons pedestrian, yield, crosswalk, bicycle and parked (3, 6, 9, 10, 11). "moving" =
v >= 0.2 m/s. Also: the steps' parity and the step gaps between an episode's recorded frames per directory.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from keipilot.data import FUTURE_HORIZONS, episode_groups, future_index, load_shards  # noqa: E402

REASON_NAMES = {0: "cruise", 1: "curve", 2: "vehicle", 3: "pedestrian", 4: "red_light", 5: "junction_blocked",
                6: "yield", 7: "crossing", 8: "stop_sign", 9: "crosswalk", 10: "bicycle", 11: "parked"}
FOCUS = (3, 6, 9, 10, 11)
MOVING_V = 0.2
GAP_BINS = (2, 4, 6, 8, 10, 20)        # gap == 2, 4, 6, 8, 10, 11-20, > 20


def train_weights(data, files, a):
    """scripts/train.py's sampling weight of every frame (0 for the validation episodes), from its args `a`."""
    from keisim.expert import YELLOW_VERSION
    from train import sample_weights
    w = sample_weights(data, a["dagger_weight"], a["kv_weight"])
    w[data["cf_ok"]] *= a["cf_weight"]
    for spec in a["dir_weight"]:
        d_, w_ = spec.rsplit("=", 1)
        in_dir = np.array([os.path.dirname(os.path.abspath(f)) == os.path.abspath(d_) for f in files])[data["src"]]
        w[in_dir] *= float(w_)
    has_prev = np.diff(data["jpg_prev_off"]) > 0
    w[has_prev] *= a.get("history_weight", 1.0)
    w[(data["tl"] == 1) & ~data["cf_ok"] & (data["label_version"] < YELLOW_VERSION)] = 0.0
    w[data["stale_creep"]] = 0.0
    rng = np.random.default_rng(a["seed"])
    eps = np.unique(data["episode"])
    rng.shuffle(eps)
    val_eps = eps[: max(1, int(len(eps) * a["val_frac"]))]
    w[np.isin(data["episode"], val_eps)] = 0.0
    return w


def share(ok, m, w=None):
    """(frames in m, share of them that are ok[, weighted share])"""
    n = int(m.sum())
    out = {"frames": n, "ok": float(ok[m].mean()) if n else None}
    if w is not None:
        ws = float(w[m].sum())
        out["weighted"] = float(w[m & ok].sum() / ws) if ws > 0 else None
    return out


def gaps(grp, step, m):
    """Histogram of the step gaps between consecutive recorded frames of an episode (frames in m)."""
    idx = np.nonzero(m)[0]
    o = np.lexsort((step[idx], grp[idx]))
    g, s = grp[idx][o], step[idx][o]
    d = np.diff(s)[np.diff(g) == 0]
    out = {}
    lo = 0
    for b in GAP_BINS:
        key = str(b) if b - lo <= 2 else f"{lo + 1}-{b}"
        out[key] = int(((d > lo) & (d <= b)).sum())
        lo = b
    out[f">{GAP_BINS[-1]}"] = int((d > GAP_BINS[-1]).sum())
    out["zero"] = int((d == 0).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/keipilot_mc14/last.pt", help="its training args give the data and weights")
    ap.add_argument("--data", nargs="*", default=None, help="data directories instead of the checkpoint's")
    ap.add_argument("--horizons", type=int, nargs="+", default=list(FUTURE_HORIZONS), help="look-aheads [steps]")
    ap.add_argument("--curve", type=int, nargs="+", default=list(range(2, 31, 2)),
                    help="single look-aheads to count one by one as well")
    ap.add_argument("--out", default="runs/future_aux/coverage.json")
    args = ap.parse_args()
    a = None
    if args.data is None:
        import torch
        a = torch.load(args.ckpt, map_location="cpu")["args"]
        args.data = a["data"]
    data, files = load_shards(args.data, with_prev=False, mmap=True)
    n = len(data["step"])
    real_paths = [os.path.realpath(f) for f in files]
    if len(set(real_paths)) != len(real_paths):
        sys.exit("a shard file is loaded twice (through two directories)")
    fut, info = future_index(data, files, args.horizons)
    ok = (fut >= 0).any(1)
    fut_c, _ = future_index(data, files, args.curve)
    w = train_weights(data, files, a) if a is not None else None
    grp = episode_groups(data, files)
    step = data["step"].astype(np.int64)
    real = ~data["virtual"].astype(bool)
    reason = data["reason"].astype(np.int64)
    focus = np.isin(reason, FOCUS)
    moving = data["v"] >= MOVING_V
    every = np.ones(n, bool)
    file_dir = np.array([os.path.dirname(os.path.abspath(f)) for f in files])[data["src"]]

    def block(m):
        b = share(ok, m, w)
        b["real"] = float(real[m].mean()) if m.any() else None
        b["focus"] = share(ok, m & focus, w)
        b["moving"] = share(ok, m & moving, w)
        b["per_horizon"] = {str(h): float((fut[m, j] >= 0).mean()) if m.any() else None
                            for j, h in enumerate(args.horizons)}
        return b

    res = {"ckpt": args.ckpt if a is not None else None, "data": list(args.data), "horizons": args.horizons,
           "frames": n, "shards": len(files), **info, "total": block(every),
           "per_reason": {}, "per_dir": {}, "curve": {}}
    for r in range(12):
        m = reason == r
        if m.any():
            res["per_reason"][f"{r} {REASON_NAMES[r]}"] = {**share(ok, m, w), "real": float(real[m].mean())}
    for j, h in enumerate(args.curve):
        res["curve"][str(h)] = {"all": float((fut_c[:, j] >= 0).mean()), "focus": float((fut_c[focus, j] >= 0).mean())}
    for d_ in args.data:
        m = file_dir == os.path.abspath(d_)
        b = block(m)
        b["episodes"] = int(len(np.unique(grp[m])))
        b["odd_steps"] = int((step[m] % 2 == 1).sum())
        b["gaps_all"] = gaps(grp, step, m)
        b["gaps_real"] = gaps(grp, step, m & real)
        res["per_dir"][d_] = b
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    def pct(x):
        return "-" if x is None else f"{100 * x:.0f}"

    t = res["total"]
    print(f"{n} frames, {len(files)} shards, {info['episodes']} episodes ({info['episodes_repeated_steps']} with a "
          f"step recorded twice: {info['frames_repeated_steps']} frames); look-aheads {args.horizons} steps")
    wk = w is not None
    print(f"ok: all {pct(t['ok'])} % (real camera {pct(t['real'])} %), focus {pct(t['focus']['ok'])} %, "
          f"moving {pct(t['moving']['ok'])} %" + (f"; of the training samples: all {pct(t['weighted'])} %, "
                                                   f"focus {pct(t['focus']['weighted'])} %" if wk else ""))
    print("\neach look-ahead alone:",
          " ".join(f"{h}: {pct(t['per_horizon'][str(h)])}" for h in args.horizons))
    print("single look-aheads, all / focus:",
          " ".join(f"{h}: {pct(c['all'])}/{pct(c['focus'])}" for h, c in res["curve"].items()))
    print("\n| reason | frames | real % | ok % |" + (" ok % of samples |" if wk else ""))
    print("|---|---:|---:|---:|" + ("---:|" if wk else ""))
    for k, b in res["per_reason"].items():
        print(f"| {k} | {b['frames']} | {pct(b['real'])} | {pct(b['ok'])} |" + (f" {pct(b['weighted'])} |" if wk else ""))
    print("\n| data | frames | episodes | real % | ok % | focus frames | focus ok % | moving ok % | "
          + " | ".join(f"h{h} %" for h in args.horizons) + " | odd steps | gaps 2 / 4 / 6-10 / >10 % |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(args.horizons) + "---:|---|")
    for d_, b in res["per_dir"].items():
        g = b["gaps_all"]
        tot = max(sum(v for k, v in g.items() if k != "zero"), 1)
        g610 = g["6"] + g["8"] + g["10"]
        g10 = g["11-20"] + g[">20"]
        print(f"| {os.path.basename(d_)} | {b['frames']} | {b['episodes']} | {pct(b['real'])} | {pct(b['ok'])} | "
              f"{b['focus']['frames']} | {pct(b['focus']['ok'])} | {pct(b['moving']['ok'])} | "
              + " | ".join(pct(b["per_horizon"][str(h)]) for h in args.horizons)
              + f" | {b['odd_steps']} | {pct(g['2'] / tot)} / {pct(g['4'] / tot)} / {pct(g610 / tot)} / "
                f"{pct(g10 / tot)} |")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
