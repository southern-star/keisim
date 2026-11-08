"""Copy a data dir without the frames of a rule, for data collected before collect.py filtered them itself.

Rule ramp_early (round 8's ramps towns, collected before --ramp_early_drop): command left and the label's path
ending RIGHT_M or more to the right within PRE_S before a left turn of the episode (a run of frames with command left
whose path ends LEFT_M or more to the left; runs more than GAP_S apart are separate turns). Those are the labels back
out of a ramp's deceleration lane for a model that moved over in its taper, before the route's change into it: about
10x as many there as before the left turns of the suburban towns, which have no ramps (11-05). The merges out of the
acceleration lane after the turn, also command left with a path to the right, are kept.
Shards are rewritten as ShardWriter writes them (np.savez); meta.json gets "dropped": {"rule", "frames"}.
  uv run python tools/drop_frames.py SRC DST [--rule ramp_early]   (fork dir)"""
import argparse
import glob
import json
import os
import shutil
from collections import defaultdict

import numpy as np

from keisim.config import CMD_LEFT

LEFT_M, RIGHT_M = 6.0, 2.5        # m: path end of a left turn's frames / of a label back to the right
PRE_S, GAP_S = 8.0, 2.0           # s before a left turn; gap between two turns' frames
STEP_S = 0.1                      # s per control step (the shards' step field)


def ramp_early(cols):
    """Drop mask over all frames (cols: per-frame arrays of every shard, concatenated)."""
    cmd, y, step = cols["cmd"], cols["path"][:, -1, 1], cols["step"]
    eps = defaultdict(list)
    for i, key in enumerate(zip(cols["town"].tolist(), cols["episode"].tolist())):
        eps[key].append(i)
    drop = np.zeros(len(cmd), bool)
    pre, gap = PRE_S / STEP_S, GAP_S / STEP_S
    for idx in eps.values():
        idx = np.array(sorted(idx, key=lambda i: step[i]))
        st, left = step[idx], cmd[idx] == CMD_LEFT
        turn = st[left & (y[idx] >= LEFT_M)]
        starts = [s for k, s in enumerate(turn) if k == 0 or s - turn[k - 1] > gap]
        back = left & (y[idx] <= -RIGHT_M)
        for s0 in starts:
            drop[idx[back & (st >= s0 - pre) & (st < s0)]] = True
    return drop


RULES = {"ramp_early": ramp_early}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--rule", default="ramp_early", choices=list(RULES))
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.src, "*.npz")))
    if not files or os.path.exists(args.dst):
        raise SystemExit(f"no shards in {args.src}, or {args.dst} exists")
    keys = ("cmd", "path", "town", "episode", "step")
    cols, sizes = {k: [] for k in keys}, []
    for f in files:
        z = np.load(f)
        for k in keys:
            cols[k].append(z[k])
        sizes.append(len(z["cmd"]))
    drop = RULES[args.rule]({k: np.concatenate(v) for k, v in cols.items()})
    os.makedirs(args.dst)
    at = 0
    for f, n in zip(files, sizes):
        keep = np.flatnonzero(~drop[at:at + n])
        at += n
        z = np.load(f)
        out = {}
        for k in z.files:
            if k.endswith("_off"):
                continue
            a = z[k]
            if k + "_off" in z.files:                          # bytes of every frame, cut at the offsets
                off = z[k + "_off"]
                parts = [a[off[i]:off[i + 1]] for i in keep]
                out[k] = np.concatenate(parts) if parts else a[:0]
                out[k + "_off"] = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int64)
            else:
                assert len(a) == n, (f, k, a.shape)
                out[k] = a[keep]
        np.savez(os.path.join(args.dst, os.path.basename(f)), **out)
    for f in glob.glob(os.path.join(args.src, "stats_w*.json")):
        shutil.copy(f, args.dst)
    meta = json.load(open(os.path.join(args.src, "meta.json")))
    meta["dropped"] = {"rule": args.rule, "frames": int(drop.sum()), "of": int(len(drop)), "src": args.src}
    json.dump(meta, open(os.path.join(args.dst, "meta.json"), "w"))
    print(f"{args.src} -> {args.dst}: {drop.sum()} of {len(drop)} frames dropped ({args.rule})")


if __name__ == "__main__":
    main()
