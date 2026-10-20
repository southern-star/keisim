"""Share of car-following time with the lead vehicle inside the horizontal FOV, from fov_log.py output.
usage: fov_analyze.py LOG.jsonl [LOG2.jsonl ...]
"""
import json
import math
import sys

import numpy as np

FOVS = (100, 120, 140, 160)
DT = 0.1


def load(paths):
    R = []
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            for x in r["rows"]:
                R.append((r["town"], r["episode"], *x[:5], *x[5], x[6], x[7]))
    a = np.array(R, float)
    return dict(town=a[:, 0], ep=a[:, 1], t=a[:, 2], v=a[:, 3], gap=a[:, 4], lv=a[:, 5], o=a[:, 6],
                X=a[:, 7:11], bearing=a[:, 11], turn=a[:, 12])


def vis(d, fov):
    T = math.tan(math.radians(fov) / 2)
    X = d["X"]
    full = (np.abs(X) <= T).all(1)
    lo, hi = X.min(1), X.max(1)
    part = (hi >= -T) & (lo <= T)           # some of the footprint inside (the box spans the image interval)
    return full, part


def table(d, mask, label):
    n = int(mask.sum())
    cells = []
    for f in FOVS:
        full, part = vis(d, f)
        cells.append(f"{100 * np.mean(~part[mask]):5.1f}% / {100 * np.mean(~full[mask]):5.1f}%")
    print(f"{label:<34} {n * DT / 60:7.1f} min  " + "  ".join(cells))


def stretches(d, fov, min_s=0.0):
    """contiguous runs (same route, same lead, consecutive steps) with the lead entirely outside the FOV"""
    _, part = vis(d, fov)
    out = ~part
    runs, i, N = [], 0, len(out)
    while i < N:
        if out[i]:
            j = i
            while j + 1 < N and out[j + 1] and d["town"][j + 1] == d["town"][i] and d["ep"][j + 1] == d["ep"][i] \
                    and abs(d["t"][j + 1] - d["t"][j] - DT) < 1e-6:
                j += 1
            dur = (j - i + 1) * DT
            if dur >= min_s:
                runs.append((int(d["town"][i]), int(d["ep"][i]), float(d["t"][i]), dur, float(d["gap"][i:j + 1].min()),
                             float(np.abs(d["turn"][i:j + 1]).max()), float(d["v"][i:j + 1].max())))
            i = j + 1
        else:
            i += 1
    return runs


if __name__ == "__main__":
    d = load(sys.argv[1:])
    print("cells: lead entirely outside the FOV / not entirely inside, for FOV " + ", ".join(f"{f} deg" for f in FOVS))
    g, v, turn = d["gap"], d["v"], np.abs(d["turn"])
    table(d, np.ones_like(g, bool), "all following")
    for lo_, hi in ((0, 6), (6, 10), (10, 20), (20, 40)):
        table(d, (g >= lo_) & (g < hi), f"gap {lo_}-{hi} m")
    for lo_, hi in ((0, 15), (15, 45), (45, 400)):
        table(d, (turn >= lo_) & (turn < hi), f"route turn next 15 m {lo_}-{hi} deg")
    table(d, (v < 0.5) & (g < 8), "stopped, gap < 8 m")
    table(d, (v < 0.5) & (g < 8) & (turn >= 45), "stopped, gap < 8 m, turn >= 45")
    table(d, (v >= 0.5) & (g < 10), "moving, gap < 10 m")
    for f in FOVS:
        r = stretches(d, f, 1.0)
        print(f"FOV {f}: {len(r)} stretches >= 1 s with the lead out of view, total {sum(x[3] for x in r) / 60:.1f} min,"
              f" median {np.median([x[3] for x in r]) if r else 0:.1f} s")
    r = sorted(stretches(d, 100, 1.0), key=lambda x: -x[3])[:8]
    print("longest at 100 deg (town, ep, t, dur s, min gap, max |turn|, max v):")
    for x in r:
        print("  ", x)
