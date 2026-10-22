"""Copy a DAgger data directory keeping only the frames whose labels the 徐行 rule (keisim/expert.py CROSSING_V,
commit 7e0730d) cannot have changed: those with yield_tta = NaN, i.e. not between 45 m before a crossing the ego gives
way at and its wait point. (In the data collected with the new expert, every "crossing" and "yield" frame has a finite
yield_tta.) The v11 Sapporo sets were dropped from v17's training for their 30 km/h approach labels; their other
frames (v0.8's states, many people walking in the road) can go back in.

usage: uv run python tools/filter_giveway.py SRC DST   (e.g. data/osm_dagger_v11 data/osm_dagger_v11_np)
"""
import glob
import json
import os
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
n_in = n_out = 0
for f in sorted(glob.glob(os.path.join(src, "*.npz"))):
    z = np.load(f)
    n = len(z["jpg_off"]) - 1
    keep = np.nonzero(~np.isfinite(z["yield_tta"]))[0]
    out = {}
    for name in ("jpg", "seg", "jpg_prev"):
        if name not in z.files:
            continue
        buf, off = z[name], z[name + "_off"]
        parts = [buf[off[i]:off[i + 1]] for i in keep]
        out[name] = np.concatenate(parts) if parts else np.zeros(0, np.uint8)
        out[name + "_off"] = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(off.dtype)
    for k in z.files:
        if k in out or k.endswith("_off") or k in ("jpg", "seg", "jpg_prev"):
            continue
        a = z[k]
        out[k] = a[keep] if a.ndim >= 1 and len(a) == n else a
    np.savez(os.path.join(dst, os.path.basename(f)), **out)
    n_in += n
    n_out += len(keep)
meta = json.load(open(os.path.join(src, "meta.json")))
meta.update(filtered_from=src, filter="yield_tta NaN (outside give-way approaches)", frames=n_out)
with open(os.path.join(dst, "meta.json"), "w") as fh:
    json.dump(meta, fh)
print(f"{src} -> {dst}: {n_out} / {n_in} frames kept")
