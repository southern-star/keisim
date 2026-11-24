"""ReconFill's training data (docs/recon_fill.md; written by web/tools/fill_dump.mjs): checks and what is in it.

    python tools/fill_data.py check DIR [--json OUT.json]
    python tools/fill_data.py table OUT.json            (the check's numbers as Markdown)

check, per run (DIR or every DIR/NAME):
* shapes and types; as many samples as meta.json lists;
* a picture of the view has the background colour exactly where its depth says "no surface" (in, loose, prev);
* the town's depth, decoded and put back into the world by keisim/render/rays.py with the chase camera of
  keipilot/filldata.py, has its road at the ground's height (the town is flat), near and far: the browser's camera and
  codes are Python's (the median height of the pixels near the ground, per distance band, within 5 mm of 0; painted
  lines and gutters are a few centimetres off, so the 95 % point is not the test);
* the holes: the share of the town's pixels within 80 m where the view has nothing (what a fill model is trained on),
  and of those the share the loose rule's picture and the fill before cover."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keipilot.filldata import ARRAYS, BG, CHASE, MAX_R, ground_depth, load_run, runs_of  # noqa: E402
from keisim.render import rays as RY  # noqa: E402

BANDS = ((0.0, 10.0), (10.0, 30.0), (30.0, 80.0))


def check_run(path, chunk=64):
    arrays, meta = load_run(path)
    n, bad = len(meta["samples"]), []
    for k, ch in ARRAYS.items():
        want = (n, CHASE.height, CHASE.width) + ((ch,) if ch else ())
        if arrays[k].shape != want or arrays[k].dtype != (np.uint8 if ch else np.uint16):
            bad.append(f"{k}: {arrays[k].shape} {arrays[k].dtype}, expected {want}")
    out = {"run": os.path.basename(os.path.normpath(path)), "town": meta["town"], "samples": n, "bad": bad,
           "mismatch": {k: 0 for k in ("in", "loose", "prev")}, "pixels": n * CHASE.height * CHASE.width,
           "omega": 0, "holes": 0, "holes_loose": 0, "holes_prev": 0, "prev_omega": 0,
           "bytes": int(sum(os.path.getsize(os.path.join(path, f"{k}.npy")) for k in ARRAYS))}
    if bad or n == 0:
        return out
    rays, bg = RY.camera_rays(CHASE), np.array(BG, np.uint8)
    gd = ground_depth()
    z_by_band = [[] for _ in BANDS]
    plane = []
    for a in range(0, n, chunk):
        sl = slice(a, min(n, a + chunk))
        for k in ("in", "loose", "prev"):
            none, is_bg = arrays[f"{k}_dep"][sl] == 0, (arrays[f"{k}_rgb"][sl] == bg).all(-1)
            out["mismatch"][k] += int((none != is_bg).sum())
        g = RY.decode_depth(arrays["gt_dep"][sl])
        omega = (g > 0) & (g <= MAX_R)
        hole = omega & (arrays["in_dep"][sl] == 0)
        out["omega"] += int(omega.sum())
        out["holes"] += int(hole.sum())
        out["holes_loose"] += int((hole & (arrays["loose_dep"][sl] > 0)).sum())
        out["holes_prev"] += int((hole & (arrays["prev_dep"][sl] > 0)).sum())
        out["prev_omega"] += int((omega & (arrays["prev_dep"][sl] > 0)).sum())
        # the road: where the town's depth is, within 2 %, the distance to the flat ground along the pixel's line of
        # sight (the sidewalk, 0.15 m up, is 4 % nearer or more); its height from the unprojected points
        for i, s in enumerate(meta["samples"][sl]):
            p = s["pose"]
            C, R = RY.level_frame(CHASE, p["x"], p["y"], p["yaw"])
            z = RY.unproject(g[i], rays, C, R)[..., 2]
            road = omega[i] & (gd > 0) & (np.abs(g[i] - gd) < 0.02 * gd)
            plane.append(float(road.sum() / max(1, omega[i].sum())))
            for b, (lo, hi) in enumerate(BANDS):
                m = road & (g[i] > lo) & (g[i] <= hi)
                if m.any():
                    z_by_band[b].append(z[m][:: max(1, int(m.sum()) // 500)])
    out["road_share"] = float(np.mean(plane))
    out["road_z"] = []
    for (lo, hi), zs in zip(BANDS, z_by_band):
        z = np.concatenate(zs) if zs else np.zeros(0)
        out["road_z"].append({"lo": lo, "hi": hi, "n": int(len(z)), "median": float(np.median(z)) if len(z) else None,
                              "within_10mm": float(np.mean(np.abs(z) < 0.01)) if len(z) else None})
    return out


def table(rs):
    pc = lambda a, b: f"{100 * a / b:.1f} %" if b else "–"
    print("| 走行 | 町 | 組 | 大きさ | 形と型 | 背景色と「面なし」が食い違う画素（入力 / 緩い規則 / 1 回前） | 地面の高さ: 0〜10 m / 10〜30 m / 30〜80 m（中央値、±10 mm に入る割合）|\n|---|---|---:|---:|---|---|---|")
    for r in rs:
        z = " / ".join("–" if b["median"] is None else f"{1000 * b['median']:+.1f} mm、{100 * b['within_10mm']:.0f} %" for b in r.get("road_z", []))
        mm = r["mismatch"]
        print(f"| {r['run']} | {r['town']} | {r['samples']:,} | {r['bytes'] / 1e6:,.0f} MB | {'、'.join(r['bad']) or '合う'} | "
              f"{mm['in']:,} / {mm['loose']:,} / {mm['prev']:,}（{r['pixels']:,} 画素のうち） | {z} |")
    s = lambda k: sum(r[k] for r in rs)
    print(f"\n合計 {s('samples'):,} 組、{s('bytes') / 1e9:.2f} GB。街の 80 m 以内の画素のうち、入力に面のない所（穴）: {pc(s('holes'), s('omega'))}。"
          f"穴のうち、緩い規則の画に面がある: {pc(s('holes_loose'), s('holes'))}、1 回前の正解のメッシュに面がある: {pc(s('holes_prev'), s('holes'))}。"
          f"1 回前の正解のメッシュが覆う割合（80 m 以内の全部）: {pc(s('prev_omega'), s('omega'))}。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("check", "table"))
    ap.add_argument("path")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.what == "table":
        table(json.load(open(args.path)))
        return
    rs = [check_run(p) for p in runs_of(args.path)]
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rs, f, indent=1)
    table(rs)
    ok = all(not r["bad"] and r["samples"] > 0 and not any(r["mismatch"].values()) and
             all(b["median"] is None or abs(b["median"]) < 0.005 for b in r["road_z"]) for r in rs)
    print("\n検査: " + ("通過" if ok else "**不合格**"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
