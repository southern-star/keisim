"""The tables of ReconView's gates G3 and G4 (docs/recon_view.md 7 and 8; the runs of docs/recon/g3.sh and g4.sh):

    python tools/recon_g3.py drive runs/eval/recon/g3_drive      (does ReconView change the drive?)
    python tools/recon_g3.py speed runs/eval/recon/g3_speed      (the page's speed in real time)

drive: per town and setting the "drive" digests of its runs (the car's state, the model's plan and its input images
every second: web/tools/recon_check.mjs) and whether a run without ReconView has the same one. speed: frames per
second of the main view, model calls per second, inference and model-frame times, the mesh's time."""
import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np

TOWNS = {"1000": "town 1000", "1015-98782ab7": "town 1015（二段階の信号）", "osm_sapporo": "札幌（OSM）"}
CFG = {"public": "recon なし、公開の ONNX", "off": "recon なし、深度つきの ONNX", "model": "`recon=model`（蓄積なし）", "split": "`recon=model&layout=split`（蓄積なし）",
       "accum": "`recon=model`（蓄積あり）"}
REAL = re.compile(r"real time (\d+) s.*?: ([\d.]+) fps \(min ([\d.]+)\), ([\d.]+) model calls/s, inference ([\d.]+) ms, model frame ([\d.]+) ms"
                  r"(?:.*?mesh ([\d.]+) ms)?, town time x([\d.]+)")


def split_name(path):
    town, cfg, i = os.path.splitext(os.path.basename(path))[0].rsplit("_", 2)
    return town, cfg, int(i)


def drive(d):
    runs = {}
    for path in sorted(glob.glob(os.path.join(d, "*.json"))):
        town, cfg, i = split_name(path)
        r = json.load(open(path))
        last = r["rows"][-1]
        runs.setdefault(town, {}).setdefault(cfg, []).append((i, r["drive"], last["s"], last["car"]["v"], r["sec"], len(r["rows"])))
    print("| 町 | 設定 | 走行の要約値（回数） | recon なしと同じ要約値の走行 | 判定 |\n|---|---|---|---:|---|")
    for town in [t for t in TOWNS if t in runs] + [t for t in runs if t not in TOWNS]:
        base = Counter(x[1] for x in runs[town].get("off", []))
        for cfg in ("off", "model", "split", "accum"):
            rs = sorted(runs[town].get(cfg, []))
            if not rs:
                continue
            count = Counter(x[1] for x in rs)
            shown = "、".join(f"`{h}`（{n} 回）" for h, n in count.items())
            common = sum(n for h, n in count.items() if h in base)
            verdict = "–" if cfg == "off" else ("通過" if common else "**共通の値なし**")
            print(f"| {TOWNS.get(town, town)} | {CFG[cfg]} | {shown} | {'–' if cfg == 'off' else f'{common} / {len(rs)} 本'} | {verdict} |")
    print("\n| 町 | 要約値 | 60 秒後の位置 | 60 秒後の速度 | その走行の設定 |\n|---|---|---:|---:|---|")
    for town, by in runs.items():
        seen = {}
        for cfg, rs in by.items():
            for _, h, s, v, _, _ in rs:
                seen.setdefault(h, {"s": s, "v": v, "cfg": Counter()})["cfg"][cfg] += 1
        for h, x in seen.items():
            cfgs = "、".join(f"{CFG[c]} × {n}" for c, n in x["cfg"].items())
            print(f"| {TOWNS.get(town, town)} | `{h}` | {x['s']:.1f} m | {3.6 * x['v']:.1f} km/h | {cfgs} |")
    n = [x for by in runs.values() for rs in by.values() for x in rs]
    print(f"\n走行 {len(n)} 本、各 {n[0][4]} 秒、要約値は {n[0][5]} 点（1 秒ごと）から。")


def speed(d):
    runs = {}
    for path in sorted(glob.glob(os.path.join(d, "*.log"))):
        town, cfg, _ = split_name(path)
        m = REAL.search(open(path).read())
        if m:
            runs.setdefault(town, {}).setdefault(cfg, []).append([float(x) if x else np.nan for x in m.groups()])
    print("| 町 | 設定 | 回数 | 主画面 [fps]、中央値 | 各回の fps | モデルの呼び出し [回/秒] | 推論 [ms] | モデルのフレーム（主スレッド）[ms] | メッシュ [ms] | 町の時間 |\n|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for town in [t for t in TOWNS if t in runs]:
        for cfg in CFG:
            rs = np.array(runs[town].get(cfg, []))
            if not len(rs):
                continue
            med = lambda k, f="{:.1f}": f.format(np.median(rs[:, k]))
            mesh = "–" if np.isnan(rs[:, 6]).all() else med(6)
            print(f"| {TOWNS[town]} | {CFG[cfg]} | {len(rs)} | {med(1)} | {'、'.join(f'{v:.0f}' for v in rs[:, 1])} | {med(3)} | {med(4)} | {med(5)} | {mesh} | {med(7, '×{:.2f}')} |")
    print("\n推論、モデルのフレーム、メッシュ、町の時間は各回の値の中央値。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("drive", "speed"))
    ap.add_argument("dir")
    args = ap.parse_args()
    (drive if args.what == "drive" else speed)(args.dir)


if __name__ == "__main__":
    main()
