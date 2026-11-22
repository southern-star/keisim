"""The tables of docs/recon_view.md from ReconView's measurements (web/tools/recon_check.mjs --out DIR).

    python tools/recon_table.py runs/eval/recon/g1 [--repeat runs/eval/recon/g1_repeat]

Prints Markdown: the fidelity per town (the view's triangle rule), by distance, by rule, the triangles, and with
--repeat how far a second run of the same arguments is from the first."""
import argparse
import glob
import json
import os

import numpy as np

TOWNS = {"1000": "town 1000", "1015-98782ab7": "town 1015（二段階の信号）", "osm_sapporo": "札幌（OSM）"}
RULES = {"design": "設計の出発点: クラス、比 1.15（2 台とも）", "class": "クラス、比 1.15（望遠は 1.04）",
         "both": "クラスかつ高さ、比 1.15（望遠は 1.04）", "heightflat": "高さ、比 1.15（2 台とも）",
         "height": "高さ、比 1.15（望遠は 1.04）", "height110": "高さ、比 1.10（望遠は 1.03）",
         "height125": "高さ、比 1.25（望遠は 1.07）"}


def load(d):
    out = {}
    for path in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(path))
        out[r["town"]] = r
    return {k: out[k] for k in list(TOWNS) + [k for k in out if k not in TOWNS] if k in out}


def pc(a, b):
    return f"{100 * a / b:.1f} %" if b else "–"


def tot(rows, key, sub=None):
    return sum((r[sub][key] if sub else r[key]) for r in rows if "omega" in r)


def line(name, rows, get=lambda r: r):
    rows = [get(r) for r in rows if "omega" in r]
    s = lambda k: sum(r[k] for r in rows)
    err = np.median([r["error"] for r in rows if r["error"] is not None])
    return (f"| {name} | {len(rows)} | {pc(s('covered'), s('omega'))} | {pc(s('seenCovered'), s('seen'))} | {pc(s('seen'), s('omega'))} | "
            f"{100 * err:.2f} % | {pc(s('wrong'), s('drawn'))} | {pc(s('through'), s('drawn'))} | {pc(s('phantom'), s('drawn'))} | "
            f"{pc(s('good'), s('omega'))} | {pc(s('seenGood'), s('seen'))} |")


HEAD = ("| {} | フレーム | 被覆率（全体） | 被覆率（いま見える範囲） | 見える範囲の割合 | 誤差（中央値） | 誤描画率 | うち透け | うち実在しない面 | "
        "正しく描けた（全体） | 正しく描けた（見える範囲） |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--repeat", default=None)
    args = ap.parse_args()
    runs = load(args.dir)
    every = [r for run in runs.values() for r in run["rows"]]
    rule = next(r["rule"] for r in every if "rule" in r)
    print(f"規則: 地面 = {rule['ground']}、比 {rule['ratio']}{'（カメラの画素の細かさで縮める）' if rule['scaled'] else ''}\n")
    print(HEAD.format("町"))
    for town, run in runs.items():
        print(line(TOWNS.get(town, town), run["rows"]))
    print(line("**合計**", every))
    print(line("うち動いているフレーム（0.5 m/s 以上）", [r for r in every if r["car"]["v"] >= 0.5]))

    print("\n| 追跡カメラからの距離 | 画素 | 被覆率（全体） | 被覆率（いま見える範囲） | 見える範囲の割合 |\n|---|---:|---:|---:|---:|")
    rows = [r for r in every if "omega" in r]
    lo = 0
    for i in range(3):
        b = [r["bands"][i] for r in rows]
        s = lambda k: sum(x[k] for x in b)
        print(f"| {lo}〜{b[0]['far']} m | {s('omega'):,} | {pc(s('covered'), s('omega'))} | {pc(s('seenCovered'), s('seen'))} | {pc(s('seen'), s('omega'))} |")
        lo = b[0]["far"]

    print("\n" + HEAD.format("三角形の規則（地面の判定、縁の比）"))
    for name, ja in RULES.items():
        if name in rows[0]["variants"]:
            print(line(ja, every, lambda r, n=name: r["variants"][n]))

    print("\n| カメラ | 頂点 | 有効な頂点 | 描いた三角形 | 捨てた: 面なし・80 m 超 | 捨てた: 縁 | 捨てた: 望遠と重なる |\n|---|---:|---:|---:|---:|---:|---:|")
    for k in range(len(rows[0]["stats"])):
        st = [r["stats"][k] for r in rows]
        m = lambda key: np.mean([x[key] for x in st])
        print(f"| {st[0]['camera']} | {st[0]['vertices']:,} | {m('valid'):,.0f} | {m('triangles'):,.0f} | {m('noSurface'):,.0f} | {m('edge'):,.0f} | {m('overlap'):,.0f} |")

    if args.repeat:
        rep = load(args.repeat)
        print("\n| 町 | 1 回目の要約値 | 2 回目の要約値 | 車の位置の差（最大） | 被覆率（全体）の差（最大のフレーム） | 合計の被覆率 1 回目 / 2 回目 |\n|---|---|---|---:|---:|---|")
        for town, run in runs.items():
            if town not in rep:
                continue
            a, b = run["rows"], rep[town]["rows"]
            dx = max(np.hypot(p["car"]["x"] - q["car"]["x"], p["car"]["y"] - q["car"]["y"]) for p, q in zip(a, b))
            dc = max(abs(p["covered"] / p["omega"] - q["covered"] / q["omega"]) for p, q in zip(a, b) if "omega" in p and "omega" in q)
            print(f"| {TOWNS.get(town, town)} | `{run['digest']}` | `{rep[town]['digest']}` | {1000 * dx:.1f} mm | {100 * dc:.2f} ポイント | "
                  f"{pc(tot(a, 'covered'), tot(a, 'omega'))} / {pc(tot(b, 'covered'), tot(b, 'omega'))} |")


if __name__ == "__main__":
    main()
