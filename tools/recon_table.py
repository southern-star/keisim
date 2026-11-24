"""The tables of docs/recon_view.md from ReconView's measurements (web/tools/recon_check.mjs --out DIR).

    python tools/recon_table.py runs/eval/recon/g1 [--repeat runs/eval/recon/g1_repeat]
    python tools/recon_table.py runs/eval/recon/g3          (?recon=model: the model's depth next to the oracle's)
    python tools/recon_table.py runs/eval/recon/g4          (with keyframes: also the current frame alone against with them)
    python tools/recon_table.py runs/eval/recon_fill/p0     (--fill: also the same frames with the fill, docs/recon_fill.md)

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


KEEP = {"all": "全部（設計の値）: 地面、建物、植物、柱、14 m まで", "noveg": "植物を外す", "nobuild": "建物を外す",
        "flat": "植物と建物を外す（地面と柱）", "ground": "地面だけ", "all10": "全部、10 m まで", "flat10": "植物と建物を外す、10 m まで"}


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


def model_tables(runs):
    """?recon=model: every frame measured with the model's depth and with the town's (rows[i]["oracle"])."""
    orc = lambda r: r["oracle"]
    every = [r for run in runs.values() for r in run["rows"] if "oracle" in r]
    rule = every[0]["rule"]
    print(f"規則: 地面 = {rule['ground']}、比 {rule['ratio']}{'（カメラの画素の細かさで縮める）' if rule['scaled'] else ''}。同じフレームを、町の深度（オラクル）とモデルの深度で測った値\n")
    print(HEAD.format("町、深度"))
    for town, run in runs.items():
        rows = [r for r in run["rows"] if "oracle" in r]
        print(line(f"{TOWNS.get(town, town)}、オラクル", rows, orc))
        print(line(f"{TOWNS.get(town, town)}、モデル", rows))
    print(line("**合計、オラクル**", every, orc))
    print(line("**合計、モデル**", every))
    moving = [r for r in every if r["car"]["v"] >= 0.5]
    print(line("動いているフレーム、オラクル", moving, orc))
    print(line("動いているフレーム、モデル", moving))

    print("\n| 追跡カメラからの距離 | 画素 | 被覆率（全体）オラクル / モデル | 被覆率（いま見える範囲）オラクル / モデル | 見える範囲の割合 |\n|---|---:|---:|---:|---:|")
    lo = 0
    for i in range(3):
        b, o = [r["bands"][i] for r in every], [r["oracle"]["bands"][i] for r in every]
        s, so = (lambda k: sum(x[k] for x in b)), (lambda k: sum(x[k] for x in o))
        print(f"| {lo}〜{b[0]['far']} m | {s('omega'):,} | {pc(so('covered'), so('omega'))} / {pc(s('covered'), s('omega'))} | "
              f"{pc(so('seenCovered'), so('seen'))} / {pc(s('seenCovered'), s('seen'))} | {pc(s('seen'), s('omega'))} |")
        lo = b[0]["far"]

    print("\n" + HEAD.format("モデルの深度での三角形の規則（地面の判定、縁の比）"))
    for name, ja in RULES.items():
        if name in every[0]["variants"]:
            print(line(ja, every, lambda r, n=name: r["variants"][n]))

    print("\n| 深度 | カメラ | 頂点 | 有効な頂点 | 描いた三角形 | 捨てた: 面なし・80 m 超 | 捨てた: 縁 | 捨てた: 望遠と重なる |\n|---|---|---:|---:|---:|---:|---:|---:|")
    for name, get in (("オラクル", orc), ("モデル", lambda r: r)):
        for k in range(len(every[0]["stats"])):
            st = [get(r)["stats"][k] for r in every]
            m = lambda key: np.mean([x[key] for x in st])
            print(f"| {name} | {st[0]['camera']} | {st[0]['vertices']:,} | {m('valid'):,.0f} | {m('triangles'):,.0f} | {m('noSurface'):,.0f} | {m('edge'):,.0f} | {m('overlap'):,.0f} |")


def accum_tables(runs):
    """&accum=1: every frame measured with the current frame alone and with the keyframes (rows[i]["accum"])."""
    acc = lambda r: r["accum"]
    every = [r for run in runs.values() for r in run["rows"] if "accum" in r]
    has_oracle = "oracle" in every[0]
    name = "モデルの深度" if has_oracle else "オラクル"
    print(f"\n蓄積（キーフレーム）の有無。同じフレームを、今のフレームだけ（A）と、キーフレームを足したもの（A+）で測った値。深度は{name}\n")
    print(HEAD.format("町、描くもの"))
    for town, run in runs.items():
        rows = [r for r in run["rows"] if "accum" in r]
        print(line(f"{TOWNS.get(town, town)}、A", rows))
        print(line(f"{TOWNS.get(town, town)}、A+", rows, acc))
    print(line("**合計、A**", every))
    print(line("**合計、A+**", every, acc))

    print("\n| 追跡カメラからの距離 | 画素 | 被覆率（全体）A / A+ | 被覆率（いま見える範囲）A / A+ |\n|---|---:|---:|---:|")
    lo = 0
    for i in range(3):
        b, o = [r["bands"][i] for r in every], [r["accum"]["bands"][i] for r in every]
        s, so = (lambda k: sum(x[k] for x in b)), (lambda k: sum(x[k] for x in o))
        print(f"| {lo}〜{b[0]['far']} m | {s('omega'):,} | {pc(s('covered'), s('omega'))} / {pc(so('covered'), so('omega'))} | "
              f"{pc(s('seenCovered'), s('seen'))} / {pc(so('seenCovered'), so('seen'))} |")
        lo = b[0]["far"]

    print("\n| 町 | キーフレームが足した画素 | うち正しい | うち透け | うち実在しない面 | 今のフレームの面より手前に出た画素（うち正しい） | 作ったキーフレーム | 残した頂点 | うち車や自転車の箱の中 | そういう頂点が 10 個以上のキーフレーム |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    tot = np.zeros(10)
    for town, run in runs.items():
        rows = [r for r in run["rows"] if "accum" in r]
        a = lambda k: sum(r["accum"][k] for r in rows)
        fin = rows[-1]["accum"]
        v = np.array([a("added"), a("addedGood"), a("addedThrough"), a("addedPhantom"), a("before"), fin["made"], fin["vertices"], fin["ghostVertices"], fin["ghostKeyframes"], a("beforeGood")], float)
        tot += v
        drawn = sum(r["drawn"] for r in rows)
        print(f"| {TOWNS.get(town, town)} | {v[0]:,.0f} | {pc(v[1], v[0])} | {pc(v[2], v[0])} | {pc(v[3], v[0])} | {v[4]:,.0f}（{pc(v[9], v[4])}） | {v[5]:.0f} | {v[6]:,.0f} | {v[7]:,.0f}（{pc(v[7], v[6])}） | {v[8]:.0f} |")
    if "variants" in every[0]["accum"]:
        print("\n" + HEAD.format("キーフレームから描くもの（同じフレーム、同じキーフレーム）"))
        print(line("（キーフレームなし: A）", every))
        for name, ja in KEEP.items():
            if name in every[0]["accum"]["variants"]:
                print(line(ja, every, lambda r, n=name: r["accum"]["variants"][n]))
        print()
    drawn = sum(r["drawn"] for r in every)
    print(f"| **合計** | {tot[0]:,.0f} | {pc(tot[1], tot[0])} | {pc(tot[2], tot[0])} | {pc(tot[3], tot[0])} | {tot[4]:,.0f}（{pc(tot[9], tot[4])}） | {tot[5]:.0f} | {tot[6]:,.0f} | {tot[7]:,.0f}（{pc(tot[7], tot[6])}） | {tot[8]:.0f} |")


FILLS = {"holes": "穴・できたて", "holesPrev": "穴・1 回前", "all": "全部・できたて", "allPrev": "全部・1 回前",
         "oracle.holes": "穴・できたて（正解の補完）", "oracle.all": "全部・できたて（正解の補完）"}


def _fill(f, k):
    """f["holes"], or f["oracle"]["holes"] for "oracle.holes"; None when the frame has none."""
    for part in k.split("."):
        f = f.get(part) if f else None
    return f


def fill_tables(runs):
    """&fill=: every frame also measured with the fill (rows[i]["fill"]): this frame's and the one before, the holes
    only (with what the cameras saw) and all of the fill alone."""
    every = [r for run in runs.values() for r in run["rows"] if "fill" in r]
    f0 = every[0]["fill"]
    base = (lambda r: r["accum"]) if "accum" in every[0] else (lambda r: r)
    name = {"oracle": "正解（街そのもの）", "model": "補完のモデル"}.get(f0["mode"], f0["mode"])
    print(f"\n補完（{name}）。同じフレームを、補完なし（A、A+）と、補完を足したもので測った値。補完は運転モデルの呼び出し {f0['every']} 回に 1 回\n")
    print(HEAD.format("町、描くもの"))
    pick = lambda rows, k: [_fill(r["fill"], k) for r in rows if _fill(r["fill"], k)]
    for town, run in list(runs.items()) + [("**合計**", {"rows": every})]:
        rows = [r for r in run["rows"] if "fill" in r]
        ja = TOWNS.get(town, town)
        if "accum" in rows[0]:
            print(line(f"{ja}、A", rows))
        print(line(f"{ja}、A+" if "accum" in rows[0] else f"{ja}、A", rows, base))
        for k, label in FILLS.items():
            if pick(rows, k):
                print(line(f"{ja}、{label}", pick(rows, k)))

    print("\n| 追跡カメラからの距離 | 画素 | 被覆率（全体）: 補完なし / 穴・できたて / 穴・1 回前 / 全部・できたて" + (" / 穴・できたて（正解の補完）" if pick(every, "oracle.holes") else "") + " |\n|---|---:|---|")
    lo = 0
    for i in range(3):
        cols = [[base(r)["bands"][i] for r in every]] + [[x["bands"][i] for x in pick(every, k)] for k in ("holes", "holesPrev", "all", "oracle.holes") if pick(every, k)]
        s = lambda b, k: sum(x[k] for x in b)
        print(f"| {lo}〜{cols[0][0]['far']} m | {s(cols[0], 'omega'):,} | " + " / ".join(pc(s(b, "covered"), s(b, "omega")) for b in cols) + " |")
        lo = cols[0][0]["far"]

    print("\n| 町 | 描くもの | 補完が足した画素（補完なしでは何もない所） | うち正しい | うち透け | うち実在しない面 |\n|---|---|---:|---:|---:|---:|")
    for town, run in list(runs.items()) + [("**合計**", {"rows": every})]:
        rows = [r for r in run["rows"] if "fill" in r]
        for k, label in FILLS.items():
            a = lambda key: sum(x[key] for x in pick(rows, k))
            if pick(rows, k):
                print(f"| {TOWNS.get(town, town)} | {label} | {a('added'):,} | {pc(a('addedGood'), a('added'))} | {pc(a('addedThrough'), a('added'))} | {pc(a('addedPhantom'), a('added'))} |")

    fl = [r["fill"]["flicker"] for r in every if r["fill"].get("flicker")]
    npx = sum(f["pixels"] for f in fl)
    flicker = (f"ちらつき（今の補完と 1 回前の補完を、同じ追跡カメラから 1 枚ずつ描き、どちらも同じ面で埋めた穴の画素の色の差、0〜255）: "
               f"{sum(f['sum'] for f in fl) / npx:.2f}（{npx:,} 画素、{len(fl)} フレーム）。" if npx else "")
    cl = [(r["t"], r["fill"]["colour"]) for r in every if r["fill"].get("colour")]
    if cl and sum(c["pixels"] for _, c in cl):
        part = lambda cs: (f"L1 {sum(c['sum'] for c in cs) / max(1, sum(c['pixels'] for c in cs)):.2f}、鮮明さ {sum(c['step'] for c in cs) / max(1e-9, sum(c['stepTown'] for c in cs)):.2f}")
        tmax = max(t for t, _ in cl)
        flicker += (f"ページの中の色（補完と、同じフレームの正解の補完を 1 枚ずつ描き、どちらも同じ面で埋めた穴の画素で。色の L1 は 0〜255、鮮明さは隣の画素との差の比）: "
                    f"全体で {part([c for _, c in cl])}、最後の 20 秒で {part([c for t, c in cl if t > tmax - 20])}（{sum(c['pixels'] for _, c in cl):,} 画素、{len(cl)} フレーム）。")
    m = lambda get: np.mean([get(r["fill"]) for r in every])
    aged = [r["fill"]["age"] for r in every if r["fill"].get("age")]
    print(f"\n補完のメッシュ（{len(every)} フレームの平均）: 有効な頂点 {m(lambda f: f['stats']['valid']):,.0f} / {f0['stats']['vertices']:,}、三角形 {m(lambda f: f['stats']['triangles']):,.0f}、"
          f"縁で捨てた三角形 {m(lambda f: f['stats']['edge']):,.0f}、穴にかかるマス {m(lambda f: f['stats']['holeCells']):,.0f}、作る時間 {m(lambda f: f['ms']):.1f} ms。"
          + (f"「1 回前」の補完は、平均でモデルのフレーム {np.mean([a['frames'] for a in aged]):.1f} 回前、{np.mean([a['metres'] for a in aged]):.2f} m 手前のもの。" if aged else "")
          + flicker
          + (f"補完のモデル: 入力を作る時間 {m(lambda f: f['prepMs']):.1f} ms、推論（往復）{m(lambda f: f['inferMs']):.1f} ms。" if "inferMs" in f0 else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--repeat", default=None)
    args = ap.parse_args()
    runs = load(args.dir)
    has_accum = any("accum" in r for run in runs.values() for r in run["rows"])
    has_fill = any("fill" in r for run in runs.values() for r in run["rows"])
    if any("oracle" in r for run in runs.values() for r in run["rows"]):
        model_tables(runs)
        if has_accum:
            accum_tables(runs)
        if has_fill:
            fill_tables(runs)
        return
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

    if has_accum:
        accum_tables(runs)
    if has_fill:
        fill_tables(runs)
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
