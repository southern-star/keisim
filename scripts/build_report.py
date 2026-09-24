#!/usr/bin/env python3
"""Build docs/report/index.html (results page) from evaluation JSONs, clips and images."""
from __future__ import annotations

import glob
import html
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "report")

AGENTS = [("expert", "特権エキスパート", "privileged (正解情報を使用)"),
          ("keipilot_bc", "KeiPilot · BC", "カメラのみ・模倣学習"),
          ("keipilot_dagger", "KeiPilot · DAgger", "カメラのみ・+ オンポリシー学習")]
SUITES = [("test", "未見の街 A", "town 1000–1004 · 20 ルート"),
          ("test2", "未見の街 B", "town 1005–1009 · 20 ルート"),
          ("train", "既知の街・新ルート", "学習時の街 5 つ · 20 ルート"),
          ("test_dense", "高密度交通", "未見の街 A · 車両/歩行者 ~2 倍")]


def load(name):
    p = os.path.join(ROOT, "runs", "eval", name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def bar_chart(ds):
    """Grouped horizontal bars of DS per suite. ds[suite][agent] -> value or None."""
    W, left, right = 640, 150, 56
    bh, gap, ggap, top = 13, 4, 18, 26
    n_ag = len(AGENTS)
    gh = n_ag * bh + (n_ag - 1) * gap
    H = top + len(SUITES) * gh + (len(SUITES) - 1) * ggap + 30
    sx = lambda v: left + (W - left - right) * v
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Driving Score の比較" class="chart">']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{H - 22}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{H - 6}" class="tick" text-anchor="middle">{t:.2f}</text>')
    y = top
    for key, label, _ in SUITES:
        out.append(f'<text x="{left - 12}" y="{y + gh / 2 + 4:.1f}" class="cat" text-anchor="end">{html.escape(label)}</text>')
        for i, (ak, an, _) in enumerate(AGENTS):
            v = ds.get(key, {}).get(ak)
            yy = y + i * (bh + gap)
            if v is None:
                out.append(f'<text x="{left + 4}" y="{yy + bh - 2}" class="val muted">n/a</text>')
                continue
            out.append(f'<rect x="{left}" y="{yy}" width="{sx(v) - left:.1f}" height="{bh}" rx="2" class="bar a{i}"/>')
            out.append(f'<text x="{sx(v) + 5:.1f}" y="{yy + bh - 2}" class="val">{v:.3f}</text>')
        y += gh + ggap
    out.append("</svg>")
    return "\n".join(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    # ---------- data
    res = {}
    for ak, _, _ in AGENTS:
        for sk, _, _ in SUITES:
            d = load(f"{ak}_{sk}")
            if d:
                res[(ak, sk)] = d["summary"]
    ds = {sk: {ak: res[(ak, sk)]["DS"] for ak, _, _ in AGENTS if (ak, sk) in res} for sk, _, _ in SUITES}
    perc = {k: (json.load(open(p)) if os.path.exists(p) else None) for k, p in
            (("bc", os.path.join(ROOT, "runs/eval/perception_bc.json")),
             ("dagger", os.path.join(ROOT, "runs/eval/perception_dagger.json")))}
    speed = json.load(open(os.path.join(ROOT, "runs/eval/speed.json"))) if os.path.exists(
        os.path.join(ROOT, "runs/eval/speed.json")) else {}

    # headline numbers: DAgger on the two unseen suites
    def mean_of(agent, key):
        vals = [res[(agent, s)][key] for s in ("test", "test2") if (agent, s) in res]
        return sum(vals) / len(vals) if vals else None

    head_ag = "keipilot_dagger" if ("keipilot_dagger", "test") in res else "keipilot_bc"
    h_ds, h_rc, h_sr = mean_of(head_ag, "DS"), mean_of(head_ag, "RC"), mean_of(head_ag, "success_rate")
    km = sum(res[(head_ag, s)]["km"] for s in ("test", "test2") if (head_ag, s) in res)
    coll = sum(res[(head_ag, s)]["infraction_counts"].get(k, 0) for s in ("test", "test2") if (head_ag, s) in res
               for k in ("vehicle", "pedestrian", "static"))

    # ---------- media
    media = []
    for src in sorted(glob.glob(os.path.join(ROOT, "docs", "clips", "*.mp4"))):
        name = os.path.basename(src)
        shutil.copy(src, os.path.join(OUT, name))
        poster = src[:-4] + ".jpg"
        if os.path.exists(poster):
            shutil.copy(poster, os.path.join(OUT, os.path.basename(poster)))
        media.append(name)
    for img in ("gallery.jpg", "town_overview.jpg"):
        p = os.path.join(ROOT, "docs", img)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(OUT, img))
    captions = json.load(open(os.path.join(ROOT, "docs", "clips", "captions.json"))) if os.path.exists(
        os.path.join(ROOT, "docs", "clips", "captions.json")) else {}

    # ---------- tables
    rows = []
    for ak, an, sub in AGENTS:
        cells = []
        for sk, _, _ in SUITES:
            s = res.get((ak, sk))
            if not s:
                cells.append('<td class="num muted">—</td>')
                continue
            cls = "good" if s["DS"] >= 0.95 else ("warn" if s["DS"] >= 0.85 else "bad")
            cells.append(f'<td class="num"><span class="ds {cls}">{s["DS"]:.3f}</span></td>')
        rows.append(f'<tr><th scope="row"><span class="agent">{an}</span><span class="sub">{sub}</span></th>{"".join(cells)}</tr>')
    ds_table = "\n".join(rows)

    det = []
    for ak, an, _ in AGENTS:
        for sk, sl, _ in SUITES:
            s = res.get((ak, sk))
            if not s:
                continue
            ipk = s["infractions_per_km"]
            st = ", ".join(f"{k} {v}" for k, v in sorted(s["status"].items()))
            det.append(
                f'<tr><td>{an}</td><td>{sl}</td><td class="num">{s["DS"]:.3f}</td><td class="num">{s["RC"]:.3f}</td>'
                f'<td class="num">{s["success_rate"] * 100:.0f}%</td><td class="num">{s["km"]:.1f}</td>'
                f'<td class="num">{ipk.get("red_light", 0):.2f}</td>'
                f'<td class="num">{ipk.get("vehicle", 0) + ipk.get("pedestrian", 0) + ipk.get("static", 0):.2f}</td>'
                f'<td class="num">{s["avg_speed_kmh"]:.1f}</td><td class="st">{html.escape(st)}</td></tr>')
    det_table = "\n".join(det)

    def pc(k, key, d=3, pct=False):
        p = perc.get(k)
        if not p:
            return "—"
        v = p[key]
        return f"{v * 100:.1f}%" if pct else f"{v:.{d}f}"

    iou_rows = ""
    if perc.get("dagger") or perc.get("bc"):
        p = perc.get("dagger") or perc.get("bc")
        names = {"road": "道路", "marking": "路面標示", "sidewalk": "歩道", "vehicle": "車両", "pedestrian": "歩行者",
                 "tl_red": "信号 赤", "tl_yellow": "信号 黄", "tl_green": "信号 青", "building": "建物", "vegetation": "植栽",
                 "pole": "ポール", "terrain": "地面", "sky": "空"}
        items = sorted(p["iou"].items(), key=lambda kv: -kv[1])
        iou_rows = "".join(
            f'<div class="iou"><span class="iou-name">{names.get(k, k)}</span><span class="iou-bar"><i style="width:{v * 100:.1f}%"></i></span>'
            f'<span class="iou-val">{v:.2f}</span></div>' for k, v in items)

    vids = []
    for m in media:
        cap = captions.get(m, m)
        vids.append(f'<figure class="clip"><video src="{m}" poster="{m[:-4]}.jpg" controls muted playsinline loop preload="metadata"></video>'
                    f'<figcaption>{html.escape(cap)}</figcaption></figure>')
    vids_html = "\n".join(vids) if vids else '<p class="muted">(動画なし)</p>'

    sp = speed
    page = TEMPLATE
    subs = {
        "{{H_DS}}": fmt(h_ds, 3), "{{H_RC}}": f"{h_rc * 100:.1f}%" if h_rc is not None else "—",
        "{{H_SR}}": f"{h_sr * 100:.0f}%" if h_sr is not None else "—", "{{H_KM}}": f"{km:.1f}",
        "{{H_COLL}}": str(coll), "{{HEAD_AG}}": "DAgger" if head_ag == "keipilot_dagger" else "BC",
        "{{SIM_X}}": fmt(sp.get("sim_x"), 1), "{{CAM_X}}": fmt(sp.get("cam_x"), 1),
        "{{SIM_SPS}}": fmt(sp.get("sim_sps"), 0), "{{CAM_SPS}}": fmt(sp.get("cam_sps"), 0),
        "{{TOWN_S}}": fmt(sp.get("town_s"), 2), "{{ASSET_S}}": fmt(sp.get("asset_s"), 2),
        "{{INF_MS}}": fmt(sp.get("inf_ms"), 1),
        "{{DS_TABLE}}": ds_table, "{{DET_TABLE}}": det_table, "{{CHART}}": bar_chart(ds), "{{VIDEOS}}": vids_html,
        "{{P_ADE_BC}}": pc("bc", "ADE_m"), "{{P_ADE_DG}}": pc("dagger", "ADE_m"),
        "{{P_SPD_BC}}": pc("bc", "speed_MAE"), "{{P_SPD_DG}}": pc("dagger", "speed_MAE"),
        "{{P_TL_BC}}": pc("bc", "tl_acc", pct=True), "{{P_TL_DG}}": pc("dagger", "tl_acc", pct=True),
        "{{P_STOP_BC}}": pc("bc", "stop_acc", pct=True), "{{P_STOP_DG}}": pc("dagger", "stop_acc", pct=True),
        "{{P_MIOU_BC}}": pc("bc", "seg_mIoU"), "{{P_MIOU_DG}}": pc("dagger", "seg_mIoU"),
        "{{IOU_ROWS}}": iou_rows,
        "{{P_TLR_BC}}": (f"{perc['bc']['tl_recall']['red'] * 100:.1f}%" if perc.get("bc") else "—"),
        "{{P_TLR_DG}}": (f"{perc['dagger']['tl_recall']['red'] * 100:.1f}%" if perc.get("dagger") else "—"),
        "{{P_TLG_BC}}": (f"{perc['bc']['tl_recall']['green'] * 100:.1f}%" if perc.get("bc") else "—"),
        "{{P_TLG_DG}}": (f"{perc['dagger']['tl_recall']['green'] * 100:.1f}%" if perc.get("dagger") else "—"),
        "{{BC_RED}}": str(sum(res[("keipilot_bc", s)]["infraction_counts"].get("red_light", 0) for s in ("test", "test2")
                              if ("keipilot_bc", s) in res)),
        "{{DG_RED}}": str(sum(res[("keipilot_dagger", s)]["infraction_counts"].get("red_light", 0) for s in ("test", "test2")
                              if ("keipilot_dagger", s) in res)),
    }
    for k, v in subs.items():
        page = page.replace(k, v)
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(page)
    print("wrote", os.path.join(OUT, "index.html"), "media:", media)


TEMPLATE = r"""<title>KeiSim × KeiPilot</title>
<meta name="description" content="軽量クローズドループ自動運転環境: 手続き生成の街シミュレータとカメラ単眼 E2E モデル">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+JP:wght@400;500;600&family=Zen+Kaku+Gothic+New:wght@700;900&display=swap">
<style>
:root{
  --bg:#EFF0EC; --surface:#FBFBF9; --ink:#1B1F23; --ink-2:#4F5860; --ink-3:#7A838B; --line:#D5D9D6;
  --asphalt:#2B3035; --paint:#FFFFFF; --kei:#F0B90B; --kei-ink:#191B1D; --kei-soft:#FBEFC7;
  --go:#0B8F6C; --caution:#B87700; --stop:#C7402C;
  --go-bg:#DCF1EA; --caution-bg:#F8EBD2; --stop-bg:#F7DFDA;
  --bar0:#8C969E; --bar1:#3E4A55; --bar2:#E0A800;
  --code:#23282D; --code-ink:#E8EBED;
  --display:"Zen Kaku Gothic New","Hiragino Kaku Gothic ProN","Noto Sans JP","Yu Gothic",sans-serif;
  --body:"IBM Plex Sans JP","Hiragino Sans","Noto Sans JP","Yu Gothic",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --bg:#14171A; --surface:#1C2024; --ink:#E6E9EA; --ink-2:#A7B0B7; --ink-3:#7F8991; --line:#2E353A;
    --asphalt:#0E1012; --kei:#F3C434; --kei-soft:#3A3217;
    --go:#3CCB9E; --caution:#F0B444; --stop:#F2735F;
    --go-bg:#15352C; --caution-bg:#3A2E14; --stop-bg:#3B1E19;
    --bar0:#7C868E; --bar1:#9FB0BF; --bar2:#F3C434;
    --code:#0F1214; --code-ink:#DDE2E5;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --bg:#14171A; --surface:#1C2024; --ink:#E6E9EA; --ink-2:#A7B0B7; --ink-3:#7F8991; --line:#2E353A;
  --asphalt:#0E1012; --kei:#F3C434; --kei-soft:#3A3217;
  --go:#3CCB9E; --caution:#F0B444; --stop:#F2735F;
  --go-bg:#15352C; --caution-bg:#3A2E14; --stop-bg:#3B1E19;
  --bar0:#7C868E; --bar1:#9FB0BF; --bar2:#F3C434;
  --code:#0F1214; --code-ink:#DDE2E5;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:15px;line-height:1.75;margin:0}
.wrap{max-width:1080px;margin:0 auto;padding-inline:20px;padding-block:28px 72px}
h1,h2,h3{font-family:var(--display);line-height:1.25;text-wrap:balance;margin:0}
h1{font-size:clamp(34px,6vw,56px);font-weight:900;letter-spacing:.01em}
h2{font-size:clamp(22px,3vw,28px);font-weight:900;margin-bottom:6px}
h3{font-size:17px;font-weight:700;margin-bottom:4px}
p{margin:0}
.prose{max-width:68ch}
div.prose{display:grid;gap:12px}
.muted{color:var(--ink-3)}
a{color:inherit;text-decoration-color:var(--kei);text-decoration-thickness:2px;text-underline-offset:3px}
code,.mono{font-family:var(--mono);font-size:.92em}
section{margin-top:64px;display:grid;gap:18px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}

/* masthead: kei-car yellow plate */
.masthead{display:grid;grid-template-columns:auto 1fr;gap:26px;align-items:center;padding-bottom:26px;border-bottom:1px solid var(--line)}
.plate{width:118px;aspect-ratio:2/1;max-width:100%;background:var(--kei);border-radius:10px;display:grid;place-items:center;
  box-shadow:inset 0 0 0 3px color-mix(in srgb,var(--kei-ink) 70%,transparent),inset 0 0 0 6px var(--kei)}
.plate span{font-family:var(--display);font-weight:900;font-size:40px;color:var(--kei-ink);line-height:1}
.lede{font-size:17px;color:var(--ink-2);max-width:62ch;margin-top:10px}
.lede strong{color:var(--ink);background:linear-gradient(transparent 62%,var(--kei-soft) 62%)}
@media (max-width:620px){.masthead{grid-template-columns:1fr}.plate{width:96px}}

/* key figures */
.figures{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:6px;overflow:hidden;margin-top:28px}
.fig{background:var(--surface);padding:16px 18px;display:grid;gap:2px;align-content:start}
.fig .n{font-family:var(--mono);font-weight:500;font-size:clamp(24px,3.4vw,34px);font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.fig .l{font-size:13px;color:var(--ink-2);line-height:1.5}
.fig .n small{font-size:.5em;color:var(--ink-3);margin-left:2px}
@media (max-width:760px){.figures{grid-template-columns:repeat(2,minmax(0,1fr))}}

/* road-lane divider under section headers */
.lane{height:6px;background:repeating-linear-gradient(90deg,var(--ink) 0 22px,transparent 22px 36px);opacity:.14;border-radius:2px;max-width:220px}

/* videos */
.clips{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:18px}
.clip{margin:0;display:grid;gap:8px}
.clip video{width:100%;max-width:100%;aspect-ratio:2/1;background:var(--asphalt);border-radius:6px;display:block}
.clip figcaption{font-size:13px;color:var(--ink-2)}

/* tables */
.tablebox{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:10px 14px;text-align:left;border-bottom:1px solid var(--line);vertical-align:middle;white-space:nowrap}
thead th{font-size:12px;font-weight:600;color:var(--ink-2);background:color-mix(in srgb,var(--surface) 70%,var(--bg))}
thead th .sub,th .sub{display:block;font-weight:400;font-size:11.5px;color:var(--ink-3)}
tbody tr:last-child>*{border-bottom:0}
.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
.agent{display:block;font-weight:600}
.ds{display:inline-block;min-width:64px;text-align:center;padding:2px 8px;border-radius:999px;font-weight:500}
.ds.good{background:var(--go-bg);color:var(--go)}
.ds.warn{background:var(--caution-bg);color:var(--caution)}
.ds.bad{background:var(--stop-bg);color:var(--stop)}
.st{font-family:var(--mono);font-size:12px;color:var(--ink-2)}
details{border:1px solid var(--line);border-radius:6px;background:var(--surface)}
details>summary{cursor:pointer;padding:10px 14px;font-weight:600;font-size:14px}
details[open]>summary{border-bottom:1px solid var(--line)}
details .tablebox{border:0;border-radius:0}

/* chart */
.chartbox{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px 14px 6px;overflow-x:auto}
.chart{width:100%;min-width:520px;height:auto;display:block}
.chart .grid{stroke:var(--line);stroke-width:1}
.chart .tick{fill:var(--ink-3);font:11px var(--mono)}
.chart .cat{fill:var(--ink);font:600 12.5px var(--body)}
.chart .val{fill:var(--ink-2);font:11px var(--mono)}
.chart .muted{fill:var(--ink-3)}
.chart .a0{fill:var(--bar0)} .chart .a1{fill:var(--bar1)} .chart .a2{fill:var(--bar2)}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--ink-2)}
.legend i{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:-1px}

/* two-column blocks */
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:22px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:16px 18px;display:grid;gap:8px;align-content:start}
.card p,.card li{font-size:14px;color:var(--ink-2)}
.card ul{margin:0;padding-left:1.1em;display:grid;gap:4px}
.kv{display:grid;grid-template-columns:auto 1fr auto;gap:4px 14px;font-size:14px;align-items:baseline}
.kv .k{color:var(--ink-2)} .kv .v{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}
.kv .arrow{color:var(--ink-3);font-family:var(--mono);font-size:12px}

.iou{display:grid;grid-template-columns:7em 1fr 3.2em;gap:10px;align-items:center;font-size:13px}
.iou-bar{height:8px;background:color-mix(in srgb,var(--line) 70%,transparent);border-radius:4px;overflow:hidden}
.iou-bar i{display:block;height:100%;background:var(--bar1)}
.iou-val{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums;color:var(--ink-2)}

/* pipeline diagram */
.diagram{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px;overflow-x:auto}
.diagram svg{width:100%;min-width:640px;height:auto;display:block}
.diagram .box{fill:var(--bg);stroke:var(--line);stroke-width:1.2}
.diagram .box.hot{fill:var(--kei-soft);stroke:var(--kei)}
.diagram .t{fill:var(--ink);font:600 13px var(--body)}
.diagram .s{fill:var(--ink-2);font:11.5px var(--body)}
.diagram .ar{stroke:var(--ink-3);stroke-width:1.4;fill:none}
.diagram .arh{fill:var(--ink-3)}
.diagram .loop{stroke:var(--kei);stroke-width:2;fill:none;stroke-dasharray:5 4}
.diagram .loopt{fill:var(--caution);font:600 12px var(--body)}

.gallery{width:100%;max-width:100%;border-radius:6px;border:1px solid var(--line);display:block}
pre{background:var(--code);color:var(--code-ink);border-radius:6px;padding:14px 16px;overflow-x:auto;font-size:13px;line-height:1.6;margin:0}
pre .c{color:#8C979F}
.note{border-left:3px solid var(--kei);padding:4px 0 4px 14px;color:var(--ink-2);font-size:14px;max-width:72ch}
footer{margin-top:72px;padding-top:18px;border-top:1px solid var(--line);font-size:12.5px;color:var(--ink-3)}
:focus-visible{outline:2px solid var(--kei);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style>

<main class="wrap">
  <header class="masthead">
    <div class="plate" aria-hidden="true"><span>軽</span></div>
    <div>
      <p class="eyebrow">軽量クローズドループ自動運転環境 · 2026-09-23</p>
      <h1>KeiSim × KeiPilot</h1>
      <p class="lede">見た目のリアルさを追う競争をやめて、<strong>シミュレータとモデルを一体で設計</strong>しました。numpy と OpenCV だけで動く街シミュレータと、その世界で特権エキスパートから学んだカメラ単眼モデル (14.9M パラメータ) の組み合わせです。</p>
    </div>
  </header>

  <div class="figures" role="list">
    <div class="fig" role="listitem"><span class="n">{{H_DS}}</span><span class="l">Driving Score<br>未見の街 40 ルート ({{HEAD_AG}})</span></div>
    <div class="fig" role="listitem"><span class="n">{{H_RC}}</span><span class="l">ルート完走率<br>{{H_KM}} km 中の衝突 {{H_COLL}} 件</span></div>
    <div class="fig" role="listitem"><span class="n">{{SIM_X}}<small>×</small></span><span class="l">実時間比 (1 プロセス)<br>カメラ描画込みで {{CAM_X}}×</span></div>
    <div class="fig" role="listitem"><span class="n">14.9<small>M</small></span><span class="l">モデルのパラメータ数<br>RTX 3060 で 1 フレーム {{INF_MS}} ms</span></div>
  </div>

  <section id="drive">
    <div class="lane" aria-hidden="true"></div>
    <h2>未見の街を走る</h2>
    <p class="prose muted">学習に一度も使っていない街でのクローズドループ走行です。左上がモデルへの入力 (カメラ 320×160)、緑がエキスパートの経路、マゼンタがモデルの予測経路。左下はモデル自身が推定したセマンティックセグメンテーション、右は BEV (評価用の真値) です。</p>
    <div class="clips">
{{VIDEOS}}
    </div>
  </section>

  <section id="bench">
    <div class="lane" aria-hidden="true"></div>
    <h2>クローズドループ評価</h2>
    <p class="prose">各スイートは <code>(town_seed, episode_seed)</code> で完全に固定された 600 m ルート 20 本です (地図・経路・交通・歩行者・信号位相・天候まで再現)。Driving Score = ルート完走率 × 違反ペナルティ (CARLA Leaderboard と同じ形。車両衝突 ×0.60、歩行者 ×0.50、静止物 ×0.65、信号無視 ×0.70)。衝突した時点でルートは打ち切りです。</p>
    <div class="tablebox">
      <table>
        <thead><tr><th>エージェント</th>
          <th class="num">未見の街 A<span class="sub">town 1000–1004</span></th>
          <th class="num">未見の街 B<span class="sub">town 1005–1009</span></th>
          <th class="num">既知の街・新ルート<span class="sub">学習時の街</span></th>
          <th class="num">高密度交通<span class="sub">車両/歩行者 ~2 倍</span></th></tr></thead>
        <tbody>
{{DS_TABLE}}
        </tbody>
      </table>
    </div>
    <div class="chartbox">
      <div class="legend"><span><i style="background:var(--bar0)"></i>特権エキスパート</span><span><i style="background:var(--bar1)"></i>KeiPilot · BC</span><span><i style="background:var(--bar2)"></i>KeiPilot · DAgger</span></div>
{{CHART}}
    </div>
    <details>
      <summary>詳細 (RC / 成功率 / 違反 per km / 平均速度 / 終了状態)</summary>
      <div class="tablebox">
        <table>
          <thead><tr><th>エージェント</th><th>スイート</th><th class="num">DS</th><th class="num">RC</th><th class="num">成功</th><th class="num">km</th><th class="num">信号無視/km</th><th class="num">衝突/km</th><th class="num">平均 km/h</th><th>終了状態</th></tr></thead>
          <tbody>
{{DET_TABLE}}
          </tbody>
        </table>
      </div>
    </details>
  </section>

  <section id="why">
    <div class="lane" aria-hidden="true"></div>
    <h2>なぜ「認識できる」のか</h2>
    <div class="prose">
      <p>CARLA は重く、MetaDrive は軽いものの、実データや CARLA で学んだモデルから見ると描画の分布が違います (ドメインギャップ)。そのため、車も信号も「見えない」。問題はシミュレータ単体ではなく、<strong>モデルと世界の組み合わせ</strong>にあります。</p>
      <p>KeiSim は、特権エキスパートとピクセル単位で正確なラベルを最初から持っています。KeiPilot はその世界のカメラ画像だけで学び、閉ループで評価されます。手続き生成なので、学習にない街での汎化もきちんと測れます。</p>
    </div>
    <div class="diagram">
      <svg viewBox="0 0 980 250" role="img" aria-label="KeiSim と KeiPilot のパイプライン">
        <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="arh"/></marker></defs>
        <rect x="10" y="30" width="150" height="86" rx="6" class="box"/>
        <text x="24" y="56" class="t">手続き生成の街</text>
        <text x="24" y="76" class="s">ジッタ格子・カーブ</text>
        <text x="24" y="94" class="s">信号交差点・横断歩道</text>
        <rect x="200" y="30" width="160" height="86" rx="6" class="box"/>
        <text x="214" y="56" class="t">World</text>
        <text x="214" y="76" class="s">IDM 交通・歩行者</text>
        <text x="214" y="94" class="s">スプリット現示信号</text>
        <rect x="400" y="30" width="170" height="86" rx="6" class="box"/>
        <text x="414" y="56" class="t">CPU レンダラ</text>
        <text x="414" y="76" class="s">地面=厳密ホモグラフィ</text>
        <text x="414" y="94" class="s">RGB + セマンティック</text>
        <rect x="610" y="30" width="170" height="86" rx="6" class="box hot"/>
        <text x="624" y="56" class="t">KeiPilot</text>
        <text x="624" y="76" class="s">ResNet-18 + Transformer</text>
        <text x="624" y="94" class="s">経路・目標速度・信号</text>
        <rect x="820" y="30" width="150" height="86" rx="6" class="box"/>
        <text x="834" y="56" class="t">制御 → 閉ループ</text>
        <text x="834" y="76" class="s">Pure Pursuit + PI</text>
        <text x="834" y="94" class="s">DS / RC / 違反</text>
        <path d="M160 73 H196" class="ar" marker-end="url(#ah)"/>
        <path d="M360 73 H396" class="ar" marker-end="url(#ah)"/>
        <path d="M570 73 H606" class="ar" marker-end="url(#ah)"/>
        <path d="M780 73 H816" class="ar" marker-end="url(#ah)"/>
        <rect x="200" y="160" width="370" height="62" rx="6" class="box"/>
        <text x="214" y="186" class="t">特権エキスパート (正解情報)</text>
        <text x="214" y="206" class="s">中心線経路 + 距離ベース目標速度 → 1 フレームから学習可能なラベル</text>
        <path d="M280 116 V156" class="ar" marker-end="url(#ah)"/>
        <path d="M570 191 H690 V120" class="ar" marker-end="url(#ah)"/>
        <text x="600" y="184" class="s">教師信号</text>
        <path d="M895 116 V238 H120 V120" class="loop" marker-end="url(#ah)"/>
        <text x="128" y="234" class="loopt">DAgger: モデルが運転した状態をエキスパートがラベル付け</text>
      </svg>
    </div>
    <div class="cols">
      <div class="card"><h3>学習できるラベル設計</h3><p>エキスパートの目標速度は「見えるものまでの距離」だけで決まります (前方車・歩行者・赤信号の停止線・カーブ)。1 フレームの画像から原理的に推定でき、速度入力が不要なので慣性問題も起きません。</p></div>
      <div class="card"><h3>誤差蓄積への対策</h3><p>DART 型ノイズ注入 (ラベルは常に修正後の計画)、任意姿勢からレンダリングする仮想カメラずらし (±1.2 m / ±9°)、DAgger。モデルとエキスパートは同じ制御器を共有するので、予測が正しければエキスパートと同じに走ります。</p></div>
      <div class="card"><h3>補助タスクで「見える」を担保</h3><p>RGB と同じ幾何で描いたセグメンテーションを補助教師に使います。信号灯は赤・黄・青を別クラスにしたので、信号の認識が画素単位で教師付けされます。</p></div>
    </div>
  </section>

  <section id="fix">
    <div class="lane" aria-hidden="true"></div>
    <h2>見つけた弱点と対処</h2>
    <div class="prose">
      <p>BC モデルは未見の街 40 ルートをすべて衝突なしで完走しましたが、<strong>信号無視が {{BC_RED}} 件</strong>ありました。軌跡を追うと、信号の認識自体は正しい (P(赤) ≈ 1.0) ものの、停止直前の 1〜2 m で距離を読み違えて止まり切れていませんでした。停止線が画面の下端から消えると、学習データ上は「停止線を越えた = 進んでよい」状態と見分けがつかず、再加速していたのです。</p>
      <p>対策は 3 つです。停止目標を線の 2.5 m 手前にして線が画面に残るようにしたこと、停止直前の速度プロファイルを √ 型から線形の徐行に変えて距離誤差への感度を下げたこと、線を少し越えた状態でも赤なら止まるようにエキスパートを直したことです。旧データは同じ式で再ラベルし、DAgger でモデル自身が陥る状態を学ばせました。その結果、信号無視は <strong>{{DG_RED}} 件</strong>になりました。</p>
      <p class="note">残っている信号無視の多くは黄信号のジレンマです。エキスパートは黄の残り時間 (特権情報) と自車速度から止まるか進むかを決めますが、モデルは速度を入力に持たない 1 フレームなので、この判断を原理的に再現できません。次の一手は、自車速度 (ドロップアウト付き) か 2〜3 フレームの時間文脈を入力に加えることです。遠方の信号を隣の灯器と取り違えるケースもあり、ここは解像度を上げるか信号用の望遠クロップで改善できる見込みです。</p>
    </div>
    <div class="cols">
      <div class="card">
        <h3>オフライン指標 (未見の街 6,000 フレーム)</h3>
        <p class="muted" style="font-size:12.5px">ラベル規約に依存しない指標のみ。信号は「自車が従うべき信号」の 4 クラス分類 (赤/黄/青/なし)。</p>
        <div class="kv">
          <span class="k">経路 ADE</span><span class="arrow">BC → DAgger</span><span class="v">{{P_ADE_BC}} → {{P_ADE_DG}} m</span>
          <span class="k">信号状態の認識</span><span class="arrow"></span><span class="v">{{P_TL_BC}} → {{P_TL_DG}}</span>
          <span class="k">　赤信号の再現率</span><span class="arrow"></span><span class="v">{{P_TLR_BC}} → {{P_TLR_DG}}</span>
          <span class="k">　青信号の再現率</span><span class="arrow"></span><span class="v">{{P_TLG_BC}} → {{P_TLG_DG}}</span>
          <span class="k">セグメンテーション mIoU</span><span class="arrow"></span><span class="v">{{P_MIOU_BC}} → {{P_MIOU_DG}}</span>
        </div>
      </div>
      <div class="card">
        <h3>クラス別 IoU (モデルが「見ている」もの)</h3>
        <div style="display:grid;gap:6px">{{IOU_ROWS}}</div>
      </div>
    </div>
  </section>

  <section id="sim">
    <div class="lane" aria-hidden="true"></div>
    <h2>KeiSim の中身</h2>
    <div class="cols" style="align-items:start">
      <figure style="margin:0;display:grid;gap:8px"><img class="gallery" src="town_overview.jpg" alt="未見の街 1001 の上面図: ジッタ格子の道路網、信号交差点、横断歩道、建物と街路樹" loading="lazy">
        <figcaption class="muted" style="font-size:13px">未見の街 1001 の上面図 (382 m × 317 m、交差点 20・うち信号 14)。シード 1 つで道路網・建物・色合いまで決まる。</figcaption></figure>
      <figure style="margin:0;display:grid;gap:8px"><img class="gallery" src="gallery.jpg" alt="未見の街 6 つのカメラ画像と、同じ幾何から生成したセマンティックラベル" loading="lazy">
        <figcaption class="muted" style="font-size:13px">カメラ画像 (上段) と、同じ幾何で描いたセマンティックラベル (下段)。晴れ・夕焼け・曇り・霧。</figcaption></figure>
    </div>
    <div class="cols">
      <div class="card"><h3>日本仕様の街</h3><ul>
        <li>左側通行・交差点奥の横型信号 (青・黄・赤)</li>
        <li>停止線 + 横断歩道、ジッタ格子の十字/T 字/斜め交差、S 字・円弧カーブ</li>
        <li>軽自動車サイズを含む車種、歩道を歩き乱横断する歩行者</li>
        <li>流入方向ごとのスプリット現示で原理的に衝突しない信号制御、交差点の詰まりを防ぐ「箱に入らない」ルール</li>
      </ul></div>
      <div class="card"><h3>CPU レンダラ</h3><ul>
        <li>地面は上面テクスチャを平面誘導ホモグラフィで透視投影 (距離帯ごとにミップ選択)</li>
        <li>物体は低ポリ凸パーツをペインタ法で描画 (背面カリング・近平面クリップ・ランバート陰影・フォグ)</li>
        <li>ブレーキランプ・窓・信号灯はデカール、晴れ/曇り/夕焼け/霧 + ランダム化</li>
        <li>任意姿勢からの再描画 = 仮想カメラ拡張がタダ</li>
      </ul></div>
      <div class="card"><h3>速度 (i5-10400F, 1 プロセス)</h3>
        <div class="kv">
          <span class="k">街の生成</span><span></span><span class="v">{{TOWN_S}} s</span>
          <span class="k">テクスチャ + メッシュ</span><span></span><span class="v">{{ASSET_S}} s</span>
          <span class="k">物理 + 交通 + エキスパート</span><span></span><span class="v">{{SIM_SPS}} step/s</span>
          <span class="k">+ カメラ 320×160 (SSAA×2)</span><span></span><span class="v">{{CAM_SPS}} step/s</span>
          <span class="k">決定論的リプレイ</span><span></span><span class="v">OK</span>
        </div>
        <p class="muted" style="font-size:12.5px">1 step = 0.1 s (物理 20 Hz / 制御 10 Hz)。ゲームエンジンも GPU も不要です。</p>
      </div>
    </div>
  </section>

  <section id="run">
    <div class="lane" aria-hidden="true"></div>
    <h2>動かし方</h2>
<pre><span class="c"># git clone https://github.com/southern-star/keisim.git &amp;&amp; cd keisim &amp;&amp; bash setup.sh</span>
gh release download v0.1.0 -R southern-star/keisim -p keipilot.pt -D runs
uv run scripts/demo.py --agent model --ckpt runs/keipilot.pt --town 1001 --episode 3 --show
uv run scripts/evaluate.py --agent model --ckpt runs/keipilot.pt --suite test --videos 3

<span class="c"># 再現: データ収集 → 学習 → DAgger → 再学習</span>
uv run scripts/collect.py --out data/expert --frames 160000 --workers 10
uv run scripts/train.py --data data/expert --out runs/keipilot_bc --epochs 12
uv run scripts/collect.py --mode dagger --ckpt runs/keipilot_bc/last.pt --out data/dagger1 --frames 60000 --workers 6
uv run scripts/train.py --data data/expert data/expert_ped data/dagger1 --init runs/keipilot_bc/last.pt --out runs/keipilot_dagger --epochs 5</pre>
<pre><span class="c"># Gym 風 API</span>
from keisim.env import KeiEnv
env = KeiEnv()
obs = env.reset(town_seed=1000, episode_seed=0)   <span class="c"># rgb, seg, speed, command, target_point, expert{...}</span>
while True:
    obs, reward, done, info = env.step(env.expert_action())   <span class="c"># [steer, throttle, brake]</span>
    if done: break
print(info["status"], info["DS"])                       <span class="c"># env.vector_obs() で物体レベルの観測も取れる</span></pre>
  </section>

  <footer>KeiSim × KeiPilot — <a href="https://github.com/southern-star/keisim">github.com/southern-star/keisim</a>。数値はすべて <span class="mono">runs/eval/*.json</span> から生成。</footer>
</main>
"""

if __name__ == "__main__":
    main()
