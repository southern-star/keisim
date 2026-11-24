"""Why does a colour net of ReconFill P2 (keipilot/fillgen.py; docs/recon_fill.md 6) look as it does?

    python tools/fill_gen_diag.py runs/reconfill_gen1/last.pt --out runs/eval/recon_fill/p2_diag_1.json [--sheet X.jpg]
    python tools/fill_gen_diag.py --table runs/eval/recon_fill/p2_diag_1.json

On every 14th eval sample, at the holes (the view drew nothing, the town has a surface within 80 m):
  steps   the colour's L1 and sharpness (tools/fill_eval.py) of samples made in 1, 2 and 4 steps of the average velocity
          and in 8 and 32 Euler steps of the instantaneous one (u with r = t), all from the model's fixed noise image
  noise   the one step's answer x (the colour difference over the scale) against the noise image e it started from and
          against the town's own difference g: variances, correlations, and the same for the fine detail alone (minus
          a 5x5 mean). "left": the share of the noise's fine detail still in the answer (the regression of x on e)."""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keipilot.filldata import FillDataset  # noqa: E402
from keipilot.fillnet import CODE_MAX_R, load_fillnet  # noqa: E402
from tools.fill_eval import Acc  # noqa: E402

KINDS = [("1 ステップ", 1, True), ("2 ステップ（平均速度）", 2, True), ("4 ステップ（平均速度）", 4, True), ("瞬間の速度で 8 ステップ", 8, False), ("瞬間の速度で 32 ステップ", 32, False)]


def sample(gen, cond, rgb1, steps, mean):
    """mean: steps of the average velocity u(z, r, t); else Euler steps of u(z, t, t)."""
    z = gen.noise.expand_as(rgb1).clone()
    ts = torch.linspace(1, 0, steps + 1, device=rgb1.device)
    for a, b in zip(ts[:-1], ts[1:]):
        t = torch.full((z.shape[0], 1, 1, 1), float(a), device=z.device)
        z = z - (a - b) * gen.velocity(z, t, torch.full_like(t, float(a - b)) if mean else torch.zeros_like(t), cond)
    return rgb1 + gen.scale * z


def table(r):
    print(f"{r['samples']} 組（評価用の 14 組に 1 組）、穴の画素\n")
    print("| 作り方 | 色の L1（0〜255） | 鮮明さ |\n|---|---:|---:|")
    for k, v in r["steps"].items():
        print(f"| {k} | {v['rgb_l1']:.1f} | {v['sharpness']:.2f} |")
    n = r["noise"]
    print(f"\n1 ステップの答え（色の差 ÷ 尺度）と、出発点の乱数の画: 相関 {n['corr_noise']:.2f}（細かい成分だけでは {n['hi_corr_noise']:.2f}、乱数の細かい成分のうち答えに残った割合 {100 * n['hi_left']:.0f} %）。"
          f"正解の色の差との相関: {n['corr_town']:.2f}。分散: 答え {n['var']:.2f}、正解 {n['var_town']:.2f}（細かい成分: {n['hi_var']:.2f}、{n['hi_var_town']:.2f}）。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--data", default="data/recon_fill/eval")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--table", default=None)
    args = ap.parse_args()
    if args.table:
        return table(json.load(open(args.table)))
    dev = args.device
    gen = load_fillnet(args.ckpt, dev)
    ds = FillDataset([args.data])
    idx = list(range(0, len(ds), 14))
    acc = {"P1（回帰）": Acc(), **{k[0]: Acc() for k in KINDS}}
    S = {k: 0.0 for k in ("n", "xx", "gg", "xe", "xg", "ee", "hxx", "hgg", "hxe", "hee")}
    blur = lambda a: F.avg_pool2d(a, 5, 1, 2)
    pics = []
    with torch.no_grad():
        for a in range(0, len(idx), 16):
            b = torch.utils.data.default_collate([ds[i] for i in idx[a:a + 16]])
            b = {k: v.to(dev) for k, v in b.items()}
            cond, rgb1, _code, surf = gen.condition(b["in_rgb"], b["in_dep"], b["loose_rgb"], b["loose_dep"])
            gt = b["gt_dep"]
            target = (gt > 0) & (gt <= CODE_MAX_R)
            hole = target & (b["in_dep"] == 0)
            acc["P1（回帰）"].add(hole, rgb1, None, None, b["gt_rgb"], None, target)
            row = [b["in_rgb"][0], rgb1[0]]
            for name, steps, mean in KINDS:
                out = sample(gen, cond, rgb1, steps, mean)
                if steps == 1:
                    x, g, e, m = (out - rgb1) / gen.scale, ((b["gt_rgb"] - rgb1) / gen.scale).clamp(-4, 4), gen.noise.expand_as(rgb1), hole.float().expand_as(rgb1)
                    hx, hg, he = x - blur(x), g - blur(g), e - blur(e)
                    for k, v in (("n", m), ("xx", x * x * m), ("gg", g * g * m), ("xe", x * e * m), ("xg", x * g * m), ("ee", e * e * m),
                                 ("hxx", hx * hx * m), ("hgg", hg * hg * m), ("hxe", hx * he * m), ("hee", he * he * m)):
                        S[k] += float(v.sum())
                acc[name].add(hole, out.clamp(0, 1), None, None, b["gt_rgb"], None, target)
                row.append(out[0].clamp(0, 1))
            row.append(b["gt_rgb"][0])
            if len(pics) < 4:
                ok = (surf[0] > 0).float()
                pics.append(np.hstack([(p * (ok if 0 < j < len(row) - 1 else 1)).permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255 for j, p in enumerate(row)]).astype(np.uint8))
    n = S["n"]
    out = {"ckpt": args.ckpt, "config": gen.config(), "samples": len(idx), "steps": {k: {m: a.result()[m] for m in ("rgb_l1", "sharpness")} for k, a in acc.items()},
           "noise": {"var": S["xx"] / n, "var_town": S["gg"] / n, "corr_noise": S["xe"] / (S["xx"] * S["ee"]) ** 0.5, "corr_town": S["xg"] / (S["xx"] * S["gg"]) ** 0.5,
                     "hi_var": S["hxx"] / n, "hi_var_town": S["hgg"] / n, "hi_corr_noise": S["hxe"] / (S["hxx"] * S["hee"]) ** 0.5, "hi_left": S["hxe"] / S["hee"]}}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
    table(out)
    if args.sheet:
        import cv2
        cv2.imwrite(args.sheet, np.vstack(pics), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{args.sheet}: columns: the view, P1, {', '.join(f'{s} steps' for _n, s, _m in KINDS)}, the town")


if __name__ == "__main__":
    main()
