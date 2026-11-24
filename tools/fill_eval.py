"""ReconFill's model against the town, offline (docs/recon_fill.md; the samples of web/tools/fill_dump.mjs from the
towns of the measurements).

    python tools/fill_eval.py runs/reconfill_a/last.pt [more.pt ...] --data data/recon_fill/eval \
        --out runs/eval/recon_fill/p1_offline.json [--sheet docs/recon_fill/p1_offline.jpg --frames 40 200 400]
    python tools/fill_eval.py --table runs/eval/recon_fill/p1_offline.json

Two kinds of pixels, both where the town has a surface within 80 m: holes (the view drew nothing there) and seen
ones. Per kind: is a surface predicted (at the holes also over the pixels where the town has none: the sky must stay
empty), the depth's AbsRel and the share within 25 % (the measurement's "drawn right"), the colour's L1 (0..255), the
sharpness: the mean colour step between neighbouring pixels over the town's own (1: as sharp as the town; less:
blurred), and "lines": how much of the detail is edges and lines (coherence() below; noise scores low, a blur high; the
town's own is in the result as "lines_gt").
Baselines: "ground": every hole is the flat ground (the depth of keipilot/filldata.py ground_depth, no colour);
"inpaint": cv2.inpaint of the view's picture (colour only); "loose": the loose rule's picture where it has a surface;
"view": the view itself at the seen pixels."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keipilot.filldata import FillDataset  # noqa: E402
from keipilot.fillnet import CODE_MAX_R, code_to_metres, load_fillnet  # noqa: E402

KEYS = ("pixels", "surface_ok", "surface_n", "rel", "within", "l1", "grad", "grad_gt", "grad_n", "covered", "co", "en", "co_gt", "en_gt")


def coherence(rgb, region, k=5):
    """How much of the picture's detail is lines and edges: the brightness's gradients (central differences, inside the
    region only) as a structure tensor over k x k pixels; returns the sums over the region of the tensor's
    anisotropy sqrt((Jxx - Jyy)^2 + 4 Jxy^2) and of its trace. Their ratio is 1 where all gradients of a
    neighbourhood point one way (an edge, a ramp) and near 0 where they point every way (noise)."""
    import torch.nn.functional as F
    y = rgb.float().mean(1, keepdim=True)
    m = region.float()
    gx = F.pad(y[..., :, 2:] - y[..., :, :-2], (1, 1)) * F.pad(m[..., :, 2:] * m[..., :, :-2], (1, 1))
    gy = F.pad(y[..., 2:, :] - y[..., :-2, :], (0, 0, 1, 1)) * F.pad(m[..., 2:, :] * m[..., :-2, :], (0, 0, 1, 1))
    pool = lambda a: F.avg_pool2d(a, k, 1, k // 2)
    xx, yy, xy = pool(gx * gx), pool(gy * gy), pool(gx * gy)
    return float((torch.sqrt((xx - yy) ** 2 + 4 * xy * xy) * m).sum()), float(((xx + yy) * m).sum())


class Acc:
    """Sums over the pixels of a region for one method."""

    def __init__(self):
        self.s = {k: 0.0 for k in KEYS}
        self.has = set()                                    # what the method gives: "surface", "depth", "rgb"

    def add(self, region, rgb, metres, surf, gt_rgb, gt_m, target, surface_region=None):
        """region: (B, 1, H, W) bool, inside the target. rgb, metres, surf may be None (a method without them)."""
        s = self.s
        s["pixels"] += float(region.sum())
        self.has |= {k for k, v in (("surface", surf), ("depth", metres), ("rgb", rgb)) if v is not None}
        if surf is not None:
            sr = region if surface_region is None else surface_region
            s["surface_ok"] += float(((surf == target) & sr).sum())
            s["surface_n"] += float(sr.sum())
            s["covered"] += float((surf & region).sum())
        if metres is not None:
            rel = (metres - gt_m).abs() / gt_m.clamp(min=1e-3)
            ok = surf if surf is not None else torch.ones_like(region)
            s["rel"] += float((rel * region).sum())
            s["within"] += float(((rel <= 0.25) & region & ok).sum())
        if rgb is not None:
            s["l1"] += float(((rgb - gt_rgb).abs().mean(1, keepdim=True) * region).sum()) * 255.0
            both = region[..., :, 1:] & region[..., :, :-1]
            g = lambda x: ((x[..., :, 1:] - x[..., :, :-1]).abs().mean(1, keepdim=True) * both).sum()
            s["grad"] += float(g(rgb)) * 255.0
            s["grad_gt"] += float(g(gt_rgb)) * 255.0
            s["grad_n"] += float(both.sum())
            for k, x in (("", rgb), ("_gt", gt_rgb)):
                co, en = coherence(x, region)
                s["co" + k], s["en" + k] = s["co" + k] + co, s["en" + k] + en

    def result(self):
        s, has = self.s, self.has
        d = lambda what, a, b: (a / b if b and what in has else None)
        return {"pixels": int(s["pixels"]), "surface_acc": d("surface", s["surface_ok"], s["surface_n"]), "covered": d("surface", s["covered"], s["pixels"]),
                "abs_rel": d("depth", s["rel"], s["pixels"]), "within25": d("depth", s["within"], s["pixels"]), "rgb_l1": d("rgb", s["l1"], s["pixels"]),
                "sharpness": d("rgb", s["grad"], s["grad_gt"]), "gt_step": d("rgb", s["grad_gt"], s["grad_n"]),
                "lines": d("rgb", s["co"], s["en"]), "lines_gt": d("rgb", s["co_gt"], s["en_gt"])}


def evaluate(model, ds, device, bs=32, workers=4, baselines=True):
    """{method: numbers at the holes, with "seen": the numbers at the seen pixels}."""
    from torch.utils.data import DataLoader
    import cv2
    names = ["model"] + (["ground", "inpaint", "loose", "view"] if baselines else [])
    holes, seen = {k: Acc() for k in names}, {k: Acc() for k in names}
    amp = device.startswith("cuda")
    model.eval()
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=bs, num_workers=workers):
            b = {k: v.to(device) for k, v in b.items()}
            gt = b["gt_dep"]
            target = (gt > 0) & (gt <= CODE_MAX_R)
            empty = b["in_dep"] == 0
            hole, seen_px = target & empty, target & ~empty
            gt_m = code_to_metres(gt)
            with torch.autocast("cuda" if amp else "cpu", dtype=torch.float16, enabled=amp):
                rgb, code, surf = model(b["in_rgb"], b["in_dep"], b["loose_rgb"], b["loose_dep"])
            rgb, m, surf = rgb.float().clamp(0, 1), code_to_metres(code.float()), surf > 0
            holes["model"].add(hole, rgb, m, surf, b["gt_rgb"], gt_m, target, empty)
            seen["model"].add(seen_px, rgb, m, surf, b["gt_rgb"], gt_m, target)
            if not baselines:
                continue
            ground = model.ground.expand_as(gt)
            holes["ground"].add(hole, None, code_to_metres(ground), ground > 0, b["gt_rgb"], gt_m, target, empty)
            lv = b["loose_dep"] > 0
            holes["loose"].add(hole, b["loose_rgb"], code_to_metres(b["loose_dep"]), lv, b["gt_rgb"], gt_m, target, empty)
            seen["view"].add(seen_px, b["in_rgb"], code_to_metres(b["in_dep"]), ~empty, b["gt_rgb"], gt_m, target)
            src = (b["in_rgb"] * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            mask = empty[:, 0].cpu().numpy().astype(np.uint8)
            painted = np.stack([cv2.inpaint(np.ascontiguousarray(s), k, 3, cv2.INPAINT_TELEA) for s, k in zip(src, mask)])
            holes["inpaint"].add(hole, torch.from_numpy(painted).to(device).permute(0, 3, 1, 2).float() / 255.0, None, None, b["gt_rgb"], gt_m, target)
    return {k: {**holes[k].result(), "seen": seen[k].result()} for k in names}


JA = {"ground": "基準: 穴を地面で埋める", "inpaint": "基準: `cv2.inpaint`", "loose": "基準: 緩い規則の画をそのまま", "view": "見えた面そのまま（入力）"}


def table(r):
    f = lambda v, s="{:.1f} %", k=100.0: "–" if v is None else s.format(k * v)
    print(f"{r['samples']:,} 組（{'、'.join(r['data'])}）。対象は、街に 80 m 以内の面がある画素\n")
    print("穴の画素（補完なしの画に何もない所）\n")
    grain = "lines" in r["models"][0]["holes"]["model"]
    g = lambda v: f" {f(v.get('lines'), '{:.2f}', 1.0)} |" if grain else ""
    print("| | 画素 | 面があるかの正答率 | 面を出した割合 | 深度の AbsRel | 25 % 以内（面あり） | 色の L1（0〜255） | 鮮明さ |" + (" 線らしさ |" if grain else "") + "\n|---|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" if grain else ""))
    rows = [(m["name"], m["holes"]["model"]) for m in r["models"]] + [(JA[k], r["models"][0]["holes"][k]) for k in ("ground", "inpaint", "loose")]
    for name, v in rows:
        print(f"| {name} | {v['pixels']:,} | {f(v['surface_acc'])} | {f(v['covered'])} | {f(v['abs_rel'])} | {f(v['within25'])} | {f(v['rgb_l1'], '{:.1f}', 1.0)} | {f(v['sharpness'], '{:.2f}', 1.0)} |{g(v)}")
    if grain:
        print(f"| 正解の画そのもの | | | | | | 0.0 | 1.00 | {r['models'][0]['holes']['model']['lines_gt']:.2f} |")
    print("\n見えた画素（補完なしの画に面がある所）\n")
    print("| | 画素 | 面を出した割合 | 深度の AbsRel | 25 % 以内（面あり） | 色の L1（0〜255） | 鮮明さ |" + (" 線らしさ |" if grain else "") + "\n|---|---:|---:|---:|---:|---:|---:|" + ("---:|" if grain else ""))
    rows = [(m["name"], m["holes"]["model"]["seen"]) for m in r["models"]] + [(JA["view"], r["models"][0]["holes"]["view"]["seen"])]
    for name, v in rows:
        print(f"| {name} | {v['pixels']:,} | {f(v['covered'])} | {f(v['abs_rel'])} | {f(v['within25'])} | {f(v['rgb_l1'], '{:.1f}', 1.0)} | {f(v['sharpness'], '{:.2f}', 1.0)} |{g(v)}")
    if grain:
        print(f"| 正解の画そのもの | | | | | 0.0 | 1.00 | {r['models'][0]['holes']['model']['seen']['lines_gt']:.2f} |")
    print("\n鮮明さ: 隣の画素との色の差の平均を、正解の画のそれで割った値（1 が正解と同じ）。")
    if grain:
        print("線らしさ: 明るさの勾配が、5×5 画素の中でどれだけ同じ向きにそろっているか（縁や線だけなら 1、画素がばらばらの乱数なら 0 に近い。ぼけた画は高く出る）。")


def sheet(path, models, ds, frames, device):
    """One row per frame: the view, the loose rule's picture, each model's colour, the town; below them the depths."""
    import cv2
    rows = []
    for i in frames:
        b = {k: v[None].to(device) for k, v in ds[i].items()}
        pics, deps = [b["in_rgb"][0], b["loose_rgb"][0]], [b["in_dep"][0, 0], b["loose_dep"][0, 0]]
        for model in models:
            with torch.no_grad():
                rgb, code, surf = model(b["in_rgb"], b["in_dep"], b["loose_rgb"], b["loose_dep"])
            ok = (surf[0, 0] > 0).float()
            pics.append(rgb[0].clamp(0, 1) * ok)
            deps.append(code[0, 0].clamp(0, 1) * ok)
        pics.append(b["gt_rgb"][0])
        deps.append(b["gt_dep"][0, 0])
        im = lambda p: (p.permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8)
        dm = lambda d: cv2.applyColorMap((255 * (1 - d.cpu().numpy() / CODE_MAX_R).clip(0, 1)).astype(np.uint8), cv2.COLORMAP_TURBO) * (d.cpu().numpy() > 0)[..., None].astype(np.uint8)
        rows += [np.hstack([im(p) for p in pics]), np.hstack([dm(d) for d in deps])]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{path}: columns: the view, the loose rule's picture, {len(models)} model(s), the town; rows: colour, depth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="*")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--data", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--frames", type=int, nargs="+", default=[])
    ap.add_argument("--table", default=None, help="print the table of this result and do not evaluate")
    args = ap.parse_args()
    if args.table:
        table(json.load(open(args.table)))
        return
    ds = FillDataset(args.data)
    models = [load_fillnet(p, args.device) for p in args.ckpt]
    names = args.names or [os.path.basename(os.path.dirname(os.path.abspath(p))) for p in args.ckpt]
    out = {"data": [os.path.basename(os.path.normpath(d)) for d in args.data], "samples": len(ds), "runs": [os.path.basename(p) for p in ds.paths], "models": []}
    for name, path, model in zip(names, args.ckpt, models):
        out["models"].append({"name": name, "ckpt": path, "config": model.config(), "holes": evaluate(model, ds, args.device, bs=args.bs, baselines=True)})
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    table(out)
    if args.sheet:
        sheet(args.sheet, models, ds, args.frames or list(np.linspace(0, len(ds) - 1, 6).astype(int)[1:-1]), args.device)


if __name__ == "__main__":
    main()
