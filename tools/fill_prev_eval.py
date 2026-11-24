"""ReconFill P2c offline (docs/recon_fill.md 6.14): a colour net that takes the fill before as a condition, on the eval
towns' samples, at the holes.

    python tools/fill_prev_eval.py runs/reconfill_genp1/last.pt --out runs/eval/recon_fill/p2c_offline_p1.json [--sheet X.jpg]
    python tools/fill_prev_eval.py --table runs/eval/recon_fill/p2c_offline_p1.json
    python tools/fill_prev_eval.py CKPT --rounds 1 8 16 32 --out ...      (more rounds; a sheet needs rounds 1 and 8)

Rows: without a fill before; with the town's own fill before (the training data's: nearly the answer, so no measure of
the page); and the rounds: the net's own answer given back as the fill before, 1, 2, 4 and 8 times, from the same
viewpoint (where the data's fill before has a surface). Per row the colour's L1, the sharpness and "lines"
(tools/fill_eval.py). "change": how far a round's colours are from the round before, where the fill before has a
surface (0..255). The depth and "a surface here" must be P1's in every row."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keipilot.filldata import FillDataset  # noqa: E402
from keipilot.fillnet import CODE_MAX_R, load_fillnet  # noqa: E402
from tools.fill_eval import Acc  # noqa: E402

ROUNDS = (1, 2, 4, 8)


def table(r):
    print(f"{r['samples']} 組（評価用）、穴の画素。深度と「面があるか」の P1 との差の最大: {r['shape_diff']:g}\n")
    print("| 1 回前の画 | 色の L1（0〜255） | 鮮明さ | 線らしさ | 前の回からの色の変化 |\n|---|---:|---:|---:|---:|")
    for k, v in r["rows"].items():
        print(f"| {k} | {v['rgb_l1']:.1f} | {v['sharpness']:.2f} | {v['lines']:.2f} | {'–' if v.get('change') is None else format(v['change'], '.1f')} |")
    print(f"| 正解の画そのもの | 0.0 | 1.00 | {next(iter(r['rows'].values()))['lines_gt']:.2f} | – |")
    print("\n前の回からの色の変化: 1 回前の画に面がある穴の画素で、その回の色と前の回の色の差の平均。")


def main():
    global ROUNDS
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--data", default="data/recon_fill/eval")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--frames", type=int, nargs="+", default=[60, 300, 520, 700])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--table", default=None)
    ap.add_argument("--rounds", type=int, nargs="+", default=list(ROUNDS))
    args = ap.parse_args()
    ROUNDS = tuple(sorted(args.rounds))
    if args.table:
        return table(json.load(open(args.table)))
    dev = args.device
    gen = load_fillnet(args.ckpt, dev)
    assert getattr(gen, "prev", False), "a colour net with the fill before as a condition"
    ds = FillDataset([args.data], prev=True)
    names = ["なし（最初の 1 枚）", "正解の 1 回前の画（学習データのもの）"] + [f"自分の出力を {k} 回" for k in ROUNDS]
    acc = {k: Acc() for k in names}
    change, pixels, shape_diff = {k: 0.0 for k in ROUNDS}, 0.0, 0.0
    pics = {}
    with torch.no_grad():
        for a in range(0, len(ds), args.bs):
            idx = list(range(a, min(len(ds), a + args.bs)))
            b = torch.utils.data.default_collate([ds[i] for i in idx])
            b = {k: v.to(dev) for k, v in b.items()}
            gt = b["gt_dep"]
            target = (gt > 0) & (gt <= CODE_MAX_R)
            hole = target & (b["in_dep"] == 0)
            run = lambda pr=None, pd=None: gen(b["in_rgb"], b["in_dep"], b["loose_rgb"], b["loose_dep"], prev_rgb=pr, prev_dep=pd)
            _rgb1, code1, surf1 = gen.shape(b["in_rgb"], b["in_dep"], b["loose_rgb"], b["loose_dep"])
            add = lambda name, rgb: acc[name].add(hole, rgb.clamp(0, 1), None, None, b["gt_rgb"], None, target)
            rgb, code, surf = run()
            shape_diff = max(shape_diff, float((code - code1).abs().max()), float((surf - surf1).abs().max()))
            add(names[0], rgb)
            first = rgb.clamp(0, 1)
            truth = run(b["prev_rgb"], b["prev_dep"])[0]
            add(names[1], truth)
            there = hole & (b["prev_dep"] > 0)
            pixels += float(there.sum())
            out, kept = first, {}
            for k in range(1, ROUNDS[-1] + 1):
                new = run(out, b["prev_dep"])[0].clamp(0, 1)
                if k in ROUNDS:
                    add(f"自分の出力を {k} 回", new)
                    change[k] += float(((new - out).abs().mean(1, keepdim=True) * there).sum()) * 255.0
                    kept[k] = new
                out = new
            for j, i in enumerate(idx):
                if i in args.frames:
                    ok = (surf[j] > 0).float()
                    pics[i] = [b["in_rgb"][j], gen.condition(b["in_rgb"][j:j + 1], b["in_dep"][j:j + 1])[1][0] * ok, first[j] * ok, kept[1][j] * ok, kept[8][j] * ok,
                               truth[j].clamp(0, 1) * ok, b["prev_rgb"][j] * (b["prev_dep"][j] > 0), b["gt_rgb"][j]]
    rows = {}
    for k in names:
        v = acc[k].result()
        rows[k] = {m: v[m] for m in ("rgb_l1", "sharpness", "lines", "lines_gt")}
    for k in ROUNDS:
        rows[f"自分の出力を {k} 回"]["change"] = change[k] / max(1.0, pixels)
    out = {"ckpt": args.ckpt, "config": gen.config(), "samples": len(ds), "shape_diff": shape_diff, "prev_pixels": pixels, "rows": rows}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    table(out)
    if args.sheet and pics:
        import cv2
        im = lambda p: (p.permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8)
        cv2.imwrite(args.sheet, np.vstack([np.hstack([im(p) for p in pics[i]]) for i in sorted(pics)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{args.sheet}: columns: the view, P1, no fill before, the own answer given back once, 8 times, with the town's fill before, the town's fill before, the town")


if __name__ == "__main__":
    main()
