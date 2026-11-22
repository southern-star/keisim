"""Where a depth head's error is (ReconView G2, docs/recon_view.md): inside surfaces or next to a depth step, too near
or too far, per camera; and pictures of the labels, the head's depth and the error.

    python tools/depth_detail.py runs/keipilot_mc14d/last.pt --data data/recon_depth_eval_sapporo \
        --out runs/eval/recon/g2_depth_detail.json [--sheet docs/recon/g2_depth.jpg --frames 150 700 1250 1800]
    python tools/depth_detail.py --table runs/eval/recon/g2_depth_detail.json

Pixels within 30 m, per class group. "Edge": the labels within 2 pixels (half resolution) differ by more than a factor
1.15, or one of them has no surface. Too near / too far: off by more than a factor 1.25. Bias: the median of
ln(pred / gt)."""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keipilot.depthstats import GROUPS, codes_to_metres  # noqa: E402
from keipilot.model import DEPTH_NEAR  # noqa: E402

JA = {"road": "道路・白線・歩道", "terrain": "地面（その他）", "building": "建物", "vegetation": "植物", "pole": "柱・信号",
      "vehicle": "車", "pedestrian": "人"}
PARTS = {"all": "全部", "inside": "面の内側", "edge": "深度の段差のそば"}
FAR, STEP, OFF = 30.0, 1.15, 1.25
BINS = np.linspace(-3.0, 3.0, 1201)          # histogram of ln(pred / gt): the medians come from it


def edges(gt):
    """(B, K, h, w) bool: a depth step (or no surface) within 2 pixels of the label gt [m]."""
    lg = torch.where(gt > 0, torch.log(gt.clamp(min=0.01)), torch.full_like(gt, 20.0)).flatten(0, 1)[:, None]
    return ((F.max_pool2d(lg, 5, 1, 2) + F.max_pool2d(-lg, 5, 1, 2)) > math.log(STEP)).view_as(gt)


def predict(model, img, shape, amp):
    with torch.autocast(img.device.type, dtype=torch.float16, enabled=amp):
        y = model.forward_depth(img)
    return (DEPTH_NEAR * torch.exp(y[:, 0].float().clamp(0.0, 7.4))).view(shape), (y[:, 1] > 0).view(shape)


def cell(h_signed, h_abs, n, rel):
    if n == 0:
        return None
    mid = lambda h: float(np.interp(0.5, np.cumsum(h) / h.sum(), BINS[1:]))
    c = 0.5 * (BINS[1:] + BINS[:-1])
    return {"pixels": int(n), "abs_rel": float(rel / n), "median_rel": math.expm1(mid(h_abs)), "bias": mid(h_signed),
            "near": float(h_signed[c < -math.log(OFF)].sum() / n), "far": float(h_signed[c > math.log(OFF)].sum() / n)}


def table(r):
    f = lambda v: f"{100 * v:.1f} %"
    print(f"{r['frames']:,} フレーム（{', '.join(r['data'])}）、0〜{FAR:.0f} m の画素\n")
    print("| クラス | 画素の場所 | 画素 | AbsRel（平均） | 誤差の中央値 | 1.25 倍より近すぎ | 1.25 倍より遠すぎ | 偏り（ln の中央値） |\n|---|---|---:|---:|---:|---:|---:|---:|")
    for g in GROUPS:
        for part, ja in PARTS.items():
            c = r["groups"][g][part]
            if c:
                print(f"| {JA[g]} | {ja} | {c['pixels']:,} | {f(c['abs_rel'])} | {f(c['median_rel'])} | {f(c['near'])} | {f(c['far'])} | {100 * c['bias']:+.1f} % |")
    print("\n| クラス | カメラ | 画素（面の内側） | AbsRel（面の内側） | 誤差の中央値 | 遠すぎ |\n|---|---|---:|---:|---:|---:|")
    for g in ("building", "vehicle", "pedestrian"):
        for k, name in enumerate(r["cameras"]):
            c = r["groups"][g]["cameras"][k]
            if c:
                print(f"| {JA[g]} | {name} | {c['pixels']:,} | {f(c['abs_rel'])} | {f(c['median_rel'])} | {f(c['far'])} |")


def sheet(path, model, ds, frames, device, amp):
    """One row per frame and camera: the image, the label, the head's depth, the error (blue: too near, red: too far)."""
    import cv2
    rows = []
    for i in frames:
        b = ds[i]
        img, gt = b["img"][None].to(device), codes_to_metres(b["dep"][None].to(device))
        with torch.no_grad():
            pred, surf = predict(model, img, gt.shape, amp)
        for k in range(gt.shape[1]):
            rgb = b["img"][k].numpy().transpose(1, 2, 0)[:, :, ::-1]
            g, p, s = gt[0, k].cpu().numpy(), pred[0, k].cpu().numpy(), surf[0, k].cpu().numpy()

            def colour(d, ok):
                t = np.clip(np.log(np.maximum(d, 1.0)) / math.log(80.0), 0, 1)
                im = cv2.applyColorMap((255 * (1 - t)).astype(np.uint8), cv2.COLORMAP_TURBO)
                im[~ok] = 0
                return im
            e = np.clip(np.log(np.maximum(p, 1e-3) / np.maximum(g, 1e-3)) / math.log(1.5), -1, 1)
            err = np.full(g.shape + (3,), 255, np.uint8)
            err[..., 1] = err[..., 1] * (1 - np.abs(e))
            err[..., 0] = np.where(e > 0, 255 * (1 - e), 255)             # too far: red (BGR)
            err[..., 2] = np.where(e < 0, 255 * (1 + e), 255)             # too near: blue
            err[g <= 0] = 60
            up = lambda im: cv2.resize(im, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            rows.append(np.hstack([rgb, up(colour(g, g > 0)), up(colour(p, s)), up(err)]))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{path}: {len(rows)} rows (image, label, head, error: blue too near, red too far, full colour at a factor 1.5)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--data", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sheet", default=None, help="a picture of --frames (indices into the data's frames)")
    ap.add_argument("--frames", type=int, nargs="+", default=[])
    ap.add_argument("--table", default=None, help="print the table of this result and do not evaluate")
    args = ap.parse_args()
    if args.table:
        table(json.load(open(args.table)))
        return
    from torch.utils.data import DataLoader

    from keipilot.agent import load_model
    from keipilot.data import DrivingDataset, load_shards
    from keisim.config import CAMERA_RIGS, SEM
    model = load_model(args.ckpt, args.device)
    assert model.aux_depth, "the checkpoint has no depth head"
    data, _ = load_shards(args.data, with_prev=False)
    assert data.get("depth"), "the data has no depth images"
    idx = np.nonzero(~np.asarray(data["virtual"]).astype(bool))[0]
    ds = DrivingDataset(data, idx, train=False)
    amp = args.device.startswith("cuda")
    lut = torch.full((len(SEM),), -1, dtype=torch.long)
    for g, names in enumerate(GROUPS.values()):
        for name in names:
            lut[SEM[name]] = g
    lut = lut.to(args.device)
    K, G, NB = model.n_cams, len(GROUPS), len(BINS) - 1
    signed, absolute = np.zeros((K, G, 2, NB)), np.zeros((K, G, 2, NB))
    n, rel = np.zeros((K, G, 2)), np.zeros((K, G, 2))
    edge_of = torch.tensor(BINS, device=args.device, dtype=torch.float32)
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=args.bs, num_workers=args.workers):
            img, q, seg = b["img"].to(args.device), b["dep"].to(args.device), b["seg"].to(args.device)
            gt = codes_to_metres(q)
            pred, _ = predict(model, img, gt.shape, amp)
            grp, e = lut[seg.long()], edges(gt).long()
            ok = (gt > 0) & (gt < FAR) & (grp >= 0)
            cam = torch.arange(K, device=gt.device).view(1, K, 1, 1).expand_as(gt)
            cell_id = ((cam * G + grp) * 2 + e)[ok]
            p, g = pred[ok].clamp(min=1e-3), gt[ok]
            ln = torch.log(p / g)
            n += torch.bincount(cell_id, minlength=K * G * 2).view(K, G, 2).cpu().numpy()
            rel += torch.bincount(cell_id, weights=((p - g).abs() / g).double(), minlength=K * G * 2).view(K, G, 2).cpu().numpy()
            for hist, val in ((signed, ln), (absolute, torch.log1p((p - g).abs() / g))):
                k = (torch.bucketize(val.clamp(BINS[0], BINS[-1] - 1e-6), edge_of, right=True) - 1).clamp(0, NB - 1)
                hist += torch.bincount(cell_id * NB + k, minlength=K * G * 2 * NB).view(K, G, 2, NB).cpu().numpy()
    out = {"ckpt": args.ckpt, "data": [os.path.basename(os.path.normpath(d)) for d in args.data], "frames": int(len(idx)),
           "cameras": [c.name for c in CAMERA_RIGS[model.rig]], "groups": {}}
    for gi, g in enumerate(GROUPS):
        pick = lambda cams, parts: cell(signed[cams][:, gi][:, parts].sum((0, 1)), absolute[cams][:, gi][:, parts].sum((0, 1)),
                                        n[cams][:, gi][:, parts].sum(), rel[cams][:, gi][:, parts].sum())
        every = list(range(K))
        out["groups"][g] = {"all": pick(every, [0, 1]), "inside": pick(every, [0]), "edge": pick(every, [1]),
                            "cameras": [pick([k], [0]) for k in every]}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    table(out)
    if args.sheet:
        sheet(args.sheet, model, ds, args.frames or list(np.linspace(0, len(ds) - 1, 6).astype(int)[1:-1]), args.device, amp)


if __name__ == "__main__":
    main()
