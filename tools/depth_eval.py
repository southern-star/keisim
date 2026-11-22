"""Accuracy of a depth-head checkpoint (scripts/train_depth.py) on data with depth images, against the baseline
"every pixel is the flat ground" (ReconView G2, docs/recon_view.md).

    python tools/depth_eval.py runs/keipilot_mc14d/last.pt --data data/recon_depth_eval_sapporo [--out runs/eval/recon/g2_depth.json]
    python tools/depth_eval.py --table runs/eval/recon/g2_depth.json
    python tools/depth_eval.py --log runs/eval/recon/g2_train_log.json     (train_depth.py's log.json: the error per epoch)

AbsRel = mean |pred - gt| / gt, d1 = share with max(pred / gt, gt / pred) < 1.25, RMSE of ln(pred / gt), over the
labelled pixels within 80 m, per class group, distance band (0-10, 10-30, 30-80 m) and camera. Frames from a virtual
camera are left out."""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keipilot.agent import load_model  # noqa: E402
from keipilot.data import DrivingDataset, load_shards  # noqa: E402
from keipilot.depthstats import GROUPS, DepthStats, codes_to_metres, ground_baseline  # noqa: E402
from keipilot.model import DEPTH_NEAR  # noqa: E402

JA = {"road": "道路・白線・歩道", "terrain": "地面（その他）", "building": "建物", "vegetation": "植物", "pole": "柱・信号",
      "vehicle": "車", "pedestrian": "人"}


def table(r):
    f = lambda c, k="abs_rel": "–" if c is None else f"{100 * c[k]:.1f} %"
    print(f"{r['frames']:,} フレーム（{', '.join(r['data'])}）、「面がある」の正答率 {100 * r['model'].get('surface', float('nan')):.2f} %\n")
    print("| クラス | 画素（0〜30 m） | AbsRel 0〜30 m | 基準（全部が地面） | 0〜10 m | 10〜30 m | 30〜80 m | δ < 1.25（0〜80 m） | log の RMSE |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows = [(JA[g], r["model"]["groups"][g], r["ground"]["groups"][g]) for g in GROUPS] + [("**全部**", r["model"]["all"], r["ground"]["all"])]
    rows += [(f"カメラ {name}", m, b) for name, m, b in zip(r["cameras"], r["model"]["cameras"], r["ground"]["cameras"])]
    for name, m, b in rows:
        px = m["0-30"]["pixels"] if m["0-30"] else 0
        print(f"| {name} | {px:,} | {f(m['0-30'])} | {f(b['0-30'])} | {f(m['0-10'])} | {f(m['10-30'])} | {f(m['30-80'])} | "
              f"{f(m['all'], 'd1')} | {'–' if m['all'] is None else format(m['all']['rmse_log'], '.3f')} |")


def log_table(log):
    """The validation error (AbsRel 0-30 m) after every epoch of scripts/train_depth.py."""
    names = ("road", "building", "vehicle", "pedestrian")
    print("| エポック | 学習の損失（ln の L1） | " + " | ".join(JA[g] for g in names) + " |\n|---:|---:|" + "---:|" * len(names))
    for e in log:
        print(f"| {e['epoch']} | {e['train']['depth_l1']:.4f} | " + " | ".join(f"{100 * e['val']['groups'][g]['0-30']['abs_rel']:.1f} %" for g in names) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--data", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--table", default=None, help="print the table of this result and do not evaluate")
    ap.add_argument("--log", default=None, help="print the per-epoch table of this training log and do not evaluate")
    args = ap.parse_args()
    if args.log:
        log_table(json.load(open(args.log)))
        return
    if args.table:
        table(json.load(open(args.table)))
        return
    model = load_model(args.ckpt, args.device)
    assert model.aux_depth, "the checkpoint has no depth head"
    data, _ = load_shards(args.data, with_prev=False)
    assert data.get("depth"), "the data has no depth images"
    idx = np.nonzero(~np.asarray(data["virtual"]).astype(bool))[0]
    if args.max_frames:
        idx = idx[np.linspace(0, len(idx) - 1, min(args.max_frames, len(idx))).astype(int)]
    dl = DataLoader(DrivingDataset(data, idx, train=False), batch_size=args.bs, num_workers=args.workers)
    ms, gs = DepthStats(model.n_cams), DepthStats(model.n_cams)
    ground = ground_baseline(model.rig).to(args.device)
    amp = args.device.startswith("cuda")
    with torch.no_grad():
        for batch in dl:
            img, q, seg = batch["img"].to(args.device), batch["dep"].to(args.device), batch["seg"].to(args.device)
            with torch.autocast(args.device, dtype=torch.float16, enabled=amp):
                y = model.forward_depth(img)
            gt = codes_to_metres(q)
            pred = (DEPTH_NEAR * torch.exp(y[:, 0].float().clamp(0.0, 7.4))).unflatten(0, q.shape[:2])
            ms.add(pred, gt, seg, (y[:, 1] > 0).unflatten(0, q.shape[:2]))
            gs.add(ground[None].expand_as(gt), gt, seg)
    from keisim.config import CAMERA_RIGS
    r = {"ckpt": args.ckpt, "data": [os.path.basename(os.path.normpath(d)) for d in args.data], "frames": int(len(idx)),
         "cameras": [c.name for c in CAMERA_RIGS[model.rig]], "model": ms.summary(), "ground": gs.summary()}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(r, f, indent=1)
    table(r)


if __name__ == "__main__":
    main()
