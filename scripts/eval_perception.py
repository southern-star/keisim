#!/usr/bin/env python3
"""Open-loop perception/planning metrics of a checkpoint on a held-out dataset.

  python scripts/eval_perception.py --ckpt runs/keipilot_bc/best.pt --data data/heldout_test
"""
from __future__ import annotations

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
from keisim.config import N_SEM, SEM_NAMES, TL_NAMES  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", nargs="+", default=["data/heldout_test"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_speed", action="store_true", help="hide the ego speed from a speed-input model")
    args = ap.parse_args()
    d, _ = load_shards(args.data)
    ds = DrivingDataset(d, np.arange(len(d["cmd"])), train=False)
    dl = DataLoader(ds, batch_size=128, num_workers=6)
    model = load_model(args.ckpt).to(memory_format=torch.channels_last)
    conf = torch.zeros(N_SEM, N_SEM, dtype=torch.long)
    tl_conf = np.zeros((4, 4), int)
    ade, fde, spd, stop_ok, n = 0.0, 0.0, 0.0, 0, 0
    for b in dl:
        b = {k: v.cuda() for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b["img"].contiguous(memory_format=torch.channels_last), b["cmd"], b["tp"],
                        speed=None if args.no_speed else b["v"])
        err = torch.linalg.norm(out["path"] - b["path"], dim=-1)
        ade += err.mean(-1).sum().item()
        fde += err[:, -1].sum().item()
        v, _ = model.decode_speed(out["speed_logits"])
        spd += (v - b["speed"]).abs().sum().item()
        stop_ok += ((v < 0.05) == (b["speed"] < 0.05)).sum().item()
        pr = out["tl_logits"].argmax(-1).cpu().numpy()
        for g, p in zip(b["tl"].cpu().numpy(), pr):
            tl_conf[g, p] += 1
        seg = torch.nn.functional.interpolate(out["seg"].float(), size=b["seg"].shape[-2:], mode="bilinear",
                                              align_corners=False).argmax(1)
        conf += torch.bincount((b["seg"].flatten() * N_SEM + seg.flatten()).cpu(), minlength=N_SEM ** 2).view(N_SEM, N_SEM)
        n += len(pr)
    inter = conf.diag().float()
    union = conf.sum(0).float() + conf.sum(1).float() - inter
    iou = (inter / union.clamp(min=1)).numpy()
    present = (conf.sum(1) > 0).numpy()
    res = {
        "frames": n, "speed_input": bool(model.speed_input and not args.no_speed), "ADE_m": ade / n, "FDE_m": fde / n, "speed_MAE": spd / n, "stop_acc": stop_ok / n,
        "tl_acc": float(np.trace(tl_conf) / tl_conf.sum()),
        "tl_recall": {TL_NAMES[i]: float(tl_conf[i, i] / max(1, tl_conf[i].sum())) for i in range(4)},
        "tl_confusion": tl_conf.tolist(),
        "seg_mIoU": float(iou[present].mean()),
        "iou": {SEM_NAMES[i]: float(iou[i]) for i in range(N_SEM) if present[i]},
    }
    print(json.dumps(res, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
