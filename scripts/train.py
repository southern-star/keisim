#!/usr/bin/env python3
"""Train KeiPilot by imitation of the privileged expert.

  python scripts/train.py --data data/expert --out runs/keipilot_bc
  python scripts/train.py --data data/expert data/dagger1 --init runs/keipilot_bc/best.pt --out runs/keipilot_dagger1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keipilot.data import DrivingDataset, load_shards  # noqa: E402
from keipilot.model import KeiPilot, compute_loss  # noqa: E402
from keisim.config import N_SEM  # noqa: E402


def sample_weights(d, dagger_weight=1.0, kv_weight=1.0):
    w = np.ones(len(d["cmd"]), np.float64)
    w[d["on_policy"]] *= dagger_weight          # on-policy (DAgger) states
    w[d["keiview"]] *= kv_weight                # KeiView-rendered frames (domain balance)
    reason = d["reason"]
    w[reason == 3] *= 3.0                       # pedestrian
    w[reason == 4] *= 1.5                       # red light
    w[d["tl"] == 1] *= 2.0                      # yellow
    start = (d["v"] < 1.0) & (d["speed"] > 1.0)
    w[start] *= 3.0                             # moving off (anti-inertia)
    w[d["cmd"] != 1] *= 1.3                     # turns
    return w


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    agg = {}
    n = 0
    ade = 0.0
    spd_err = 0.0
    stop_ok = 0
    tl_ok = 0
    conf = torch.zeros(N_SEM, N_SEM, dtype=torch.long)
    for b in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b["img"].contiguous(memory_format=torch.channels_last), b["cmd"], b["tp"])
        _, logs = compute_loss(model, out, b)
        bs = b["img"].shape[0]
        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + v * bs
        n += bs
        ade += torch.linalg.norm(out["path"] - b["path"], dim=-1).mean(-1).sum().item()
        v, _ = model.decode_speed(out["speed_logits"])
        spd_err += (v - b["speed"]).abs().sum().item()
        stop_ok += ((v < 0.05) == (b["speed"] < 0.05)).sum().item()
        tl_ok += (out["tl_logits"].argmax(-1) == b["tl"]).sum().item()
        seg = torch.nn.functional.interpolate(out["seg"].float(), size=b["seg"].shape[-2:], mode="bilinear",
                                              align_corners=False).argmax(1)
        idx = b["seg"].flatten() * N_SEM + seg.flatten()
        conf += torch.bincount(idx.cpu(), minlength=N_SEM * N_SEM).view(N_SEM, N_SEM)
    res = {k: v / n for k, v in agg.items()}
    inter = conf.diag().float()
    union = conf.sum(0).float() + conf.sum(1).float() - inter
    present = conf.sum(1) > 0
    iou = (inter / union.clamp(min=1))
    res.update({"ADE_m": ade / n, "speed_MAE": spd_err / n, "stop_acc": stop_ok / n, "tl_acc": tl_ok / n,
                "seg_mIoU": iou[present].mean().item(), "iou": iou.tolist()})
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/expert"])
    ap.add_argument("--out", default="runs/keipilot_bc")
    ap.add_argument("--init", default=None)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--val_frac", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_iters", type=int, default=0, help="debug: stop after N iterations per epoch")
    ap.add_argument("--dagger_weight", type=float, default=2.0, help="sampling weight of on-policy frames")
    ap.add_argument("--kv_weight", type=float, default=1.0, help="sampling weight of KeiView-rendered frames")
    ap.add_argument("--samples_per_epoch", type=int, default=0, help="0 = one pass over the training frames")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda"

    data, files = load_shards(args.data)
    N = len(data["cmd"])
    rng = np.random.default_rng(args.seed)
    eps = np.unique(data["episode"])
    rng.shuffle(eps)
    val_eps = eps[: max(1, int(len(eps) * args.val_frac))]
    is_val = np.isin(data["episode"], val_eps)
    tr_idx, va_idx = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
    print(f"{N} frames from {len(files)} shards, {len(eps)} episodes; train {len(tr_idx)} / val {len(va_idx)}", flush=True)

    ds_tr = DrivingDataset(data, tr_idx, train=True)
    ds_va = DrivingDataset(data, va_idx, train=False)
    w = sample_weights(data, args.dagger_weight, args.kv_weight)[tr_idx]
    n_epoch = args.samples_per_epoch or len(tr_idx)
    sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=n_epoch, replacement=True)
    kv = data["keiview"][tr_idx]
    print(f"KeiView frames {int(kv.sum())} / {len(tr_idx)}; expected KeiView share per batch {w[kv].sum() / w.sum():.2f}", flush=True)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, sampler=sampler, num_workers=args.workers, pin_memory=True,
                       drop_last=True, persistent_workers=True, prefetch_factor=4)
    dl_va = DataLoader(ds_va, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    model = KeiPilot(pretrained=args.init is None)
    print("backbone init:", model.init_info, flush=True)
    if args.init:
        sd = torch.load(args.init, map_location="cpu")
        model.load_state_dict(sd["model"])
        print("initialised from", args.init)
    model = model.to(device).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f} M", flush=True)
    backbone = [p for n, p in model.named_parameters() if n.startswith(("stem", "layer"))]
    heads = [p for n, p in model.named_parameters() if not n.startswith(("stem", "layer"))]
    lr = args.lr if args.init is None else args.lr * 0.5
    opt = torch.optim.AdamW([{"params": backbone, "lr": lr * 0.5}, {"params": heads, "lr": lr}], weight_decay=args.wd)
    iters = len(dl_tr) if not args.max_iters else min(len(dl_tr), args.max_iters)
    total = iters * args.epochs
    warm = min(1000, total // 10)
    base = [g["lr"] for g in opt.param_groups]

    def set_lr(it):
        f = it / warm if it < warm else 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, total - warm)))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * max(f, 0.02)

    hist = []
    best = float("inf")
    it = 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        run = {}
        for i, b in enumerate(dl_tr):
            if args.max_iters and i >= args.max_iters:
                break
            set_lr(it)
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(b["img"].contiguous(memory_format=torch.channels_last), b["cmd"], b["tp"])
            loss, logs = compute_loss(model, out, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            it += 1
            for k, v in logs.items():
                run[k] = 0.98 * run.get(k, v) + 0.02 * v
            if i % 200 == 0:
                el = time.time() - t0
                print(f"ep {ep} it {i}/{iters} " + " ".join(f"{k}={v:.3f}" for k, v in run.items()) +
                      f"  {(i + 1) * args.bs / max(el, 1e-6):.0f} img/s", flush=True)
        val = evaluate(model, dl_va, device)
        val["epoch"] = ep
        val["train"] = run
        val["time"] = time.time() - t0
        hist.append(val)
        print(f"== ep {ep} val loss={val['loss']:.3f} ADE={val['ADE_m']:.3f}m speedMAE={val['speed_MAE']:.3f} "
              f"stop_acc={val['stop_acc']:.3f} tl_acc={val['tl_acc']:.3f} mIoU={val['seg_mIoU']:.3f} ({val['time']:.0f}s)",
              flush=True)
        ck = {"model": model.state_dict(), "epoch": ep, "val": val, "model_cfg": {}, "args": vars(args)}
        torch.save(ck, os.path.join(args.out, "last.pt"))
        if val["loss"] < best:
            best = val["loss"]
            torch.save(ck, os.path.join(args.out, "best.pt"))
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)
    print("done. best val loss", best)


if __name__ == "__main__":
    main()
