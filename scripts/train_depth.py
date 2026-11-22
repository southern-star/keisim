#!/usr/bin/env python3
"""Train the depth head of a KeiPilot rig model on top of the frozen driving model (ReconView, docs/recon_view.md).

  python scripts/train_depth.py --init runs/keipilot_mc14/last.pt --data data/recon_depth_a data/recon_depth_b \
      --out runs/keipilot_mc14d [--epochs 15] [--bs 64] [--lr 2e-3] [--width 48 32 16 24] [--mmap]

The data has depth images (collect.py --depth 1, KeiView). The driving model does not change: its parameters do not
get gradients, and the whole model stays in eval() (in train() the batch-norm statistics of the trunk would move even
without gradients). Only the parameters named aux_depth_* are trained; at the end every other tensor of the model is
compared with --init, and the script fails if one differs. Frames from a virtual (displaced) camera are left out: the
head's lines of sight are those of the rig's own cameras.
Writes OUT/last.pt (model_cfg with aux_depth) and OUT/log.json (per epoch: the loss and the validation accuracy).
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keipilot.data import DrivingDataset, load_shards  # noqa: E402
from keipilot.depthstats import DepthStats, codes_to_metres  # noqa: E402
from keipilot.model import DEPTH_NEAR, KeiPilot, depth_loss  # noqa: E402


def freeze(model):
    """Only the depth head trains; the model stays in eval(). Returns the head's parameters."""
    model.eval()
    head = []
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("aux_depth_"))
        if p.requires_grad:
            head.append(p)
    return head


def unchanged(model, init_sd):
    """The tensors of the model that are not the depth head's and differ from the checkpoint it started from."""
    return [k for k, v in model.state_dict().items()
            if not k.startswith("aux_depth_") and not torch.equal(v.cpu(), init_sd[k].to(v.dtype))]


@torch.no_grad()
def validate(model, loader, device, max_batches=None):
    stats = DepthStats(model.n_cams)
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        img, q, seg = batch["img"].to(device), batch["dep"].to(device), batch["seg"].to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=device.startswith("cuda")):
            y = model.forward_depth(img)
        pred = (DEPTH_NEAR * torch.exp(y[:, 0].float().clamp(0.0, 7.4))).unflatten(0, q.shape[:2])
        stats.add(pred, codes_to_metres(q), seg, (y[:, 1] > 0).unflatten(0, q.shape[:2]))
    return stats.summary()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="the driving model's checkpoint (a rig model)")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--width", type=int, nargs=4, default=[48, 32, 16, 24],
                    help="channels: the two layers at 1/4, the stem's 1/2 features, the layer at 1/2")
    ap.add_argument("--val_frac", type=float, default=0.05, help="share of the episodes kept for validation")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mmap", action="store_true", help="read the shards from disk (keipilot.data.load_shards)")
    ap.add_argument("--max_steps", type=int, default=0, help="stop after this many steps (tests)")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    device = args.device

    data, files = load_shards(args.data, with_prev=False, mmap=args.mmap)
    if not data.get("depth"):
        sys.exit("the data has no depth images (collect.py --depth 1)")
    rig = {json.load(open(os.path.join(d, "meta.json"))).get("rig", "front") for d in args.data}
    if len(rig) != 1:
        sys.exit(f"data of different camera rigs: {sorted(rig)}")
    rig = rig.pop()
    real = ~np.asarray(data["virtual"]).astype(bool)
    rng = np.random.default_rng(args.seed)
    eps = np.unique(data["episode"])
    rng.shuffle(eps)
    is_val = np.isin(data["episode"], eps[: max(1, int(len(eps) * args.val_frac))])
    tr_idx, va_idx = np.nonzero(real & ~is_val)[0], np.nonzero(real & is_val)[0]
    print(f"{len(real)} frames from {len(files)} shards ({int((~real).sum())} virtual ones left out), {len(eps)} episodes; "
          f"train {len(tr_idx)} / val {len(va_idx)}", flush=True)

    sd = torch.load(args.init, map_location="cpu")
    cfg = dict(sd.get("model_cfg", {}))
    if sd.get("rig", "front") != rig or cfg.get("n_cams", 1) != data["n_cams"]:
        sys.exit(f"--init is a {sd.get('rig', 'front')} model, the data is {rig}")
    cfg.update(aux_depth=True, depth_width=tuple(args.width), depth_rig=rig)
    model = KeiPilot(pretrained=False, **cfg)
    new = model.load_compatible(sd["model"])
    assert all(k.startswith("aux_depth_") for k in new), new
    model.to(device)
    head = freeze(model)
    print(f"depth head {tuple(args.width)}: {sum(p.numel() for p in head):,} parameters; the driving model "
          f"({sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in head):,}) is frozen", flush=True)

    ds_tr = DrivingDataset(data, tr_idx, train=True)          # its photometric augmentation; no geometric one
    ds_va = DrivingDataset(data, va_idx, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
                       persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=128, shuffle=False, num_workers=min(args.workers, 4), pin_memory=True)
    opt = torch.optim.AdamW(head, lr=args.lr, weight_decay=args.wd)
    total = args.max_steps or args.epochs * len(dl_tr)
    warm = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=amp)
    os.makedirs(args.out, exist_ok=True)
    log, step, t0 = [], 0, time.time()
    for ep in range(args.epochs):
        sums, n = {}, 0
        for batch in dl_tr:
            img, q, seg = batch["img"].to(device, non_blocking=True), batch["dep"].to(device), batch["seg"].to(device)
            with torch.autocast(device, dtype=torch.float16, enabled=amp):
                y = model.forward_depth(img)
            loss, logs = depth_loss(y, q.flatten(0, 1), seg.flatten(0, 1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            n += 1
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v
            if step % 200 == 0:
                print(f"  step {step}/{total}  depth_l1 {sums['depth_l1'] / n:.4f}  surface {sums['surface'] / n:.4f}  "
                      f"{step * args.bs / (time.time() - t0):.0f} frames/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        val = validate(model, dl_va, device, max_batches=2 if args.max_steps else None)
        a = val["all"]["0-30"]
        g = {k: (v["0-30"]["abs_rel"] if v["0-30"] else None) for k, v in val["groups"].items()}
        print(f"epoch {ep + 1}/{args.epochs}: train depth_l1 {sums['depth_l1'] / max(n, 1):.4f}; val AbsRel 0-30 m "
              f"{a['abs_rel']:.4f}" + "".join(f"  {k} {v:.3f}" for k, v in g.items() if v is not None) +
              f"  surface {val.get('surface', float('nan')):.4f}  ({time.time() - t0:.0f} s)", flush=True)
        log.append({"epoch": ep + 1, "step": step, "train": {k: v / max(n, 1) for k, v in sums.items()}, "val": val})
        torch.save({"model": model.state_dict(), "epoch": ep, "val": val, "model_cfg": cfg, "args": vars(args), "rig": rig,
                    "depth_init": os.path.abspath(args.init)}, os.path.join(args.out, "last.pt"))
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
        if args.max_steps and step >= args.max_steps:
            break
    changed = unchanged(model, sd["model"])
    if changed:
        sys.exit(f"the driving model changed: {changed[:8]}{' ...' if len(changed) > 8 else ''}")
    print(f"the driving model's {len(sd['model'])} tensors are the ones of {args.init}", flush=True)


if __name__ == "__main__":
    main()
