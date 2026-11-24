"""Train ReconFill's model (keipilot/fillnet.py; docs/recon_fill.md) on the samples of web/tools/fill_dump.mjs.

    python scripts/train_fill.py --data data/recon_fill/train --val data/recon_fill/eval --out runs/reconfill_a
        [--epochs 30] [--bs 32] [--lr 2e-3] [--width 16 32 64 128] [--no_loose] [--norm none|gn] [--workers 6]

Writes OUT/last.pt (the weights and the net's configuration) and OUT/log.json (per epoch: the training loss's parts
and, on --val, the numbers of tools/fill_eval.py at the holes)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keipilot.filldata import FillDataset  # noqa: E402
from keipilot.fillnet import FillNet, fill_loss, macs  # noqa: E402


def to_device(b, device):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--width", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--no_loose", action="store_true")
    ap.add_argument("--norm", default="none", choices=("none", "gn"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_steps", type=int, default=0, help="stop after this many steps (a smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    ds = FillDataset(args.data, train=True)
    val = FillDataset(args.val) if args.val else None
    model = FillNet(width=args.width, loose=not args.no_loose, norm=args.norm).to(device).to(memory_format=torch.channels_last)
    print(f"{len(ds):,} samples of {len(ds.runs)} runs; the net: {sum(p.numel() for p in model.parameters()):,} parameters, {macs(model) / 1e9:.2f} G multiply-adds")
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True,
                        persistent_workers=args.workers > 0, worker_init_fn=lambda i: np.random.seed(args.seed * 1000 + i))
    steps = args.epochs * len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = min(500, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(1, steps)))))
    amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=amp)
    os.makedirs(args.out, exist_ok=True)
    log, step, t0 = [], 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        acc, n = {}, 0
        for b in loader:
            b = to_device(b, device)
            with torch.autocast("cuda" if amp else "cpu", dtype=torch.float16, enabled=amp):
                loss, parts = fill_loss(model, b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            n += 1
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + v
            if step % 200 == 0:
                print(f"  step {step}/{steps}  " + "  ".join(f"{k} {v / n:.4f}" for k, v in acc.items()) + f"  {step * args.bs / (time.time() - t0):.0f} samples/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        row = {"epoch": epoch + 1, "train": {k: v / max(1, n) for k, v in acc.items()}, "seconds": time.time() - t0}
        if val is not None and len(val):
            from tools.fill_eval import evaluate
            row["val"] = evaluate(model, val, device, bs=args.bs, workers=min(4, args.workers), baselines=False)["model"]
            v = row["val"]
            print(f"epoch {epoch + 1}/{args.epochs}: holes: surface {100 * v['surface_acc']:.2f} %  depth AbsRel {100 * v['abs_rel']:.2f} %  "
                  f"colour L1 {v['rgb_l1']:.4f}  sharpness {v['sharpness']:.3f}  ({row['seconds']:.0f} s)", flush=True)
        else:
            print(f"epoch {epoch + 1}/{args.epochs}: " + "  ".join(f"{k} {v:.4f}" for k, v in row["train"].items()) + f"  ({row['seconds']:.0f} s)", flush=True)
        log.append(row)
        torch.save({"model": model.state_dict(), "config": model.config(), "args": vars(args), "epoch": epoch + 1}, os.path.join(args.out, "last.pt"))
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump({"args": vars(args), "samples": len(ds), "runs": [os.path.basename(p) for p in ds.paths], "log": log}, f, indent=1)
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
