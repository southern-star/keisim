"""Train ReconFill's colour net (keipilot/fillgen.py; docs/recon_fill.md 6) on the samples of web/tools/fill_dump.mjs,
on top of a trained fill model of scripts/train_fill.py (P1; not changed).

    python scripts/train_fillgen.py --shape runs/reconfill_b/last.pt --data data/recon_fill/train --val data/recon_fill/eval
        --out runs/reconfill_gen1 [--epochs 100] [--bs 32] [--lr 6e-4] [--width 16 32 64 128]
        [--mu -2.0 --sigma 2.0 --ratio 0.75 --p 0.75]   (the (r, t) sampler, the share of r != t, the loss's power)
        [--ema 0.999] [--eval_every 5]

Writes OUT/last.pt (P1's net, the colour net's moving average, the configuration; loads with
keipilot.fillnet.load_fillnet) and OUT/log.json (per epoch the loss's parts and, every --eval_every epochs, the
numbers of tools/fill_eval.py at the holes for the moving average)."""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keipilot.filldata import FillDataset  # noqa: E402
from keipilot.fillgen import FillGen, colour_scale, meanflow_loss  # noqa: E402
from keipilot.fillnet import load_fillnet, macs  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, help="P1's checkpoint")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--width", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--mu", type=float, default=-2.0)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--ratio", type=float, default=0.75, help="the share of samples with r != t")
    ap.add_argument("--p", type=float, default=0.75, help="the adaptive weight's power")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval_every", type=int, default=5)
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
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True,
                        persistent_workers=args.workers > 0, worker_init_fn=lambda i: np.random.seed(args.seed * 1000 + i))
    gen = FillGen(load_fillnet(args.shape, device), width=args.width).to(device).to(memory_format=torch.channels_last)
    gen.scale = colour_scale(gen, loader, device)
    gen.train()
    params = list(gen.colour.parameters())
    print(f"{len(ds):,} samples of {len(ds.runs)} runs; the colour net: {sum(p.numel() for p in params):,} parameters, with P1's net {macs(gen) / 1e9:.2f} G multiply-adds; "
          f"the colour difference's scale {gen.scale:.4f}", flush=True)
    ema = copy.deepcopy(gen).eval()
    steps = args.epochs * len(loader)
    opt = torch.optim.Adam(params, lr=args.lr)
    warm = min(1000, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)))
    amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=amp)
    os.makedirs(args.out, exist_ok=True)
    log, step, t0 = [], 0, time.time()
    for epoch in range(args.epochs):
        acc, cnt = {}, {}
        for b in loader:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast("cuda" if amp else "cpu", dtype=torch.float16, enabled=amp):
                loss, parts = meanflow_loss(gen, b, args.mu, args.sigma, args.ratio, args.p)
            if not torch.isfinite(loss):
                raise SystemExit(f"the loss is not finite at step {step}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            d = min(args.ema, (1 + step) / (10 + step))     # a short memory at first
            with torch.no_grad():
                for pe, pm in zip(ema.colour.parameters(), gen.colour.parameters()):
                    pe.lerp_(pm, 1.0 - d)
            for k, v in parts.items():
                acc[k], cnt[k] = acc.get(k, 0.0) + v, cnt.get(k, 0) + 1
            if step % 200 == 0:
                print(f"  step {step}/{steps}  " + "  ".join(f"{k} {v / cnt[k]:.4f}" for k, v in acc.items()) + f"  {step * args.bs / (time.time() - t0):.0f} samples/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        row = {"epoch": epoch + 1, "train": {k: v / cnt[k] for k, v in acc.items()}, "seconds": time.time() - t0}
        last = epoch + 1 == args.epochs or (args.max_steps and step >= args.max_steps)
        if val is not None and len(val) and ((epoch + 1) % args.eval_every == 0 or last):
            from tools.fill_eval import evaluate
            row["val"] = evaluate(ema, val, device, bs=args.bs, workers=min(4, args.workers), baselines=False)["model"]
            v = row["val"]
            print(f"epoch {epoch + 1}/{args.epochs}: " + "  ".join(f"{k} {x:.4f}" for k, x in row["train"].items()) +
                  f"  holes: colour L1 {v['rgb_l1']:.2f}  sharpness {v['sharpness']:.3f}  ({row['seconds']:.0f} s)", flush=True)
        else:
            print(f"epoch {epoch + 1}/{args.epochs}: " + "  ".join(f"{k} {x:.4f}" for k, x in row["train"].items()) + f"  ({row['seconds']:.0f} s)", flush=True)
        log.append(row)
        ema.scale = gen.scale
        torch.save({"model": ema.state_dict(), "config": ema.config(), "args": vars(args), "epoch": epoch + 1}, os.path.join(args.out, "last.pt"))
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump({"args": vars(args), "samples": len(ds), "scale": gen.scale, "runs": [os.path.basename(p) for p in ds.paths], "log": log}, f, indent=1)
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
