"""Train ReconFill's colour net the other way (docs/recon_fill.md 6.10): no noise, an L1 loss and a patch
discriminator (keipilot/fillgen.py: FillGen with param "g", PatchDisc), on top of a trained fill model of
scripts/train_fill.py (P1; not changed).

    python scripts/train_fillgan.py --shape runs/reconfill_b/last.pt --data data/recon_fill/train --val data/recon_fill/eval
        --out runs/reconfill_gan1 [--epochs 60] [--bs 32] [--lr 2e-4] [--lam 100] [--width 16 32 64 128] [--dwidth 32 64 128]

Writes OUT/last.pt (P1's net, the colour net's moving average, the configuration; loads with
keipilot.fillnet.load_fillnet) and OUT/log.json (per epoch the losses' parts and, every --eval_every epochs, the
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
from keipilot.fillgen import FillGen, PatchDisc, colour_scale, disc_loss, gan_picture, gen_loss  # noqa: E402
from keipilot.fillnet import load_fillnet, macs  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, help="P1's checkpoint")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam", type=float, default=100.0, help="the L1 loss's weight (colours in 0..1) against the adversarial loss")
    ap.add_argument("--width", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--dwidth", type=int, nargs="+", default=[32, 64, 128])
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
    gen = FillGen(load_fillnet(args.shape, device), width=args.width, param="g").to(device).to(memory_format=torch.channels_last)
    gen.scale = colour_scale(gen, loader, device)
    gen.train()
    ncond = (11 if gen.loose else 6) + 5
    disc = PatchDisc(ncond, args.dwidth).to(device).to(memory_format=torch.channels_last)
    gp, dp = list(gen.colour.parameters()), list(disc.parameters())
    print(f"{len(ds):,} samples of {len(ds.runs)} runs; the colour net: {sum(p.numel() for p in gp):,} parameters, with P1's net {macs(gen) / 1e9:.2f} G multiply-adds; "
          f"the discriminator: {sum(p.numel() for p in dp):,} parameters; the colour difference's scale {gen.scale:.4f}", flush=True)
    ema = copy.deepcopy(gen).eval()
    steps = args.epochs * len(loader)
    og = torch.optim.Adam(gp, lr=args.lr, betas=(0.5, 0.999))
    od = torch.optim.Adam(dp, lr=args.lr, betas=(0.5, 0.999))
    amp = device.startswith("cuda")
    sg, sd = torch.amp.GradScaler(enabled=amp), torch.amp.GradScaler(enabled=amp)
    cast = lambda: torch.autocast("cuda" if amp else "cpu", dtype=torch.float16, enabled=amp)
    os.makedirs(args.out, exist_ok=True)
    log, step, t0 = [], 0, time.time()
    for epoch in range(args.epochs):
        acc, cnt = {}, {}
        for b in loader:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with cast():
                cond, fake, comp, w = gan_picture(gen, b)
                ld, parts = disc_loss(disc, cond, comp, b["gt_rgb"], w)
            od.zero_grad(set_to_none=True)
            sd.scale(ld).backward()
            sd.step(od)
            sd.update()
            for p in dp:
                p.requires_grad_(False)
            with cast():
                lg, pg = gen_loss(disc, cond, fake, comp, b["gt_rgb"], w, args.lam)
            for p in dp:
                p.requires_grad_(True)
            if not (torch.isfinite(lg) and torch.isfinite(ld)):
                raise SystemExit(f"a loss is not finite at step {step}")
            og.zero_grad(set_to_none=True)
            sg.scale(lg).backward()
            sg.unscale_(og)
            torch.nn.utils.clip_grad_norm_(gp, 5.0)
            sg.step(og)
            sg.update()
            step += 1
            d = min(args.ema, (1 + step) / (10 + step))     # a short memory at first
            with torch.no_grad():
                for pe, pm in zip(ema.colour.parameters(), gen.colour.parameters()):
                    pe.lerp_(pm, 1.0 - d)
            for k, v in {**parts, **pg, "d": ld.item()}.items():
                acc[k], cnt[k] = acc.get(k, 0.0) + v, cnt.get(k, 0) + 1
            if step % 200 == 0:
                print(f"  step {step}/{steps}  " + "  ".join(f"{k} {v / cnt[k]:.4f}" for k, v in acc.items()) + f"  {step * args.bs / (time.time() - t0):.0f} samples/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        row = {"epoch": epoch + 1, "train": {k: v / cnt[k] for k, v in acc.items()}, "seconds": time.time() - t0}
        last = epoch + 1 == args.epochs or (args.max_steps and step >= args.max_steps)
        text = f"epoch {epoch + 1}/{args.epochs}: " + "  ".join(f"{k} {x:.4f}" for k, x in row["train"].items())
        if val is not None and len(val) and ((epoch + 1) % args.eval_every == 0 or last):
            from tools.fill_eval import evaluate
            v = row["val"] = evaluate(ema, val, device, bs=args.bs, workers=min(4, args.workers), baselines=False)["model"]
            text += f"  holes: colour L1 {v['rgb_l1']:.2f}  sharpness {v['sharpness']:.3f}  lines {v['lines']:.3f}"
        print(text + f"  ({time.time() - t0:.0f} s)", flush=True)
        log.append(row)
        ema.scale = gen.scale
        torch.save({"model": ema.state_dict(), "config": ema.config(), "args": vars(args), "epoch": epoch + 1}, os.path.join(args.out, "last.pt"))
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump({"args": vars(args), "samples": len(ds), "scale": gen.scale, "runs": [os.path.basename(p) for p in ds.paths], "log": log}, f, indent=1)
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
