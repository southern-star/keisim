#!/usr/bin/env python3
"""Export ReconFill's model (keipilot/fillnet.py; docs/recon_fill.md) to ONNX for the browser (web/src/fill.js).

    python scripts/export_fill_onnx.py runs/reconfill_a/last.pt web/models/reconfill.onnx
    python scripts/export_fill_onnx.py OUT.onnx --random 16 32 64 128 [--no_loose] [--norm gn]   (random weights: timing)
    python scripts/export_fill_onnx.py runs/reconfill_gen1/last.pt web/models/reconfill_gen.onnx   (P2: P1's net, the colour
        net of keipilot/fillgen.py and its fixed noise image in one file; the same inputs and outputs)
    python scripts/export_fill_onnx.py --same A.onnx B.onnx      (P2 keeps P1's shape: the two files' depths on eval samples)

Inputs, all from ReconView's chase camera, float32:
  in_rgb    (1, 3, 160, 320)  the view without a fill, RGB 0..255 (the background where there is no surface)
  in_dep    (1, 1, 160, 320)  its depth as 16 bit codes 0..65535 (web/src/unproject.js; 0: no surface)
  loose_rgb, loose_dep        the current frame alone by the loose triangle rule, the same way (models with loose input)
  prev_rgb, prev_dep          the fill before this one drawn alone from this frame's chase camera, the same way; all
                              zero when there is none (models with it: keipilot/fillgen.py FillGen with prev)
Outputs:
  rgb    (3, 160, 320) float32  the fill's colours, 0..255
  depth  (80, 160) float32      metres from the chase camera along the lines of sight of the pixels (2i, 2j); 0: no
                                surface (none predicted, or farther than 80 m). The promise of the driving model's depth.
The ground's depth and every normalisation are inside the graph."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keipilot.filldata import CHASE, MAX_R  # noqa: E402
from keipilot.fillnet import FillNet, code_to_metres, load_fillnet, macs  # noqa: E402


class FillOnnx(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.names = ["in_rgb", "in_dep"] + (["loose_rgb", "loose_dep"] if net.loose else []) + (["prev_rgb", "prev_dep"] if getattr(net, "prev", False) else [])

    def forward(self, *args):
        a = {k: v / (255.0 if k.endswith("rgb") else 65535.0) for k, v in zip(self.names, args)}
        more = {k: a[k] for k in ("prev_rgb", "prev_dep") if k in a}
        rgb, code, surf = self.net(a["in_rgb"], a["in_dep"], a.get("loose_rgb"), a.get("loose_dep"), **more)
        metres = code_to_metres(code)
        depth = torch.where((surf > 0) & (metres <= MAX_R), metres, torch.zeros_like(metres))
        return rgb[0].clamp(0.0, 1.0) * 255.0, depth[0, 0, ::2, ::2]


def same_shape(a, b, data, every=9):
    """P2 keeps P1's shape (docs/recon_fill.md 6.4): the depth outputs of two ONNX files on every 9th eval sample."""
    import onnxruntime as ort
    from keipilot.filldata import FillDataset
    ds = FillDataset([data])
    sa, sb = (ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for p in (a, b))
    worst, flips, rgb, n = 0.0, 0, 0.0, 0
    for i in range(0, len(ds), every):
        x = ds[i]
        feeds = {"in_rgb": (x["in_rgb"][None] * 255.0).numpy().round(), "in_dep": (x["in_dep"][None] * 65535.0).numpy().round(),
                 "loose_rgb": (x["loose_rgb"][None] * 255.0).numpy().round(), "loose_dep": (x["loose_dep"][None] * 65535.0).numpy().round()}
        feeds.update(prev_rgb=np.zeros_like(feeds["in_rgb"]), prev_dep=np.zeros_like(feeds["in_dep"]))    # no fill before
        (ra, da), (rb, db) = (s.run(None, {k.name: feeds[k.name] for k in s.get_inputs()}) for s in (sa, sb))
        worst, flips, rgb, n = max(worst, float(np.abs(da - db).max())), flips + int(((da > 0) != (db > 0)).sum()), rgb + float(np.abs(ra - rb).mean()), n + 1
    print(f"{os.path.basename(a)} against {os.path.basename(b)} on {n} eval samples: depth differs by at most {worst:g} m, pixels whose surface differs: {flips}; "
          f"colours {rgb / n:.1f} apart (0..255)")
    if worst > 0 or flips:
        sys.exit("the shapes differ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="CKPT DST, or DST with --random")
    ap.add_argument("--random", type=int, nargs="+", default=None, help="widths of a net with random weights")
    ap.add_argument("--no_loose", action="store_true")
    ap.add_argument("--norm", default="none")
    ap.add_argument("--same", action="store_true", help="A.onnx B.onnx: do the two give the same depth (the same shape)? On the eval samples")
    ap.add_argument("--data", default="data/recon_fill/eval")
    args = ap.parse_args()
    if args.same:
        return same_shape(args.paths[0], args.paths[1], args.data)
    args.dst = args.paths[-1]
    assert len(args.paths) == (1 if args.random else 2), "CKPT DST, or DST with --random"
    if args.random:
        torch.manual_seed(0)
        model = FillNet(width=args.random, loose=not args.no_loose, norm=args.norm).eval()
        nn.init.normal_(model.head.weight, std=0.02)        # not the zeros a fresh net starts from: every layer counts
    else:
        model = load_fillnet(args.paths[0])
    net = FillOnnx(model).eval()
    H, W = CHASE.height, CHASE.width
    rng = np.random.default_rng(0)
    dep = rng.integers(20000, 52000, (1, 1, H, W)).astype(np.float32)
    dep[rng.random(dep.shape) < 0.2] = 0
    feeds = {"in_rgb": rng.uniform(0, 255, (1, 3, H, W)).astype(np.float32), "in_dep": dep}
    if model.loose:
        feeds.update(loose_rgb=rng.uniform(0, 255, (1, 3, H, W)).astype(np.float32), loose_dep=np.roll(dep, 7, 3))
    if getattr(model, "prev", False):
        feeds.update(prev_rgb=rng.uniform(0, 255, (1, 3, H, W)).astype(np.float32), prev_dep=np.roll(dep, 11, 2))
    assert list(feeds) == net.names
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.onnx.export(net, tuple(torch.from_numpy(v) for v in feeds.values()), args.dst, opset_version=17, dynamo=False,
                      input_names=list(feeds), output_names=["rgb", "depth"])
    import onnxruntime as ort
    sess = ort.InferenceSession(args.dst, providers=["CPUExecutionProvider"])
    got = sess.run(None, feeds)
    with torch.no_grad():
        want = net(*(torch.from_numpy(v) for v in feeds.values()))
    for name, g, w in zip(("rgb", "depth"), got, want):
        w = w.numpy()
        flips = float(np.mean((g > 0) != (w > 0))) if name == "depth" else 0.0
        same = (g > 0) == (w > 0)
        print(f"{name:6s} shape {g.shape}  max |torch - onnx| {np.abs(g - w)[same].max():.2e}" + (f"  (pixels whose surface differs: {100 * flips:.3f} %)" if name == "depth" else ""))
    print(f"{args.dst}: {os.path.getsize(args.dst) / 1e6:.2f} MB, {sum(p.numel() for p in model.parameters()):,} parameters, {macs(model) / 1e9:.2f} G multiply-adds, "
          f"inputs {', '.join(feeds)}")


if __name__ == "__main__":
    main()
