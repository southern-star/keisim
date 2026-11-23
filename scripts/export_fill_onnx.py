#!/usr/bin/env python3
"""Export ReconFill's model (keipilot/fillnet.py; docs/recon_fill.md) to ONNX for the browser (web/src/fill.js).

    python scripts/export_fill_onnx.py runs/reconfill_a/last.pt web/models/reconfill.onnx
    python scripts/export_fill_onnx.py OUT.onnx --random 16 32 64 128 [--no_loose] [--norm gn]   (random weights: timing)

Inputs, all from ReconView's chase camera, float32:
  in_rgb    (1, 3, 160, 320)  the view without a fill, RGB 0..255 (the background where there is no surface)
  in_dep    (1, 1, 160, 320)  its depth as 16 bit codes 0..65535 (web/src/unproject.js; 0: no surface)
  loose_rgb, loose_dep        the current frame alone by the loose triangle rule, the same way (models with loose input)
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

    def forward(self, in_rgb, in_dep, loose_rgb=None, loose_dep=None):
        if self.net.loose:
            rgb, code, surf = self.net(in_rgb / 255.0, in_dep / 65535.0, loose_rgb / 255.0, loose_dep / 65535.0)
        else:
            rgb, code, surf = self.net(in_rgb / 255.0, in_dep / 65535.0)
        metres = code_to_metres(code)
        depth = torch.where((surf > 0) & (metres <= MAX_R), metres, torch.zeros_like(metres))
        return rgb[0].clamp(0.0, 1.0) * 255.0, depth[0, 0, ::2, ::2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="CKPT DST, or DST with --random")
    ap.add_argument("--random", type=int, nargs="+", default=None, help="widths of a net with random weights")
    ap.add_argument("--no_loose", action="store_true")
    ap.add_argument("--norm", default="none")
    args = ap.parse_args()
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
