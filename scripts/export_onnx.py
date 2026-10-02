#!/usr/bin/env python3
"""Export a KeiPilot checkpoint to ONNX for in-browser inference (web/, onnxruntime-web).

  uv run --with onnx --with onnxruntime scripts/export_onnx.py runs/keipilot_v5/last.pt web/models/keipilot.onnx [--fp32]

Weights are stored as float16 and cast back to float32 inside the graph (onnxruntime folds the casts when the
session loads), so the file is half the size (30 MB) while every backend still computes in float32. These are the
same values as the released float16 checkpoint. --fp32 keeps float32 weights.

The small condition features are computed by the caller (web/src/pilot.js does it in JavaScript), so the graph
only holds plain tensor ops:
  img   (1,3,160,320) float32  RGB 0..255
  cond  (1,7)  float32  KeiPilot.encode_cond(cmd, target_point)
  speed (1,3)  float32  KeiPilot.encode_speed(v)        (speed-input models)
outputs: path (1,10,2) ego-frame waypoints [m], speed_probs (1,12) over 0..11 m/s, tl_probs (1,4) red/yellow/green/none,
seg (1,40,80) int64 class ids (quarter resolution).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keipilot.agent import load_model  # noqa: E402
from keipilot.model import N_PATH  # noqa: E402


class Exported(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, img, cond, speed):
        m = self.m
        x = (img / 255.0 - m.mean) / m.std
        x = m.stem(x)
        c1 = m.layer1(x)
        c2 = m.layer2(c1)
        c3 = m.layer3(c2)
        c4 = m.layer4(c3)
        p = m.lat[3](c4)
        p = m.lat[2](c3) + F.interpolate(p, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p = m.lat[1](c2) + F.interpolate(p, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p = m.lat[0](c1) + F.interpolate(p, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        seg = m.seg_head(p).argmax(1)
        t3 = (m.proj3(c3) + m.pos3).flatten(2).transpose(1, 2)
        t4 = (m.proj4(c4) + m.pos4).flatten(2).transpose(1, 2)
        mem = torch.cat([t3, t4], 1)
        c = m.cond(cond)
        if m.speed_input:
            c = c + m.speed_enc(speed)
        q = m.queries[None] + c[:, None]
        h = m.decoder(q, mem)
        path = m.path_prior[None] + m.path_head(h[:, :N_PATH]) * 2.0
        return path, m.speed_head(h[:, N_PATH]).softmax(-1), m.tl_head(h[:, N_PATH + 1]).softmax(-1), seg


def fp16_weights(path, min_size=256):
    """Store the float32 initializers with at least min_size elements as float16, each followed by a Cast."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m = onnx.load(path)
    g = m.graph
    casts = []
    for init in list(g.initializer):
        if init.data_type != TensorProto.FLOAT or int(np.prod(init.dims)) < min_size:
            continue
        half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "_fp16")
        g.initializer.remove(init)
        g.initializer.append(half)
        casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast"))
    for node in reversed(casts):
        g.node.insert(0, node)
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return len(casts)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src, dst = args[0], args[1]
    torch.backends.mha.set_fastpath_enabled(False)          # export the plain attention ops
    model = load_model(src, "cpu").float().eval()
    assert not model.history, "history models are not exported (the browser demo feeds single frames)"
    net = Exported(model).eval()
    rng = np.random.default_rng(0)
    img = torch.from_numpy(rng.integers(0, 256, (1, 3, 160, 320)).astype(np.float32))
    cmd, tp, v = torch.tensor([1]), torch.tensor([[18.0, -3.0]]), torch.tensor([5.0])
    cond, spd = model.encode_cond(cmd, tp), model.encode_speed(v)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.onnx.export(net, (img, cond, spd), dst, opset_version=17, dynamo=False,
                      input_names=["img", "cond", "speed"], output_names=["path", "speed_probs", "tl_probs", "seg"])
    if "--fp32" not in sys.argv:
        print(f"float16 storage for {fp16_weights(dst)} weight tensors")
    # parity check against PyTorch (float32 weights)
    import onnxruntime as ort
    with torch.no_grad():
        ref = [t.numpy() for t in net(img, cond, spd)]
    out = ort.InferenceSession(dst).run(None, {"img": img.numpy(), "cond": cond.numpy(), "speed": spd.numpy()})
    for name, a, b in zip(["path", "speed_probs", "tl_probs", "seg"], ref, out):
        err = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if name != "seg" else float((a != b).mean())
        print(f"{name:12s} shape {tuple(b.shape)}  max |torch - onnx| {err:.2e}" + ("  (fraction of differing pixels)" if name == "seg" else ""))
    print(f"{dst}: {os.path.getsize(dst) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
