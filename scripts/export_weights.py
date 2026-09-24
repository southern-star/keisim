#!/usr/bin/env python3
"""Export a training checkpoint as a compact fp16 inference checkpoint.

  uv run scripts/export_weights.py runs/keipilot_dagger/last.pt runs/keipilot.pt
"""
from __future__ import annotations

import os
import sys

import torch


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu")
    model = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["model"].items()}
    val = ck.get("val", {})
    info = {"source": os.path.basename(os.path.dirname(src)) + "/" + os.path.basename(src),
            "epoch": ck.get("epoch"), "val_ADE_m": val.get("ADE_m"), "val_tl_acc": val.get("tl_acc")}
    torch.save({"model": model, "model_cfg": ck.get("model_cfg", {}), "info": info}, dst)
    print(f"{dst}: {os.path.getsize(dst) / 1e6:.1f} MB", info)


if __name__ == "__main__":
    main()
