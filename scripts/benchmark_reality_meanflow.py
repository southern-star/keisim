#!/usr/bin/env python3
"""Budget a first-frame conditional MeanFlow U-Net with 100 real GPU updates. No test data or rollout is used."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, write_json
from keireality.meanflow import meanflow_terms
from keireality.multiview import MultiViewFrames
from keireality.flow_model import FlowUNet as BudgetUNet
from eval_reality_temporal import load_renderer
from train_reality import on_device, predict


def benchmark(args):
    torch.manual_seed(91)
    torch.set_num_threads(2)
    ds = MultiViewFrames(args.data, args.features, "train", views=(0,1,2), temporal=True, limit=12)
    renderer = load_renderer(args.temporal, ds.cache["checkpoint_sha256"], "cuda")
    conditions, teachers = [], []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=4):
            b = on_device(batch, "cuda")
            rgb, _, logits = predict(renderer, b)
            support = logits.softmax(1)[:, 1:3].sum(1, keepdim=True)
            current = F.avg_pool2d(torch.cat([rgb*2-1, support], 1), 2)
            conditions.append(torch.cat((current, torch.zeros_like(current)), 1))
            teachers.append(F.avg_pool2d(b["rgb"], 2)*2-1)
    condition, data = torch.cat(conditions), torch.cat(teachers)
    del renderer
    field = BudgetUNet().to(device="cuda", memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(field.parameters(), lr=1e-4)
    times, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(100):
        i = torch.randint(len(data), (4,), device="cuda")
        x, c = data[i], condition[i]
        pair = torch.rand(2, len(i), device="cuda")
        r, t = pair.amin(0), pair.amax(0)
        r = torch.where(torch.rand_like(t)<.25, t, r)
        noise = torch.randn_like(x)
        torch.cuda.synchronize()
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        u, target, _ = meanflow_terms(field, x, noise, r, t, c)
        loss = (u-target).square().mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(field.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter()-tick)
        losses.append(float(loss.detach()))
        if step in (0, 24, 49, 74, 99):
            print(f"update {step+1}: {times[-1]:.3f}s, loss {losses[-1]:.4f}", flush=True)
    if not np.isfinite(losses).all() or not torch.isfinite(norm):
        raise ValueError("non-finite numerical probe")
    result = dict(updates=len(times), warmup_excluded=10, parameters=sum(p.numel() for p in field.parameters()),
                  train_ticks=12, train_tick_views=len(data), batch_size=4, size=[64,128], precision="FP32", seed=91,
                  scope="first-frame optimizer throughput; frozen Phase 2 conditions precomputed; no quality evaluation",
                  gpu=torch.cuda.get_device_name(), peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  step_ms={f"p{p}":float(np.percentile(times[10:],p)*1000) for p in (50,95,99)},
                  mean_step_s=float(np.mean(times[10:])), all_updates_s=sum(times), all_finite=True,
                  dataset_sha256=ds.collection["contract_sha256"], renderer_sha256=file_hash(args.temporal))
    args.out.mkdir(parents=True,exist_ok=True)
    write_json(args.out / "throughput.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("data","features","temporal","out"):
        p.add_argument("--" + arg, type=Path, required=True)
    benchmark(p.parse_args())
