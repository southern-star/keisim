#!/usr/bin/env python3
"""Phase 3 numerical preflight on training frames. Check FP32 JVP, gradients and noise-to-data sampling."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, write_json
from keireality.meanflow import broadcast_time, meanflow_terms, sample_meanflow
from keireality.multiview import MultiViewFrames
from eval_reality_temporal import load_renderer
from train_reality import predict


class ProbeField(nn.Module):
    """A small smooth conditional field for differentiation checks, including dependence on both interval endpoints."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(9, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, 3, 1))

    def forward(self, z, r, t, condition):
        times = [broadcast_time(x, z).expand(-1, 1, *z.shape[-2:]) for x in (r, t)]
        return self.net(torch.cat([z, condition, *times], 1))


def check(args):
    torch.set_num_threads(2)
    torch.manual_seed(37)
    ds = MultiViewFrames(args.data, args.features, "train", views=(0,), temporal=True, limit=4)
    b = next(iter(DataLoader(ds, batch_size=4)))
    renderer = load_renderer(args.temporal, ds.cache["checkpoint_sha256"], "cpu")
    with torch.no_grad():
        image, _, logits = predict(renderer, b)
        support = logits.softmax(1)[:, 1:3].sum(1, keepdim=True)
        condition = F.avg_pool2d(torch.cat([image, support], 1), 4)
    data = F.avg_pool2d(b["rgb"], 4) * 2 - 1
    noise = torch.randn_like(data)
    field = ProbeField()
    t = torch.tensor([.27, .49, .71, .87])
    r = torch.tensor([.03, .18, .22, .36])
    output, target, jvp = meanflow_terms(field, data, noise, r, t, condition)
    velocity = noise - data
    z = data + broadcast_time(t, data) * velocity
    h = .002
    with torch.no_grad():
        fd = (field(z+h*velocity, r, t+h, condition) - field(z-h*velocity, r, t-h, condition))/(2*h)
        fd_error = float((fd-jvp).norm() / jvp.norm())
    loss = (output-target).square().mean()
    loss.backward()
    gradients = [p.grad for p in field.parameters()]
    grad_norm = float(torch.cat([g.flatten() for g in gradients]).norm())
    optimizer = torch.optim.AdamW(field.parameters(), lr=.001)
    optimizer.step()
    _, boundary, _ = meanflow_terms(field, data, noise, t, t, condition)
    # A known constant transport field has an exact reconstruction at every number of steps.
    oracle_errors, calls = {}, {}
    for count in (1, 4, 8):
        actual_calls = []
        def oracle(z_, r_, t_, c_):
            actual_calls.append(1)
            return velocity
        result = sample_meanflow(oracle, noise, condition, count)
        oracle_errors[count] = float((result-data).abs().max())
        calls[count] = len(actual_calls)
    learned = sample_meanflow(field, noise, condition)
    direct = noise - field(noise, torch.zeros_like(t), torch.ones_like(t), condition)
    endpoint_error = float((learned-direct.detach()).abs().max())
    criteria = dict(fp32=data.dtype == jvp.dtype == torch.float32, finite_difference=fd_error < .001,
                    target_stop_gradient=not target.requires_grad, finite_parameter_gradients=grad_norm > 0 and
                    all(bool(torch.isfinite(g).all()) for g in gradients), instantaneous_boundary=torch.equal(boundary, velocity),
                    one_call_endpoint=endpoint_error == 0, oracle=max(oracle_errors.values()) < 3e-6,
                    nfe_counts=all(k == v for k, v in calls.items()))
    report = dict(status="PASS" if all(criteria.values()) else "FAIL", criteria=criteria,
                  source="https://arxiv.org/html/2505.13447v1#S4.SS1", split="train", training_ticks=len(data),
                  shape=list(data.shape), optimization_steps=1, condition="frozen Phase 2 RGB and estimated support",
                  seed=37, dataset_sha256=ds.collection["contract_sha256"], renderer_sha256=file_hash(args.temporal),
                  finite_difference_relative_l2=fd_error, finite_difference_step=h, gradient_norm=grad_norm,
                  loss=float(loss.detach()), endpoint_max_error=endpoint_error,
                  analytical_max_error=oracle_errors, function_evaluations=calls)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "numerics.json", report)
    print(json.dumps(report))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "features", "temporal", "out"):
        p.add_argument("--" + arg, type=Path, required=True)
    check(p.parse_args())
