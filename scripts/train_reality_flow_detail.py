#!/usr/bin/env python3
"""Fine-tune the frozen G3 checkpoint through the full one- and four-interval trajectories."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, write_json, writer_lock
from keireality.flow_data import FlowSplit
from keireality.flow_detail import image_losses
from keireality.flow_model import ConditionalVelocity, make_condition
from keireality.meanflow import integrate_meanflow, meanflow_terms
from keisim.config import SEM
from train_reality import save_checkpoint
from train_reality_flow import lr_at, select_window, training_windows


def train(args):
    torch.set_num_threads(2)
    cfg = {**json.loads(args.base_config.read_text()), **json.loads(args.config.read_text())}
    batch, window = cfg["batch_size"], cfg["window"]
    if cfg["updates"] % window or cfg["checkpoint_every"] % window:
        raise ValueError("a resumed run must start at a window boundary")
    ck = torch.load(args.init, map_location="cpu")
    ds = FlowSplit(args.data, args.cache, "train", args.device)
    base = ck["contract"]
    if (base["mode"] != "meanflow" or base["protocol"] != file_hash(args.base_config)
            or base["train"] != file_hash(ds.cache / "manifest.json") or ds.angles != cfg["train_angles"]):
        raise ValueError("the original training data, renderer and architecture must match")
    contract = dict(mode="meanflow", init=file_hash(args.init), protocol=file_hash(args.config),
                    original_protocol=base["protocol"], renderer=base["renderer"], train=base["train"])
    net = ConditionalVelocity(ck["sigma"]).to(device=args.device, memory_format=torch.channels_last)
    net.load_state_dict(ck["model"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    blocks = training_windows(ds, cfg)
    with writer_lock(args.out):
        last = args.out / "last.pt"
        start, elapsed, log = 0, 0., []
        seen = np.zeros((len(ds.current), len(ds.angles)), bool)
        if last.exists():
            resume = torch.load(last, map_location=args.device)
            if resume["contract"] != contract:
                raise ValueError("existing fine-tuning checkpoint has a different contract")
            start, elapsed, log = resume["step"], resume["elapsed_s"], resume["log"]
            net.load_state_dict(resume["model"])
            opt.load_state_dict(resume["optimizer"])
            seen[:] = resume["seen"].cpu().numpy()
            del resume
        final = args.out / "model.pt"
        if start == cfg["updates"] and (args.out / "training.json").exists():
            if file_hash(final) != json.loads((args.out / "training.json").read_text())["model_sha256"]:
                raise ValueError("finished detail checkpoint failed verification")
            return
        tick = time.perf_counter()
        sums = np.zeros(5)
        lanes = torch.arange(batch, device=args.device)
        for k in range(start, cfg["updates"]):
            pos = k % window
            if pos == 0:
                starts, angles = select_window(blocks, k // window, cfg, len(ds.angles))
                previous = torch.zeros(batch, 3, 64, 128, device=args.device)
                old_view = torch.zeros(batch, 11, device=args.device)
                use_four = (lanes + k // window) % 2 == 0
            indices, views = starts + pos, angles[pos]
            seen[indices, views] = True
            ti = torch.as_tensor(indices, device=args.device)
            vi = torch.as_tensor(views, device=args.device)
            view = ds.poses[ti, vi]
            rng = torch.Generator(device=args.device).manual_seed(cfg["seed"] * 1000003 + k)
            keep = (torch.rand(batch, device=args.device, generator=rng) >= cfg["previous_dropout"]) & (pos > 0)
            cond = make_condition(ds.current[ti, vi], view, previous, old_view, ds.delta[ti],
                                  view.new_full((batch,), .1), keep)
            x = ds.target[ti, vi].float()
            eps = torch.randn(x.shape, device=args.device, generator=rng)
            r, t = torch.rand(2, batch, device=args.device, generator=rng).sort(0).values
            draw = torch.rand(batch, device=args.device, generator=rng)
            r = torch.where(draw < cfg["equal_times_probability"], t, r)
            endpoint = draw > 1 - cfg["endpoint_probability"]
            r, t = torch.where(endpoint, 0., r), torch.where(endpoint, 1., t)
            field, target, _ = meanflow_terms(net, x, eps, r, t, cond)
            flow_loss = (field - target).square().mean()
            source = torch.randn(x.shape, device=args.device, generator=rng)
            one = integrate_meanflow(net, source, cond, 1)
            four = integrate_meanflow(net, source, cond, 4)
            weights = torch.where(ds.seg[ti, vi] >= SEM["vehicle"], cfg["endpoint_dynamic_pixel_weight"], 1.)
            rgb1, edge1 = image_losses(one, x, weights, cfg["edge_threshold_rgb01"], cfg["strong_edge_weight"])
            rgb4, edge4 = image_losses(four, x, weights, cfg["edge_threshold_rgb01"], cfg["strong_edge_weight"])
            loss = (flow_loss + cfg["rgb_weight_one"] * rgb1 + cfg["rgb_weight_four"] * rgb4
                    + cfg["structure_weight_one"] * edge1 + cfg["structure_weight_four"] * edge4)
            opt.param_groups[0]["lr"] = lr_at(k + 1, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            opt.step()
            previous = torch.where(use_four[:, None, None, None], four.detach(), one.detach()).clamp(-1, 1)
            old_view = view.detach()
            sums += [float(v.detach()) for v in (flow_loss, rgb1, rgb4, edge1, edge4)]
            step = k + 1
            if step % 50 == 0:
                log.append(dict(step=step, **dict(zip(("flow", "rgb1", "rgb4", "edge1", "edge4"),
                                                      (sums / 50).tolist(), strict=True))))
                print(step, "four-step RGB", round(log[-1]["rgb4"], 4), flush=True)
                sums[:] = 0
            if step % cfg["checkpoint_every"] == 0 or step == cfg["updates"]:
                save_checkpoint(last, dict(contract=contract, model=net.state_dict(), optimizer=opt.state_dict(),
                                          step=step, seen=torch.from_numpy(seen.copy()), log=log,
                                          elapsed_s=elapsed + time.perf_counter() - tick))
        save_checkpoint(final, dict(contract=contract, model=net.state_dict(), sigma=ck["sigma"], step=cfg["updates"]))
        write_json(args.out / "training.json", dict(contract=contract, updates=cfg["updates"],
                   model_sha256=file_hash(final), samples=cfg["updates"]*batch, unique_pairs=int(seen.sum()),
                   seconds=elapsed + time.perf_counter() - tick, log=log))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("base-config", "config", "data", "cache", "init", "out"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    train(p.parse_args())
