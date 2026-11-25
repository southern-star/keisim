#!/usr/bin/env python3
"""Train only the KeiReality decoder from a read-only feature cache. Select checkpoints on validation."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, write_json, writer_lock
from keireality.geometry import ORBIT_ANGLES
from keireality.learning import RealityFrames, objective
from keireality.multiview import MultiViewFrames
from keireality.renderer import SingleFrameRenderer
from keireality.spatial import SpatialRenderer
from keisim.config import SEM


def save_checkpoint(path, obj):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        torch.save(obj, tmp)
    os.replace(tmp.name, path)


def on_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def predict(model, b):
    if isinstance(model, SpatialRenderer):
        return model(b["mem"], b["view_pose"], b["c2"], b["c3"], b.get("past"), b.get("motion"), b.get("valid"))
    rgb, logits = model(b["mem"].float(), b["view_pose"])
    return rgb, logits, None


def training_loss(rgb, logits, support, b, weights, cfg):
    loss = objective(rgb, logits, b, weights, cfg.get("past_rgb_weight", 1.))
    if support is not None:
        target = b["support"].long().masked_fill(b["support"] == 255, 3)
        ws = weights.new_tensor(cfg["support_class_weights"])
        loss = loss + cfg["support_loss_weight"] * torch.nn.functional.cross_entropy(support.float(), target, weight=ws)
    return loss


@torch.inference_mode()
def validate(model, dataset, bs, weights, device, cfg, ablate=False):
    model.eval()
    totals = dict(loss=0., l1=0.)
    intersection = {"road": 0, "vehicle": 0}
    union = dict(intersection)
    if ablate:
        totals.update(shuffled_l1=0., zero_l1=0.)
    for batch in DataLoader(dataset, batch_size=bs):
        b = on_device(batch, device)
        rgb, logits, support = predict(model, b)
        n = len(rgb)
        totals["loss"] += n * training_loss(rgb, logits, support, b, weights, cfg).item()
        totals["l1"] += n * (rgb - b["rgb"]).abs().mean().item()
        labels = logits.argmax(1)
        visible = b["visibility"] == 1
        for name in intersection:
            wanted = b["semantic"] == SEM[name]
            found = labels == SEM[name]
            intersection[name] += (visible & wanted & found).sum().item()
            union[name] += (visible & (wanted | found)).sum().item()
        if ablate:
            # Roll over the whole set, including a final singleton batch.
            other = torch.stack([dataset.mem_at((i + max(1, len(dataset) // 2)) % len(dataset))
                                 for i in batch["index"].tolist()]).to(device)
            for label, mem in (("shuffled_l1", other), ("zero_l1", torch.zeros_like(b["mem"]))):
                out, _ = model(mem, b["view_pose"])
                totals[label] += n * (out - b["rgb"]).abs().mean().item()
    result = {k: v / len(dataset) for k, v in totals.items()}
    result.update({name + "_visible_iou": intersection[name] / n if n else None for name, n in union.items()})
    return result


def train(args):
    cfg = json.loads(Path(args.config).read_text())
    for key in ("max_steps", "epochs", "batch_size", "limit"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
        if key in cfg and cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    device = args.device
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(2)
    spatial, temporal = args.arch != "mem", args.arch == "temporal"
    def dataset(split):
        if spatial:
            views = tuple(ORBIT_ANGLES.index(a) for a in cfg["train_views_deg"]) if args.labels else (0,)
            return MultiViewFrames(args.data, args.features, split, views, args.labels, temporal, cfg.get("limit"))
        return RealityFrames(args.data, args.features, split, cfg.get("limit"))
    ds = dataset("train")
    tiny = "limit" in cfg
    val = ds if tiny else dataset("val")
    cfg.update(data_sha256=ds.collection["contract_sha256"], pilot_sha256=ds.cache["checkpoint_sha256"],
               device=device, amp=device == "cuda", validation_split="train-overfit" if tiny else "val")
    model = SpatialRenderer(bool(args.labels), temporal) if spatial else SingleFrameRenderer()
    if spatial:
        cfg.update(arch=args.arch, feature_manifest_sha256=file_hash(Path(args.features) / "manifest.json"),
                   labels_manifest_sha256=file_hash(Path(args.labels) / "manifest.json") if args.labels else None)
    if args.init:
        state = torch.load(args.init, map_location="cpu")
        if state["config"]["pilot_sha256"] != cfg["pilot_sha256"]:
            raise ValueError("warm start and training cache use different frozen encoders")
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if unexpected or any(not k.startswith(("support.", "history.")) for k in missing):
            raise ValueError("warm start is incompatible with this renderer")
        cfg["init_sha256"] = file_hash(args.init)
    model = model.to(device=device, memory_format=torch.channels_last)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    weights = torch.tensor(cfg["class_weights"], device=device, dtype=torch.float32)
    bs, cap = cfg["batch_size"], cfg.get("max_steps", 2**40)
    out = Path(args.out)
    with writer_lock(out):
        config_file = out / "config.json"
        if config_file.exists() and json.loads(config_file.read_text()) != cfg:
            raise ValueError("output directory has a different training configuration")
        write_json(config_file, cfg)
        tick = time.perf_counter()
        state_file = out / "last.pt"
        history, step, epoch, best = [], 0, 0, float("inf")
        start_metric = None
        if state_file.exists():
            state = torch.load(state_file, map_location=device)
            model.load_state_dict(state["model"])
            optim.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            history, step, epoch, best = state["history"], state["step"], state["epoch"], state["best"]
            start_metric = state["initial"]
            if (out / "training.json").exists() and (epoch >= cfg["epochs"] or step >= cap):
                saved = json.loads((out / "training.json").read_text())
                if saved["steps"] == step and (out / "best.pt").is_file():
                    print(f"Verified finished run, {step} steps")
                    return
        if tiny and start_metric is None:
            start_metric = validate(model, val, bs, weights, device, cfg)
        while epoch < cfg["epochs"] and step < cap:
            model.train()
            sampler_rng = torch.Generator().manual_seed(cfg["seed"] + epoch)
            loader = DataLoader(ds, batch_size=bs, shuffle=True, generator=sampler_rng, pin_memory=device == "cuda")
            total, n = 0., 0
            for batch in loader:
                b = on_device(batch, device)
                optim.zero_grad(set_to_none=True)
                with torch.autocast(device, enabled=device == "cuda", dtype=torch.float16):
                    rgb, logits, support = predict(model, b)
                loss = training_loss(rgb, logits, support, b, weights, cfg)
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite loss at step {step}")
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                total += loss.item() * len(rgb)
                n += len(rgb)
                step += 1
                if step % 100 == 0:
                    print(f"step {step}, training loss {total / n:.4f}", flush=True)
                if step >= cap:
                    break
            epoch += 1
            # Tiny epochs contain only a few steps. Run their validation and checkpoint at regular intervals.
            if tiny and epoch % 10 and step < cap and epoch < cfg["epochs"]:
                continue
            score = validate(model, val, bs, weights, device, cfg)
            history.append(dict(epoch=epoch, step=step, train_loss=total / n, **score))
            print(f"epoch {epoch} step {step} validation {score}", flush=True)
            if score["loss"] < best:
                best = score["loss"]
                save_checkpoint(out / "best.pt", dict(model=model.state_dict(), config=cfg, epoch=epoch, step=step))
            save_checkpoint(state_file, dict(model=model.state_dict(), optimizer=optim.state_dict(), scaler=scaler.state_dict(),
                                             epoch=epoch, step=step, best=best, history=history, initial=start_metric))
            with (out / "train_log.jsonl").open("w") as log:
                for row in history:
                    print(json.dumps(row), file=log)
        model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
        final = validate(model, val, bs, weights, device, cfg, ablate=not spatial)
        report = dict(steps=step, epochs=epoch, parameters=sum(p.numel() for p in model.parameters()),
                      train_frames=len(ds), validation_frames=len(val), initial=start_metric, selected=final,
                      elapsed_this_invocation_s=time.perf_counter() - tick,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None)
        if tiny and not spatial:
            report["overfit_pass"] = (final["l1"] < .7 * start_metric["l1"] and
                                      final["l1"] < min(final["shuffled_l1"], final["zero_l1"]))
        write_json(out / "training.json", report)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "features", "out"):
        parser.add_argument("--" + name, required=True)
    for name in ("max-steps", "epochs", "batch-size", "limit"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--arch", choices=("mem", "spatial", "temporal"), default="mem")
    parser.add_argument("--labels", help="causal support supervision for multiview training")
    parser.add_argument("--init", help="initialize common weights from a single-frame checkpoint")
    train(parser.parse_args())
