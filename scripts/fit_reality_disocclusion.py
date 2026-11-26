#!/usr/bin/env python3
"""Fit a localized background repair, then pick the method and strength on separate development towns."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from collect_reality_resolution import OUT as ORIGINAL, ROOT, TRACKS, open_fields
from keireality.dataset import file_hash, write_json, writer_lock
from keireality.disocclusion import BackgroundRepair, actor_support
from keisim.config import SEM
from prepare_reality_disocclusion import MANIFEST, OUT, verify_release
from score_reality_resolution import Scores, bytes_rgb
from train_reality import save_checkpoint


CONFIG = ROOT / "configs/reality/p2_disocclusion.json"
CFG = json.loads(CONFIG.read_text())
FIT = OUT / "fit"


def sources(towns):
    return {str(t): file_hash(OUT / "cache" / f"town{t}" / "metadata.json") for t in towns}


def fields(town):
    cache = open_fields(ORIGINAL / "features" / f"town{town}")
    raw = open_fields(ORIGINAL / "capture" / f"town{town}")
    return dict(**open_fields(OUT / "cache" / f"town{town}"), inputs=cache["inputs"],
                warp=cache["warp"], visible=cache["visible"], gt=raw["rgb"], seg=raw["labels"])


class Patches:
    def __init__(self):
        self.maps = {t: fields(t) for t in CFG["fit_towns"]}
        self.rng = np.random.default_rng(CFG["seed"])
        pools = {i: [] for i in (SEM["pedestrian"], SEM["vehicle"])}
        for town, maps in self.maps.items():
            for t, pair in enumerate(maps["vacated"]):
                for slot, mask in enumerate(pair):
                    for actor, pool in pools.items():
                        points = np.argwhere(mask == actor)
                        if len(points):
                            pool.append((town, t, slot, points))
        if not all(pools.values()):
            raise ValueError("Both pedestrian and vehicle disocclusion samples are needed")
        self.pools = list(pools.values())

    def batch(self):
        names = ("release", "inputs", "context", "gt", "seg", "vacated")
        buffers = {key: [] for key in names}
        size = CFG["patch"]
        for i in range(CFG["batch"]):
            pool = self.pools[i % len(self.pools)]
            town, t, slot, points = pool[self.rng.integers(len(pool))]
            arrays = self.maps[town]
            height, width = arrays["seg"].shape[-2:]
            y, x = points[self.rng.integers(len(points))]
            y, x = np.clip([y, x] - self.rng.integers(size//3, 2*size//3, 2), 0, [height-size, width-size])
            for key in names:
                buffers[key].append(arrays[key][t, slot, y:y+size, x:x+size])
        result = {}
        for key, values in buffers.items():
            v = torch.from_numpy(np.stack(values)).cuda()
            result[key] = v.permute(0, 3, 1, 2).float() if v.ndim == 4 else v[:, None]
        result["gt"] /= 255
        return result


def masked_mean(x, weight):
    return (x * weight).sum() / weight.sum().clamp_min(1)


def objective(pred, batch):
    ref, gt, seg = (batch[k] for k in ("release", "gt", "seg"))
    vac = batch["vacated"] != 0
    interior = torch.zeros_like(vac)
    interior[:, :, 6:-6, 6:-6] = True
    error = (pred - gt).abs().mean(1, keepdim=True)
    target = sum(masked_mean(error, (batch["vacated"] == actor) & interior)
                 for actor in (SEM["pedestrian"], SEM["vehicle"])) / 2
    support = actor_support(batch["context"])
    foreground = (seg == SEM["vehicle"]) | (seg == SEM["pedestrian"]) | (seg == SEM["marking"])
    guard = (interior & ~vac) * support * (1 + 4 * foreground.float())
    residual = pred - ref
    preserve = masked_mean(residual.square().mean(1, keepdim=True), guard)
    near = (F.max_pool2d(vac.float(), 7, 1, 3) > 0) & ~foreground & interior
    edge = pred.new_zeros(())
    for axis in (-1, -2):
        length = seg.size(axis)-1
        g = gt.diff(dim=axis)
        strong = (g.abs().mean(1, keepdim=True) >= .10) & interior.narrow(axis, 0, length)
        direction = F.normalize(g, dim=1, eps=1e-5)
        discrepancy = ((pred.diff(dim=axis) - g) * direction).sum(1, keepdim=True).abs()
        for cls in (SEM["vehicle"], SEM["marking"]):
            label = seg == cls
            boundary = label.narrow(axis, 0, length) ^ label.narrow(axis, 1, length)
            edge += masked_mean(discrepancy, strong & boundary)
    return target + .20 * masked_mean(error, near) + 2.5 * preserve + .35 * edge / 4


def fit():
    torch.manual_seed(CFG["seed"])
    contract = dict(release=file_hash(MANIFEST), config=file_hash(CONFIG), fit_script=file_hash(__file__),
                    model=file_hash(ROOT / "keireality/disocclusion.py"), towns=sources(CFG["fit_towns"]))
    path = FIT / "repair.pt"
    if path.exists():
        if torch.load(path, map_location="cpu")["contract"] != contract:
            raise ValueError("Existing repair has different training inputs")
        return
    data = Patches()
    model = BackgroundRepair().cuda().to(memory_format=torch.channels_last)
    optim = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, CFG["steps"], eta_min=.00012)
    scaler = torch.amp.GradScaler("cuda")
    losses = []
    for step in range(CFG["steps"]):
        batch = data.batch()
        optim.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(batch["release"], batch["inputs"], batch["context"])
        loss = objective(pred, batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite background repair objective")
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optim)
        scaler.update()
        schedule.step()
        losses.append(float(loss.detach()))
        if (step+1) % 150 == 0:
            print("fit", step+1, round(float(np.mean(losses[-150:])), 5), flush=True)
    save_checkpoint(path, dict(model=model.cpu().state_dict(), contract=contract))
    write_json(FIT / "training.json", dict(checkpoint=file_hash(path), updates=CFG["steps"],
               parameters=sum(p.numel() for p in model.parameters()), first100=float(np.mean(losses[:100])),
               last100=float(np.mean(losses[-100:])), contract=contract))


def trained():
    checkpoint = torch.load(FIT / "repair.pt", map_location="cpu")
    if checkpoint["contract"]["model"] != file_hash(ROOT / "keireality/disocclusion.py"):
        raise ValueError("The repair implementation differs from the fitted model")
    model = BackgroundRepair().cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def compare_native(town, candidates, model):
    data = fields(town)
    names = ["release384", *[f"{method}-{strength}" for method, strength in candidates]]
    reports = {}
    for slot, track in enumerate(TRACKS):
        score = Scores(names)
        for t in range(len(data["inputs"])):
            def image(key):
                return torch.from_numpy(data[key][t, slot].copy()).cuda().permute(2, 0, 1)[None].float()
            x, ref, ctx = (image(key) for key in ("inputs", "release", "context"))
            with torch.autocast("cuda", dtype=torch.float16):
                out = [ref]
                for method, strength in candidates:
                    value = (model(ref, x, ctx, strength) if method == "learned"
                             else torch.lerp(ref, ctx[:, :3], strength * actor_support(ctx)))
                    out.append(value)
            pred = torch.cat(out)
            gt = data["gt"][t, slot]
            score.frame(bytes_rgb(pred), gt, data["seg"][t, slot], data["visible"][t, slot], data["warp"][t, slot])
            truth = torch.from_numpy(gt.copy()).cuda().permute(2, 0, 1).float()/255
            error = (pred - truth).abs().mean(1).cpu().numpy()
            mask = data["vacated"][t, slot]
            for key, keep in (("vacated_l1", mask > 0), ("after_pedestrian", mask == SEM["pedestrian"]),
                              ("after_vehicle", mask == SEM["vehicle"])):
                count = keep.sum()
                score.add(key, error[:, keep].sum(1, dtype=np.float64), count, count)
        reports[track] = score.results()
    return reports


def rank(towns, key):
    gain, chase, protect = [], [], []
    for town in towns.values():
        for track, pair in town.items():
            old, new = pair["methods"]["release384"], pair["methods"][key]
            improvement = 1 - new["vacated_l1"]/old["vacated_l1"]
            gain.append(improvement)
            if track == "chase":
                chase.append(improvement)
            for name in ("vehicle_contrast", "white_line_contrast", "static_flicker"):
                ratio = new[name] / old[name]
                protect.append(ratio - 1 if name == "static_flicker" else 1 - ratio)
    guarded = max(protect) <= CFG["relative_guard"]
    return [int(guarded and min(chase) > 0), int(guarded), min(chase), float(np.mean(gain))]


def select():
    candidates = [(m, s) for m in ("coarse", "learned") for s in CFG["strengths"]]
    model = trained()
    towns = {t: compare_native(t, candidates, model) for t in CFG["selection_towns"]}
    ranks = {f"{m}-{s}": rank(towns, f"{m}-{s}") for m, s in candidates}
    method, strength = max(candidates, key=lambda c: ranks[f"{c[0]}-{c[1]}"])
    info = dict(method=method, strength=strength, eligible=bool(ranks[f"{method}-{strength}"][0]),
                ranks=ranks, selection_towns=CFG["selection_towns"], sources=sources(CFG["selection_towns"]),
                release=file_hash(MANIFEST), repair=file_hash(FIT / "repair.pt"), config=file_hash(CONFIG),
                model=file_hash(ROOT / "keireality/disocclusion.py"), selection_script=file_hash(__file__))
    write_json(FIT / "development.json", towns)
    write_json(FIT / "selection.json", dict(**info, development=file_hash(FIT / "development.json")))
    print("selected", method, strength, ranks, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("fit", "select", "all"), default="all")
    args = p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    verify_release()
    with writer_lock(FIT):
        if args.stage != "select":
            fit()
        if args.stage != "fit":
            select()
