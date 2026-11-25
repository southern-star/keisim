#!/usr/bin/env python3
"""Estimate the four-step output's linear response to injected noise, using paired draws on training towns."""
import argparse
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, write_json
from keireality.flow_data import FlowSplit
from keireality.flow_detail import highpass, subtract_source_component
from keireality.flow_model import frame_noise, generate, make_condition
from rollout_reality_flow import load_flow


def paired_frames(ds, field, seeds, device, training):
    for ep in ds.episodes:
        length = 16 if training else ep.length
        start = ep.start + (ep.length-length)//2
        views = len(ds.angles)
        previous = torch.zeros(views, 3, 64, 128, device=device)
        old_view = torch.zeros(views, 11, device=device)
        for i in range(start, start+length):
            view = ds.poses[i].to(device)
            keep = torch.full((views,), i > start, device=device, dtype=torch.bool)
            cond = make_condition(ds.current[i].to(device), view, previous, old_view, ds.delta[i].to(device).expand(views,-1),
                                  view.new_full((views,), .1), keep)
            noise = torch.cat([frame_noise(ep.seed, ds.steps[i], ds.angles, s, device) for s in seeds])
            image = generate(field, noise, torch.cat((cond,cond)), 4)
            previous, old_view = image[:views], view
            yield noise, (image + 1) * .5


@torch.inference_mode()
def fit(args):
    torch.set_num_threads(2)
    base, detail = [json.loads(path.read_text()) for path in (args.base_config, args.config)]
    field, ck = load_flow(args.model, args.device)
    train = FlowSplit(args.data, args.cache, "train", teachers=False)
    val = FlowSplit(args.data, args.cache, "val", teachers=False)
    if (ck["train"] != file_hash(train.cache/"manifest.json") or ck["protocol"] != file_hash(args.config)
            or {e.town for e in train.episodes} & {e.town for e in val.episodes}):
        raise ValueError("source-response fit must use the model's original training towns")
    size, stride, ridge = 5, 3, .001
    weights = 3 * size * size
    xx = torch.zeros(weights, weights, dtype=torch.float64, device=args.device)
    xy = torch.zeros(weights, 3, dtype=torch.float64, device=args.device)
    total = 0
    seeds = [75613, 75613+detail["paired_seed_offset"]]
    for noise, rgb in paired_frames(train, field, seeds, args.device, training=True):
        na, nb = noise.chunk(2)
        a, b = rgb.chunk(2)
        x = F.unfold(na-nb, size, stride=stride).transpose(1,2).reshape(-1, weights)
        r = size//2
        y = (a-b)[:, :, r:-r:stride, r:-r:stride].permute(0,2,3,1).reshape(-1,3)
        xx += (x.T @ x).double()
        xy += (x.T @ y).double()
        total += len(x)
    k = torch.linalg.solve(xx/total + ridge*torch.eye(weights,device=args.device), xy/total)
    k = k.T.contiguous().float().reshape(3, 3, size, size)
    raw_energy, clean_energy, count = 0., 0., 0
    seeds_val = [base["validation_seed"], base["validation_seed"]+detail["paired_seed_offset"]]
    for noise, rgb in paired_frames(val, field, seeds_val, args.device, training=False):
        clean = subtract_source_component(rgb, noise, k)
        a,b = rgb.chunk(2)
        c,d = clean.chunk(2)
        raw_energy += highpass(a-b).square().mean().item()*.5
        clean_energy += highpass(c-d).square().mean().item()*.5
        count += 1
    record = dict(model=file_hash(args.model), training_cache=ck["train"], train_towns=[e.town for e in train.episodes],
                  ticks_per_training_town=16, views=train.angles, fit_seed_pair=seeds, training_pixel_pairs=total,
                  kernel_size=size, fit_stride=stride, ridge=ridge, kernel=k.cpu().tolist(), validation_seed_pair=seeds_val,
                  validation=dict(ticks=count, views=val.angles, raw_rms=math.sqrt(raw_energy/count),
                                  cleaned_rms=math.sqrt(clean_energy/count)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, record)
    print(record["validation"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("base-config", "config", "data", "cache", "model", "out"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--device", default="cuda", choices=("cpu","cuda"))
    fit(p.parse_args())
