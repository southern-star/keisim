#!/usr/bin/env python3
"""Change only the source noise, holding the camera, current features and previous generated image fixed."""
import argparse
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, write_json
from keireality.flow_data import FlowSplit
from keireality.flow_detail import highpass
from keireality.flow_model import frame_noise, make_condition
from keireality.meanflow import integrate_meanflow
from rollout_reality_flow import load_flow


@torch.inference_mode()
def diagnose(args):
    torch.set_num_threads(2)
    cfg = json.loads(args.base_config.read_text())
    detail = json.loads(args.config.read_text())
    ds = FlowSplit(args.data, args.cache, "val", args.device, teachers=False)
    model, contract = load_flow(args.model, args.device)
    if contract["renderer"] != ds.meta["renderer"]:
        raise ValueError("diagnostic conditions do not match the renderer")
    seed = cfg["validation_seed"]
    count = 0
    totals = {}
    nviews = len(ds.angles)
    def add(key, batched):
        a, b = batched.chunk(2)
        # Divide squared disagreement by two for the variance of one independent draw.
        value = highpass(.5 * (a - b)).square().mean() * .5
        totals[key] = totals.get(key, 0.) + value.double()
    for ep in ds.episodes:
        previous = torch.zeros(nviews, 3, 64, 128, device=args.device)
        old_view = ds.poses[ep.start].new_zeros(nviews, 11)
        for i in range(ep.start, ep.start + ep.length):
            views = ds.poses[i]
            good = torch.full((nviews,), i > ep.start, dtype=torch.bool, device=args.device)
            cond = make_condition(ds.current[i], views, previous, old_view, ds.delta[i].expand(nviews, -1),
                                  views.new_full((nviews,), .1), good)
            cond = torch.cat((cond, cond))
            sources = [frame_noise(ep.seed, ds.steps[i], ds.angles, s, args.device)
                       for s in (seed, seed + detail["paired_seed_offset"])]
            noise = torch.cat(sources)
            add("source_t1", noise)
            one = integrate_meanflow(model, noise, cond, 1)
            add("one_display", one.clamp(-1, 1))
            stages = []
            four = integrate_meanflow(model, noise, cond, 4, trace=stages)
            for j, stage in enumerate(stages, 1):
                add(f"four_stage_{j}", stage)
            four = four.clamp(-1, 1)
            add("four_display", four)
            previous = four[:nviews]
            old_view = views
            count += 1
    report = dict(paired_seed_highpass_rms={k:math.sqrt(float(v) / count) for k, v in totals.items()},
                  ticks=count, views=ds.angles, samples=count*nviews, model=file_hash(args.model),
                  cache=file_hash(ds.cache / "manifest.json"), seeds=[seed, seed + detail["paired_seed_offset"]],
                  history="same previous four-step image from the first seed for both branches and both NFE",
                  units="RGB [0,1] scale, unclipped intermediate states; final display is clipped")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("base-config", "config", "data", "cache", "model", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    diagnose(p.parse_args())
