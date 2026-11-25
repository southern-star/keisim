#!/usr/bin/env python3
"""Fit only the native display decoder on towns 0–5, then freeze the mix on towns 6–7."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from collect_reality_resolution import CFG, OUT, ROOT, TRACKS, open_fields
from keireality.dataset import file_hash, write_json, writer_lock
from keireality.resolution import ResolutionHead
from keisim.config import SEM
from score_reality_resolution import changes, quality_check, score_town
from train_reality import save_checkpoint


FIT = OUT / "fit"


def fit_contract():
    return dict(config=file_hash(ROOT/"configs/reality/p2_resolution.json"),program=file_hash(Path(__file__)),
                model=file_hash(ROOT/"keireality/resolution.py"), data={
                    town:file_hash(OUT/"features"/f"town{town}/features.json") for town in CFG["fit_towns"]})


def make_model():
    return ResolutionHead().to("cuda", memory_format=torch.channels_last)


def read_head():
    state = torch.load(FIT/"head.pt",map_location="cpu")
    head = make_model()
    head.load_state_dict(state["model"])
    return head.eval().requires_grad_(False)


class Pairs:
    def __init__(self):
        self.maps = {t: (open_fields(OUT/"features"/f"town{t}"),open_fields(OUT/"capture"/f"town{t}"))
                     for t in CFG["fit_towns"]}
        self.rng = np.random.default_rng(CFG["seed"])

    def batch(self):
        inputs, colors, labels, warps = [], [], [], []
        for town in self.rng.choice(CFG["fit_towns"],2,replace=False):
            f,r = self.maps[town]
            t = int(self.rng.integers(1,len(f["inputs"])))
            slot = int(self.rng.integers(0,len(TRACKS)))
            inputs.extend(f["inputs"][t-1:t+1,slot])
            colors.extend(r["rgb"][t-1:t+1,slot])
            labels.extend(r["labels"][t-1:t+1,slot])
            warps.append(f["warp"][t,slot])
        def upload(items):
            return torch.from_numpy(np.stack(items)).to("cuda")
        return (upload(inputs).permute(0,3,1,2).float(),upload(colors).permute(0,3,1,2).float()/255,
                upload(labels),upload(warps).long())


def objective(pred, gt, labels, warp):
    car, line = labels==SEM["vehicle"], labels==SEM["marking"]
    weight = 1.+3.*car+3.*line
    error = pred-gt
    rgb_loss = (error.abs().mean(1)*weight).sum()/weight.sum()
    grad_loss = pred.new_zeros(())
    edge_loss = pred.new_zeros(())
    for axis in (-1,-2):
        g, p = gt.diff(dim=axis), pred.diff(dim=axis)
        magnitude = g.abs().mean(1)
        edge = magnitude >= CFG["edge_threshold"]
        w = .2 + 2*edge.float()
        grad_loss += ((p-g).abs().mean(1)*w).sum()/w.sum()
        # The signed RGB gradient distinguishes a correctly placed silhouette from an extra dark halo.
        direction = g/g.square().sum(1,keepdim=True).clamp_min(1e-5).sqrt()
        wrong = ((p-g)*direction).sum(1).abs()
        for mask in (car,line):
            length = mask.size(axis)-1
            boundary = (mask.narrow(axis,1,length) != mask.narrow(axis,0,length)) & edge
            edge_loss += (wrong*boundary).sum()/boundary.sum().clamp_min(1)
    old = error[::2].flatten(2).gather(2,warp.flatten(1).clamp_min(0)[:,None].expand(-1,3,-1)).reshape_as(error[1::2])
    valid = warp>=0
    static = labels[1::2]
    temporal_weight = valid*(1.+.8*((static==SEM["building"])|(static==SEM["vegetation"])))
    flicker = ((error[1::2]-old).abs().mean(1)*temporal_weight).sum()/temporal_weight.sum().clamp_min(1)
    return rgb_loss + .45*grad_loss/2 + .23*edge_loss/4 + 1.20*flicker


def fit():
    provenance = fit_contract()
    path = FIT/"head.pt"
    if path.exists():
        if torch.load(path,map_location="cpu")["contract"]!=provenance:
            raise ValueError("Different training sources")
        return
    torch.manual_seed(CFG["seed"])
    model = make_model()
    data = Pairs()
    opt = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.0002)
    steps = CFG["training_steps"]
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.00015)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    for i in range(steps):
        x,gt,seg,warp = data.batch()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.float16):
            pred = model(x)
        loss = objective(pred,gt,seg,warp)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(opt)
        scaler.update()
        schedule.step()
        history.append(float(loss.detach()))
        if (i+1)%120==0:
            print("native decoder",i+1,"/ ",steps,"loss",round(float(np.mean(history[-120:])),5),flush=True)
    save_checkpoint(path,dict(model=model.cpu().state_dict(),contract=provenance,steps=steps))
    write_json(FIT/"training.json",dict(towns=CFG["fit_towns"],steps=steps,parameters=sum(p.numel() for p in model.parameters()),
               checkpoint=file_hash(path),loss_first60=float(np.mean(history[:60])),loss_last60=float(np.mean(history[-60:]))))


def select():
    head = read_head()
    strengths = CFG["strengths"]
    reports = {town:score_town(town,head,strengths)[0] for town in CFG["selection_towns"]}
    ranks = {}
    for strength in strengths:
        key = str(strength)
        rows = [r for pair in reports.values() for r in pair.values()]
        checks = [quality_check(r,key) for r in rows]
        deltas = [changes(r,key) for r in rows]
        ranks[key] = [sum(all(c.values()) for c in checks),sum(sum(c.values()) for c in checks),
                      min(min(v["vehicle_contrast"],v["white_line_contrast"]) for v in deltas),
                      -float(np.mean([v["static_flicker"] for v in deltas]))]
    strength = max(strengths,key=lambda s:ranks[str(s)])
    write_json(FIT/"selection.json",dict(strength=strength, ranks=ranks,towns=CFG["selection_towns"],
               methods=reports, checkpoint=file_hash(FIT/"head.pt"),model=file_hash(ROOT/"keireality/resolution.py"),
               scorer=file_hash(Path(__file__).with_name("score_reality_resolution.py"))))
    print("training towns selected",strength,ranks,flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("fit","select","all"),default="all")
    a = p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    with writer_lock(FIT):
        if a.stage in ("fit","all"):
            fit()
        if a.stage in ("select","all"):
            select()
