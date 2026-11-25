#!/usr/bin/env python3
"""Learn a source-depth correction from training cameras only, with two development towns."""
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.depth_refinement import SourceDepthRefiner
from keireality.encoder import extract_mem, frozen_pilot, state_hash
from keisim.config import SEM
from keisim.render import rays
from diagnose_reality_blur import DATA, OUT, WEIGHTS
from prepare_reality_projection import checked_fields

CFG = ROOT/"configs/reality/p2_blur.json"


def camera_samples(towns, stride, pilot, head, device):
    chunks = []
    rows = [row for row in load_manifest(DATA)["shards"] if row["split"]=="train" and row["town_seed"] in towns]
    for row in rows:
        raw = checked_fields(DATA,row,("rgb_in","depth_in","seg_in"))
        for start in range(0,row["count"],4*stride):
            pick = slice(start,min(start+4*stride,row["count"]),stride)
            with torch.inference_mode():
                img = torch.from_numpy(raw["rgb_in"][pick].copy()).permute(0,1,4,2,3).to(device)
                f = extract_mem(pilot,img,pyramid=True,scene=True)
                est = head.depth_metres(head.forward_depth(img)).reshape(len(img),2,80,160)
                z = rays.decode_depth(raw["depth_in"][pick,:,::2,::2]).astype(np.float32)
                seg = raw["seg_in"][pick,:,::2,::2].copy()
                chunks.append((f["stem"].half().cpu(),f["c2"].half().cpu(),f["source_logits"].half().cpu(),
                               est.cpu(),torch.from_numpy(z),torch.from_numpy(seg)))
    return tuple(torch.cat(col) for col in zip(*chunks,strict=True))


def objective(log_z, surface, z, seg):
    valid = z>0
    target = z.clamp_min(.25).log()
    border = torch.zeros_like(valid)
    for d in (-2,-1):
        a, b = target.narrow(d,1,target.size(d)-1), target.narrow(d,0,target.size(d)-1)
        diff = (a-b).abs()>.18
        border.narrow(d,1,target.size(d)-1).logical_or_(diff)
        border.narrow(d,0,target.size(d)-1).logical_or_(diff)
    actors = (seg==SEM["vehicle"]) | (seg==SEM["pedestrian"])
    background = (seg==SEM["building"]) | (seg==SEM["vegetation"])
    weights = valid*(1.+2.*border+.6*background+2.*actors)
    err = F.smooth_l1_loss(log_z,target,reduction="none",beta=.08)
    error = (err*weights).sum()/weights.sum().clamp_min(1.)
    # Distance discontinuities are important for placing silhouettes in another camera.
    slope_loss = log_z.new_zeros(())
    for d in (-2,-1):
        pred_diff = torch.diff(log_z,dim=d)
        gt_diff = torch.diff(target,dim=d)
        both = valid.narrow(d,1,valid.size(d)-1)&valid.narrow(d,0,valid.size(d)-1)
        weight = both * (1 + (gt_diff.abs()>.18)*3.)
        slope_loss += (weight*F.smooth_l1_loss(pred_diff,gt_diff,reduction="none",beta=.08)).sum()/weight.sum().clamp_min(1.)
    label = F.binary_cross_entropy_with_logits(surface,valid.float(),reduction="none")
    return error+.16*slope_loss+.24*(label*(1.+border)).mean()


@torch.no_grad()
def evaluate(model, samples, device):
    sums = np.zeros(5)
    for start in range(0,len(samples[0]),4):
        stem,c2,logits,est,z,_ = [a[start:start+4].to(device) for a in samples]
        log_z,p = model(stem,c2,logits,est)
        valid = z>0
        # A missed surface receives the same 400m substitute in both depth comparisons.
        known = torch.where(est>0,est,400.).log()
        refined = torch.where(p>0, log_z,math.log(400.))
        sums += [(known[valid]-z[valid].log()).abs().sum().item(),
                 (refined[valid]-z[valid].log()).abs().sum().item(),valid.sum().item(),
                 ((p>0)==valid).sum().item(),valid.numel()]
    return dict(original_log_mae=sums[0]/sums[2],refined_log_mae=sums[1]/sums[2],surface_accuracy=sums[3]/sums[4])


def train(device):
    cfg=json.loads(CFG.read_text())
    diagnosis=OUT/"diagnosis/metrics.json"
    if json.loads(diagnosis.read_text())["focus"]!="depth":
        raise ValueError("depth was not selected by the training factorial")
    dest=OUT/"fit"
    with writer_lock(dest):
        lineage = {k:file_hash(ROOT/k) for k in ("configs/reality/p2_blur.json","keireality/depth_refinement.py",
                    "scripts/fit_reality_depth_refiner.py",*WEIGHTS[:2])}
        lineage["diagnosis"]=file_hash(diagnosis)
        if (dest/"depth.pt").exists() and (dest/"training.json").exists():
            log=json.loads((dest/"training.json").read_text())
            if log["contract"]!=lineage or log["weight_sha256"]!=file_hash(dest/"depth.pt"):
                raise ValueError("incompatible existing fit")
            print("verified depth fit",flush=True)
            return
        torch.manual_seed(cfg["seed"])
        pilot,head=[frozen_pilot(ROOT/k,device) for k in WEIGHTS[:2]]
        before=[state_hash(m) for m in (pilot,head)]
        fit=camera_samples(cfg["fit_towns"],cfg["train_stride"],pilot,head,device)
        dev=camera_samples(cfg["development_towns"],4,pilot,head,device)
        assert [state_hash(m) for m in (pilot,head)]==before
        del pilot,head
        model=SourceDepthRefiner().to(device)
        n=len(fit[0])
        opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=.0002)
        epochs,bs=cfg["epochs"],cfg["batch"]
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs*math.ceil(n/bs),eta_min=.00008)
        history=[]
        for epoch in range(epochs):
            losses=[]
            for ids in torch.randperm(n).split(bs):
                stem,c2,logits,est,z,seg=[s[ids].to(device) for s in fit]
                log_z,p=model(stem,c2,logits,est)
                loss=objective(log_z,p,z,seg)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite depth objective")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
                opt.step()
                sched.step()
                losses.append(loss.item())
            row=dict(epoch=epoch+1,loss=float(np.mean(losses)),development=evaluate(model,dev,device))
            history.append(row)
            print(row,flush=True)
        with tempfile.TemporaryDirectory(dir=dest) as tmp:
            stage = Path(tmp)/"depth.pt"
            torch.save(dict(model=model.state_dict(),contract=lineage),stage)
            stage.replace(dest/"depth.pt")
        write_json(dest/"training.json",dict(contract=lineage,weight_sha256=file_hash(dest/"depth.pt"),frozen_states=before,
                   fit_towns=cfg["fit_towns"],development_towns=cfg["development_towns"],ticks=n,
                   updates=epochs*math.ceil(n/bs),parameters=sum(p.numel() for p in model.parameters()),epochs=history))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    train(p.parse_args().device)
