#!/usr/bin/env python3
"""Fit a frozen RGB-to-label evaluator using real training teachers, with selection on different validation towns."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.flow_data import teacher_shard
from keireality.geometry import ORBIT_ANGLES
from keireality.rgb_probe import RGBProbe, probe_scores
from keisim.config import N_SEM
from train_reality import save_checkpoint


def teachers(root,rows,angles,device):
    images,labels=[],[]
    for r in rows:
        for k in ("rgb_target","seg_target"):
            if file_hash(root/r["path"]/(k+".npy"))!=r["fields"][k]["sha256"]:
                raise ValueError("corrupt probe teacher")
        x,y=teacher_shard(root,r,[ORBIT_ANGLES.index(a) for a in angles],device)
        images.append(x.flatten(0,1).half())
        labels.append(y.flatten(0,1))
    return torch.cat(images),torch.cat(labels)


@torch.inference_mode()
def assess(model,images,labels,blur=False):
    m=torch.zeros(N_SEM*N_SEM,dtype=torch.long,device=images.device)
    model.eval()
    for i in range(0,len(images),16):
        x=images[i:i+16].float()
        if blur:
            x=F.avg_pool2d(F.pad(x,(2,2,2,2),mode="reflect"),5,1)
        pred=model(x).argmax(1)
        m+=torch.bincount((labels[i:i+16].long()*N_SEM+pred).flatten(),minlength=N_SEM*N_SEM)
    return probe_scores(m.reshape(N_SEM,N_SEM).cpu().numpy())


def train(args):
    protocol=json.loads(args.config.read_text())
    cfg=protocol["phantom_probe"]
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(2)
    m=load_manifest(args.data)
    splits=[[r for r in m["shards"] if r["split"]==s] for s in ("train","val")]
    if {r["town_seed"] for r in splits[0]} & {r["town_seed"] for r in splits[1]}:
        raise ValueError("training and validation towns overlap")
    contract=dict(protocol=file_hash(args.config),data=m["contract_sha256"],seed=cfg["seed"],
                  train_towns=sorted({r["town_seed"] for r in splits[0]}),val_towns=sorted({r["town_seed"] for r in splits[1]}))
    with writer_lock(args.out):
        if (args.out/"training.json").exists():
            old=json.loads((args.out/"training.json").read_text())
            if old["contract"]!=contract or file_hash(args.out/"best.pt")!=old["sha256"]:
                raise ValueError("incompatible RGB probe")
            print("Verified complete RGB probe"); return
        (x,y),(vx,vy)=[teachers(args.data,rows,protocol["train_angles"],args.device) for rows in splits]
        net=RGBProbe().to(device=args.device,memory_format=torch.channels_last)
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        counts=torch.bincount(y.flatten().long(),minlength=N_SEM).float()
        w=(counts.max()/counts.clamp_min(1)).sqrt().clamp(1,10)
        best,log=-1.,[]
        tick=time.perf_counter()
        for k in range(cfg["updates"]):
            net.train()
            i=torch.randint(len(x),(cfg["batch"],),device=x.device)
            raw=x[i].float()
            kernel=3 if k%2 else 5
            blur=F.avg_pool2d(F.pad(raw,(kernel//2,)*4,mode="reflect"),kernel,1)
            mix=torch.rand(len(i),1,1,1,device=x.device)
            raw=raw*(1-mix)+mix*blur
            raw=(raw*(.85+.3*torch.rand(len(i),1,1,1,device=x.device))+.04*torch.randn_like(raw)).clamp(-1,1)
            pred=net(raw)
            loss=F.cross_entropy(pred,y[i].long(),weight=w)
            opt.param_groups[0]["lr"]=.0001+.0009*(.5+.5*math.cos(math.pi*k/cfg["updates"]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),3.,error_if_nonfinite=True)
            opt.step()
            if (k+1)%300==0 or k+1==cfg["updates"]:
                clean,blurred=assess(net,vx,vy),assess(net,vx,vy,True)
                score=.5*(clean["dynamic_iou"]+blurred["dynamic_iou"])
                rec=dict(step=k+1,clean=clean,blurred=blurred)
                log.append(rec)
                if score>best:
                    best,selected=score,rec
                    save_checkpoint(args.out/"best.pt",dict(model=net.state_dict(),contract=contract,selected=selected))
                print("probe",k+1,clean["dynamic_iou"],blurred["dynamic_iou"],flush=True)
        write_json(args.out/"training.json",dict(contract=contract,steps=cfg["updates"],selected=selected,log=log,
                   sha256=file_hash(args.out/"best.pt"),elapsed_s=time.perf_counter()-tick,train_images=len(x),val_images=len(vx)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","config","out"): p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    train(p.parse_args())
