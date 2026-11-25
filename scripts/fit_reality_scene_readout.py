#!/usr/bin/env python3
"""Fit a native-camera color readout from the frozen driving stem on training towns only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.encoder import extract_mem, frozen_pilot, state_hash
from prepare_reality_projection import checked_fields
from train_reality import save_checkpoint


@torch.inference_mode()
def fit(args):
    cfg = json.loads(args.config.read_text())
    raw = load_manifest(args.data)
    rows = [r for r in raw["shards"] if r["split"] == "train"]
    if sorted({r["town_seed"] for r in rows}) != cfg["train_towns"]:
        raise ValueError("unexpected readout training towns")
    contract = dict(protocol=file_hash(args.config), pilot=file_hash(args.pilot), collection=raw["contract_sha256"],
                    trainer=file_hash(Path(__file__)), towns=cfg["train_towns"])
    torch.set_num_threads(2)
    rng = np.random.default_rng(cfg["readout_seed"])
    with writer_lock(args.out):
        record = args.out/"training.json"
        model_file = args.out/"readout.pt"
        if record.exists():
            old = json.loads(record.read_text())
            if old["contract"] != contract or old["model"] != file_hash(model_file):
                raise ValueError("incompatible completed stem readout")
            return
        pilot = frozen_pilot(args.pilot,args.device)
        before = state_hash(pilot)
        size = 64*9+1
        gram = torch.zeros(size,size,device=args.device,dtype=torch.float64)
        cross = gram.new_zeros(size,12)
        y_energy = gram.new_zeros(())
        ticks, samples = 0, 0
        starts = {}
        for row in rows:
            data = checked_fields(args.data,row,("step","rgb_in"))
            start = starts.setdefault(row["episode_id"], int(data["step"][0]))
            ids = np.flatnonzero((data["step"]-start) % cfg["training_tick_stride"] == 0)
            for begin in range(0,len(ids),4):
                selected = ids[begin:begin+4]
                rgb = torch.from_numpy(data["rgb_in"][selected]).permute(0,1,4,2,3).to(args.device)
                stem = extract_mem(pilot,rgb,parity=ticks==0,pyramid=True,scene=True)["stem"].flatten(0,1)
                target = F.pixel_unshuffle(rgb.flatten(0,1).float()/255,2).permute(0,2,3,1)
                padded = F.pad(stem,(1,1,1,1),mode="replicate").permute(0,2,3,1)
                cells = torch.tensor(np.stack([rng.choice(80*160,cfg["cells_per_camera_tick"],replace=False)
                                                for _ in range(len(stem))]), device=args.device)
                y,x = cells//160, cells%160
                k = torch.arange(len(stem),device=args.device)[:,None]
                patches = torch.stack([padded[k,y+dy,x+dx] for dy in range(3) for dx in range(3)],-1)
                features = patches.reshape(-1,size-1).double()
                features = F.pad(features,(0,1),value=1.)
                target = target[k,y,x].reshape(-1,12).double()
                gram.addmm_(features.T,features)
                cross.addmm_(features.T,target)
                y_energy += target.square().sum()
                ticks += len(selected)
                samples += len(features)
        penalty = gram.diag().clamp_min(1e-8)*cfg["ridge"]
        penalty[-1] = 0
        coef = torch.linalg.solve(gram+torch.diag(penalty),cross)
        mse = (y_energy - 2*(coef*cross).sum() + (coef*(gram@coef)).sum())/(samples*12)
        if not torch.isfinite(coef).all() or state_hash(pilot) != before:
            raise ValueError("invalid fit or mutated driving encoder")
        save_checkpoint(model_file,dict(weight=coef[:-1].T.reshape(12,64,3,3).float().contiguous().cpu(),
                                        bias=coef[-1].float().cpu(),contract=contract))
        result = dict(contract=contract, model=file_hash(model_file), ticks=ticks, patches=samples,
                      parameters=12*size, source_training_rmse=float(mse.sqrt()), frozen_pilot=before)
        write_json(record,result)
        print(result,flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k,v in dict(config="configs/reality/p2_scene_detail.json",data="data/reality/p2",pilot="runs/keipilot_mc.pt",
                    out="runs/reality/p2_scene_detail/train").items():
        p.add_argument("--"+k,type=Path,default=Path(v))
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    fit(p.parse_args())
