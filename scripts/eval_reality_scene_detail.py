#!/usr/bin/env python3
"""Compare full-frame native detail, object silhouette and temporal errors on the same causal drives."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.depth_projection import DepthProjectionBranch
from keireality.encoder import extract_mem, frozen_pilot, state_hash
from keireality.geometry import ORBIT_ANGLES, orbit_camera, view_pose
from keireality.metrics import local_ssim
from keireality.scene_detail import SceneDetail, SceneDetailDisplay
from keireality.temporal_metrics import correspondence
from keireality.video import quad, write_video
from keisim.config import N_SEM, SEM, SEM_NAMES
from keisim.render import rays
from eval_reality_projection import bytes_rgb, decision
from eval_reality_temporal import load_renderer
from prepare_reality_projection import checked_fields


NAMES = ("all_l1","ssim","scene_edges","silhouette","static_flicker","vacated_l1")
MODELS = ("A","B","C")


class SceneScores:
    def __init__(self):
        self.num = np.zeros((3,len(NAMES)))
        self.den = np.zeros(len(NAMES))
        self.counts = np.zeros(len(NAMES),dtype=np.int64)
        self.class_n = np.zeros(N_SEM)
        self.class_abs = np.zeros((3,N_SEM))
        self.class_energy = np.zeros(N_SEM)
        self.class_dot = np.zeros((3,N_SEM))
        self.static_count = np.zeros(N_SEM)
        self.static_error = np.zeros((3,N_SEM))
        self.under_one_level = np.zeros(2,np.int64)

    def add(self, images, gt, seg, seen, edge_threshold, warp, old_error):
        err = images-gt
        pixels = seg.size
        mae = np.abs(err).mean(-1)
        self.num[:,0] += mae.sum(axis=(1,2),dtype=np.float64)
        self.den[:2] += pixels
        self.counts[:2] += pixels
        self.under_one_level += (np.abs(images[1:]-images[0]).mean(-1)<1/255).sum(axis=(1,2))
        self.class_n += np.bincount(seg.ravel(),minlength=N_SEM)
        for i,x in enumerate(images):
            self.num[i,1] += local_ssim(x,gt).sum(dtype=np.float64)
            self.class_abs[i] += np.bincount(seg.ravel(),weights=mae[i].ravel(),minlength=N_SEM)
        for a,b in ((np.s_[:-1,:],np.s_[1:,:]),(np.s_[:,:-1],np.s_[:,1:])):
            g = gt[b]-gt[a]
            edge = np.abs(g).mean(-1) >= edge_threshold
            change = images[(slice(None),*b)][:,edge]-images[(slice(None),*a)][:,edge]
            gradient = g[edge]
            dot = (change*gradient).sum(-1)
            energy = (gradient*gradient).sum(-1)
            left,right = seg[a][edge],seg[b][edge]
            sil = seen[a][edge] & seen[b][edge] & ((left==SEM["vehicle"]) ^ (right==SEM["vehicle"]))
            self.num[:,2] += dot.sum(1,dtype=np.float64)
            self.den[2] += energy.sum(dtype=np.float64)
            self.counts[2] += len(energy)
            self.num[:,3] += dot[:,sil].sum(1,dtype=np.float64)
            self.den[3] += energy[sil].sum(dtype=np.float64)
            self.counts[3] += sil.sum()
            for cls in (left,right):
                self.class_energy += np.bincount(cls,weights=.5*energy,minlength=N_SEM)
                for i in range(3):
                    self.class_dot[i] += np.bincount(cls,weights=.5*dot[i],minlength=N_SEM)
        flat = err.reshape(3,pixels,3)
        if warp is not None:
            (now,old),(vacated,_) = warp
            difference = flat[:,now]-old_error[:,old]
            labels = seg.ravel()[now]
            self.static_count += np.bincount(labels,minlength=N_SEM)
            for i,v in enumerate(np.abs(difference).mean(-1)):
                self.static_error[i] += np.bincount(labels,weights=v,minlength=N_SEM)
            for k,x in ((4,difference),(5,flat[:,vacated])):
                self.num[:,k] += np.abs(x).sum(axis=(1,2),dtype=np.float64)
                self.den[k] += x.shape[1]*3
                self.counts[k] += x.shape[1]
        return flat

    def merge(self, other):
        for name in vars(self):
            getattr(self,name)[:] += getattr(other,name)

    def report(self, models=MODELS):
        metrics = {m:{k:float(self.num[i,j]/self.den[j]) if self.den[j] else None for j,k in enumerate(NAMES)}
                   for i,m in enumerate(models)}
        classes = {name:dict(pixels=int(self.class_n[k]),l1={m:self.class_abs[i,k]/self.class_n[k] for i,m in enumerate(models)},
                             edge={m:float(self.class_dot[i,k]/self.class_energy[k]) if self.class_energy[k] else None
                                   for i,m in enumerate(models)})
                   for k,name in enumerate(SEM_NAMES) if self.class_n[k]}
        static = {name:dict(pixels=int(n), **{m:float(self.static_error[i,k]/n) for i,m in enumerate(models)})
                  for k,(name,n) in enumerate(zip(SEM_NAMES,self.static_count)) if n}
        return dict(**metrics,counts=dict(zip(NAMES,self.counts.tolist())),classes=classes,static_by_class=static,
                    fraction_mean_rgb_change_below_one_8bit_level_vs_A=dict(zip(models[1:],(self.under_one_level/self.counts[0]).tolist())))


def release(row):
    original = decision(dict(A=row["A"],B=row["C"]))
    if original["status"]=="INCONCLUSIVE":
        return original
    checks = dict(original["checks"],all_l1=row["C"]["all_l1"]<row["A"]["all_l1"],
                  ssim=row["C"]["ssim"]>row["A"]["ssim"],scene_edges=row["C"]["scene_edges"]>row["A"]["scene_edges"])
    return dict(status="PASS" if all(checks.values()) else "FAIL",checks=checks)


@torch.inference_mode()
def produce(a, rows, pilot, head, display, cameras, path, n):
    views=[torch.tensor(view_pose(c),device=a.device)[None] for c in cameras]
    frames=np.lib.format.open_memmap(path,"w+",np.float32,(n,len(views),3,128,256,3))
    offset,old_step,old_t,source_sq,source_n=0,None,None,0.,0
    for row in rows:
        raw=checked_fields(a.data,row,("step","t_sim","delta_pose","rgb_in"))
        for j,t in enumerate(raw["t_sim"]):
            step=int(raw["step"][j])
            if offset and (step!=old_step+1 or not np.isclose(t-old_t,.1,rtol=0,atol=1e-6)):
                raise ValueError("discontinuous recorded drive")
            image=torch.from_numpy(raw["rgb_in"][j].copy()).permute(0,3,1,2)[None].to(a.device)
            features=extract_mem(pilot,image,pyramid=True,scene=True,parity=offset==0)
            for k in ("mem","c2","c3"):
                features[k]=features[k].half()
            features["estimated_depth"]=head.depth_metres(head.forward_depth(image)).reshape(1,2,80,160)
            display.observe(features,row["episode_id"],t,raw["delta_pose"][j])
            if display.reference.valid.sum().item()!=min(offset,7):
                raise ValueError("incorrect causal history")
            colors=display.encoded[0]
            source_sq+=(colors-image.float()/255).square().sum(dtype=torch.float64).item()
            source_n+=colors.numel()
            for vi,view in enumerate(views):
                values=display.render(view,slot=vi,comparisons=True)
                frames[offset,vi]=torch.cat(values).permute(0,2,3,1).cpu().numpy()
            offset,old_step,old_t=offset+1,step,t
    if offset!=n or not np.isfinite(frames).all():
        raise ValueError("incomplete rollout or nonfinite output")
    frames.flush()
    return frames,float(np.sqrt(source_sq/source_n))


def score_drive(a, rows, frames, cameras, pooled, cfg, dest, *, models=MODELS, check=release,
                captions=("A　現行の Phase 2","B　前回の深度投影","C　浅い運転特徴から全景を再構成"),
                tail_ticks=0):
    total=[SceneScores() for _ in cameras]
    directions=[rays.camera_rays(c) for c in cameras]
    previous=[None]*len(cameras)
    errors=[None]*len(cameras)
    town=rows[0]["town_seed"]
    tick,movie=0,[]
    tail = SceneScores() if tail_ticks else None
    for row in rows:
        truth=checked_fields(a.data,row,("view_pose","rgb_target","seg_target","visibility_now","depth_target","ego_pose_world"))
        for j in range(row["count"]):
            for k,c in enumerate(cameras):
                if not np.allclose(truth["view_pose"][j,k],view_pose(c),rtol=0,atol=1e-6):
                    raise ValueError("teacher camera mismatch")
                gt=truth["rgb_target"][j,k]
                seg=truth["seg_target"][j,k]
                geom=(seg,truth["depth_target"][j,k],truth["ego_pose_world"][j])
                warp=correspondence(c,directions[k],geom,previous[k]) if previous[k] is not None else None
                args = (frames[tick,k],gt.astype(np.float32)/255,seg,truth["visibility_now"][j,k]==1,
                        cfg["edge_threshold"],warp,errors[k])
                errors[k]=total[k].add(*args)
                if k==0 and tail is not None and tick>=len(frames)-tail_ticks:
                    tail.add(*args)
                previous[k]=geom
                if k==0 and dest is not None:
                    aa,bb,cc=bytes_rgb(frames[tick,k])
                    movie.append((gt.copy(),aa,bb,cc))
            tick+=1
    def panels():
        for i,(gt,aa,bb,cc) in enumerate(movie):
            image=quad([(gt,f"正解　町{town}　{i/10:.1f}秒"),*zip((aa,bb,cc),captions,strict=True)])
            if i==len(movie)//2:
                Image.fromarray(image).save(dest/f"town{town}.png")
            yield image
    if dest is not None and write_video(dest/f"town{town}.mp4",panels(),10)!=len(frames):
        raise ValueError("A/B/C movie incomplete")
    for result,other in zip(pooled,total,strict=True):
        result.merge(other)
    reports={str(v):s.report(models) for v,s in zip(cfg["eval_angles"],total,strict=True)}
    result = dict(ticks=tick,views=reports,chase_check=check(reports["0"]),still_offset=tick//2)
    if tail is not None:
        result["last_ticks"] = dict(ticks=tail_ticks,chase=tail.report(models))
    return result


def evaluate(a):
    cfg=json.loads(a.config.read_text())
    raw=load_manifest(a.data)
    plan=json.loads((a.data/"plan.json").read_text())
    fitted=torch.load(a.readout,map_location="cpu")
    previous=json.loads((a.previous/"val/metrics.json").read_text())
    pilot_hash=file_hash(a.pilot)
    contract=dict(config=file_hash(a.config),readout=file_hash(a.readout),pilot=pilot_hash,
                  depth_head=file_hash(a.depth_head),phase2=file_hash(a.phase2),previous_branch=file_hash(a.previous/"train/model.pt"),
                  renderer=file_hash(ROOT/"keireality/scene_detail.py"))
    if (fitted["contract"]["protocol"]!=contract["config"] or fitted["contract"]["pilot"]!=pilot_hash or
            any(previous["contract"][k]!=contract[k] for k in ("pilot","phase2","depth_head")) or
            previous["contract"]["branch"]!=contract["previous_branch"]):
        raise ValueError("mismatched controls or readout")
    towns=cfg["validation_towns"] if a.split=="val" else cfg["fresh_towns"]
    if a.split=="test":
        val=json.loads((a.out/"val/metrics.json").read_text())
        if val["contract"]!=contract or val["status"]!="PASS":
            raise ValueError("fresh drives require a validation pass on this fixed candidate")
    rows=[r for r in raw["shards"] if r["split"]==a.split]
    cameras=[orbit_camera(v) for v in cfg["eval_angles"]]
    if (list(dict.fromkeys(r["town_seed"] for r in rows))!=towns or tuple(cfg["eval_angles"])!=ORBIT_ANGLES
            or [asdict(c) for c in cameras]!=plan["calibration"]["targets"]):
        raise ValueError("unexpected towns or calibration")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    dest=a.out/a.split
    with writer_lock(dest):
        metrics=dest/"metrics.json"
        if metrics.exists():
            old=json.loads(metrics.read_text())
            if (old["contract"]!=contract or old["collection"]!=raw["contract_sha256"]
                    or any(file_hash(dest/k)!=v for k,v in old["media"].items())):
                raise ValueError("prior results do not match this experiment")
            print("verified",old["status"])
            return
        pilot,head=frozen_pilot(a.pilot,a.device),frozen_pilot(a.depth_head,a.device)
        phase2=load_renderer(a.phase2,pilot_hash,a.device)
        branch=DepthProjectionBranch().to(a.device,memory_format=torch.channels_last).eval()
        branch.load_state_dict(torch.load(a.previous/"train/model.pt",map_location="cpu")["model"])
        detail=SceneDetail(cfg).to(a.device).eval()
        detail.camera_color.load_state_dict({k:fitted[k] for k in ("weight","bias")})
        models=(pilot,head,phase2,branch,detail)
        before=[state_hash(m) for m in models]
        evidence,pooled={},[SceneScores() for _ in cameras]
        for town in towns:
            episode=[r for r in rows if r["town_seed"]==town]
            n=sum(r["count"] for r in episode)
            if n!=(96 if a.split=="val" else 320) or len({r["episode_id"] for r in episode})!=1:
                raise ValueError("incomplete town")
            display=SceneDetailDisplay(phase2,detail,branch)
            with tempfile.TemporaryDirectory(dir=dest) as tmp:
                path=Path(tmp)/"frames.npy"
                frames,rmse=produce(a,episode,pilot,head,display,cameras,path,n)
                report=score_drive(a,episode,frames,cameras,pooled,cfg,dest)
                report.update(source_camera_rmse=rmse,rollout_sha256=file_hash(path))
                if a.split=="val":
                    ref=previous["towns"][str(town)]["views"]
                    for v,scores in report["views"].items():
                        for m in ("A","B"):
                            for metric in ("all_l1","silhouette","static_flicker","vacated_l1"):
                                if abs(scores[m][metric]-ref[v][m][metric])>1e-5:
                                    raise ValueError(f"control changed for {town}, {v}, {m}, {metric}")
                evidence[str(town)]=report
            print(town,report["chase_check"],{k:report["views"]["0"][k] for k in MODELS},flush=True)
        if before!=[state_hash(m) for m in models]:
            raise ValueError("inference changed a frozen model")
        status="PASS" if all(t["chase_check"]["status"]=="PASS" for t in evidence.values()) else "HOLD"
        write_json(metrics,dict(status=status,contract=contract,collection=raw["contract_sha256"],split=a.split,
                                ticks=sum(t["ticks"] for t in evidence.values()),frozen_states=before,towns=evidence,
                                pooled={str(v):s.report() for v,s in zip(cfg["eval_angles"],pooled,strict=True)},
                                media={f.name:file_hash(f) for f in sorted(dest.iterdir()) if f.suffix in (".mp4",".png")}))
        print(a.split,status,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k,v in dict(config="configs/reality/p2_scene_detail.json",data="data/reality/p2",pilot="runs/keipilot_mc.pt",
                    phase2="runs/reality/p2/temporal/best.pt",depth_head="runs/reference/mc14-depth.pt",
                    previous="runs/reality/p2_projection",readout="runs/reality/p2_scene_detail/train/readout.pt",
                    out="runs/reality/p2_scene_detail").items():
        p.add_argument("--"+k.replace("_","-"),type=Path,default=Path(v))
    p.add_argument("--split",choices=("val","test"),required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    evaluate(p.parse_args())
