#!/usr/bin/env python3
"""Batch-one, synchronized camera-tensor-to-display timing and live-vs-cached parity on four full test drives."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, load_field, load_manifest, write_json
from keireality.encoder import extract_mem, frozen_pilot, state_hash
from keireality.flow_data import phase2_readout
from keireality.flow_model import FlowRollout, frame_noise
from keireality.geometry import CHASE_CAMERA, view_pose
from keireality.stream import TemporalDisplay
from eval_reality_temporal import load_renderer
from rollout_reality_flow import load_flow


@torch.inference_mode()
def profile_method(args, cfg, data, nfe):
    pilot=frozen_pilot(args.pilot,"cuda")
    before=state_hash(pilot)
    renderer=load_renderer(args.renderer,file_hash(args.pilot),"cuda")
    flow=load_flow(args.model,"cuda")[0] if nfe else None
    camera=torch.tensor(view_pose(CHASE_CAMERA),device="cuda")[None]
    ts={k:[] for k in ("upload","encoder","history","phase2","flow","download","total")}
    comparisons=[]
    previous=None
    global_pos=0
    native=np.load(args.cache/"test"/"native.npy",mmap_mode="r") if not nfe else None
    torch.cuda.reset_peak_memory_stats()
    def mark():
        torch.cuda.synchronize()
        return time.perf_counter()
    for row in data["shards"]:
        ep=row["episode_id"]
        if ep!=previous:
            display=TemporalDisplay(renderer)
            stream=FlowRollout(flow,nfe) if nfe else None
            reference=np.load(args.rollouts/ep/f"nfe{nfe}.npy",mmap_mode="r") if nfe else None
            local=0
            previous=ep
        a={k:load_field(args.data,row,k) for k in ("rgb_in","delta_pose","t_sim","step")}
        for i,t in enumerate(a["t_sim"]):
            t0=mark()
            raw=torch.from_numpy(a["rgb_in"][i].copy()).permute(0,3,1,2)[None].cuda()
            t1=mark()
            features={k:v.half() for k,v in extract_mem(pilot,raw,pyramid=True).items()}
            t2=mark()
            delta=a["delta_pose"][i]
            display.observe(features,ep,t,delta)
            b=dict(**features,view_pose=camera,past=display.past,motion=display.motion,valid=display.valid)
            t3=mark()
            feat,rgb,support=phase2_readout(renderer,b)
            t4=mark()
            if nfe:
                current=F.avg_pool2d(torch.cat((rgb*2-1,support,feat),1),2).half()
                noise=frame_noise(row["episode_seed"],a["step"][i],[0],cfg["evaluation_seed"],"cuda")
                move=torch.from_numpy(delta.copy()).cuda()[None]
                pred=stream.render(current,camera,move,ep,t,noise)
                output=(F.interpolate(pred,scale_factor=2,mode="bilinear",align_corners=False)+1)*.5
            else:
                output=rgb
            t5=mark()
            torch.cat((output,support),1).mul(255).round().clamp(0,255).byte().cpu()
            t6=mark()
            if local>=8:
                for k,value in zip(("upload","encoder","history","phase2","flow","download"),np.diff((t0,t1,t2,t3,t4,t5,t6)),strict=True):
                    ts[k].append(1000*value)
                ts["total"].append(1000*(t6-t0))
            if local in (0,159,319):
                if nfe:
                    diff=.5*np.abs(pred[0].float().cpu().numpy()-reference[local,0].astype(np.float32))
                else:
                    diff=np.abs(output[0].cpu().numpy()-native[global_pos,0].astype(np.float32))
                comparisons.append(dict(episode=ep,offset=local,mean=float(diff.mean()),max=float(diff.max())))
            local+=1
            global_pos+=1
    if state_hash(pilot)!=before: raise RuntimeError("driving encoder changed during replay")
    stats={k+"_ms":{f"p{p}":float(np.percentile(v,p)) for p in (50,95,99)} for k,v in ts.items()}
    return dict(**stats,samples=len(ts["total"]),ticks=global_pos,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                over_100ms_fraction=float(np.mean(np.array(ts["total"])>100)),live_vs_cached=comparisons)


def profile(args):
    torch.set_num_threads(2)
    data=load_manifest(args.data)
    cfg=json.loads(args.config.read_text())
    run=json.loads((args.rollouts/"manifest.json").read_text())
    if (run["contract"]["model"]!=file_hash(args.model) or run["contract"]["renderer"]!=file_hash(args.renderer)
            or run["contract"]["cache"]!=file_hash(args.cache/"test"/"manifest.json")):
        raise ValueError("latency models or reference rollouts do not match")
    methods={}
    for n in (0,*cfg["test_nfe"]):
        gc.collect()
        torch.cuda.empty_cache()
        name=f"nfe{n}" if n else "phase2"
        methods[name]=profile_method(args,cfg,data,n)
        print(name,methods[name]["total_ms"],flush=True)
    write_json(args.out,dict(methods=methods,device=torch.cuda.get_device_name(),precision="FP32; frozen caches FP16",
               display="128x256 RGB plus support copied to CPU; pre-recorded input cameras; no simulation or monitor refresh",
               batch_size=1,warmup_per_episode=8,collection=data["contract_sha256"],flow=file_hash(args.model),
               renderer=file_hash(args.renderer),pilot=file_hash(args.pilot)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","cache","config","pilot","model","renderer","rollouts","out"):
        p.add_argument("--"+k,type=Path,required=True)
    profile(p.parse_args())
