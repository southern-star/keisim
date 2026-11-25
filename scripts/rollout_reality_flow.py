#!/usr/bin/env python3
"""Generate complete held-out videos causally, before loading targets for scoring."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, write_json, writer_lock
from keireality.flow_data import FlowSplit
from keireality.flow_model import ConditionalVelocity, FlowRollout, frame_noise, generate, make_condition


SWEEP = (0,-18,-36,-18,0,18,36,18)


def load_flow(path,device):
    ck = torch.load(path,map_location="cpu")
    net = ConditionalVelocity(ck["sigma"])
    net.load_state_dict(ck["model"])
    return net.eval().requires_grad_(False).to(device=device,memory_format=torch.channels_last),ck["contract"]


@torch.inference_mode()
def rollout(args):
    torch.set_num_threads(2)
    cfg=json.loads(args.config.read_text())
    ds=FlowSplit(args.data,args.cache,"test",args.device,teachers=False)
    field,trained=load_flow(args.model,args.device)
    if (ds.meta["renderer"]!=trained["renderer"] or ds.angles!=cfg["test_angles"] or
            trained["protocol"]!=file_hash(args.config) or trained["mode"]!="meanflow"):
        raise ValueError("training and held-out condition contracts differ")
    if ([e.town for e in ds.episodes]!=cfg["eval_towns"] or
            any(e.length!=cfg["frames_per_town"] for e in ds.episodes)):
        raise ValueError("G3 requires the four complete planned episodes")
    test_towns={e.town for e in ds.episodes}
    training_meta=json.loads((args.training_data/"plan.json").read_text())
    if test_towns & {e["town_seed"] for e in training_meta["episodes"]}:
        raise ValueError("held-out town was used by an earlier Phase 2 split")
    contract=dict(version=1,model=file_hash(args.model),protocol=file_hash(args.config),
                  cache=file_hash(ds.cache/"manifest.json"),evaluation_seed=cfg["evaluation_seed"],
                  angles=ds.angles,sweep=list(SWEEP),renderer=ds.meta["renderer"],nfe=cfg["test_nfe"])
    with writer_lock(args.out):
        records={}
        for ep in ds.episodes:
            dest=args.out/ep.name
            if dest.exists():
                m=json.loads((dest/"episode.json").read_text())
                if (m["contract"]!=contract or m["ticks"]!=ep.length or
                        any(file_hash(dest/(k+".npy"))!=h for k,h in m["arrays"].items())):
                    raise ValueError(f"invalid previous rollout {ep.name}")
                records[ep.name]=m
                continue
            tick=time.perf_counter()
            with tempfile.TemporaryDirectory(dir=args.out,prefix=".writing-") as tmp:
                stage=Path(tmp)/"episode"
                stage.mkdir()
                arrays={}
                def output(k,shape,dtype=np.float16):
                    arr=np.lib.format.open_memmap(stage/(k+".npy"),mode="w+",dtype=dtype,shape=shape)
                    arrays[k]=arr
                    return arr
                indices=output("views",(ep.length,len(ds.angles)+1),np.uint8)
                streams={n:FlowRollout(field,n) for n in cfg["test_nfe"]}
                for n in streams:
                    output(f"nfe{n}",(ep.length,len(ds.angles)+1,3,64,128))
                    output(f"no_previous{n}",(ep.length,3,64,128))
                used={str(n):0 for n in streams}
                for offset in range(ep.length):
                    i=ep.start+offset
                    selected=[*range(len(ds.angles)),ds.angles.index(SWEEP[offset*len(SWEEP)//ep.length])]
                    indices[offset]=selected
                    cams=[ds.angles[j] for j in selected]
                    vi=torch.tensor(selected,device=args.device)
                    current,view=ds.current[i,vi],ds.poses[i,vi]
                    delta=ds.delta[i].expand(len(vi),-1)
                    noise=frame_noise(ep.seed,ds.steps[i],cams,cfg["evaluation_seed"],args.device)
                    empty=make_condition(current[:1],view[:1],torch.zeros_like(noise[:1]),torch.zeros_like(view[:1]),
                                         delta[:1],view.new_zeros(1),torch.zeros(1,dtype=torch.bool,device=view.device))
                    for n,stream in streams.items():
                        img=stream.render(current,view,delta,ep.name,ds.times[i],noise)
                        if not torch.isfinite(img).all() or stream.used_previous!=(offset>0):
                            raise ValueError("invalid image or autoregressive history discontinuity")
                        used[str(n)]+=int(stream.used_previous)
                        arrays[f"nfe{n}"][offset]=img.half().cpu().numpy()
                        no_history=generate(field,noise[:1],empty,n)
                        arrays[f"no_previous{n}"][offset]=no_history[0].half().cpu().numpy()
                for arr in arrays.values(): arr.flush()
                record=dict(contract=contract,ticks=ep.length,first_step=int(ds.steps[ep.start]),
                            first_time=float(ds.times[ep.start]),last_time=float(ds.times[ep.start+ep.length-1]),
                            feedback_transitions_per_track=used,seconds=time.perf_counter()-tick,
                            arrays={k:file_hash(stage/(k+".npy")) for k in arrays})
                write_json(stage/"episode.json",record)
                stage.rename(dest)
                records[ep.name]=record
            print(ep.name,ep.length,"causal ticks",flush=True)
        write_json(args.out/"manifest.json",dict(contract=contract,episodes=records))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","cache","config","model","out","training-data"): p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    rollout(p.parse_args())
