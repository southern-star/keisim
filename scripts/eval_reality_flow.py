#!/usr/bin/env python3
"""Score saved causal videos against synchronized teachers, geometry, an independent RGB probe and the Phase 2 reference."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, load_field, write_json, writer_lock
from keireality.flow_data import FlowSplit
from keireality.flow_metrics import (confusion, display_image, finalize, g3_decision, measure, projected_scores,
                                     region_masks, region_scores, sample_region)
from keireality.geometry import orbit_camera
from keireality.rgb_probe import RGBProbe, probe_scores
from keireality.temporal_metrics import Totals, correspondence
from keisim.config import N_SEM
from keisim.render import rays


def load_json(path):
    return json.loads(Path(path).read_text())


def rgb_unit(x):
    return (np.asarray(x,dtype=np.float32).transpose(1,2,0)+1)*.5


@torch.inference_mode()
def classify(probe, images, device):
    x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).to(device)
    return probe(x*2-1).argmax(1).byte().cpu().numpy()


def score(args):
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    cfg=load_json(args.config)
    ds=FlowSplit(args.data,args.cache,"test",teachers=False)
    run=load_json(args.rollouts/"manifest.json")
    profile=load_json(args.profile)
    if (run["contract"]["cache"]!=file_hash(ds.cache/"manifest.json") or
            run["contract"]["protocol"]!=file_hash(args.config) or profile["flow"]!=run["contract"]["model"] or
            profile["renderer"]!=ds.meta["renderer"] or profile["collection"]!=ds.meta["collection"]):
        raise ValueError("rollout, latency and source identities differ")
    sm=load_json(args.labels/"manifest.json")
    if sm["collection_sha256"]!=ds.meta["collection"]:
        raise ValueError("history labels and targets differ")
    probe_state=torch.load(args.probe,map_location="cpu")
    if (probe_state["contract"]["protocol"]!=file_hash(args.config) or
            set(probe_state["contract"]["train_towns"]+probe_state["contract"]["val_towns"]) & set(cfg["eval_towns"])):
        raise ValueError("RGB probe uses incompatible splits or protocol")
    probe=RGBProbe().to(args.device).eval()
    probe.load_state_dict(probe_state["model"])
    native=np.load(ds.cache/"native.npy",mmap_mode="r")
    chase=np.load(ds.cache/"chase.npy",mmap_mode="r")
    cams=[orbit_camera(a) for a in ds.angles]
    directions=[rays.camera_rays(c) for c in cams]
    scores=defaultdict(lambda:defaultdict(Totals))
    semantics=defaultdict(lambda:defaultdict(lambda:np.zeros((N_SEM,N_SEM),np.int64)))
    calibration=np.zeros((N_SEM,N_SEM),np.int64)
    support_totals=Totals()
    traces=[]
    coverage={}
    tick=time.perf_counter()
    for ep in ds.episodes:
        loc=args.rollouts/ep.name
        record=load_json(loc/"episode.json")
        if record!=run["episodes"][ep.name] or record["contract"]!=run["contract"]:
            raise ValueError("unpaired episode manifest")
        for key,digest in record["arrays"].items():
            if file_hash(loc/(key+".npy"))!=digest:
                raise ValueError("corrupt saved prediction "+key)
        gen={key:np.load(loc/(key+".npy"),mmap_mode="r") for key in record["arrays"]}
        coverage[ep.name]=dict(ticks=record["ticks"],transitions=record["feedback_transitions_per_track"],
                               time_span_s=record["last_time"]-record["first_time"])
        old_geometry,old_frames={},{}
        last_stop=None
        for row,base in zip(ds.rows,ds.offsets,strict=True):
            if row["episode_id"]!=ep.name: continue
            keys=("rgb_target","seg_target","depth_target","ego_pose_world","junction","ego_speed","visibility_now")
            for key in keys:
                if file_hash(args.data/row["path"]/(key+".npy"))!=row["fields"][key]["sha256"]:
                    raise ValueError(f"invalid teacher field {key} in {row['path']}")
            fields={k:load_field(args.data,row,k) for k in keys}
            spath=args.labels/row["path"]/"visibility.npy"
            if file_hash(spath)!=sm["shards"][row["path"]]["sha256"]:
                raise ValueError("corrupt support labels")
            supports=np.load(spath,mmap_mode="r")
            for j in range(row["count"]):
                i=base+j
                off=i-ep.start
                time_now=ds.times[i]
                speed=float(fields["ego_speed"][j])
                if speed<.4: last_stop=time_now
                views=gen["views"][off].tolist()
                inputs=[]; predictions=[]
                for track,v in enumerate(views):
                    gt=fields["rgb_target"][j,v].astype(np.float32)/255
                    gt_low=cv2.resize(gt,(128,64),interpolation=cv2.INTER_AREA)
                    low={"phase2":rgb_unit(ds.current[i,v,:3].numpy())}
                    low.update({f"nfe{n}":rgb_unit(gen[f"nfe{n}"][off,track]) for n in cfg["test_nfe"]})
                    if track==0:
                        low["chase_spatial"]=cv2.resize(chase[i].astype(np.float32).transpose(1,2,0),(128,64),interpolation=cv2.INTER_AREA)
                        low.update({f"no_previous{n}":rgb_unit(gen[f"no_previous{n}"][off]) for n in cfg["test_nfe"]})
                    inputs.extend([gt_low,*low.values()])
                    predictions.append((gt,gt_low,low))
                classified=classify(probe,inputs,args.device)
                cursor=0
                trace=dict(episode=ep.name,offset=off,step=int(ds.steps[i]),seconds=off/10,speed=speed,chase={})
                central_frames={}
                for track,v in enumerate(views):
                    gt,gt_low,low=predictions[track]
                    real_seg=fields["seg_target"][j,v]
                    seg=sample_region(real_seg)
                    mask=region_masks(sample_region(supports[j,v]))
                    if not np.array_equal(mask["visible"],sample_region(fields["visibility_now"][j,v])==1):
                        raise ValueError("current support supervision was altered")
                    is_fixed=track<len(ds.angles)
                    if is_fixed:
                        calibration+=confusion(seg,classified[cursor])
                        prob=ds.current[i,v,3].float().numpy()
                        known=~mask["unknown"]; supported=mask["visible"]|mask["past"]
                        support_totals.add({"known_brier":measure((prob[known]-supported[known])**2),
                                            "predicted_support_fraction":measure((prob>=.5)[known])})
                    cursor+=1
                    geom=(real_seg,fields["depth_target"][j,v],fields["ego_pose_world"][j])
                    warp=None
                    if off:
                        old_v,old=old_geometry[track]
                        warp=correspondence(cams[v],directions[v],geom,old,cams[old_v])
                    old_geometry[track]=(v,geom)
                    other=None
                    if 0<track<len(ds.angles):
                        other=correspondence(cams[v],directions[v],geom,old_geometry[0][1],cams[0])[0]
                    if is_fixed:
                        groups=["fixed",ep.name,f"view{ds.angles[v]:+d}",f"quarter{off*4//ep.length}"]
                        if track==0:
                            groups.append("chase")
                            if bool(fields["junction"][j]): groups.append("chase_junction")
                            if last_stop is not None and 0<time_now-last_stop<=3 and speed>1:
                                groups.append("chase_restart")
                    else:
                        groups=["view_switch_stream"]
                        if off and gen["views"][off-1,track]!=v:
                            groups.append("view_switch_transition")
                    for mode,img in low.items():
                        pred_seg=classified[cursor]
                        cursor+=1
                        common_up=display_image(img)
                        if mode=="phase2":
                            display=native[i,v].astype(np.float32).transpose(1,2,0)
                        elif mode=="chase_spatial":
                            display=chase[i].astype(np.float32).transpose(1,2,0)
                        else:
                            display=common_up
                        stats=region_scores(img,gt_low,pred_seg,seg,mask)
                        stats["native_l1"]=measure(np.abs(display-gt).mean(-1))
                        seg_up=cv2.resize(pred_seg,(256,128),interpolation=cv2.INTER_NEAREST)
                        frame=(common_up-gt,gt,seg_up)
                        if warp is not None:
                            stats.update(projected_scores(frame,old_frames[track,mode],warp))
                        old_frames[track,mode]=frame
                        if track==0:
                            central_frames[mode]=frame
                            trace["chase"][mode]={k:(float(s/n) if n else None) for k,(s,n) in stats.items()
                                                  if k in ("all_l1","native_l1","warped_residual_l1","phantom_fpr")}
                        elif other is not None:
                            a,b=other
                            delta=frame[0].reshape(-1,3)[a]-central_frames[mode][0].reshape(-1,3)[b]
                            stats["cross_view_residual_l1"]=measure(np.abs(delta).mean(-1))
                        mode_groups=groups
                        if mode.startswith("no_previous") or mode=="chase_spatial":
                            mode_groups=[g for g in groups if g.startswith("chase") or g=="view+0"]
                        for g in mode_groups: scores[g][mode].add(stats)
                        if is_fixed:
                            for name,region in mask.items():
                                semantics[mode][name]+=confusion(seg,pred_seg,region)
                if cursor!=len(classified): raise RuntimeError("RGB probe batching mismatch")
                traces.append(trace)
        print("scored",ep.name,flush=True)
    summary={g:{m:finalize(tot) for m,tot in modes.items()} for g,modes in scores.items()}
    test_probe=probe_scores(calibration)
    valid_iou=probe_state["selected"]["clean"]["dynamic_iou"]
    minimum_iou = cfg["phantom_probe"]["minimum_dynamic_iou_on_real_validation_and_test"]
    calibration_ok = all(iou is not None and iou >= minimum_iou for iou in (valid_iou, test_probe["dynamic_iou"]))
    complete=(set(coverage)==set(run["episodes"]) and len(coverage)==4 and
              all(v["ticks"]==cfg["frames_per_town"] and v["time_span_s"]>=30 and
                  all(k==v["ticks"]-1 for k in v["transitions"].values()) for v in coverage.values()) and
              all(summary["fixed"][f"nfe{n}"]["frames"]==len(ds.angles)*len(ds.current) for n in cfg["test_nfe"]))
    decisions=g3_decision(cfg,summary,profile,calibration_ok,complete)
    status="PASS" if any(d["status"]=="PASS" for d in decisions.values()) else "FAIL"
    if not calibration_ok or not complete: status="INCONCLUSIVE"
    result=dict(status=status,decisions=decisions,scores=summary,coverage=coverage,
                probe=dict(validation_iou=valid_iou,test=test_probe,reliable=calibration_ok,sha256=file_hash(args.probe)),
                semantics={m:{r:probe_scores(mat) for r,mat in regions.items()} for m,regions in semantics.items()},
                observability=support_totals.report(),protocol_sha256=file_hash(args.config),
                rollouts_sha256=file_hash(args.rollouts/"manifest.json"),runtime_sha256=file_hash(args.profile),
                support_sha256=file_hash(args.labels/"manifest.json"),seconds=time.perf_counter()-tick)
    write_json(args.out/"metrics.json",result)
    write_json(args.out/"trajectory.json",traces)
    print(json.dumps(dict(status=status,decisions=decisions)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","cache","config","probe","profile","labels","rollouts","out"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=p.parse_args()
    with writer_lock(args.out): score(args)
