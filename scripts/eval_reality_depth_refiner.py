#!/usr/bin/env python3
"""Compare source-depth repairs with the frozen A/D controls on fixed and moving cameras."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.depth_refinement import SourceDepthRefiner
from keireality.encoder import state_hash
from keireality.geometry import orbit_camera, view_pose
from keireality.refined_display import RefinedDrive, VARIANTS
from keireality.temporal_metrics import correspondence
from keireality.video import quad, write_video
from keisim.render import rays
from collect_reality_sweep import CFG, OUT, split_info, sweep
from diagnose_reality_blur import WEIGHTS, frozen, inputs
from eval_reality_projection import bytes_rgb
from eval_reality_scene_detail import NAMES, SceneScores, release
from prepare_reality_projection import checked_fields

TRACKS=("chase","sweep")
SOURCES=("keireality/depth_refinement.py","keireality/refined_display.py","keireality/inverse_surface.py","keireality/background_history.py",
         "keireality/scene_detail.py","keireality/depth_projection.py","keireality/surface_observation.py",
         "scripts/eval_reality_depth_refiner.py","scripts/collect_reality_sweep.py",
         "scripts/eval_reality_scene_detail.py","scripts/diagnose_reality_blur.py","configs/reality/p2_blur.json",
         "configs/reality/p2_surface_history.json","runs/reality/p2_blur/fit/depth.pt",
         "runs/reality/p2_blur/diagnosis/metrics.json",*WEIGHTS)


def contract():
    return {name:file_hash(ROOT/name) for name in SOURCES}


def read_refiner(device):
    model=SourceDepthRefiner().to(device).eval()
    model.load_state_dict(torch.load(OUT/"fit/depth.pt",map_location="cpu")["model"])
    return model.requires_grad_(False)


def contours(row, method):
    marking=row["classes"].get("marking",{}).get("edge",{})
    result={}
    for key,v,d in (("vehicle",row[method]["silhouette"],row["D"]["silhouette"]),
                    ("marking",marking.get(method),marking.get("D"))):
        result[key]=d is not None and v is not None and v>=d-1e-5
    return result


def gate(tracks,method):
    chase=tracks["chase"]
    six=release(dict(A=chase["A"],C=chase[method]))
    guards={key:contours(tracks[key],method) for key in TRACKS}
    passed=six["status"]=="PASS" and all(all(g.values()) for g in guards.values())
    return dict(status="PASS" if passed else "HOLD",six_vs_A=six,contours_vs_D=guards)


def to_rgb(tensor):
    return tensor[0].permute(1,2,0).cpu().numpy()


@torch.inference_mode()
def generate(data,rows,modules,refiner,method,names,device,path):
    pilot,head,model,detail=modules
    drive=RefinedDrive(model,detail,refiner,method,compare="all" if len(names)>3 else True)
    views={a:torch.tensor(view_pose(orbit_camera(a)),device=device)[None] for a in range(-36,37,3)}
    n=sum(r["count"] for r in rows)
    frames=np.lib.format.open_memmap(path,"w+",np.float32,(n,2,len(names),128,256,3))
    tick=0
    roundtrips=[]
    for row in rows:
        raw=checked_fields(data,row,("step","rgb_in","t_sim","delta_pose"))
        for j,t in enumerate(raw["t_sim"]):
            if raw["step"][j]!=rows[0]["first"]+tick:
                raise ValueError("nonconsecutive input drive")
            _,f=inputs(pilot,head,raw["rgb_in"][j],device,tick==0)
            drive.observe(f,row["episode_id"],t,raw["delta_pose"][j])
            for slot,angle in enumerate((0,sweep(tick))):
                outputs=drive.render(views[angle],slot)
                frames[tick,slot]=torch.cat([outputs[m] for m in names]).permute(0,2,3,1).cpu().numpy()
            if tick in (0,n//2,n-1):
                # Use the chase slot at the same observation; restore it before observing the next tick.
                for angle in (18,36,18,0,-18,-36,-18,0):
                    outputs=drive.render(views[angle],0)
                    if angle==0:
                        values={m:float(np.abs(to_rgb(outputs[m])-frames[tick,0,k]).mean())
                                for k,m in enumerate(names)}
                        roundtrips.append(dict(tick=tick,mean_absolute_return_difference=values))
            tick+=1
    if not np.isfinite(frames).all() or max(max(r["mean_absolute_return_difference"].values()) for r in roundtrips)>1/255:
        raise ValueError("invalid rollout or state changed during same-time orbit")
    frames.flush()
    return frames,roundtrips


def scores(names,totals):
    """SceneScores measures three images together. Combine their independent sums on identical samples."""
    groups=[totals[i//3].report(tuple(names[i:i+3])) for i in range(0,len(names),3)]
    result=groups[0]
    del result["fraction_mean_rgb_change_below_one_8bit_level_vs_A"]
    for start,other in enumerate(groups[1:],1):
        if result["counts"]!=other["counts"]:
            raise ValueError("different scoring samples for candidates")
        for m in names[3*start:3*start+3]:
            result[m]=other[m]
            for cls,values in other["classes"].items():
                for key in ("edge","l1"):
                    result["classes"][cls][key][m]=values[key][m]
            for cls,values in other["static_by_class"].items():
                result["static_by_class"][cls][m]=values[m]
    return result


def score(data,rows,frames,names,truth_path,dest,method):
    n=len(frames)
    groups=len(names)//3
    total=[[SceneScores() for _ in range(groups)] for _ in TRACKS]
    angles=sorted({sweep(t) for t in range(48)})
    per_angle={a:[SceneScores() for _ in range(groups)] for a in angles}
    tail=SceneScores() if n==320 else None
    cameras={a:orbit_camera(a) for a in angles}
    directions={a:rays.camera_rays(cam) for a,cam in cameras.items()}
    previous=[None,None]
    errors=[[None]*groups for _ in TRACKS]
    tick=0
    clips=[[],[]]
    town=rows[0]["town_seed"]
    with np.load(truth_path) as saved:
        moving={k:saved[k] for k in saved.files}
    for row in rows:
        truth=checked_fields(data,row,("rgb_target","depth_target","seg_target","view_pose","visibility_now","ego_pose_world"))
        for j in range(row["count"]):
            if moving["angle"][tick]!=sweep(tick) or not np.allclose(truth["view_pose"][j,0],view_pose(cameras[0]),atol=1e-6,rtol=0):
                raise ValueError("teacher view does not match requested camera")
            pose=truth["ego_pose_world"][j]
            for k,angle in enumerate((0,sweep(tick))):
                if k==0:
                    gt,seg,depth,seen=(truth[key][j,0] for key in ("rgb_target","seg_target","depth_target","visibility_now"))
                else:
                    gt,seg,depth,seen=(moving[key][tick] for key in ("rgb","seg","depth","seen"))
                now=(seg,depth,pose)
                warp=None
                if previous[k] is not None:
                    old,old_camera=previous[k]
                    warp=correspondence(cameras[angle],directions[angle],now,old,previous_camera=old_camera)
                target=gt.astype(np.float32)/255
                for group in range(groups):
                    predictions=frames[tick,k,3*group:3*group+3]
                    args=(predictions,target,seg,seen==1,.10,warp,errors[k][group])
                    if k==1:
                        per_angle[angle][group].add(*args)
                    if k==0 and tail is not None and tick>=n-80:
                        tail.add(*args)
                    errors[k][group]=total[k][group].add(*args)
                previous[k]=now,cameras[angle]
                if dest is not None:
                    clips[k].append((gt.copy(),*bytes_rgb(frames[tick,k])))
            tick+=1
    if tick!=n:
        raise ValueError("incomplete score")
    tracks={name:scores(names,group) for name,group in zip(TRACKS,total,strict=True)}
    angles_report={}
    for angle,group in per_angle.items():
        s=scores(names,group)
        angles_report[angle]=dict(count=s["counts"]["silhouette"],
                                 silhouette={m:s[m]["silhouette"] for m in names},
                                 marking=s["classes"].get("marking",{}).get("edge",{}))
    if dest is not None:
        for k,track in enumerate(TRACKS):
            def panels():
                for t,(gt,a,d,e) in enumerate(clips[k]):
                    angle=0 if k==0 else sweep(t)
                    panel=quad([(gt,f"正解　町{town}　{t/10:.1f}秒　{angle:+d}°"),(a,"A　元のPhase 2"),
                                (d,"D　現在の表示"),(e,"E　深度補正")])
                    if t==n//2:
                        Image.fromarray(panel).save(dest/f"town{town}-{track}.png")
                    yield panel
            if write_video(dest/f"town{town}-{track}.mp4",panels())!=n:
                raise ValueError("incomplete movie")
    return tracks,angles_report,tail.report(names) if tail is not None else None


def rank(towns, m):
    gates=[gate(t["tracks"],m) for t in towns.values()]
    passed=sum(g["status"]=="PASS" for g in gates)
    guards=sum(sum(v.values()) for g in gates for v in g["contours_vs_D"].values())
    six=sum(sum(g["six_vs_A"].get("checks",{}).values()) for g in gates)
    edge=float(np.mean([f[m]["scene_edges"]-f["D"]["scene_edges"] for t in towns.values() for f in t["tracks"].values()]))
    flicker=float(np.mean([f["D"]["static_flicker"]-f[m]["static_flicker"] for t in towns.values() for f in t["tracks"].values()]))
    return [passed,guards,six,edge,flicker]


def freeze():
    dest=OUT/"development"
    with writer_lock(dest):
        data=json.loads((dest/"metrics.json").read_text())
        if contract()!=data["contract"]:
            raise ValueError("development candidate changed")
        ranks={m:rank(data["towns"],m) for m in VARIANTS}
        winner=max(ranks,key=ranks.get)
        write_json(dest/"selection.json",dict(variant=winner,rank_order=["towns_passing","contour_checks","six_checks",
            "mean_edge_gain","mean_flicker_reduction"],ranks=ranks,development=file_hash(dest/"metrics.json"),contract=contract()))
        print("selected",winner,ranks,flush=True)


def evaluate(stage,device):
    data,split,towns,ticks=split_info(stage)
    raw=load_manifest(data)
    hashes=contract()
    method=VARIANTS[0]
    if stage!="development":
        selection=json.loads((OUT/"development/selection.json").read_text())
        if selection["contract"]!=hashes or selection["development"]!=file_hash(OUT/"development/metrics.json"):
            raise ValueError("candidate changed since development")
        method=selection["variant"]
        hashes["selection"]=file_hash(OUT/"development/selection.json")
        if stage=="fresh" and json.loads((OUT/"val/metrics.json").read_text())["contract"]!=hashes:
            raise ValueError("candidate changed since validation")
    names=("A","D",*VARIANTS) if stage=="development" else ("A","D",method)
    modules=frozen(device)
    refiner=read_refiner(device)
    before=[state_hash(m) for m in (*modules,refiner)]
    dest=OUT/stage
    with writer_lock(dest):
        baseline=None
        if stage!="fresh":
            baseline=json.loads((ROOT/f"runs/reality/p2_surface_history/{split}/metrics.json").read_text())
        reports={}
        for town in towns:
            rows=[r for r in raw["shards"] if r["split"]==split and r["town_seed"]==town]
            if sum(r["count"] for r in rows)!=ticks or len({r["episode_id"] for r in rows})!=1:
                raise ValueError("incomplete episode")
            truth=dest/"sweep_truth"/f"town{town}.npz"
            meta=json.loads(truth.with_suffix(".json").read_text())
            if file_hash(truth)!=meta["sha256"] or meta["contract"]["collection"]!=raw["contract_sha256"]:
                raise ValueError("sweep teacher provenance mismatch")
            with tempfile.TemporaryDirectory(dir=dest) as tmp:
                path=Path(tmp)/"outputs.npy"
                frames,returns=generate(data,rows,modules,refiner,method,names,device,path)
                tracks,angles,tail=score(data,rows,frames,names,truth,None if stage=="development" else dest,method)
                rollout=file_hash(path)
                del frames
            if baseline is not None:
                old=baseline["towns"][str(town)]["views"]["0"]
                for m in ("A","D"):
                    for key in NAMES:
                        if old[m][key] is not None and abs(tracks["chase"][m][key]-old[m][key])>1e-5:
                            raise ValueError(f"changed {m}/{key} control on town {town}")
            report=dict(ticks=ticks,tracks=tracks,sweep_by_angle=angles,roundtrips=returns,tail=tail,
                        rollout_sha256=rollout,sweep_teacher_sha256=meta["sha256"],
                        candidates={m:gate(tracks,m) for m in names[2:]})
            reports[town]=report
            write_json(dest/f"town{town}.json",report)
            print(stage,town,report["candidates"],flush=True)
        if [state_hash(m) for m in (*modules,refiner)]!=before:
            raise ValueError("frozen weights changed")
        status="DEVELOPMENT" if stage=="development" else ("PASS" if all(t["candidates"][method]["status"]=="PASS" for t in reports.values()) else "HOLD")
        write_json(dest/"metrics.json",dict(status=status,stage=stage,method=method if stage!="development" else None,
                 contract=hashes,collection=raw["contract_sha256"],frozen_states=before,towns=reports,
                 media={p.name:file_hash(p) for p in dest.iterdir() if p.suffix in (".png",".mp4")}))
        print(stage,status,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("development","val","fresh","freeze"),required=True)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if args.stage=="freeze":
        freeze()
    else:
        evaluate(args.stage,args.device)
