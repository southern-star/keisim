#!/usr/bin/env python3
"""Run the diagnosed three-frame background repair and the predeclared long-drive gate."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from keireality.background_history import BackgroundHistoryDisplay
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.encoder import frozen_pilot, state_hash
from keireality.geometry import ORBIT_ANGLES, orbit_camera
from keireality.surface_observation import ObservedDetail
from diagnose_reality_surface_gaps import read_detail
from eval_reality_scene_detail import NAMES, SceneScores, produce, release, score_drive
from eval_reality_temporal import load_renderer


MODELS = ("A","C","D")
CAPTIONS = ("A　現行の Phase 2","C　全景の深度投影","D　直近3フレームの背景を補完")
OUT = ROOT/"runs/reality/p2_surface_history"


def decision(row):
    scores = release({"A":row["A"],"C":row["D"]})
    marking = row["classes"].get("marking",{}).get("edge",{})
    guard = {}
    for label,c,d in (("vehicle",row["C"]["silhouette"],row["D"]["silhouette"]),
                      ("marking",marking.get("C"),marking.get("D"))):
        guard[label] = c is not None and d is not None and d >= c-1e-5
    return dict(status="PASS" if scores["status"]=="PASS" and all(guard.values()) else "HOLD",
                six_vs_A=scores, contours_vs_C=guard)


def contract():
    files = ("configs/reality/p2_surface_history.json", "configs/reality/p2_scene_detail.json",
             "keireality/background_history.py","keireality/surface_observation.py","keireality/scene_detail.py",
             "scripts/eval_reality_surface_history.py","scripts/eval_reality_scene_detail.py",
             "runs/reality/p2_scene_detail/train/readout.pt","runs/keipilot_mc.pt","runs/reference/mc14-depth.pt",
             "runs/reality/p2/temporal/best.pt","runs/reality/p2_scene_detail/val/metrics.json",
             "runs/reality/p2_surface_history/train/diagnosis.json")
    return {name:file_hash(ROOT/name) for name in files}


def evaluate(args):
    cfg = json.loads((ROOT/"configs/reality/p2_surface_history.json").read_text())
    previous = json.loads((ROOT/"runs/reality/p2_scene_detail/val/metrics.json").read_text())
    diagnosis = json.loads((OUT/"train/diagnosis.json").read_text())
    if (diagnosis["status"]!="SUPPORTED" or diagnosis["contract"]["protocol"]!=file_hash(ROOT/"configs/reality/p2_surface_history.json")
            or diagnosis["contract"]["trace"]!=file_hash(ROOT/"keireality/surface_observation.py")):
        raise ValueError("a matching positive training diagnosis is required")
    hashes = contract()
    if args.split!="train":
        selection_file=OUT/"train/selection.json"
        selection=json.loads(selection_file.read_text())
        if selection["model_sha256"]!=hashes["keireality/background_history.py"]:
            raise ValueError("the renderer differs from the one selected on training towns")
        hashes["training_selection"]=file_hash(selection_file)
    if args.split=="test":
        val = json.loads((OUT/"val/metrics.json").read_text())
        if val["status"]!="PASS" or val["contract"]!=hashes:
            raise ValueError("the frozen candidate has not passed on both validation towns")
    args.data = ROOT/("data/reality/p2_surface_fresh" if args.split=="test" else "data/reality/p2")
    raw = load_manifest(args.data)
    plan = json.loads((args.data/"plan.json").read_text())
    rows = [r for r in raw["shards"] if r["split"]==args.split]
    towns = cfg[{"train":"train_towns","val":"validation_towns","test":"fresh_towns"}[args.split]]
    ticks = {"train":128,"val":96,"test":320}[args.split]
    angles = [0] if args.split=="train" else ORBIT_ANGLES
    cameras = [orbit_camera(v) for v in angles]
    if ([asdict(orbit_camera(v)) for v in ORBIT_ANGLES]!=plan["calibration"]["targets"]
            or list(dict.fromkeys(r["town_seed"] for r in rows))!=towns):
        raise ValueError("unexpected cameras or town split")
    if args.split=="test":
        expected = [(t,s,i,320) for t,s,i in zip(towns,cfg["fresh_seeds"],cfg["fresh_starts"],strict=True)]
        if [(e["town_seed"],e["episode_seed"],e.get("start_step",0),e["frames"]) for e in plan["episodes"]]!=expected:
            raise ValueError("held-out routes differ from reserved drives")
    pilot = frozen_pilot(ROOT/"runs/keipilot_mc.pt", args.device)
    depth = frozen_pilot(ROOT/"runs/reference/mc14-depth.pt",args.device)
    phase2 = load_renderer(ROOT/"runs/reality/p2/temporal/best.pt",file_hash(ROOT/"runs/keipilot_mc.pt"),args.device)
    detail = read_detail(ObservedDetail,args.device)
    modules = (pilot,depth,phase2,detail)
    before = [state_hash(m) for m in modules]
    dest = OUT/args.split
    with writer_lock(dest):
        output = dest/"metrics.json"
        if output.exists():
            saved=json.loads(output.read_text())
            if (saved["contract"]!=hashes or saved["collection"]!=raw["contract_sha256"]
                    or any(file_hash(dest/f)!=h for f,h in saved["media"].items())):
                raise ValueError("incompatible saved experiment")
            print("verified",saved["status"],flush=True)
            return
        evidence, pooled = {}, [SceneScores() for _ in cameras]
        for town in towns:
            sequence=[r for r in rows if r["town_seed"]==town]
            if sum(r["count"] for r in sequence)!=ticks or len({r["episode_id"] for r in sequence})!=1:
                raise ValueError("incomplete town")
            display=BackgroundHistoryDisplay(phase2,detail)
            with tempfile.TemporaryDirectory(dir=dest) as tmp:
                path=Path(tmp)/"frames.npy"
                frames,rmse=produce(args,sequence,pilot,depth,display,cameras,path,ticks)
                result=score_drive(args,sequence,frames,cameras,pooled,
                                   {"eval_angles":angles,"edge_threshold":.10},dest if args.split!="train" else None,
                                   models=MODELS,check=decision,captions=CAPTIONS,
                                   tail_ticks=80 if args.split=="test" else 0)
                result.update(source_rmse=rmse, rollout_sha256=file_hash(path))
                del frames
            if args.split=="val":
                reference=previous["towns"][str(town)]["views"]
                for angle,sc in result["views"].items():
                    for method in ("A","C"):
                        if any(abs(sc[method][k]-reference[angle][method][k])>1e-5 for k in NAMES):
                            raise ValueError(f"changed {method} control on {town}/{angle}")
            row=result["views"]["0"]
            print(town,result["chase_check"],"static flicker",[row[m]["static_flicker"] for m in MODELS],
                  "marking",row["classes"].get("marking",{}).get("edge",{}),flush=True)
            evidence[town]=result
        if [state_hash(m) for m in modules]!=before:
            raise ValueError("frozen model state changed")
        status="PASS" if all(v["chase_check"]["status"]=="PASS" for v in evidence.values()) else "HOLD"
        write_json(output,dict(status=status,split=args.split,contract=hashes,collection=raw["contract_sha256"],
                               frozen_states=before,towns=evidence,pooled={str(v):p.report(MODELS) for v,p in zip(angles,pooled)},
                               media={f.name:file_hash(f) for f in dest.iterdir() if f.suffix in (".mp4",".png")}))
        print(args.split,status,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",choices=("train","val","test"),required=True)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    evaluate(args)
