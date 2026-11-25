#!/usr/bin/env python3
"""Independently score the two prespecified sharpness/stability choices using the identical release gate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import cv2
import torch

from eval_reality_depth_refiner import (ROOT, OUT, TRACKS, NAMES, contract, file_hash, frozen, gate, generate,
                                      load_manifest, read_refiner, score, split_info, state_hash, write_json, writer_lock)

PAIR=OUT/"development/paired-selection.json"


def register():
    selection_path=OUT/"development/selection.json"
    s=json.loads(selection_path.read_text())
    ranks=s["ranks"]
    eligible=[m for m,v in ranks.items() if v[:3]==ranks[s["variant"]][:3] and v[3]>0]
    secondary=max(eligible,key=lambda m:ranks[m][4])
    info=dict(development=file_hash(OUT/"development/metrics.json"),primary_selection=file_hash(selection_path),
              primary=s["variant"],secondary=secondary,rank=ranks[secondary],weights_and_software=s["contract"])
    if PAIR.exists():
        saved=json.loads(PAIR.read_text())
        if any(saved[k]!=v for k,v in info.items()):
            raise ValueError("changed paired selection")
        return saved
    if any((OUT/"val").glob("town*.json")) or (OUT/"val/metrics.json").exists():
        raise ValueError("register the secondary before either validation score")
    result=dict(**info,created_utc=datetime.now(timezone.utc).isoformat(),
                selection="Among development variants tying on pass counts with positive edge gain, prefer the greatest flicker reduction. Primary has first release priority.")
    write_json(PAIR,result)
    return result


def verify_pair():
    pair=register()
    if pair["weights_and_software"]!=contract():
        raise ValueError("candidate weights or source changed")
    return pair


def secondary_contract():
    return dict(contract(),selection=file_hash(PAIR),comparison_runner=file_hash(Path(__file__)))


def run_secondary(device):
    pair=verify_pair()
    method=pair["secondary"]
    primary=json.loads((OUT/"val/metrics.json").read_text())
    if primary["method"]!=pair["primary"] or primary["contract"]!=dict(contract(),selection=pair["primary_selection"]):
        raise ValueError("primary validation does not match the registered pair")
    data,split,towns,ticks=split_info("val")
    raw=load_manifest(data)
    names=("A","D",method)
    modules=frozen(device)
    refiner=read_refiner(device)
    before=[state_hash(m) for m in (*modules,refiner)]
    dest=OUT/"val-stability"
    reports={}
    with writer_lock(dest):
        for town in towns:
            rows=[r for r in raw["shards"] if r["split"]==split and r["town_seed"]==town]
            if sum(r["count"] for r in rows)!=ticks:
                raise ValueError("incomplete validation drive")
            truth=OUT/"val/sweep_truth"/f"town{town}.npz"
            existing=primary["towns"][str(town)]
            if file_hash(truth)!=existing["sweep_teacher_sha256"]:
                raise ValueError("sweep teacher changed")
            with tempfile.TemporaryDirectory(dir=dest) as tmp:
                path=Path(tmp)/"frames.npy"
                frames,returns=generate(data,rows,modules,refiner,method,names,device,path)
                tracks,angles,tail=score(data,rows,frames,names,truth,dest,method)
                rollout=file_hash(path)
                del frames
            for track in TRACKS:
                for m in ("A","D"):
                    for key in NAMES:
                        if abs(tracks[track][m][key]-existing["tracks"][track][m][key])>1e-5:
                            raise ValueError(f"changed {track}/{m}/{key} control")
            report=dict(ticks=ticks,tracks=tracks,sweep_by_angle=angles,roundtrips=returns,tail=tail,
                        rollout_sha256=rollout,sweep_teacher_sha256=file_hash(truth),candidates={method:gate(tracks,method)})
            reports[town]=report
            write_json(dest/f"town{town}.json",report)
            print("stability",town,report["candidates"],flush=True)
        if before!=[state_hash(m) for m in (*modules,refiner)]:
            raise ValueError("frozen model state changed")
        status="PASS" if all(r["candidates"][method]["status"]=="PASS" for r in reports.values()) else "HOLD"
        write_json(dest/"metrics.json",dict(status=status,stage="val-stability",method=method,contract=secondary_contract(),
                    collection=raw["contract_sha256"],frozen_states=before,towns=reports,
                    media={p.name:file_hash(p) for p in dest.iterdir() if p.suffix in (".mp4",".png")}))
    return release()


def release():
    pair=verify_pair()
    candidates={}
    for role,stage,expected in (("primary","val",dict(contract(),selection=pair["primary_selection"])),
                                ("secondary","val-stability",secondary_contract())):
        path=OUT/stage/"metrics.json"
        data=json.loads(path.read_text())
        method=pair[role]
        if data["method"]!=method or data["contract"]!=expected or set(data["towns"])!={"40","41"}:
            raise ValueError("wrong validation evidence")
        checks={t:gate(r["tracks"],method) for t,r in data["towns"].items()}
        status="PASS" if all(c["status"]=="PASS" for c in checks.values()) else "HOLD"
        if status!=data["status"]:
            raise ValueError("stored and recomputed decisions differ")
        candidates[role]=dict(method=method,status=status,checks=checks,validation=stage,metrics_sha256=file_hash(path))
    winner=next((r for r,v in candidates.items() if v["status"]=="PASS"),None)
    decision=dict(status="PASS" if winner else "HOLD",winner=winner,candidates=candidates,preregistration=file_hash(PAIR))
    write_json(OUT/"release.json",decision)
    return decision


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--register",action="store_true")
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if args.register:
        print(register()["secondary"])
    else:
        print(run_secondary(args.device)["status"])
