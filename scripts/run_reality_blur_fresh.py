#!/usr/bin/env python3
"""Collect and score the reserved drives only for a winner of the prespecified paired validation."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from collect_reality_sweep import CFG, FRESH, OUT, ROOT, sweep
from eval_reality_blur_pair import release
from eval_reality_depth_refiner import frozen, gate, generate, read_refiner, score
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.encoder import state_hash
from keireality.geometry import ORBIT_ANGLES, orbit_camera, visibility_now
from keireality.snapshot import state_digest
from keisim.config import EnvConfig, set_camera_rig
from keisim.env import KeiEnv
from prepare_reality_projection import checked_fields


def moving_teacher(env,angles,ep,rows,path,*,data=FRESH):
    env.reset(town_seed=ep["town_seed"],episode_seed=ep["episode_seed"],render=False)
    for _ in range(ep["start_step"]):
        env.step(env.expert_action())
    kv=env.keiview()
    if kv.gl!="hw":
        raise ValueError("hardware renderer expected")
    samples={k:[] for k in ("angle","rgb","seg","depth","seen")}
    for row in rows:
        raw=checked_fields(data,row,("state_sha256","depth_in","depth_target","seg_target","rgb_target"))
        for j in range(row["count"]):
            t=len(samples["angle"])
            angle=sweep(t)
            pose=tuple(env.world.ego.pose)
            scene=env.world.scene_state(include_ego=False)
            if state_digest(pose,env.world.t,scene)!=bytes(raw["state_sha256"][j]).decode():
                raise ValueError("fresh sweep differs from recorded driving state")
            images,z=kv.render_rig_depth(pose,scene,env.world.t,which=[2+angles.index(angle)])
            bgr,seg=images[0]
            rgb=np.ascontiguousarray(bgr[...,::-1])
            if angle in ORBIT_ANGLES:
                i=ORBIT_ANGLES.index(angle)
                if (not np.array_equal(z[0],raw["depth_target"][j,i]) or not np.array_equal(seg,raw["seg_target"][j,i])
                        or np.abs(rgb.astype(np.int16)-raw["rgb_target"][j,i]).mean()>1):
                    raise ValueError("fresh camera calibration mismatch")
            for key,value in dict(angle=angle,rgb=rgb,seg=seg,depth=z[0],
                    seen=visibility_now(orbit_camera(angle),z[0],env.cfg.cameras[:2],raw["depth_in"][j])).items():
                samples[key].append(value)
            if t+1<ep["frames"]:
                env.step(env.expert_action())
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        draft=Path(tmp)/path.name
        np.savez(draft,**{key:np.stack(v) for key,v in samples.items()})
        draft.replace(path)


def run():
    decision=release()
    if decision["status"]!="PASS":
        print("HOLD: neither registered candidate passed both towns; fresh collection not opened",flush=True)
        return
    winner=decision["candidates"][decision["winner"]]
    method=winner["method"]
    original=json.loads((ROOT/json.loads(CFG.read_text())["split_contract"]).read_text())
    episodes=[dict(split="test",town_seed=t,episode_seed=s,start_step=i,frames=320) for t,s,i in
              zip(original["fresh_towns"],original["fresh_seeds"],original["fresh_starts"],strict=True)]
    plan=OUT/"fresh-episodes.json"
    write_json(plan,episodes)
    subprocess.run([sys.executable,"-u",str(ROOT/"scripts/collect_reality.py"),"--gl","hw","--orbit","--audit",
                    "--plan",str(plan),"--out",str(FRESH)],cwd=ROOT,check=True)
    manifest=load_manifest(FRESH)
    if json.loads((FRESH/"plan.json").read_text())["episodes"]!=episodes:
        raise ValueError("unexpected reserved drives")
    dest=OUT/"fresh"
    angles=sorted(set(sweep(t) for t in range(48)))
    cfg=set_camera_rig(EnvConfig(renderer="keiview",weather="noon",keiview_gl="hw"),"wide_narrow")
    cfg.extra_cameras=(*cfg.extra_cameras,*(orbit_camera(a) for a in angles))
    env=KeiEnv(cfg)
    modules=frozen("cuda")
    refiner=read_refiner("cuda")
    before=[state_hash(m) for m in (*modules,refiner)]
    reports={}
    names=("A","D",method)
    try:
        with writer_lock(dest):
            for ep in episodes:
                town=ep["town_seed"]
                rows=[r for r in manifest["shards"] if r["town_seed"]==town and r["split"]=="test"]
                if sum(r["count"] for r in rows)!=320:
                    raise ValueError("incomplete reserved drive")
                with tempfile.TemporaryDirectory(dir=dest) as tmp:
                    file=Path(tmp)/"frames.npy"
                    frames,returns=generate(FRESH,rows,modules,refiner,method,names,"cuda",file)
                    truth=dest/f"town{town}-teacher.npz"
                    moving_teacher(env,angles,ep,rows,truth)
                    tracks,angles_report,tail=score(FRESH,rows,frames,names,truth,dest,method)
                    report=dict(ticks=320,tracks=tracks,sweep_by_angle=angles_report,tail=tail,roundtrips=returns,
                                candidates={method:gate(tracks,method)},rollout_sha256=file_hash(file),sweep_teacher_sha256=file_hash(truth))
                    del frames
                reports[town]=report
                write_json(dest/f"town{town}.json",report)
                print(town,report["candidates"],flush=True)
            if [state_hash(m) for m in (*modules,refiner)]!=before:
                raise ValueError("frozen weights changed")
            write_json(dest/"metrics.json",dict(method=method,release=file_hash(OUT/"release.json"),frozen_states=before,
                collection=manifest["contract_sha256"],towns=reports,
                status="PASS" if all(r["candidates"][method]["status"]=="PASS" for r in reports.values()) else "HOLD"))
    finally:
        env.close()


if __name__=="__main__":
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    run()
