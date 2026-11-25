#!/usr/bin/env python3
"""Record the registered moving teacher on the same simulated states as the stored camera inputs."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.geometry import ORBIT_ANGLES, orbit_camera, visibility_now
from keireality.snapshot import state_digest
from keisim.config import EnvConfig, set_camera_rig
from keisim.env import KeiEnv
from prepare_reality_projection import checked_fields

OUT=ROOT/"runs/reality/p2_blur"
CFG=ROOT/"configs/reality/p2_blur.json"
FRESH=ROOT/"data/reality/p2_surface_fresh"


def sweep(tick):
    """A 4.8 second orbit: center, left, center, right, center, at 3 degrees per tick."""
    t=tick%48
    return 3*(t if t<=12 else 24-t if t<=36 else t-48)


def split_info(stage):
    cfg=json.loads(CFG.read_text())
    original=json.loads((ROOT/cfg["split_contract"]).read_text())
    if cfg["sweep_degrees_per_tick"]!=3 or cfg["max_angle"]!=36:
        raise ValueError("unknown orbit")
    if stage=="fresh":
        val=json.loads((OUT/"val/metrics.json").read_text())
        if val["status"]!="PASS":
            raise ValueError("fresh drives require both validation towns to pass")
        return FRESH,"test",original["fresh_towns"],320
    if stage=="val":
        if not (OUT/"development/selection.json").is_file():
            raise ValueError("freeze candidate before validating")
        return ROOT/"data/reality/p2","val",original["validation_towns"],96
    return ROOT/"data/reality/p2","train",cfg["development_towns"],128


def collect(stage):
    data,split,towns,ticks=split_info(stage)
    raw=load_manifest(data)
    angles=sorted(set(sweep(t) for t in range(48)))
    cfg=set_camera_rig(EnvConfig(renderer="keiview",weather="noon",keiview_gl="hw"),"wide_narrow")
    cfg.extra_cameras=(*cfg.extra_cameras,*(orbit_camera(a) for a in angles))
    env=KeiEnv(cfg)
    dest=OUT/stage/"sweep_truth"
    try:
        with writer_lock(dest):
            for town in towns:
                path=dest/f"town{town}.npz"
                meta=dest/f"town{town}.json"
                contract=dict(collection=raw["contract_sha256"],town=town,ticks=ticks,sweep=[sweep(i) for i in range(48)],
                              program=file_hash(Path(__file__)))
                if path.exists() and meta.exists():
                    result=json.loads(meta.read_text())
                    if result["contract"]!=contract or result["sha256"]!=file_hash(path):
                        raise ValueError("different sweep capture")
                    continue
                rows=[r for r in raw["shards"] if r["split"]==split and r["town_seed"]==town]
                if sum(r["count"] for r in rows)!=ticks:
                    raise ValueError("wrong sweep length")
                env.reset(town_seed=town,episode_seed=rows[0]["episode_seed"],render=False)
                kv=env.keiview()
                if kv.gl!="hw":
                    raise ValueError("hardware renderer expected")
                for _ in range(rows[0]["first"]):
                    env.step(env.expert_action())
                fields={k:[] for k in ("rgb","seg","depth","seen","angle")}
                max_input=anchor_rgb=anchor_depth=anchor_seg=anchors=0
                input_abs=anchor_abs=chase_abs=0
                chase_depth=chase_seg=0
                for row in rows:
                    source=checked_fields(data,row,("step","rgb_in","depth_in","state_sha256","rgb_target","seg_target","depth_target"))
                    for j in range(row["count"]):
                        t=len(fields["angle"])
                        angle=sweep(t)
                        pose=tuple(env.world.ego.pose)
                        scene=env.world.scene_state(include_ego=False)
                        digest=state_digest(pose,env.world.t,scene)
                        if digest!=bytes(source["state_sha256"][j]).decode():
                            raise ValueError(f"unmatched state {town}/{t}")
                        which=[0,1,2+angles.index(0)]
                        if angle!=0:
                            which.append(2+angles.index(angle))
                        imgs,depths=kv.render_rig_depth(pose,scene,env.world.t,which=which)
                        rgb=np.ascontiguousarray(imgs[-1][0][...,::-1])
                        seg=imgs[-1][1]
                        input_rgb=np.stack([bgr[...,::-1] for bgr,_ in imgs[:2]])
                        input_diff=np.abs(input_rgb.astype(np.int16)-source["rgb_in"][j])
                        max_input=max(max_input,int(input_diff.max()))
                        input_abs+=int(input_diff.sum())
                        chase_abs+=int(np.abs(imgs[2][0][...,::-1].astype(np.int16)-source["rgb_target"][j,0]).sum())
                        chase_depth+=int(np.count_nonzero(depths[2]!=source["depth_target"][j,0]))
                        chase_seg+=int(np.count_nonzero(imgs[2][1]!=source["seg_target"][j,0]))
                        if angle in ORBIT_ANGLES:
                            vi=ORBIT_ANGLES.index(angle)
                            anchors+=1
                            anchor_diff=np.abs(rgb.astype(np.int16)-source["rgb_target"][j,vi])
                            anchor_rgb=max(anchor_rgb,int(anchor_diff.max()))
                            anchor_abs+=int(anchor_diff.sum())
                            anchor_depth+=int(np.count_nonzero(depths[-1]!=source["depth_target"][j,vi]))
                            anchor_seg+=int(np.count_nonzero(seg!=source["seg_target"][j,vi]))
                        fields["angle"].append(angle)
                        fields["rgb"].append(rgb)
                        fields["seg"].append(seg)
                        fields["depth"].append(depths[-1])
                        fields["seen"].append(visibility_now(orbit_camera(angle),depths[-1],cfg.cameras[:2],source["depth_in"][j]))
                        if t+1<ticks:
                            env.step(env.expert_action())
                input_mae=input_abs/(ticks*2*160*320*3)
                anchor_mae=anchor_abs/(anchors*128*256*3)
                chase_mae=chase_abs/(ticks*128*256*3)
                # Simulated states and geometry agree exactly. Chrome versions may change final
                # RGB rounding; measure the differences against the archived cameras in 8-bit units.
                if anchor_depth or anchor_seg or chase_depth or chase_seg or max(input_mae,anchor_mae,chase_mae)>1:
                    raise ValueError(f"sweep recapture mismatch: {anchor_depth=} {anchor_seg=} {chase_depth=} {chase_seg=} {input_mae=} {chase_mae=}")
                with tempfile.TemporaryDirectory(dir=dest) as tmp:
                    draft=Path(tmp)/path.name
                    np.savez(draft,**{k:np.stack(v) for k,v in fields.items()})
                    draft.replace(path)
                write_json(meta,dict(contract=contract,sha256=file_hash(path),anchors=anchors,
                                     max_input_difference=max_input,anchor_rgb_max=anchor_rgb,input_mae_8bit=input_mae,
                                     anchor_mae_8bit=anchor_mae,chase_mae_8bit=chase_mae,
                                     chase_depth_mismatches=chase_depth,chase_label_mismatches=chase_seg,
                                     anchor_depth_mismatches=anchor_depth,anchor_label_mismatches=anchor_seg))
                print(stage,town,ticks,"verified states;",anchors,"matching anchor geometries; RGB MAE",anchor_mae,flush=True)
    finally:
        env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("development","val","fresh"),required=True)
    collect(p.parse_args().stage)
