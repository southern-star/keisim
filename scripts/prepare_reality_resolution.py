#!/usr/bin/env python3
"""Cache the real driving feature pass for both display sizes and offline-only scoring geometry."""
import argparse
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch

from collect_reality_resolution import CFG, OUT, H, W, ROOT, ANGLES, camera, open_fields, sweep
from keireality.dataset import file_hash, write_json, writer_lock
from keireality.geometry import relative_pose, view_pose, visibility_now
from keireality.refined_display import RefinedDrive
from keireality.resolution import CHANNELS, SIZE, ResolutionDrive
from keireality.temporal_metrics import correspondence
from keisim.config import CAMERA_RIGS
from keisim.render import rays
from reality_closed_loop import Runtime


def source_spec(runtime):
    return dict(baseline=runtime.methods["stable"], model=file_hash(ROOT/"keireality/resolution.py"),
                prepare=file_hash(Path(__file__)), validation=file_hash(ROOT/"runs/reality/p2_blur/val-stability/metrics.json"),
                driver_wrapper=file_hash(ROOT/"keipilot/agent.py"))


@torch.inference_mode()
def prepare(runtime, town, model_spec):
    source = OUT / "capture" / f"town{town}"
    meta = json.loads((source/"capture.json").read_text())
    spec = dict(**model_spec, capture=file_hash(source/"capture.json"))
    dest = OUT / "features" / f"town{town}"
    if dest.exists():
        if json.loads((dest/"features.json").read_text())["spec"] != spec:
            raise ValueError("Feature sources or model changed")
        return
    if SIZE != (H,W) or CFG["baseline"] != runtime.methods["stable"]:
        raise ValueError("Invalid baseline or resolution")
    if any(file_hash(source/k) != v for k,v in meta["files"].items()):
        raise ValueError("Incomplete capture")
    raw = open_fields(source)
    n = len(raw["pose"])
    display = ResolutionDrive(RefinedDrive(runtime.renderer, runtime.detail, runtime.refiner, CFG["baseline"]))
    views = {a:torch.tensor(view_pose(camera(a)),device=runtime.device)[None] for a in ANGLES}
    directions = {a:rays.camera_rays(camera(a)) for a in ANGLES}
    runtime.agent.reset()
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        folder = Path(tmp) / "prepared"
        folder.mkdir()
        def array(name, shape, dtype):
            return np.lib.format.open_memmap(folder/(name+".npy"), "w+", dtype, (n,2,*shape))
        features = array("inputs",(H,W,CHANNELS),np.float16)
        native = array("baseline",(128,256,3),np.uint8)
        warp = array("warp",(H,W),np.int32)
        visible = array("visible",(H,W),np.bool_)
        for t in range(n):
            tm, speed, cmd, tx, ty = raw["nav"][t]
            pose = raw["pose"][t]
            _, plan = runtime.agent.act(list(raw["cameras"][t]), int(cmd), (tx,ty), speed, runtime.dt, with_reality=True)
            shared = plan["reality"]
            f = {k:shared[k].half() for k in ("mem","c2","c3")}
            f.update(stem=shared["stem"], source_logits=shared["source_logits"])
            f["estimated_depth"] = runtime.depth.depth_metres(runtime.depth.forward_depth(shared["rgb_in"])).reshape(1,2,80,160)
            display.observe(f, town, tm, relative_pose(raw["pose"][t-1] if t else None, pose))
            for slot, a in enumerate((0,sweep(t))):
                low, x = display.prepare(views[a],slot)
                if not torch.isfinite(x).all():
                    raise ValueError(f"Nonfinite native features {town}/{t}/{slot}")
                features[t,slot] = x[0].permute(1,2,0).half().cpu().numpy()
                native[t,slot] = (low[0].permute(1,2,0).clamp(0,1)*255).round().byte().cpu().numpy()
                z, seg = raw["depth"][t,slot], raw["labels"][t,slot]
                cam = camera(a)
                visible[t,slot] = visibility_now(cam,z,CAMERA_RIGS["wide_narrow"],raw["source_depth"][t]) == 1
                warp[t,slot].fill(-1)
                if t:
                    old = (raw["labels"][t-1,slot], raw["depth"][t-1,slot], raw["pose"][t-1])
                    (now,prev), _ = correspondence(cam, directions[a], (seg,z,pose), old,
                                                   camera(0 if slot==0 else sweep(t-1)))
                    warp[t,slot].reshape(-1)[now] = prev
        for field in (features, native, warp, visible):
            field.flush()
        write_json(folder/"features.json",dict(spec=spec,ticks=n,channels=CHANNELS))
        folder.rename(dest)
    print("paired inputs",town,n,"ticks × 2 tracks",flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("all","fit","selection","validation"), default="all")
    a = p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    runtime = Runtime()
    try:
        with writer_lock(OUT/"features"):
            spec = source_spec(runtime)
            for group in (("fit","selection","validation") if a.stage == "all" else (a.stage,)):
                for town in CFG[group+"_towns"]:
                    prepare(runtime,town,spec)
    finally:
        runtime.env.close()
