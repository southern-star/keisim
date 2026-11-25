#!/usr/bin/env python3
"""Check cache/live parity, then time the two display resolutions on independent full closed loops."""
import json

import cv2
import numpy as np
import torch

from benchmark_reality_closed_loop import compare
from collect_reality_resolution import CFG, H, W, OUT, ROOT, camera, open_fields
from keireality.dataset import file_hash, write_json, writer_lock
from keireality.drive_metrics import summarize
from keireality.encoder import state_hash
from keireality.geometry import relative_pose, view_pose
from reality_closed_loop import Drive, Runtime


@torch.inference_mode()
def parity(runtime):
    """Compare the public live rendering path to the exact RGB scored for the fixed-camera replay."""
    town = 41
    raw = open_fields(OUT/"capture"/f"town{town}")
    cache = open_fields(OUT/"features"/f"town{town}")
    baseline,high = runtime.display("stable"),runtime.display("stable384")
    view = torch.tensor(view_pose(camera(0)),device="cuda")[None]
    runtime.agent.reset()
    absolute = np.zeros(2)
    maxima = np.zeros(2)
    count = 24
    for t in range(count):
        tm,speed,cmd,tx,ty = raw["nav"][t]
        _,plan = runtime.agent.act(list(raw["cameras"][t]),int(cmd),(tx,ty),speed,runtime.dt,with_reality=True)
        shared = plan["reality"]
        f = {k:shared[k].half() for k in ("mem","c2","c3")}
        f.update(stem=shared["stem"],source_logits=shared["source_logits"])
        f["estimated_depth"] = runtime.depth.depth_metres(runtime.depth.forward_depth(shared["rgb_in"])).reshape(1,2,80,160)
        delta = relative_pose(raw["pose"][t-1] if t else None,raw["pose"][t])
        for display in (baseline, high):
            display.observe(f,town,tm,delta)
        low, hi = baseline.render(view), high.render(view)
        x = torch.from_numpy(cache["inputs"][t,0].copy()).permute(2,0,1)[None].cuda().float()
        with torch.autocast("cuda",dtype=torch.float16):
            archived = runtime.resolution_head(x.contiguous(memory_format=torch.channels_last),runtime.resolution_strength)
        def byte_rgb(pred):
            return pred[0].permute(1,2,0).clamp(0,1).mul(255).round().byte().cpu().numpy()
        for k,(pred,ref) in enumerate(((low,cache["baseline"][t,0]),(hi,byte_rgb(archived)))):
            diff = np.abs(byte_rgb(pred).astype(np.int16)-ref.astype(np.int16))
            absolute[k] += diff.mean()
            maxima[k] = max(maxima[k],diff.max())
    mae = absolute/count
    if max(mae)>.10:
        raise ValueError(f"Scoring and live paths disagree: {mae}")
    return dict(town=town, ticks=count, modes=["stable","stable384"], mean_abs_rgb8=mae.tolist(),max_abs_rgb8=maxima.tolist())


def run():
    runtime = Runtime()
    before = state_hash(runtime.agent.model)
    quality = json.loads((OUT/"quality.json").read_text())
    if quality["selection"] != file_hash(OUT/"fit/selection.json"):
        raise ValueError("Display selection changed since quality scoring")
    runs,checks,pairs = {},{},{}
    try:
        agreement = parity(runtime)
        for town,modes in ((40,("stable","stable384")),(41,("stable384","stable"))):
            drives,runs[town],checks[town] = {},{},{}
            for mode in modes:
                drive = Drive(runtime,town,mode,capture_trace=True)
                drive.run(CFG["validation_ticks"])
                drives[mode] = drive
                if len(drive.steps)!=CFG["validation_ticks"] or drive.latest[1].shape[:2] != ((H,W) if mode=="stable384" else (128,256)):
                    raise ValueError("Incomplete drive or wrong native output shape")
                report = summarize(drive,16)
                report["over_100ms_all"] = sum(r["work_ms"]>100 for r in drive.steps)
                shown = {r[0] for r in drive.rendered}
                report["skipped_ticks"] = [t for t in range(len(drive.steps)) if t not in shown]
                runs[town][mode] = report
                checks[town][mode] = report["wall_control_hz"]>=9.9 and report["control_work_ms"]["p95"]<=100
                write_json(OUT/"timing"/f"town{town}-{mode}.json",dict(summary=report,steps=drive.steps,generated=drive.rendered))
                print(town,mode,"Hz",round(report["wall_control_hz"],3),"p95 ms",round(report["control_work_ms"]["p95"],1),
                      "misses",report["over_100ms"],"generated fps",round(report["generated_fps"],3),flush=True)
            base,high = drives["stable"],drives["stable384"]
            pairs[town] = compare(base.steps,high.steps,base.result,high.result)
        if before != state_hash(runtime.agent.model):
            raise ValueError("Driver weights or statistics changed")
        write_json(OUT/"timing.json",dict(gpu=torch.cuda.get_device_name(),warmup=16,control_dt=runtime.dt,
                   selection=quality["selection"],driver_state=before,parity=agreement,runs=runs,paired=pairs,
                   checks=checks,status="PASS" if all(row["stable384"] for row in checks.values()) else "HOLD"))
    finally:
        runtime.env.close()


if __name__=="__main__":
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    with writer_lock(OUT/"timing"):
        run()
