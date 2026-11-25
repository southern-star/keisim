#!/usr/bin/env python3
"""Factorial source-color and source-depth intervention on identical training drives."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keireality.background_history import BackgroundHistoryDisplay
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.encoder import extract_mem, frozen_pilot, state_hash
from keireality.geometry import orbit_camera, view_pose
from keireality.surface_observation import ObservedDetail
from keireality.temporal_metrics import correspondence
from keireality.video import quad, write_video
from keisim.render import rays
from diagnose_reality_surface_gaps import read_detail
from eval_reality_projection import bytes_rgb
from eval_reality_scene_detail import NAMES, SceneScores
from eval_reality_temporal import load_renderer
from prepare_reality_projection import checked_fields

OUT = ROOT / "runs/reality/p2_blur"
DATA = ROOT / "data/reality/p2"
NAMES4 = ("D", "color", "depth", "both")
GROUPS = ((0, 1, 2), (0, 2, 3))
TITLES = ("現行D", "色だけ元画像", "深度だけ正解値", "色と深度の両方")
WEIGHTS = ("runs/keipilot_mc.pt", "runs/reference/mc14-depth.pt",
           "runs/reality/p2/temporal/best.pt", "runs/reality/p2_scene_detail/train/readout.pt")


def frozen(device):
    pilot, head = (frozen_pilot(ROOT / name, device) for name in WEIGHTS[:2])
    renderer = load_renderer(ROOT / WEIGHTS[2], file_hash(ROOT / WEIGHTS[0]), device)
    return pilot, head, renderer, read_detail(ObservedDetail, device)


def inputs(pilot, head, rgb, device, first):
    image = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2)[None].to(device)
    features = extract_mem(pilot, image, pyramid=True, scene=True, parity=first)
    for key in ("mem", "c2", "c3"):
        features[key] = features[key].half()
    features["estimated_depth"] = head.depth_metres(head.forward_depth(image)).reshape(1, 2, 80, 160)
    return image, features


def factorial(row):
    d, c, z, both = (row[key]["scene_edges"] for key in NAMES4)
    return dict(color_only=c-d, depth_only=z-d, color_marginal=((c-d)+(both-z))/2,
                depth_marginal=((z-d)+(both-c))/2, interaction=both-c-z+d)


@torch.inference_mode()
def generate(data, rows, modules, device, path):
    pilot, head, model, detail = modules
    displays = [BackgroundHistoryDisplay(model, detail) for _ in NAMES4]
    view = torch.as_tensor(view_pose(orbit_camera(0)), device=device)[None]
    n = sum(r["count"] for r in rows)
    frames = np.lib.format.open_memmap(path, "w+", np.float32, (n, 4, 128, 256, 3))
    tick = 0
    first = rows[0]["first"]
    for row in rows:
        raw = checked_fields(data, row, ("rgb_in", "depth_in", "step", "t_sim", "delta_pose"))
        for j, t in enumerate(raw["t_sim"]):
            if raw["step"][j] != first+tick:
                raise ValueError("nonconsecutive diagnostic")
            image, features = inputs(pilot, head, raw["rgb_in"][j], device, tick==0)
            measured = rays.decode_depth(raw["depth_in"][j, :, ::2, ::2]).astype(np.float32)
            oracle = dict(features, estimated_depth=torch.from_numpy(measured)[None].to(device))
            for k, display in enumerate(displays):
                display.observe(oracle if k in (2,3) else features, row["episode_id"], t, raw["delta_pose"][j])
                if k in (1,3):
                    colors = image.float()/255
                    display.encoded = colors, display.encoded[1]
                    # The history keeps the exact same source, with its background probability in channel four.
                    display.observation.colors = torch.cat((colors, display.observation.colors[:,:,3:]), 2)
                frame = display.render(view)[0][0]
                frames[tick,k] = frame.permute(1,2,0).cpu().numpy()
            tick += 1
    if tick!=n or not np.isfinite(frames).all():
        raise ValueError("incomplete factorial")
    return frames


def scored(data, rows, frames, pooled, dest):
    cam = orbit_camera(0)
    dirs = rays.camera_rays(cam)
    totals = [SceneScores() for _ in GROUPS]
    errors = [None, None]
    tick, previous = 0, None
    town = rows[0]["town_seed"]
    for row in rows:
        truth = checked_fields(data,row,("rgb_target","seg_target","visibility_now","depth_target","ego_pose_world","view_pose"))
        for j in range(row["count"]):
            if not np.allclose(truth["view_pose"][j,0], view_pose(cam), atol=1e-6, rtol=0):
                raise ValueError("wrong teacher camera")
            gt = truth["rgb_target"][j,0].astype(np.float32)/255
            seg = truth["seg_target"][j,0]
            now = (seg, truth["depth_target"][j,0],truth["ego_pose_world"][j])
            warp = correspondence(cam,dirs,now,previous) if previous is not None else None
            for k, group in enumerate(GROUPS):
                errors[k] = totals[k].add(frames[tick,list(group)], gt, seg,
                                           truth["visibility_now"][j,0]==1, .10, warp, errors[k])
            previous = now
            tick += 1
    def panels():
        for i, values in enumerate(frames):
            labels = (f"町{town}　{i/10:.1f}秒　{TITLES[0]}", *TITLES[1:])
            panel = quad(zip(bytes_rgb(values), labels, strict=True))
            if i==len(frames)//2:
                Image.fromarray(panel).save(dest/f"town{town}.png")
            yield panel
    if write_video(dest/f"town{town}.mp4", panels())!=tick:
        raise ValueError("incomplete diagnostic movie")
    for total, target in zip(totals, pooled, strict=True):
        target.merge(total)
    return report(totals)


def report(totals):
    a, b = [total.report(tuple(NAMES4[i] for i in group)) for total,group in zip(totals,GROUPS,strict=True)]
    a["both"] = b["both"]
    for values, other in ((a["classes"], b["classes"]), (a["static_by_class"], b["static_by_class"])):
        for key, value in values.items():
            if "edge" in value:
                for metric in ("edge","l1"):
                    value[metric]["both"] = other[key][metric]["both"]
            else:
                value["both"] = other[key]["both"]
    del a["fraction_mean_rgb_change_below_one_8bit_level_vs_A"]
    a["factorial_scene_edges"] = factorial(a)
    return a


def run(device):
    dest = OUT / "diagnosis"
    with writer_lock(dest):
        raw = load_manifest(DATA)
        codes = ("keireality/background_history.py","keireality/surface_observation.py","keireality/scene_detail.py",
                 "keireality/depth_projection.py","scripts/diagnose_reality_blur.py","scripts/eval_reality_scene_detail.py")
        contract = {k:file_hash(ROOT/k) for k in (*WEIGHTS,*codes)}
        contract["collection"] = raw["contract_sha256"]
        modules = frozen(device)
        before = [state_hash(m) for m in modules]
        old = json.loads((ROOT/"runs/reality/p2_surface_history/train/metrics.json").read_text())
        pooled = [SceneScores(),SceneScores()]
        towns = {}
        for town in range(8):
            rows = [r for r in raw["shards"] if r["split"]=="train" and r["town_seed"]==town]
            if sum(r["count"] for r in rows)!=128:
                raise ValueError("wrong training split")
            with tempfile.TemporaryDirectory(dir=dest) as tmp:
                frames = generate(DATA, rows, modules, device, Path(tmp)/"outputs.npy")
                result = scored(DATA,rows,frames,pooled,dest)
                del frames
            old_d = old["towns"][str(town)]["views"]["0"]["D"]
            if any(abs(old_d[k]-result["D"][k])>1e-5 for k in NAMES if old_d[k] is not None):
                raise ValueError(f"D baseline differs in town {town}")
            towns[town]=result
            write_json(dest/f"town{town}.json",result)
            print(town, result["factorial_scene_edges"],flush=True)
        merged = report(pooled)
        effects = merged["factorial_scene_edges"]
        focus = max(("depth","color"),key=lambda k:effects[k+"_marginal"])
        if effects[focus+"_marginal"] <= 0:
            focus="coverage"
        if [state_hash(m) for m in modules]!=before:
            raise ValueError("changed frozen weights")
        write_json(dest/"metrics.json",dict(towns=towns,pooled=merged,focus=focus,contract=contract,ticks=1024,
                                            media={p.name:file_hash(p) for p in dest.iterdir() if p.suffix in (".mp4",".png")}))
        print("focus",focus,"pooled",effects,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    run(p.parse_args().device)
