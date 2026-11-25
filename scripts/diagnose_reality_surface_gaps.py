#!/usr/bin/env python3
"""Paired support transitions and a controlled alpha intervention on training drives."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.encoder import extract_mem, frozen_pilot, state_hash
from keireality.geometry import CHASE_CAMERA, view_pose
from keireality.scene_detail import SceneDetail
from keireality.surface_observation import ObservedDetail
from keisim.config import SEM
from prepare_reality_projection import checked_fields


CLASSES = ("building", "vegetation")
GROUPS = ("weak_both", "gained", "lost", "supported_both")
MEASURES = ("A", "C", "equal_alpha")


def read_detail(model_type, device):
    protocol = ROOT / "configs/reality/p2_scene_detail.json"
    checkpoint = ROOT / "runs/reality/p2_scene_detail/train/readout.pt"
    state = torch.load(checkpoint, map_location="cpu")
    if state["contract"]["protocol"] != file_hash(protocol):
        raise ValueError("stem readout protocol mismatch")
    model = model_type(json.loads(protocol.read_text())).to(device).eval()
    model.camera_color.load_state_dict({k: state[k] for k in ("weight", "bias")})
    return model


def as_rgb(tensor):
    return tensor[0].permute(1, 2, 0).contiguous().cpu().numpy()


def pair_sums(counts, thresholds, warp, labels, now, old, *, exact_gaps=False):
    index = np.flatnonzero(warp > 0)
    last = warp[index] - 1
    alpha, previous_alpha = now[2][index], old[2][last]
    common = np.minimum(alpha, previous_alpha)
    delta_c = now[1][index] - old[1][last]
    # Each raw frame contains alpha * (projected_color - base) on top of the
    # other layers. Remove only the difference in alpha on this matched pair.
    removed = ((alpha-common)[:, None] * now[3][index]
               - (previous_alpha-common)[:, None] * old[3][last])
    errors = (np.abs(now[0][index]-old[0][last]).mean(1),
              np.abs(delta_c).mean(1), np.abs(delta_c-removed).mean(1))
    labels = labels.ravel()[index]
    for k, name in enumerate(CLASSES):
        for i, cutoff in enumerate(thresholds):
            mask = labels == SEM[name]
            if exact_gaps:
                # Exclude partial coverage on either endpoint. A zero means that the
                # projected-color layer contributed exactly zero to the output RGB.
                mask &= ((alpha==0) | (alpha>=cutoff)) & ((previous_alpha==0) | (previous_alpha>=cutoff))
            group = (2*(previous_alpha[mask] >= cutoff) + (alpha[mask] >= cutoff)).astype(int)
            counts[i,k,:,0] += np.bincount(group, minlength=4)
            for j, error in enumerate(errors, 1):
                counts[i,k,:,j] += np.bincount(group, weights=error[mask], minlength=4)


def summarize(counts, thresholds):
    def row(sums):
        n, *total = sums
        return dict(pairs=int(n), **{m: float(s/n) if n else None for m,s in zip(MEASURES,total)})
    output = {}
    for i, threshold in enumerate(thresholds):
        classes = {}
        for j, name in enumerate(CLASSES):
            sums = counts[i,j]
            groups = {n: row(s) for n,s in zip(GROUPS,sums)}
            switching = row(sums[1:3].sum(0))
            steady = groups["supported_both"]
            supported = (switching["pairs"] > 0 and steady["pairs"] > 0
                         and switching["C"]-switching["A"] > steady["C"]-steady["A"]
                         and switching["equal_alpha"] < switching["C"])
            classes[name] = dict(**groups, switching=switching, all=row(sums.sum(0)),
                                 supports_gap_hypothesis=supported)
        output[str(threshold)] = classes
    return output


@torch.inference_mode()
def diagnose(device, out, *, exact_gaps=False):
    cfg_path = ROOT / "configs/reality/p2_surface_history.json"
    cfg = json.loads(cfg_path.read_text())
    thresholds = [cfg["primary_alpha"], *cfg["sensitivity_alpha"]]
    data = ROOT / "data/reality/p2"
    source = ROOT / "data/reality/p2_projection/train"
    manifest = load_manifest(data)
    meta = json.loads((source/"manifest.json").read_text())
    rows = [r for r in manifest["shards"] if r["split"] == "train"]
    pilot_path = ROOT / "runs/keipilot_mc.pt"
    if (meta["contract"]["collection"] != manifest["contract_sha256"]
            or meta["contract"]["pilot"] != file_hash(pilot_path)
            or meta["contract"]["angles"][0] != 0
            or meta["contract"]["rows"] != [r["path"] for r in rows]
            or sorted({r["town_seed"] for r in rows}) != cfg["train_towns"]):
        raise ValueError("training cache does not match the experiment")
    files = ("base", "estimated", "training_warp")
    if any(file_hash(source/(n+".npy")) != meta["files"][n+".npy"] for n in files):
        raise ValueError("corrupt training cache")
    bases, depths, warps = [np.load(source/(n+".npy"), mmap_mode="r") for n in files]
    pilot = frozen_pilot(pilot_path, device)
    detail = read_detail(ObservedDetail, device)
    control = read_detail(SceneDetail, device)
    pilot_state = state_hash(pilot)
    view = torch.tensor(view_pose(CHASE_CAMERA), device=device)[None]
    counters, per_town_ticks = {}, {}
    offset = 0
    parity_max = 0.
    replay_max = 0.
    for town in cfg["train_towns"]:
        total = np.zeros((len(thresholds), len(CLASSES), 4, 4), np.float64)
        previous = old = last_step = None
        ticks = 0
        for row in (r for r in rows if r["town_seed"] == town):
            raw = checked_fields(data, row, ("rgb_in","step","delta_pose","seg_target","rgb_target"))
            for start in range(0, row["count"], 4):
                images = torch.from_numpy(raw["rgb_in"][start:start+4].copy()).permute(0,1,4,2,3).to(device)
                features = extract_mem(pilot, images, pyramid=True, scene=True, parity=ticks==0)
                colors, probs = detail.encode(features["stem"], features["source_logits"])
                for b in range(len(images)):
                    j = start+b
                    if last_step is not None and raw["step"][j] != last_step+1:
                        raise ValueError("nonconsecutive training drive")
                    base = torch.tensor(bases[offset,0:1], device=device)
                    estimated = torch.tensor(depths[offset:offset+1], device=device)
                    encoded = colors[b:b+1], probs[b:b+1]
                    frame = detail.inspect(base, encoded, view, estimated)
                    delta = torch.tensor(raw["delta_pose"][j], device=device)
                    c = detail.temporal(frame.rgb, frame.depth, frame.static, frame.direction,
                                        view, delta, previous)
                    if ticks == 0:
                        ref = control.frame(base, encoded, view, estimated)[0]
                        replay_max = max(replay_max, (ref-frame.rgb).abs().max().item())
                        # Reusing identical splats isolates the trace from CUDA scatter-add ordering.
                        with patch.object(control, "surfaces", return_value=detail.last_surface):
                            same = control.frame(base, encoded, view, estimated)[0]
                        parity_max = max(parity_max, (same-frame.rgb).abs().max().item())
                    gt = raw["rgb_target"][j,0].reshape(-1,3).astype(np.float32)/255
                    now = (as_rgb(base).reshape(-1,3)-gt, as_rgb(c).reshape(-1,3)-gt,
                           as_rgb(frame.alpha).ravel(), as_rgb(frame.surface_rgb-base).reshape(-1,3))
                    if old is not None:
                        pair_sums(total,thresholds,warps[offset,0].ravel(),raw["seg_target"][j,0],now,old,
                                  exact_gaps=exact_gaps)
                    old = now
                    previous = (frame.rgb, frame.depth, frame.static, view)
                    last_step = raw["step"][j]
                    ticks += 1
                    offset += 1
        counters[town] = total
        per_town_ticks[town] = ticks
        print(f"train town {town}: {ticks} frames", flush=True)
    if (offset != meta["ticks"] or offset != sum(r["count"] for r in rows)
            or parity_max > 1e-6 or state_hash(pilot) != pilot_state):
        raise ValueError(f"ticks={offset}/{meta['ticks']}, identical-splat error={parity_max}, "
                         f"independent replay error={replay_max}, pilot stable={state_hash(pilot)==pilot_state}")
    pooled = summarize(sum(counters.values()), thresholds)
    by_town = {k:summarize(v,thresholds) for k,v in counters.items()}
    key = str(cfg["primary_alpha"])
    result = dict(status="SUPPORTED" if all(v["supports_gap_hypothesis"] for v in pooled[key].values())
                  else "NOT_SUPPORTED", pooled=pooled, towns=by_town, ticks=per_town_ticks,
                  parity_max=parity_max, independent_replay_pixel_max=replay_max, pilot_state=pilot_state,
                  contract=dict(protocol=file_hash(cfg_path), diagnosis=file_hash(Path(__file__)),
                                trace=file_hash(ROOT/"keireality/surface_observation.py"),
                                cache=file_hash(source/"manifest.json"),
                                readout=file_hash(ROOT/"runs/reality/p2_scene_detail/train/readout.pt")))
    if exact_gaps:
        result["selection"] = "Each endpoint must have alpha exactly zero or at least the threshold."
    path = out/("exact-gaps.json" if exact_gaps else "diagnosis.json")
    if path.exists():
        saved = json.loads(path.read_text())
        if (saved["status"]!=result["status"] or saved["pilot_state"]!=result["pilot_state"]
                or any(saved["contract"][k]!=result["contract"][k] for k in ("protocol","trace","cache","readout"))):
            raise ValueError("different saved diagnosis; select a separate --out")
        # Preserve the file referenced by the selection and validation when a CUDA
        # replay differs only by the reduction order of the scattered point colors.
        for town, levels in by_town.items():
            for level, classes in levels.items():
                for name, groups in classes.items():
                    original = saved["towns"][str(town)][level][name]
                    if original["supports_gap_hypothesis"]!=groups["supports_gap_hypothesis"]:
                        raise ValueError("town-level diagnosis changed")
                    for group in (*GROUPS, "switching", "all"):
                        a,b = original[group],groups[group]
                        if a["pairs"]!=b["pairs"] or any(abs(a[m]-b[m])>1e-6 for m in MEASURES if a["pairs"]):
                            raise ValueError("saved support transitions differ; select a separate --out")
        print("verified",path,flush=True)
        return
    write_json(path,result)
    print(result["status"], json.dumps(pooled[key]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    p.add_argument("--out", type=Path, default=ROOT/"runs/reality/p2_surface_history/train")
    p.add_argument("--exact-gaps", action="store_true", help="exclude intermediate nonzero projection weights")
    args = p.parse_args()
    torch.set_num_threads(2)
    with writer_lock(args.out):
        diagnose(args.device, args.out, exact_gaps=args.exact_gaps)
