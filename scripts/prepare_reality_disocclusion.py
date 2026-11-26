#!/usr/bin/env python3
"""Diagnose history on training towns and cache causal cues separately from offline exposure labels."""
import argparse
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch

from collect_reality_resolution import ANGLES, CFG, OUT as ORIGINAL, ROOT, TRACKS, camera, open_fields, sweep
from download_reality_release import MANIFEST, verify
from keireality.dataset import file_hash, write_json, writer_lock
from keireality.disocclusion import ActorContext, actor_support
from keireality.geometry import relative_pose, view_pose
from keireality.resolution import SIZE
from keireality.temporal_metrics import correspondence
from keisim.config import SEM
from keisim.render import rays
from reality_closed_loop import Runtime


OUT = ROOT / "runs/reality/disocclusion384"
ABLATIONS = ("release", "no_native_history", "no_display_history", "coarse", "current_ground")


def verify_release():
    for item in json.loads(MANIFEST.read_text())["files"]:
        verify(ROOT / item["path"], item)


def native_head(native, x):
    with torch.autocast("cuda", dtype=torch.float16):
        return native.head(x.contiguous(memory_format=torch.channels_last), native.strength)


@torch.inference_mode()
def prepare(runtime, town, diagnose):
    raw_dir = ORIGINAL / "capture" / f"town{town}"
    feature_dir = ORIGINAL / "features" / f"town{town}"
    capture_sha = file_hash(raw_dir / "capture.json")
    if json.loads((feature_dir / "features.json").read_text())["spec"]["capture"] != capture_sha:
        raise ValueError("Cached features do not match the captured drive")
    contract = dict(town=town, capture=capture_sha, features=file_hash(feature_dir / "features.json"),
                    release=file_hash(MANIFEST), program=file_hash(__file__),
                    context=file_hash(ROOT / "keireality/disocclusion.py"), diagnose=diagnose)
    dest = OUT / "cache" / f"town{town}"
    if dest.exists():
        if json.loads((dest / "metadata.json").read_text())["contract"] != contract:
            raise ValueError(f"Incompatible existing cache for town {town}")
        return
    raw, cached = open_fields(raw_dir), open_fields(feature_dir)
    count = len(raw["pose"])
    warm = runtime.display("stable384")
    cold = runtime.display("stable384") if diagnose else None
    context = ActorContext()
    views = {a: torch.from_numpy(view_pose(camera(a)))[None].cuda() for a in ANGLES}
    directions = {a: rays.camera_rays(camera(a)) for a in ANGLES}
    runtime.agent.reset()
    sums = np.zeros((2, 3, len(ABLATIONS)), np.float64)
    pixels = np.zeros((2, 3), np.int64)
    support_sum = np.zeros_like(pixels, dtype=np.float64)
    previous = [None, None]
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        folder = Path(tmp) / "town"
        folder.mkdir()
        def field(name, shape, dtype):
            return np.lib.format.open_memmap(folder / (name + ".npy"), "w+", dtype, (count, 2, *shape))
        cues = field("context", (*SIZE, 5), np.float16)
        released = field("release", (*SIZE, 3), np.float16)
        exposed = field("vacated", SIZE, np.uint8)
        parity = 0.
        for t, (tm, speed, cmd, tx, ty) in enumerate(raw["nav"]):
            _, plan = runtime.agent.act(list(raw["cameras"][t]), int(cmd), (tx, ty), speed, runtime.dt, with_reality=True)
            shared = plan["reality"]
            f = {key: shared[key].half() for key in ("mem", "c2", "c3")}
            f.update(stem=shared["stem"], source_logits=shared["source_logits"])
            f["estimated_depth"] = runtime.depth.depth_metres(runtime.depth.forward_depth(shared["rgb_in"])).reshape(1, 2, 80, 160)
            pose = raw["pose"][t]
            warm.observe(f, town, tm, relative_pose(raw["pose"][t-1] if t else None, pose))
            context.observe(town, tm)
            if cold is not None:
                # New episode identity at every tick clears all renderer-side temporal buffers.
                cold.observe(f, (town, t), tm, np.zeros(3, np.float32))
            for slot, angle in enumerate((0, sweep(t))):
                view = views[angle]
                _, x = warm.prepare(view, slot)
                ctx = context.features(warm, view, slot)
                out = native_head(warm, x)
                cues[t, slot] = ctx[0].permute(1, 2, 0).half().cpu().numpy()
                released[t, slot] = out[0].permute(1, 2, 0).half().cpu().numpy()
                if t in (0, count // 2, count-1):
                    recorded = torch.from_numpy(cached["inputs"][t, slot].copy()).cuda().permute(2, 0, 1)[None]
                    parity = max(parity, float((x.half() - recorded).abs().float().mean()))
                seg, depth = raw["labels"][t, slot], raw["depth"][t, slot]
                now = (seg, depth, pose)
                mask = exposed[t, slot]
                mask.fill(0)
                if previous[slot] is not None:
                    old, old_cam = previous[slot]
                    _, (idx, old_idx) = correspondence(camera(angle), directions[angle], now, old, old_cam)
                    mask.ravel()[idx] = old[0].ravel()[old_idx]
                previous[slot] = now, camera(angle)
                if cold is None or not mask.any():
                    continue
                without_history = x.clone()
                without_history[:, 13:17] = 0
                values = torch.cat((out, native_head(warm, without_history), cold.render(view, slot),
                                    ctx[:, :3], (x[:, :3] + x[:, 8:11]).clamp(0, 1)))
                truth = torch.from_numpy(raw["rgb"][t, slot].copy()).cuda().permute(2, 0, 1).float()/255
                errors = (values - truth).abs().mean(1).cpu().numpy()
                support = actor_support(ctx)[0, 0].cpu().numpy()
                for i, group in enumerate((mask > 0, mask == SEM["vehicle"], mask == SEM["pedestrian"])):
                    pixels[slot, i] += group.sum()
                    sums[slot, i] += errors[:, group].sum(1, dtype=np.float64)
                    support_sum[slot, i] += support[group].sum(dtype=np.float64)
        if parity > 1e-4:
            raise ValueError(f"Driving features differ from original replay: {parity}")
        for arr in (released, exposed, cues):
            arr.flush()
        write_json(folder / "metadata.json", dict(contract=contract, ticks=count, rgb_input_cache_mae=parity,
                   groups=["all", "previous_vehicle", "previous_pedestrian"], tracks=TRACKS,
                   methods=ABLATIONS, counts=pixels.tolist(), error_sums=sums.tolist(), support_sums=support_sum.tolist()))
        folder.rename(dest)
    print("town", town, "exposed pixels by view", pixels[:, 0].tolist(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "selection", "validation"), required=True)
    stage = parser.parse_args().stage
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    verify_release()
    runtime = Runtime()
    try:
        with writer_lock(OUT / "cache"):
            for town in CFG[stage + "_towns"]:
                prepare(runtime, town, stage == "fit")
    finally:
        runtime.env.close()
