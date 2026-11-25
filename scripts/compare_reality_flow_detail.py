#!/usr/bin/env python3
"""Generate causal detail candidates, then measure grain on flat surfaces and contrast on real edges."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, load_field, write_json, writer_lock
from keireality.flow_data import FlowSplit
from keireality.flow_detail import detail_sums, native_display, subtract_source_component
from keireality.flow_model import FlowRollout, frame_noise
from keireality.geometry import ORBIT_ANGLES, orbit_camera
from keireality.temporal_metrics import correspondence
from keireality.video import quad, write_video
from keisim.config import SEM
from keisim.render import rays
from rollout_reality_flow import load_flow


def read(path):
    return json.loads(Path(path).read_text())


def predict_episode(ds, ep, fields, seed, device):
    shape = (ep.length, len(ds.angles), 3, 64, 128)
    images = {}
    streams = {}
    for name, field in fields.items():
        for n in (1, 4):
            key = name + str(n)
            streams[key] = FlowRollout(field, n)
            images[key] = np.empty(shape, np.float16)
    for offset in range(ep.length):
        i = ep.start + offset
        current = ds.current[i].to(device)
        pose = ds.poses[i].to(device)
        delta = ds.delta[i].to(device).expand(len(pose), -1)
        noise = frame_noise(ep.seed, ds.steps[i], ds.angles, seed, device)
        for key, stream in streams.items():
            pred = stream.render(current, pose, delta, ep.name, ds.times[i], noise)
            if stream.used_previous != (offset > 0) or not torch.isfinite(pred).all():
                raise ValueError("discontinuous or non-finite autoregressive generation")
            images[key][offset] = pred.half().cpu().numpy()
    return images


def load_original_rollout(root, run, ep, views):
    loc = root / ep.name
    record = read(loc / "episode.json")
    if record != run["episodes"][ep.name] or record["ticks"] != ep.length:
        raise ValueError("original rollout identity or length differs")
    for k in ("nfe1", "nfe4", "views"):
        if file_hash(loc / (k + ".npy")) != record["arrays"][k]:
            raise ValueError("original rollout checksum differs")
    angles = np.load(loc / "views.npy", mmap_mode="r")[:, :views]
    if not np.all(angles == np.arange(views)):
        raise ValueError("original predictions are ordered differently")
    return {"old"+str(n):np.load(loc / f"nfe{n}.npy", mmap_mode="r")[:, :views] for n in (1, 4)}


class Scores:
    def __init__(self, names, device):
        self.names = list(names)
        self.total = torch.zeros(len(names), 3, dtype=torch.float64, device=device)
        self.denom = torch.zeros(3, dtype=torch.float64, device=device)

    def add(self, preds, truth, cfg):
        num, den = detail_sums(torch.stack([preds[k] for k in self.names]), truth,
                               cfg["flat_threshold_rgb01"], cfg["edge_threshold_rgb01"])
        self.total += num
        self.denom += den

    def report(self):
        names = ("l1", "flat_highpass_mae", "aligned_edge_contrast")
        den = self.denom.tolist()
        return {k:{m:float(v)/d if d else None for m, v, d in zip(names, row, den, strict=True)}
                for k, row in zip(self.names, self.total.tolist(), strict=True)}


@torch.inference_mode()
def compare(args):
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    cfg, base = read(args.config), read(args.base_config)
    split = args.split
    ds = FlowSplit(args.data, args.cache, split, teachers=False)
    fine, fine_contract = load_flow(args.candidate, args.device)
    orig_sha = file_hash(args.original)
    if (fine_contract["init"] != orig_sha or fine_contract["renderer"] != ds.meta["renderer"] or
            fine_contract["protocol"] != file_hash(args.config) or fine_contract["original_protocol"] != file_hash(args.base_config)):
        raise ValueError("candidate and experiment contracts differ")
    kernel = None
    if args.seed_kernel:
        response = read(args.seed_kernel)
        if response["model"] != file_hash(args.candidate) or response["training_cache"] != fine_contract["train"]:
            raise ValueError("noise-response kernel was not fitted on this model and training split")
        kernel = torch.tensor(response["kernel"], device=args.device)
        if kernel.shape != (3, 3, 5, 5) or not torch.isfinite(kernel).all():
            raise ValueError("invalid noise-response kernel")
    seed = base["validation_seed"] if split == "val" else base["evaluation_seed"]
    fields = {"new":fine}
    native = None
    if split == "test":
        run = read(args.original_rollouts / "manifest.json")
        if (run["contract"]["model"] != orig_sha or run["contract"]["cache"] != file_hash(ds.cache / "manifest.json")
                or run["contract"]["angles"] != ds.angles or run["contract"]["evaluation_seed"] != seed):
            raise ValueError("original rollout inputs differ from this comparison")
        native = np.load(ds.cache / "native.npy", mmap_mode="r")
    else:
        fields["old"] = load_flow(args.original, args.device)[0]
    expected = base["eval_towns"] if split == "test" else [40, 41]
    if [ep.town for ep in ds.episodes] != expected:
        raise ValueError("unexpected evaluation towns")
    cameras = [orbit_camera(a) for a in ds.angles]
    directions = [rays.camera_rays(cam) for cam in cameras]
    vi = [ORBIT_ANGLES.index(a) for a in ds.angles]
    methods = ["phase2", "old1", "old4", "new1", "new4"]
    if kernel is not None:
        methods.append("clean4")
    displays = list(methods) + (["phase2_native", "new1_native", "new4_native"] if native is not None else [])
    if kernel is not None and native is not None:
        displays.append("clean4_native")
    display_stats = [*displays, "teacher_down_up"]
    joined_low, joined_high = Scores(methods, args.device), Scores(display_stats, args.device)
    total_flicker = np.zeros(len(displays), np.float64)
    total_n = 0
    records, movie = {}, []
    crop = None
    for ep in ds.episodes:
        # Only inference inputs are read until all candidate images in this episode have been generated.
        bank = predict_episode(ds, ep, fields, seed, args.device)
        if split == "test":
            bank.update(load_original_rollout(args.original_rollouts, run, ep, len(vi)))
        low_scores, high_scores = Scores(methods, args.device), Scores(display_stats, args.device)
        flicker, n = np.zeros(len(displays), np.float64), 0
        old_geometry = old_errors = None
        for row, base_i in zip(ds.rows, ds.offsets, strict=True):
            if row["episode_id"] != ep.name:
                continue
            keys = ("rgb_target", "depth_target", "seg_target", "ego_pose_world")
            for key in keys:
                if file_hash(args.data / row["path"] / (key + ".npy")) != row["fields"][key]["sha256"]:
                    raise ValueError(f"corrupt scoring field {key}")
            rgb, depth, seg, pose = [load_field(args.data, row, key) for key in keys]
            for j in range(row["count"]):
                i = base_i + j
                off = i - ep.start
                truth = torch.from_numpy(rgb[j, vi].copy()).to(args.device).permute(0, 3, 1, 2).float() / 255
                target = F.avg_pool2d(truth, 2)
                low = {"phase2":(ds.current[i, :, :3].to(args.device).float() + 1) * .5}
                for key in bank:
                    low[key] = (torch.from_numpy(np.array(bank[key][off], copy=True)).to(args.device).float() + 1) * .5
                if kernel is not None:
                    noise = frame_noise(ep.seed, ds.steps[i], ds.angles, seed, args.device)
                    low["clean4"] = subtract_source_component(low["new4"], noise, kernel)
                high = {k:F.interpolate(v, size=truth.shape[-2:], mode="bilinear", align_corners=False) for k,v in low.items()}
                if native is not None:
                    p2 = torch.from_numpy(native[i].copy()).to(args.device).float()
                    high.update(phase2_native=p2, new1_native=native_display(low["new1"], p2),
                                new4_native=native_display(low["new4"], p2))
                    if kernel is not None:
                        high["clean4_native"] = native_display(low["clean4"], p2)
                high["teacher_down_up"] = F.interpolate(target, size=truth.shape[-2:], mode="bilinear", align_corners=False)
                low_scores.add(low, target, cfg)
                high_scores.add(high, truth, cfg)
                errors = (torch.stack([high[k] for k in displays]) - truth).permute(1, 0, 3, 4, 2).contiguous()
                errors = errors.cpu().numpy().reshape(len(vi), len(displays), -1, 3)
                geometry = [(seg[j,v], depth[j,v], pose[j]) for v in vi]
                if old_geometry is not None:
                    for v, cam in enumerate(cameras):
                        (a,b), _ = correspondence(cam, directions[v], geometry[v], old_geometry[v])
                        difference = np.take(errors[v], a, axis=1) - np.take(old_errors[v], b, axis=1)
                        flicker += np.abs(difference).sum(axis=(1, 2), dtype=np.float64)
                        n += 3 * len(a)
                old_geometry, old_errors = geometry, errors
                if ep.town == cfg["video_town"] and native is not None:
                    bottom_right = "new4_native" if kernel is None else "clean4_native"
                    items = (truth[0], high["old4"][0], high["new4"][0], high[bottom_right][0])
                    movie.append((torch.stack(items).permute(0, 2, 3, 1).clamp(0, 1)*255).round().byte().cpu().numpy())
                    if off == cfg["video_offset"]:
                        mask = (seg[j, vi[0]] == SEM["vehicle"]).astype(np.uint8)
                        count, _, boxes, centers = cv2.connectedComponentsWithStats(mask)
                        if count > 1:
                            largest = 1 + boxes[1:, cv2.CC_STAT_AREA].argmax()
                            cx, cy = centers[largest]
                            x = int(np.clip(round(cx) - 48, 0, mask.shape[1] - 96))
                            y = int(np.clip(round(cy) - 24, 0, mask.shape[0] - 48))
                            crop = (x, y, 96, 48)
        for joined, part in ((joined_low, low_scores), (joined_high, high_scores)):
            joined.total += part.total
            joined.denom += part.denom
        total_flicker += flicker
        total_n += n
        records[ep.name] = dict(ticks=ep.length, transitions=ep.length-1, low=low_scores.report(), native=high_scores.report(),
                                flicker=dict(zip(displays, (flicker/n).tolist(), strict=True)))
        print(split, ep.name, "complete", flush=True)
    evidence = dict(split=split, candidate=file_hash(args.candidate), original=orig_sha, seed=seed,
                    protocol=file_hash(args.config), cache=file_hash(ds.cache / "manifest.json"), angles=ds.angles,
                    images=len(ds.current)*len(ds.angles), low=joined_low.report(), native=joined_high.report(),
                    flicker=dict(zip(displays, (total_flicker/total_n).tolist(), strict=True)),
                    reprojected_static_pixels=total_n//3, episodes=records)
    if args.seed_kernel:
        evidence["seed_kernel_sha256"] = file_hash(args.seed_kernel)
    if movie:
        path = args.out / f"town{cfg['video_town']}.mp4"
        def frames():
            for off, panels in enumerate(movie):
                right = "追加学習後＋Phase 2 の細部" if kernel is None else "乱数成分を除去＋Phase 2 の細部"
                titles = (f"教師　町{cfg['video_town']}　{off/10:.1f} 秒", "元の 4 回生成", "追加学習後の 4 回生成", right)
                frame = quad(zip(panels, titles, strict=True))
                if off == cfg["video_offset"]:
                    Image.fromarray(frame).save(args.out / "comparison.png")
                    if crop is not None:
                        x, y, width, height = crop
                        tiles = panels[:, y:y + height, x:x + width]
                        Image.fromarray(quad(zip(tiles, titles, strict=True))).save(args.out / "crop.png")
                yield frame
        count = write_video(path, frames(), 10)
        info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames,r_frame_rate,duration", "-of", "json", str(path)]))["streams"][0]
        if count != base["frames_per_town"] or int(info["nb_read_frames"]) != count or float(info["duration"]) != count/10:
            raise ValueError("comparison video did not contain the full drive")
        evidence["video"] = dict(**info, file=path.name, sha256=file_hash(path),
                                 sample_offset=cfg["video_offset"], crop_xywh=crop)
    write_json(args.out / (split + ".json"), evidence)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("val", "test"), required=True)
    for k in ("base-config", "config", "data", "cache", "original", "candidate", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--original-rollouts", type=Path)
    p.add_argument("--seed-kernel", type=Path, help="optional trained output response to this frame's original noise")
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = p.parse_args()
    if args.split == "test" and args.original_rollouts is None:
        p.error("--original-rollouts is required for the previously evaluated long drives")
    with writer_lock(args.out):
        compare(args)
