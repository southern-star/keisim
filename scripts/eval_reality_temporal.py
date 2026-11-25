#!/usr/bin/env python3
"""Paired held-out evaluation of spatial and temporal renderers, including falsifiable history/view controls."""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, load_field, write_json, writer_lock
from keireality.geometry import ORBIT_ANGLES, orbit_camera
from keireality.multiview import MultiViewFrames
from keireality.renderer import SingleFrameRenderer
from keireality.spatial import SpatialRenderer
from keireality.temporal_metrics import Totals, correspondence, pixel_scores
from keireality.video import quad, write_video
from keisim.render import rays
from train_reality import on_device, predict


def load_renderer(path, pilot_hash, device):
    ck = torch.load(path, map_location="cpu")
    cfg = ck["config"]
    if cfg["pilot_sha256"] != pilot_hash:
        raise ValueError("checkpoint and input encoder differ")
    arch = cfg.get("arch", "mem")
    model = (SingleFrameRenderer() if arch == "mem" else
             SpatialRenderer(bool(cfg.get("labels_manifest_sha256")), temporal=arch == "temporal"))
    model.load_state_dict(ck["model"])
    return model.eval().to(device=device, memory_format=torch.channels_last)


def as_uint8(v):
    return np.rint(np.clip(v, 0, 1)*255).astype(np.uint8)


def variant(model, batch, mode):
    b = batch
    if mode == "spatial_zero":
        b = dict(b, c2=torch.zeros_like(b["c2"]), c3=torch.zeros_like(b["c3"]))
    elif mode == "no_history":
        b = dict(b, valid=torch.zeros_like(b["valid"]))
    elif mode == "no_motion":
        motion = b["motion"].clone()
        motion[..., :3], motion[..., 3] = 0, 1
        b = dict(b, motion=motion)
    elif mode == "swapped_pose":
        view = b["view_pose"].clone()
        view[:, [1, 4, 7]] *= -1
        b = dict(b, view_pose=view)
    rgb, logits, support = predict(model, b)
    return (rgb.permute(0, 2, 3, 1).cpu().numpy(), logits.argmax(1).byte().cpu().numpy(),
            support.softmax(1).permute(0, 2, 3, 1).cpu().numpy() if support is not None else None)


@torch.inference_mode()
def evaluation(args):
    torch.set_num_threads(2)
    temporal = args.temporal is not None
    views = tuple(range(5)) if temporal else (0,)
    ds = MultiViewFrames(args.data, args.features, "test", views=views, labels=args.labels, temporal=temporal)
    paths = {"p1": args.p1, "spatial": args.spatial}
    if temporal:
        paths.update(single=args.single, temporal=args.temporal)
    models = {k: load_renderer(p, ds.cache["checkpoint_sha256"], args.device) for k, p in paths.items()}
    cfg = json.loads(args.protocol.read_text())
    gate = cfg["gate"]
    clips = json.loads(args.clips.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    totals = defaultdict(Totals)
    resets, reset_diff = 0, 0.
    intervals = {}
    tick = time.perf_counter()

    @lru_cache(maxsize=8)
    def fields(j):
        names = ["step", "t_sim", "junction", "ego_speed"]
        names += ["depth_target", "ego_pose_world"] if temporal else ["rgb_in"]
        return {k: load_field(ds.root, ds.rows[j], k) for k in names}

    def score_view(view, angle):
        nonlocal resets, reset_diff
        indices = range(view, len(ds), len(views))
        cam = orbit_camera(angle)
        directions = rays.camera_rays(cam)
        previous, old_geometry, old_episode, old_time = {}, None, None, None
        last_stop = None
        modes = {"p1": "p1", "spatial": "spatial", "spatial_zero": "spatial"}
        if temporal:
            modes = {"single": "single", "temporal": "temporal"}
            if angle == 0:
                modes.update(p1="p1", spatial="spatial", no_history="temporal", no_motion="temporal")
            if abs(angle) == 18:
                modes["swapped_pose"] = "temporal"
        for batch in DataLoader(Subset(ds, indices), batch_size=args.bs):
            b = on_device(batch, args.device)
            preds = {mode: variant(models[model], b, mode) for mode, model in modes.items()}
            for n, index in enumerate(batch["index"].tolist()):
                j, i, _ = ds.frame_view(index)
                episode = ds.rows[j]["episode_id"]
                a = fields(j)
                step, t = int(a["step"][i]), float(a["t_sim"][i])
                reset = episode != old_episode or old_time is None or abs(t-old_time-.1) > 1e-6
                if reset:
                    previous, old_geometry, last_stop = {}, None, None
                if a["ego_speed"][i] < .4:
                    last_stop = t
                truth = batch["rgb"][n].permute(1, 2, 0).numpy()
                seg = batch["semantic"][n].numpy()
                vis = batch["visibility"][n].numpy()
                support = batch["support"][n].numpy() if temporal else None
                geometry = (seg, a["depth_target"][i, view], a["ego_pose_world"][i]) if temporal else None
                warp = correspondence(cam, directions, geometry, old_geometry) if old_geometry is not None and angle == 0 else None
                groups = ["all", episode]
                if angle == 0 and temporal:
                    if reset:
                        groups.append("reset")
                        resets += 1
                        reset_diff = max(reset_diff, float(np.abs(preds["temporal"][0][n] - preds["no_history"][0][n]).max()))
                    if a["junction"][i]:
                        groups.append("junction")
                    if last_stop is not None and .0 < t-last_stop <= 3 and a["ego_speed"][i] > 1.:
                        groups.append("restart")
                for mode, (rgb, predicted, prob) in preds.items():
                    stats = pixel_scores(rgb[n], predicted[n], truth, seg, support, vis,
                                         prob[n] if prob is not None else None, warp, previous.get(mode))
                    for group in groups:
                        totals[angle, mode, group].add(stats)
                    previous[mode] = (rgb[n], truth)
                old_geometry, old_time, old_episode = geometry, t, episode
                if temporal:
                    intervals.setdefault(episode, [step, step])[1] = step
                    p = preds["temporal"][2][n, ..., 1:3].sum(-1)
                    img = quad([(as_uint8(truth), f"教師画像　{episode}　時点 {step}　視点 {angle:+d}°"),
                                (as_uint8(preds["single"][0][n]), "現在の1時点から生成"),
                                (as_uint8(preds["temporal"][0][n]), "直近8時点から生成　0.7秒の履歴"),
                                (as_uint8(np.repeat(p[..., None], 3, axis=-1)), "推定した観測支持度　白ほど高い")])
                    sample_steps = {"t70-e8770": 212, "t71-e8771": 224, "t72-e8772": 203, "t73-e8773": 290}
                    if step == sample_steps.get(episode):
                        cv2.imwrite(str(args.out / f"sample_{episode}_{angle:+d}.png"), img[..., ::-1])
                    yield img
                else:
                    clip = next((c for c in clips if c["episode"] == episode and c["first"] <= step < c["first"] + c["frames"]), None)
                    if clip:
                        img = quad([(a["rgb_in"][i, 0], f'広角入力　{episode}　時点 {step}'),
                                    (as_uint8(truth), "KeiView の教師画像　固定追跡視点"),
                                    (as_uint8(preds["p1"][0][n]), "Phase 1　運転トークンから生成"),
                                    (as_uint8(preds["spatial"][0][n]), "高解像度特徴 c2・c3 と運転トークンから生成")])
                        if step == clip["first"] + clip["frames"] // 2:
                            cv2.imwrite(str(args.out / (clip["name"] + ".png")), img[..., ::-1])
                        yield img

    for view in views:
        angle = ORBIT_ANGLES[view]
        path = args.out / (f"orbit_{angle:+d}.mp4" if temporal else "compare.mp4")
        count = write_video(path, score_view(view, angle))
        print(f'{path}: {count} comparison frames', flush=True)
    def metric(angle, mode, key):
        return totals[angle, mode, "all"].report()["metrics"].get(key)
    def compare(angle, key, a, b, ratio=1., offset=0., strict=False):
        x, y = metric(angle, a, key), metric(angle, b, key)
        if x is None or y is None:
            return None
        return bool(x < ratio*y + offset if strict else x <= ratio*y + offset)

    if temporal:
        criteria = dict(
            past_vs_single=compare(0, "past_static_l1", "temporal", "single", gate["chase_past_static_l1_ratio_max_vs_single"]),
            past_vs_original=compare(0, "past_static_l1", "temporal", "p1", strict=True),
            past_vs_removed=compare(0, "past_static_l1", "temporal", "no_history", strict=True),
            warped_residual=compare(0, "warped_residual_l1", "temporal", "single", gate["chase_warped_temporal_l1_ratio_max_vs_single"]),
            vacated_rgb=compare(0, "vacated_rgb_l1", "temporal", "single", offset=gate["chase_vacated_rgb_l1_increase_max"]),
            phantom=compare(0, "phantom_fpr", "temporal", "single", offset=gate["chase_phantom_fpr_increase_max"]),
            interpolated_left=compare(-18, "all_l1", "temporal", "swapped_pose", strict=True),
            interpolated_right=compare(18, "all_l1", "temporal", "swapped_pose", strict=True),
            reset=resets > 0 and reset_diff == 0,
            restart_coverage=totals[0, "temporal", "restart"].frames > 0,
            junction_coverage=totals[0, "temporal", "junction"].frames > 0)
    else:
        criteria = dict(vehicle_rgb=compare(0, "visible_vehicle_l1", "spatial", "p1", gate["spatial_visible_vehicle_l1_ratio_max"]),
                        visible_rgb=compare(0, "visible_l1", "spatial", "p1", gate["spatial_visible_l1_ratio_max"]),
                        road=compare(0, "visible_road_iou", "p1", "spatial", offset=gate["spatial_road_iou_drop_max"]))
    scores = {}
    for (angle, mode, group), value in totals.items():
        scores.setdefault(str(angle), {}).setdefault(mode, {})[group] = value.report()
    result = dict(status="PASS" if all(v is True for v in criteria.values()) else "FAIL", criteria=criteria,
                  seconds=time.perf_counter()-tick, data_sha256=ds.collection["contract_sha256"],
                  protocol_sha256=file_hash(args.protocol), checkpoint_sha256={k: file_hash(p) for k,p in paths.items()},
                  test_frame_counts={r["episode_id"]: ds.collection["episodes"][r["episode_id"]]["count"] for r in ds.rows},
                  intervals=intervals, reset_count=resets, max_reset_difference=reset_diff, scores=scores)
    write_json(args.out / "metrics.json", result)
    (args.out / "protocol.json").write_bytes(args.protocol.read_bytes())
    print(json.dumps(dict(status=result["status"], criteria=criteria)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "features", "p1", "spatial", "out"):
        p.add_argument("--" + arg, required=True, type=Path)
    for arg in ("labels", "single", "temporal"):
        p.add_argument("--" + arg, type=Path)
    p.add_argument("--protocol", type=Path, default=Path("configs/reality/p2_pilot.json"))
    p.add_argument("--clips", type=Path, default=Path("configs/reality/p1_clips.json"))
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = p.parse_args()
    if any((args.temporal, args.single, args.labels)) and not all((args.temporal, args.single, args.labels)):
        p.error("temporal evaluation requires --temporal, --single and --labels together")
    with writer_lock(args.out):
        evaluation(args)
