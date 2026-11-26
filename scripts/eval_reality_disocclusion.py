#!/usr/bin/env python3
"""Replay the selected repair and the release on identical observations and archived teacher surfaces."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from collect_reality_sweep import FRESH, split_info, sweep
from diagnose_reality_blur import inputs
from eval_reality_depth_refiner import NAMES as METRICS, gate
from eval_reality_release_gate import OUT as AUDIT, PRIOR, load_release
from eval_reality_scene_detail import SceneScores
from fit_reality_disocclusion import CFG, FIT, MANIFEST, OUT, compare_native, trained
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.disocclusion import DisocclusionDrive
from keireality.geometry import orbit_camera, view_pose
from keireality.refined_display import RefinedDrive
from keireality.resolution import ResolutionDrive
from keireality.temporal_metrics import correspondence
from keireality.video import quad, title_font, write_video
from keisim.config import SEM
from keisim.render import rays
from prepare_reality_projection import checked_fields
from reality_closed_loop import load_background_repair
from score_reality_resolution import Scores, bytes_rgb


METHODS = ("A", "release384", "repair384")
TRACKS = ("chase", "sweep")


def crop_board(path, gt, mask, predictions, crop, tick):
    x, y, w, h = crop
    marked = gt.copy()
    marked[mask] = (255, 222, 0)
    source = ((gt, "現在の正解"), (marked, "黄色　新しく見えた背景"),
              (predictions[1], "公開384版"), (predictions[2], "背景補正版"))
    board = Image.new("RGB", (2*w*4, 32+2*(h*4+24)), (24, 27, 29))
    draw = ImageDraw.Draw(board)
    draw.text((8, 4), f"町20401　固定視点　{tick/10:.1f}秒　前回と同じ採点画素を4倍に拡大", font=title_font())
    for i, (image, label) in enumerate(source):
        left, top = (i%2)*w*4, 32+(i//2)*(h*4+24)
        draw.text((left+8, top), label, font=title_font())
        patch = Image.fromarray(image[y:y+h, x:x+w]).resize((4*w, 4*h), Image.Resampling.NEAREST)
        board.paste(patch, (left, top+24))
    board.save(path)


@torch.inference_mode()
def town_report(data, rows, models, selected, moving_path, archived, dest, focus):
    pilot, depth, renderer, detail, refiner, head, variant, strength = models
    native = ResolutionDrive(RefinedDrive(renderer, detail, refiner, variant), head, strength)
    display = DisocclusionDrive(native, *selected)
    views = {a: torch.from_numpy(view_pose(orbit_camera(a)))[None].cuda() for a in range(-36, 37, 3)}
    directions = {a: rays.camera_rays(orbit_camera(a)) for a in views}
    with np.load(moving_path) as archive:
        moving = {k: archive[k] for k in ("angle", "rgb", "seg", "depth", "seen")}
    totals = [SceneScores(), SceneScores()]
    edges = [Scores(METHODS), Scores(METHODS)]
    previous, errors = [None, None], [None, None]
    subgroups, movies = {}, []
    t = 0
    count = sum(r["count"] for r in rows)
    for row in rows:
        raw = checked_fields(data, row, ("step", "t_sim", "delta_pose", "rgb_in", "rgb_target", "seg_target",
                                         "depth_target", "visibility_now", "ego_pose_world"))
        for j, tm in enumerate(raw["t_sim"]):
            if raw["step"][j] != rows[0]["first"]+t or moving["angle"][t] != sweep(t):
                raise ValueError("Unmatched driving state or sweep angle")
            _, features = inputs(pilot, depth, raw["rgb_in"][j], "cuda", t == 0)
            display.observe(features, row["episode_id"], tm, raw["delta_pose"][j])
            for slot, angle in enumerate((0, sweep(t))):
                camera, view = orbit_camera(angle), views[angle]
                release, repaired = display.paired(view, slot)
                a = native.baseline.control.reference.render(view)[0]
                high = torch.cat((release, repaired))
                low = F.interpolate(high.float(), size=a.shape[-2:], mode="bicubic", align_corners=False,
                                    antialias=True).clamp(0, 1)
                pred = torch.cat((a, low))
                images = pred.permute(0, 2, 3, 1).cpu().numpy()
                gt, seg, z, visible = ((raw["rgb_target"][j, 0], raw["seg_target"][j, 0],
                                         raw["depth_target"][j, 0], raw["visibility_now"][j, 0]) if slot == 0 else
                                        (moving["rgb"][t], moving["seg"][t], moving["depth"][t], moving["seen"][t]))
                now = (seg, z, raw["ego_pose_world"][j])
                warp = None
                age_map = np.full(seg.shape, -1, np.int32)
                vacmask = np.zeros(seg.shape, bool)
                if previous[slot] is not None:
                    old, old_camera = previous[slot]
                    warp = correspondence(camera, directions[angle], now, old, old_camera)
                    (now_idx, old_idx), (vacated, hidden_by) = warp
                    age_map.ravel()[now_idx] = old_idx
                    vacmask.ravel()[vacated] = True
                    if slot == 0:
                        err = np.abs(images.reshape(3, -1, 3)[:, vacated] -
                                     gt.reshape(-1, 3)[vacated].astype(np.float32)/255.).mean(-1)
                        actor = old[0].ravel()[hidden_by]
                        groups = {"after_pedestrian": actor == SEM["pedestrian"], "after_vehicle": actor == SEM["vehicle"]}
                        if t >= count-80:
                            groups["last_80"] = np.ones_like(actor, bool)
                        if focus:
                            first, last = focus["window_ticks"]
                            if first <= t <= last:
                                groups["registered_window"] = np.ones_like(actor, bool)
                            if t == focus["tick"]:
                                x, y, w, h = focus["crop_xywh"]
                                iy, ix = np.divmod(vacated, seg.shape[1])
                                groups["registered_crop"] = (ix>=x) & (ix<x+w) & (iy>=y) & (iy<y+h)
                        for name, keep in groups.items():
                            sums = subgroups.setdefault(name, np.zeros(4, np.float64))
                            sums[:3] += err[:, keep].sum(1, dtype=np.float64)
                            sums[3] += keep.sum()
                previous[slot] = now, camera
                errors[slot] = totals[slot].add(images, gt.astype(np.float32)/255., seg, visible==1, .10, warp, errors[slot])
                byte = bytes_rgb(pred)
                edges[slot].frame(byte, gt, seg, visible==1, age_map)
                if focus and slot == 0:
                    before, after = bytes_rgb(high)
                    marked = gt.copy()
                    marked[vacmask] = (255, 222, 0)
                    movies.append(quad(((gt, f"町20401　正解　{t/10:.1f}秒"), (marked, "黄色　直前まで隠れていた背景"),
                                        (before, "公開384版"), (after, "背景補正版"))))
                    if t == focus["tick"]:
                        crop_board(dest/"town20401-crop.png", gt, vacmask, byte, focus["crop_xywh"], t)
            t += 1
    if t != count or (focus and write_video(dest/"town20401-chase.mp4", movies) != count):
        raise ValueError("Incomplete matched replay")
    tracks, white, guards = {}, {}, {}
    for slot, key in enumerate(TRACKS):
        result = totals[slot].report(METHODS)
        old = archived["tracks"][key]
        if result["counts"] != old["counts"]:
            raise ValueError("Scoring pixels differ from the published-drive audit")
        for method in METHODS[:2]:
            for k in METRICS:
                a, b = result[method][k], old[method][k]
                if (a is None) != (b is None) or (a is not None and abs(a-b)>1e-5):
                    raise ValueError(f"Historical control mismatch on {key}/{method}/{k}: {a}, {b}")
        result["D"] = old["D"]
        result["classes"]["marking"]["edge"]["D"] = old["classes"]["marking"]["edge"]["D"]
        tracks[key] = result
        row = edges[slot].results()
        ref, new = (row["methods"][m] for m in METHODS[1:])
        original = archived["white"][key]
        if abs(ref["white_line_contrast"] - original["values"]["release384"]) > 1e-5:
            raise ValueError("White line reference differs")
        white[key] = dict(pixels=row["counts"]["white_line_contrast"], historical_D=original["values"]["D"],
                          release=ref["white_line_contrast"], repaired=new["white_line_contrast"])
        guards[key] = dict(vehicle=new["vehicle_contrast"]/ref["vehicle_contrast"]-1,
                           white=new["white_line_contrast"]/ref["white_line_contrast"]-1,
                           flicker=new["static_flicker"]/ref["static_flicker"]-1)
    original_gate = gate(tracks, "repair384")
    outline = all(v["repaired"] >= v["historical_D"]-1e-5 for v in white.values())
    bound = CFG["relative_guard"]
    maintained = all(r["vehicle"] >= -bound and r["white"] >= -bound and r["flicker"] <= bound for r in guards.values())
    return dict(ticks=t, original_gate=original_gate, white_vs_D=outline, within_release_guard=maintained,
                status="PASS" if original_gate["status"]=="PASS" and outline and maintained else "HOLD",
                tracks=tracks, white=white, relative_to_release=guards,
                exposed={k: dict(pixels=int(v[3]), l1={m: float(v[i]/v[3]) if v[3] else None
                          for i, m in enumerate(METHODS)}) for k,v in subgroups.items()})


def evaluate():
    selected_path = FIT/"selection.json"
    selection = json.loads(selected_path.read_text())
    models, repair = load_release(), load_background_repair()
    focus = json.loads((AUDIT/"fresh/diagnostics/town20401.json").read_text())
    dest = OUT/"replay"
    results = {}
    with writer_lock(dest):
        for stage in ("val", "fresh"):
            data, split, towns = (split_info("val")[:3] if stage == "val" else
                                  (FRESH, "test", [e["town_seed"] for e in json.loads((AUDIT/"fresh/episodes.json").read_text())]))
            manifest = load_manifest(data)
            for town in towns:
                rows = [r for r in manifest["shards"] if r["town_seed"] == town and r["split"] == split]
                prior = AUDIT/stage/f"town{town}.json"
                archived = json.loads(prior.read_text())
                teacher = PRIOR/f"val/sweep_truth/town{town}.npz" if stage=="val" else AUDIT/f"fresh/town{town}-teacher.npz"
                if (file_hash(teacher) != archived["teacher_sha256"] or sum(r["count"] for r in rows) != archived["ticks"]
                        or (town == 20401 and file_hash(prior) != focus["quality"])):
                    raise ValueError("Archived case and teacher do not match")
                result = town_report(data, rows, models, repair, teacher, archived, dest, focus if town == 20401 else None)
                result.update(selection=file_hash(selected_path), archive=file_hash(prior), collection=manifest["contract_sha256"])
                path = dest/f"town{town}.json"
                write_json(path, result)
                results[town] = dict(status=result["status"], sha256=file_hash(path))
                r = result["tracks"]["chase"]
                print(town, result["status"], "vacated", r["release384"]["vacated_l1"], r["repair384"]["vacated_l1"],
                      "guards", result["relative_to_release"], flush=True)
        write_json(dest/"summary.json", dict(selection=selection, selection_sha256=file_hash(selected_path),
                   release=file_hash(MANIFEST), evaluator=file_hash(__file__), towns=results))


def native_validation():
    selection = json.loads((FIT/"selection.json").read_text())
    dest = OUT / "native"
    with writer_lock(dest):
        for town in (40, 41):
            result = compare_native(town, [(selection["method"], selection["strength"])], trained())
            write_json(dest/f"town{town}.json", dict(town=town, selection=file_hash(FIT/"selection.json"), tracks=result))
            print("native", town, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if args.native:
        native_validation()
    else:
        evaluate()
