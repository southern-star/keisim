#!/usr/bin/env python3
"""Run the paired live Phase 2 paths causally, then score their saved frames using synchronized teachers."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import time

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.depth_projection import DepthProjectionBranch
from keireality.encoder import extract_mem, frozen_pilot, state_hash
from keireality.geometry import ORBIT_ANGLES, orbit_camera, view_pose
from keireality.stream import TemporalDisplay
from keireality.temporal_metrics import correspondence
from keireality.video import quad, write_video
from keisim.config import SEM
from keisim.render import rays
from eval_reality_temporal import load_renderer
from prepare_reality_projection import checked_fields


NAMES = ("silhouette", "vehicle_edges", "vehicle_l1", "all_l1", "static_flicker", "vacated_l1")


class Scores:
    def __init__(self):
        self.sums = np.zeros((2, len(NAMES)), np.float64)
        self.denom = np.zeros(len(NAMES), np.float64)
        self.counts = np.zeros(len(NAMES), np.int64)

    def edges(self, rgb, truth, vehicle, seen, threshold):
        for a, b in ((np.s_[:-1,:],np.s_[1:,:]), (np.s_[:,:-1],np.s_[:,1:])):
            delta = truth[b] - truth[a]
            strong = seen[a] & seen[b] & (np.abs(delta).mean(-1) >= threshold)
            for k, mask in enumerate((strong & (vehicle[a] ^ vehicle[b]), strong & (vehicle[a] | vehicle[b]))):
                g = delta[mask]
                self.counts[k] += len(g)
                self.denom[k] += (g*g).sum(dtype=np.float64)
                change = rgb[(slice(None),*b)][:,mask] - rgb[(slice(None),*a)][:,mask]
                self.sums[:,k] += (change*g).sum(axis=(1,2), dtype=np.float64)

    def error(self, k, values):
        array = values.reshape(2,-1,3)
        self.sums[:,k] += np.abs(array).sum(axis=(1,2), dtype=np.float64)
        self.denom[k] += array.shape[1]*3
        self.counts[k] += array.shape[1]

    def merge(self, other):
        self.sums += other.sums
        self.denom += other.denom
        self.counts += other.counts

    def report(self):
        return dict(counts=dict(zip(NAMES, self.counts.tolist())),
                    **{key: {name: float(self.sums[m,k]/self.denom[k]) if self.denom[k] else None
                             for k,name in enumerate(NAMES)} for m,key in enumerate(("A","B"))})


def decision(report):
    a, b = report["A"], report["B"]
    required = ("silhouette","static_flicker","vacated_l1")
    if any(a[k] is None or b[k] is None for k in required):
        return dict(status="INCONCLUSIVE")
    checks = dict(silhouette=b["silhouette"] > a["silhouette"],
                  static_flicker=b["static_flicker"] <= a["static_flicker"],
                  vacated_l1=b["vacated_l1"] <= a["vacated_l1"])
    return dict(status="PASS" if all(checks.values()) else "FAIL", checks=checks)


def bytes_rgb(a):
    return np.rint(a.clip(0,1)*255).astype(np.uint8)


@torch.inference_mode()
def rollout(data, rows, pilot, head, base, branch, cameras, device, output, expected):
    display = TemporalDisplay(base, branch)
    views = [torch.tensor(view_pose(c), device=device)[None] for c in cameras]
    frames = np.lib.format.open_memmap(output, "w+", np.float32, (expected,len(views),2,128,256,3))
    sync = torch.cuda.synchronize if device == "cuda" else lambda: None
    n, prev_step, prev_t, timings = 0, None, None, []
    for row in rows:
        inputs = checked_fields(data, row, ("rgb_in","t_sim","step","delta_pose"))
        for i, t in enumerate(inputs["t_sim"]):
            step = int(inputs["step"][i])
            if n and (step != prev_step+1 or not np.isclose(t-prev_t,.1,rtol=0,atol=1e-6)):
                raise ValueError("nonconsecutive recorded control ticks")
            prev_step, prev_t = step, t
            sync()
            start = time.perf_counter()
            rgb = torch.from_numpy(inputs["rgb_in"][i].copy()).permute(0,3,1,2)[None].to(device)
            feats = {k:v.half() for k,v in extract_mem(pilot,rgb,pyramid=True,parity=n==0).items()}
            feats["estimated_depth"] = head.depth_metres(head.forward_depth(rgb)).reshape(1,2,80,160)
            display.observe(feats, row["episode_id"], t, inputs["delta_pose"][i])
            if display.valid.sum().item() != min(n,7):
                raise ValueError("streaming history did not advance exactly once per tick")
            for v, view in enumerate(views):
                b, _, _, a = display.render(view, return_reference=True)
                frames[n,v] = torch.cat((a,b)).permute(0,2,3,1).cpu().numpy()
                if v == 0 and n >= 8:
                    sync()
                    timings.append(1000*(time.perf_counter()-start))
            n += 1
    if n != expected or not np.isfinite(frames).all():
        raise ValueError("incomplete or nonfinite rollout")
    frames.flush()
    return frames, timings


def score_town(data, rows, cameras, output, out, threshold, pooled):
    bank = [Scores() for _ in cameras]
    quarters = [Scores() for _ in range(4)]
    dirs = [rays.camera_rays(c) for c in cameras]
    previous = [None] * len(cameras)
    old_errors = [None] * len(cameras)
    town = rows[0]["town_seed"]
    i, movie, best, biggest = 0, [], None, -1
    for row in rows:
        fields = checked_fields(data, row, ("rgb_target","seg_target","depth_target","visibility_now",
                                            "ego_pose_world","rgb_in","step","view_pose"))
        for j, step in enumerate(fields["step"]):
            for v, camera in enumerate(cameras):
                if not np.allclose(fields["view_pose"][j,v], view_pose(camera), rtol=0, atol=1e-6):
                    raise ValueError("scoring camera and generated view differ")
                gt = fields["rgb_target"][j,v].astype(np.float32)/255
                seg = fields["seg_target"][j,v]
                visible = fields["visibility_now"][j,v] == 1
                vehicle = seg == SEM["vehicle"]
                rgb = output[i,v]
                err = rgb - gt
                frame = Scores()
                frame.edges(rgb, gt, vehicle, visible, threshold)
                frame.error(2, err[:,vehicle & visible])
                frame.error(3, err)
                geom = (seg, fields["depth_target"][j,v], fields["ego_pose_world"][j])
                if previous[v] is not None:
                    (now, old), (vacated, _) = correspondence(camera, dirs[v], geom, previous[v])
                    flat = err.reshape(2,-1,3)
                    frame.error(4, flat[:,now] - old_errors[v][:,old])
                    frame.error(5, flat[:,vacated])
                previous[v], old_errors[v] = geom, err.reshape(2,-1,3)
                bank[v].merge(frame)
                if v == 0:
                    quarters[min(3,4*i//len(output))].merge(frame)
                    teacher = fields["rgb_target"][j,0]
                    wide = fields["rgb_in"][j,0]
                    aa, bb = bytes_rgb(rgb)
                    movie.append((wide.copy(),teacher.copy(),aa,bb))
                    _, components, boxes, centers = cv2.connectedComponentsWithStats(vehicle.astype(np.uint8))
                    areas = np.bincount(components[vehicle & visible], minlength=len(boxes))
                    if len(areas)>1 and areas[1:].max() > biggest:
                        component = 1+areas[1:].argmax()
                        biggest = int(areas[component])
                        cx, cy = centers[component]
                        width = min(256, max(96, 2*int(boxes[component,3]+8), int(boxes[component,2]+12)))
                        width += width % 2
                        height = width//2
                        x,y = int(np.clip(round(cx)-width//2,0,256-width)),int(np.clip(round(cy)-height//2,0,128-height))
                        crop = np.s_[y:y+height,x:x+width]
                        best = dict(tick=i, step=int(step), confirmed=biggest, crop=[x,y,width,height])
                        Image.fromarray(quad([(teacher[crop],f"追跡視点の正解　町{town}　{i/10:.1f}秒"),
                                              (wide,"同時刻の広角入力"), (aa[crop],"A　現行の Phase 2"),
                                              (bb[crop],"B　推定深度の投影を追加")])).save(out/f"town{town}.png")
            i += 1
    if i != len(output):
        raise ValueError("teacher and rollout lengths differ")
    for totals, item in zip(pooled, bank, strict=True):
        totals.merge(item)
    def panels():
        for off,(wide,gt,a,b) in enumerate(movie):
            yield quad([(wide,f"広角入力　町{town}　{off/10:.1f}秒"), (gt,"追跡視点の正解"),
                        (a,"A　現行の Phase 2"), (b,"B　推定深度の投影を追加")])
    if write_video(out/f"town{town}.mp4", panels(), 10) != i:
        raise ValueError("incomplete A/B movie")
    reports = {str(a):score.report() for a,score in zip(ORBIT_ANGLES,bank,strict=True)}
    return dict(ticks=i, transitions=i-1, views=reports, chase_quarters=[q.report() for q in quarters],
                chase_check=decision(reports["0"]), crop=best)


def evaluate(a):
    cfg = json.loads(a.config.read_text())
    raw = load_manifest(a.data)
    plan = json.loads((a.data/"plan.json").read_text())
    ck = torch.load(a.branch, map_location="cpu")
    protocol, pilot_sha = file_hash(a.config), file_hash(a.pilot)
    contract = dict(protocol=protocol, pilot=pilot_sha, branch=file_hash(a.branch), phase2=file_hash(a.phase2),
                    depth_head=file_hash(a.depth_head), architecture=file_hash(ROOT/"keireality/depth_projection.py"))
    if any(ck["contract"][k] != contract[k] for k in ("protocol","pilot","phase2","depth_head","architecture")):
        raise ValueError("candidate, frozen references and protocol do not match")
    if a.split == "test":
        val = json.loads((a.out/"val/metrics.json").read_text())
        if val["contract"] != contract or val["status"] != "PASS":
            raise ValueError("validation did not release this model for the fresh drives")
    towns = cfg["validation_towns" if a.split=="val" else "fresh_towns"]
    expected = cfg["validation_ticks" if a.split=="val" else "fresh_ticks"]
    rows = [r for r in raw["shards"] if r["split"] == a.split]
    if list(dict.fromkeys(r["town_seed"] for r in rows)) != towns:
        raise ValueError("unexpected evaluation towns")
    cameras = [orbit_camera(angle) for angle in cfg["evaluation_angles"]]
    if [asdict(c) for c in cameras] != plan["calibration"]["targets"] or tuple(cfg["evaluation_angles"]) != ORBIT_ANGLES:
        raise ValueError("evaluation camera calibration differs")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    pilot, head = frozen_pilot(a.pilot,a.device), frozen_pilot(a.depth_head,a.device)
    base = load_renderer(a.phase2,pilot_sha,a.device).requires_grad_(False)
    branch = DepthProjectionBranch().to(a.device,memory_format=torch.channels_last).eval().requires_grad_(False)
    branch.load_state_dict(ck["model"])
    frozen = [state_hash(net) for net in (pilot,head,base,branch)]
    evidence, pooled, times = {}, [Scores() for _ in cameras], []
    out = a.out/a.split
    with writer_lock(out):
        meta = out/"metrics.json"
        if meta.exists():
            old = json.loads(meta.read_text())
            if old["contract"] != contract or old["collection"] != raw["contract_sha256"]:
                raise ValueError("output has results from a different experiment")
            if any(file_hash(out/file) != sha for file,sha in old["media"].items()):
                raise ValueError("corrupt saved comparison")
            print("verified", a.split, old["status"])
            return
        for town in towns:
            episode = [r for r in rows if r["town_seed"]==town]
            if len({r["episode_id"] for r in episode}) != 1 or sum(r["count"] for r in episode) != expected:
                raise ValueError(f"town {town} does not contain one full {expected}-tick episode")
            with tempfile.TemporaryDirectory(dir=out) as tmp:
                path = Path(tmp)/"paired.npy"
                frames, timing = rollout(a.data,episode,pilot,head,base,branch,cameras,a.device,path,expected)
                times.extend(timing)
                evidence[town] = score_town(a.data,episode,cameras,frames,out,cfg["edge_threshold"],pooled)
                evidence[town]["rollout_sha256"] = file_hash(path)
            print(town,evidence[town]["chase_check"],evidence[town]["views"]["0"],flush=True)
        if frozen != [state_hash(net) for net in (pilot,head,base,branch)]:
            raise ValueError("frozen inference state changed")
        checks = [r["chase_check"]["status"] for r in evidence.values()]
        status = "FAIL" if "FAIL" in checks else ("PASS" if all(s=="PASS" for s in checks) else "INCONCLUSIVE")
        media = {p.name:file_hash(p) for p in sorted(out.iterdir()) if p.suffix in (".png",".mp4")}
        result = dict(contract=contract, collection=raw["contract_sha256"], split=a.split, status=status,
                      ticks=len(towns)*expected, images=len(towns)*expected*len(cameras),
                      frozen_state_hashes=frozen, towns=evidence, media=media,
                      pooled_by_view={str(aa):s.report() for aa,s in zip(ORBIT_ANGLES,pooled,strict=True)},
                      paired_chase_camera_to_cpu_ms={f"p{q}":float(np.percentile(times,q)) for q in (50,95)})
        write_json(meta,result)
        print(a.split,status,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key, default in dict(config="configs/reality/p2_depth_projection.json", data="data/reality/p2",
            pilot="runs/keipilot_mc.pt", phase2="runs/reality/p2/temporal/best.pt", depth_head="runs/reference/mc14-depth.pt",
            branch="runs/reality/p2_projection/train/model.pt", out="runs/reality/p2_projection").items():
        p.add_argument("--"+key.replace("_","-"),type=Path,default=Path(default))
    p.add_argument("--split",choices=("val","test"), required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    evaluate(p.parse_args())
