"""ReconView G0: KeiView's depth pass measured on a few frames (docs/recon_view.md, 2 章).

    python tools/recon_g0.py [--towns 5 23 41] [--frames 4] [--out runs/eval/recon/g0.json]
    python tools/recon_g0.py --table runs/eval/recon/g0.json      # the tables of the document from a result

Per frame of the wide_narrow rig (the expert drives 2 s between frames):
(a) the depth of the road, the paint and the sidewalk against the flat ground along each pixel's line of sight, at
    full and half resolution, and for the wide camera also stitched by the nearest face pixel alone;
(b) the points of each camera in the same camera 3 m further on, and in the other camera: the share B looks through;
(c) depth and labels show the same pixels.
Needs KeiView on a GPU (about 10 s per town and 0.2 s per frame)."""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keisim.config import SEM, EnvConfig, set_camera_rig  # noqa: E402
from keisim.render.depth_check import heights, plane_errors, reproject  # noqa: E402

PAIRS = (("wide", "wide", True), ("narrow", "narrow", True), ("wide", "narrow", True), ("wide", "narrow", False),
         ("narrow", "wide", False))                              # (A, B, B is 3 m further on)


def measure(towns, frames):
    from keisim.env import KeiEnv
    env = KeiEnv(set_camera_rig(EnvConfig(renderer="keiview"), "wide_narrow"))
    cams = env.cfg.cameras
    idx = {c.name: i for i, c in enumerate(cams)}
    errs, agree, kerb, pairs = {}, [], [], {}
    try:
        for town in towns:
            env.reset(town_seed=town, episode_seed=1, weather="noon")
            kv = env.keiview()
            for _ in range(frames):
                pose, state, t = env.world.ego.pose, env.world.scene_state(), env.world.t
                views, deps = env.render_cameras_depth()
                kv.depth_stitch = "nearest"
                kv._stitchers.clear()
                near = kv.render_rig_depth(pose, state, t, which=[0])[1][0]
                kv.depth_stitch = "planar"
                kv._stitchers.clear()
                for cfg, (_, seg), dep in zip(cams, views, deps):
                    agree.append(float(np.mean((dep > 0) == (seg != SEM["sky"]))))
                    for step in (1, 2):
                        for name, e in plane_errors(cfg, dep, seg, pose, step).items():
                            errs.setdefault((cfg.name, "planar" if cfg.projection == "cylindrical" else "-", step, name), []).append(e)
                    kerb.append(heights(cfg, dep, seg, pose, "sidewalk"))
                for step in (1, 2):
                    for name, e in plane_errors(cams[0], near, views[0][1], pose, step).items():
                        errs.setdefault((cams[0].name, "nearest", step, name), []).append(e)
                x, y, yaw = pose
                pose_b = (x + 3.0 * math.cos(yaw) - 0.5 * math.sin(yaw), y + 3.0 * math.sin(yaw) + 0.5 * math.cos(yaw), yaw + 0.03)
                deps_b = kv.render_rig_depth(pose_b, state, t)[1]
                for a, b, moved in PAIRS:
                    through, n, diff = reproject(cams[idx[a]], deps[idx[a]], pose, cams[idx[b]],
                                                 (deps_b if moved else deps)[idx[b]], pose_b if moved else pose)
                    pairs.setdefault((a, b, moved), []).append((through, reproject.nearest_pixel, n, diff))
                for _ in range(20):
                    env.step(env.expert.act())
    finally:
        env.close()
    out = {"towns": list(towns), "frames": frames, "plane": [], "pairs": [], "labels_agree_min": min(agree)}
    for (cam, stitch, step, name), e in sorted(errs.items()):
        e = np.concatenate(e)
        if len(e):
            out["plane"].append({"camera": cam, "stitch": stitch, "step": step, "class": name, "pixels": int(len(e)),
                                 "median": float(np.median(e)), "p95": float(np.percentile(e, 95))})
    z = np.concatenate(kerb)
    out["sidewalk_height"] = {"pixels": int(len(z)), "median": float(np.median(z)), "p1": float(np.percentile(z, 1)),
                              "p99": float(np.percentile(z, 99)), "within": float(np.mean((z > -0.04) & (z < 0.19)))}
    for (a, b, moved), v in pairs.items():
        v = np.array(v)
        out["pairs"].append({"a": a, "b": b, "moved": moved, "points": int(v[:, 2].sum()),
                             "through": float(np.average(v[:, 0], weights=v[:, 2])), "through_max": float(v[:, 0].max()),
                             "through_nearest_pixel": float(np.average(v[:, 1], weights=v[:, 2])),
                             "smooth_median": float(np.median(v[:, 3])), "smooth_median_max": float(v[:, 3].max())})
    return out


def table(r):
    pc = lambda x: f"{100 * x:.3f} %"
    print(f"町 {', '.join(map(str, r['towns']))}、各 {r['frames']} フレーム\n")
    print("| カメラ | 貼り合わせ | 解像度 | クラス | 画素 | 中央値 | 95 % 点 |\n|---|---|---|---|---:|---:|---:|")
    ja = {"road": "道路", "marking": "白線", "sidewalk": "歩道"}
    for p in r["plane"]:
        print(f"| {p['camera']} | {p['stitch']} | {'元' if p['step'] == 1 else '半分'} | {ja[p['class']]} | {p['pixels']:,} | "
              f"{pc(p['median'])} | {pc(p['p95'])} |")
    h = r["sidewalk_height"]
    print(f"\n歩道のクラスの高さ: 中央値 {100 * h['median']:.2f} cm、1 % 点 {100 * h['p1']:.2f} cm、99 % 点 {100 * h['p99']:.2f} cm、"
          f"−4〜19 cm に {100 * h['within']:.1f} %（{h['pixels']:,} 画素）\n")
    print("| A | B | B の位置 | 点 | 素通し | 最悪のフレーム | (B の最近傍 1 画素と比べた場合) | なめらかな面での差の中央値 |\n|---|---|---|---:|---:|---:|---:|---:|")
    for p in r["pairs"]:
        print(f"| {p['a']} | {p['b']} | {'3 m 先' if p['moved'] else '同じ'} | {p['points']:,} | {100 * p['through']:.2f} % | "
              f"{100 * p['through_max']:.2f} % | {100 * p['through_nearest_pixel']:.2f} % | {pc(p['smooth_median'])} |")
    print(f"\n深度が有効な画素とラベルが sky でない画素の一致率（最小のフレーム）: {100 * r['labels_agree_min']:.3f} %")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--towns", type=int, nargs="+", default=[5, 23, 41])
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--out", default="runs/eval/recon/g0.json")
    ap.add_argument("--table", default=None, help="print the tables of this result and do not measure")
    args = ap.parse_args()
    if args.table:
        table(json.load(open(args.table)))
        return
    r = measure(args.towns, args.frames)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(r, f, indent=1)
    table(r)


if __name__ == "__main__":
    main()
