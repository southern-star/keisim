"""Pictures of a camera rig next to KeiPilot's 100 deg camera, where the expert stands at 止まれ lines in Sapporo.

The expert drives routes of the evaluation town (episodes 100..) without rendering. Each time it has just stood at a
stop line, and again when it waits at a crossing for a car with priority, every camera of the rig and the front
camera are drawn on the same frame: one PNG per moment, cameras stacked top to bottom at 2x, in runs/rig_preview/.

usage: python tools/rig_preview.py [--rig wide_narrow] [--routes 6] [--max_shots 6]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from keisim.config import CameraConfig, EnvConfig, set_camera_rig  # noqa: E402
from keisim.env import KeiEnv  # noqa: E402
from keisim.realworld import osm_town  # noqa: E402
from keisim.render.rig import make_renderer  # noqa: E402

BBOX = (43.088, 141.360, 43.097, 141.372)


def label(img, text):
    out = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rig", default="wide_narrow")
    ap.add_argument("--routes", type=int, default=6)
    ap.add_argument("--max_shots", type=int, default=6)
    ap.add_argument("--out", default="runs/rig_preview")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    town = osm_town(BBOX, cache="data/osm/osm_sapporo.json", name="osm_sapporo")
    cfg = set_camera_rig(EnvConfig(), args.rig)
    cfg.town = town.cfg
    cfg.traffic.box_rule = cfg.traffic.release_hidden = True
    cfg.traffic.max_vehicles *= 2
    cfg.traffic.max_peds *= 2
    cfg.blocked_timeout = 180.0
    env = KeiEnv(cfg)
    shots, times, front = 0, {"front": [], "rig": []}, None
    for ep in range(100, 100 + args.routes):
        env.reset(town_seed=town, episode_seed=ep, route_length=600.0, weather="noon", render=False)
        if front is None:
            _ = env.assets.cameras                 # builds the static scene that the front camera shares
            front = make_renderer(town, env.assets.tex, env.assets._static, CameraConfig())
        approach = stood = waited = False
        while not env.done and shots < args.max_shots:
            plan, e = env.plan, env.world.ego
            moment = None
            # the expert's reason is stop_sign until it has stood (below 0.05 m/s), then crossing / yield
            if plan["reason"] == "stop_sign":
                approach, stood, waited = True, False, False
            elif approach and e.v < 0.05:
                approach, stood, moment = False, True, "stop"
            elif stood and not waited and plan["reason"] == "yield" and e.v < 0.3:
                waited, moment = True, "wait"
            elif plan["reason"] not in ("crossing", "yield"):
                approach = stood = waited = False
            if moment:
                t0 = time.perf_counter()
                views = env.render_cameras(want_seg=False)
                t1 = time.perf_counter()
                rgb100, _ = front.render(e.pose, env.world.scene_state(), env.weather, want_seg=False)
                t2 = time.perf_counter()
                times["rig"].append(t1 - t0)
                times["front"].append(t2 - t1)
                rows = [label(rgb100, "front 100 deg (KeiPilot v0.10)")]
                for c, (rgb, _) in zip(cfg.cameras, views):
                    rows.append(label(rgb, f"{c.name} {c.fov_deg:g} deg {c.projection}"))
                path = os.path.join(args.out, f"ep{ep}_t{env.elapsed:05.1f}_{moment}.png")
                cv2.imwrite(path, np.concatenate(rows, 0))
                print(f"{path}: rig {1000 * (t1 - t0):.0f} ms, front {1000 * (t2 - t1):.0f} ms", flush=True)
                shots += 1
            env.step(env.expert_action())
        if shots >= args.max_shots:
            break
    for k, v in times.items():
        if v:
            print(f"{k}: {1000 * np.mean(v):.0f} ms per frame (mean of {len(v)})")


if __name__ == "__main__":
    main()
