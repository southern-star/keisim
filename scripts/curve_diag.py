#!/usr/bin/env python3
"""Step KeiPilot through part of the curve test course (keisim/curve_course.py) and log, every control step, where
the car is and what the model's path asks of the steering, next to the expert's path from the same state.

    python scripts/curve_diag.py OUT.json VARIANT fwd|rev FROM_S TO_S [options]
    python scripts/curve_diag.py runs/r16.json nolines_guard_nobld rev 1700 1880 --guard right   # the car's left rail only

The course can be changed for experiments (KeiSim renderer only):
    --guard both|left|right|none   guardrails on the road's left / right (of the course's a -> b, the forward way;
                                   driven in reverse, the road's right is the car's left)
    --beam LO HI                   guardrail beam height (m; posts up to it), default keisim/render/meshes.py
    --lead M                       straight at both ends (default 70 m)
    --light sun|flat               CameraConfig.guardrail_light (default flat)
    --texfix                       draw the rasters without Raster.fill's half-pixel shift (README, curve course)
Per step the log has route s, lateral offset (+ left), heading error, speed, the steering of the model's path and of
the expert's path (keisim.control.PurePursuit at the car's speed), and the lateral offset from the lane centre of the
model's / expert's path points 6, 10 and 20 m ahead. At the end it prints, for every curve passed, the offset towards
the curve's inside (mean / max / min) and the model path's offset 6 m ahead.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from keisim.config import EnvConfig, town_config  # noqa: E402
from keisim.control import PurePursuit  # noqa: E402
from keisim.curve_course import (  # noqa: E402
    LEAD,
    PARCEL_REACH,
    RADII,
    VARIANTS,
    course_line,
    course_route,
    curve_town,
    split_variant,
)
from keisim.geometry import world_to_local  # noqa: E402
from keisim.roadnet import Raster, RoadSpec, Town  # noqa: E402

PATH_AHEAD = (2, 4, 9)        # indices of the path points 6, 10 and 20 m ahead (KeiPilot: a point every 2 m)


def build_town(a):
    if a.lead is None and a.guard is None:
        return curve_town(a.variant)
    base, with_buildings = split_variant(a.variant)
    pts, curves = course_line(RADII, lead=a.lead or LEAD)
    kw = dict(VARIANTS[base])
    if a.guard is not None:
        kw["guardrail"] = None if a.guard == "none" else a.guard
    town = Town.from_graph(0, town_config("roadtypes"), [tuple(pts[0]), tuple(pts[-1])],
                           [(0, 1, pts, RoadSpec("residential", 3.0, 0.5, 0.0, 30 / 3.6, **kw))],
                           buildings=None if with_buildings else [], name=f"curve_{a.variant}_lead{a.lead}_g{a.guard}")
    town.course = {"variant": a.variant, "centre": pts, "curves": curves}
    town.parcel_reach, town.extra_lots = PARCEL_REACH, with_buildings
    return town


def lane_offset(route, p):
    """Lateral offset (+ left) of world point p from the route's lane centre."""
    s = float(route.poly.project(p)[0])
    q, h = route.poly.interp1(s), route.poly.heading1(s)
    return float(-(p[0] - q[0]) * math.sin(h) + (p[1] - q[1]) * math.cos(h))


def path_offsets(route, path, e):
    c, s = math.cos(e.yaw), math.sin(e.yaw)
    return [round(lane_offset(route, np.array([e.x + x * c - y * s, e.y + x * s + y * c])), 3)
            for x, y in (path[i] for i in PATH_AHEAD)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("variant")
    ap.add_argument("dir", choices=["fwd", "rev"])
    ap.add_argument("from_s", type=float)
    ap.add_argument("to_s", type=float)
    ap.add_argument("--light", default="flat", choices=["sun", "flat"])
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--weather", default="noon")
    ap.add_argument("--guard", default=None, choices=["both", "left", "right", "none"])
    ap.add_argument("--beam", type=float, nargs=2, default=None)
    ap.add_argument("--lead", type=float, default=None)
    ap.add_argument("--tp_ahead", type=float, default=50.0)
    ap.add_argument("--texfix", action="store_true")
    ap.add_argument("--ckpt", default="runs/keipilot_v10.pt")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--video", default=None)
    a = ap.parse_args()

    if a.texfix:
        import cv2

        def fill(self, img, poly, value):
            px = np.round((self.to_px(poly) - 0.5) * 8.0).astype(np.int32)
            cv2.fillPoly(img, [px.reshape(-1, 1, 2)], value, lineType=cv2.LINE_8, shift=3)
        Raster.fill = fill
    if a.beam is not None:
        from keisim.render import meshes
        meshes.GUARDRAIL_BEAM = tuple(a.beam)
        meshes.GUARDRAIL_POST = max(0.02, a.beam[1] - 0.05)
    import torch
    torch.set_num_threads(a.threads)
    from keipilot.agent import KeiPilotAgent
    from keisim.env import KeiEnv

    town = build_town(a)
    cfg = EnvConfig()
    cfg.traffic.max_vehicles = 0
    cfg.traffic.max_peds = 0
    cfg.render_seg = False
    cfg.terminate_on_collision = True
    cfg.camera.guardrail_light = a.light
    env = KeiEnv(cfg)
    route, curves = course_route(town, reverse=a.dir == "rev", start=a.from_s)
    obs = env.reset(town_seed=town, episode_seed=a.episode, weather=a.weather, route=route)
    agent = KeiPilotAgent(a.ckpt, brake_hold=True, red_hold=True)
    agent.reset()
    pp = PurePursuit()
    dt = cfg.dt * cfg.action_repeat
    vw = None
    if a.video:
        from keisim.viz import VideoWriter, compose_dashboard, info_lines
        vw = VideoWriter(a.video, fps=10)
    log, done, status = [], False, "running"
    while not done:
        e, s = env.world.ego, env.s_ego
        tp = world_to_local(route.poly.interp(min(s + a.tp_ahead, route.s_end))[None], e.x, e.y, e.yaw)[0]
        action, p = agent.act(obs["rgb"], obs["command"], tp, e.v, dt)
        ex = obs["expert"]
        h = route.poly.heading1(s)
        herr = (e.yaw - h + math.pi) % (2 * math.pi) - math.pi
        lat = lane_offset(route, np.array([e.x, e.y]))
        log.append({"s": round(float(s), 2), "lat": round(lat, 3), "herr_deg": round(math.degrees(herr), 2),
                    "v": round(float(e.v), 3), "target": round(float(p["target_speed"]), 3),
                    "steer": round(float(action[0]), 4), "steer_model": round(pp(p["path"], e.v), 4),
                    "steer_expert": round(pp(ex["path"], e.v), 4),
                    "path_lat": path_offsets(route, p["path"], e), "expert_path_lat": path_offsets(route, ex["path"], e)})
        if vw is not None:
            paths = [(p["path"], (255, 80, 255)), (ex["path"], (80, 255, 80))]
            vw.write(compose_dashboard(np.ascontiguousarray(obs["rgb"]), env.render_bev(paths=paths),
                                       np.zeros((160, 320, 3), np.uint8),
                                       info_lines(e.v, p["target_speed"], obs["command"])
                                       + [f"s {s:6.1f}  offset {lat:+.2f} (+ left)  heading {math.degrees(herr):+.1f}"],
                                       cam=env.assets.camera, pose=e.pose, paths=paths,
                                       title=f"{a.variant} {a.dir} guard={a.guard} model=pink expert=green"))
        obs, _, done, info = env.step(action)
        status = info["status"]
        if not done and (env.s_ego >= a.to_s or env.s_ego >= route.s_end - 3):
            done, status = True, "reached"
    if vw is not None:
        vw.close()
    with open(a.out, "w") as f:
        json.dump({"args": vars(a), "status": status, "s_end": float(env.s_ego), "curves": curves, "log": log}, f)
    print(f"{a.variant} {a.dir} guard={a.guard} beam={a.beam} light={a.light}: {status} at {env.s_ego:.1f} m")
    for c in curves:
        rows = [r for r in log if c["s0"] <= r["s"] <= c["s1"]]
        if len(rows) < 3:
            continue
        sg = 1.0 if c["turn"] == "left" else -1.0                    # + = towards the curve's inside
        inside = np.array([r["lat"] for r in rows]) * sg
        p6 = np.mean([r["path_lat"][0] for r in rows]) * sg
        print(f"  {c['turn']:5s} R{c['radius']:5.1f} ({c['s0']:.0f}-{c['s1']:.0f} m): inside mean {inside.mean():+.2f} "
              f"max {inside.max():+.2f} min {inside.min():+.2f}, model path 6 m ahead {p6:+.2f}")


if __name__ == "__main__":
    main()
