#!/usr/bin/env python3
"""Run one closed-loop episode and record a dashboard video.

  python scripts/demo.py --agent expert --out runs/demo_expert.mp4
  python scripts/demo.py --agent model --ckpt runs/keipilot/best.pt --town 1001 --episode 7
  python scripts/demo.py --agent model --ckpt runs/keipilot.pt --osm_town osm_sapporo --episode 110 --route 600
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keisim.config import TOWN_STYLES, EnvConfig, town_config  # noqa: E402
from keisim.env import KeiEnv  # noqa: E402
from keisim.viz import VideoWriter, colorize_seg, compose_dashboard, info_lines  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="expert", choices=["expert", "model"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--town", type=int, default=1000)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--weather", default="random")
    ap.add_argument("--route", type=float, default=None)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--out", default="runs/demo.mp4")
    ap.add_argument("--show", action="store_true", help="also show a live OpenCV window")
    ap.add_argument("--renderer", default="keisim", choices=["keisim", "keiview"], help="camera renderer")
    ap.add_argument("--keiview_roads", default="plain", choices=["plain", "detail"],
                    help="KeiView road surface: plain (as trained) or detail (markings, signs, ...)")
    ap.add_argument("--no_safety", action="store_true", help="model: without the BrakeHold / RedHold safety layers")
    ap.add_argument("--town_style", default="classic", choices=list(TOWN_STYLES),
                    help="twophase: Japan's two-phase signals, right turns wait for gaps in the oncoming traffic; "
                         "roadtypes: twophase with main roads, residential roads without sidewalks and alleys")
    ap.add_argument("--osm_town", default=None,
                    help="a real town (keisim.realworld.osm_town, cached at data/osm/<name>.json) instead of --town; "
                         "the same --episode and --route as evaluate.py --osm_town give the same route and traffic")
    ap.add_argument("--osm_bbox", type=float, nargs=4, metavar=("S", "W", "N", "E"),
                    default=None, help="--osm_town\'s box (default: the town\'s entry in keisim.realworld.OSM_BOXES)")
    args = ap.parse_args()
    if args.osm_town and args.osm_bbox is None:
        from keisim.realworld import OSM_BOXES
        if args.osm_town not in OSM_BOXES:
            ap.error(f"--osm_town {args.osm_town}: not in keisim.realworld.OSM_BOXES, give --osm_bbox S W N E")
        args.osm_bbox = list(OSM_BOXES[args.osm_town])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cfg = EnvConfig()
    cfg.renderer = args.renderer
    cfg.keiview_roads = args.keiview_roads
    town = args.town
    if args.osm_town:
        from keisim.realworld import osm_town
        town = osm_town(tuple(args.osm_bbox), cache=f"data/osm/{args.osm_town}.json", name=args.osm_town)
        cfg.town = town.cfg
        cfg.blocked_timeout = 180.0
        cfg.traffic.max_vehicles *= 2           # evaluate.py's NPC caps for two-phase towns (CAP_SCALE)
        cfg.traffic.max_peds *= 2
    else:
        cfg.town = town_config(args.town_style)
    if args.osm_town or args.town_style != "classic":       # the long2 / long3 traffic rules (box rule, hidden jam release)
        cfg.traffic.box_rule = cfg.traffic.release_hidden = True
    env = KeiEnv(cfg)
    obs = env.reset(town_seed=town, episode_seed=args.episode, route_length=args.route, weather=args.weather)
    agent = None
    if args.agent == "model":
        from keipilot.agent import KeiPilotAgent
        agent = KeiPilotAgent(args.ckpt, brake_hold=not args.no_safety, red_hold=not args.no_safety)
        agent.reset()
    vw = VideoWriter(args.out, fps=10)
    cam = env.assets.camera
    dt = cfg.dt * cfg.action_repeat
    done = False
    step = 0
    t0 = time.time()
    while not done and step < args.max_steps:
        plan = env.plan
        e = env.world.ego
        if agent is None:
            action = env.expert_action()
            paths_cam = [(plan["path"], (80, 255, 80))]
            seg_panel = colorize_seg(obs["seg"])
            lines = info_lines(e.v, plan["target_speed"], obs["command"], tl_gt=plan["tl_state"], reason=plan["reason"])
            title = "privileged expert"
        else:
            action, mp = agent.act(obs["rgb"], obs["command"], obs["target_point"], e.v, dt, with_seg=True)
            paths_cam = [(plan["path"], (80, 255, 80)), (mp["path"], (255, 80, 255))]
            seg_panel = colorize_seg(mp["seg"])
            lines = info_lines(e.v, mp["target_speed"], obs["command"], tl_pred=mp["tl"], tl_gt=plan["tl_state"],
                               extra=[f"expert target {plan['target_speed'] * 3.6:5.1f} km/h"])
            title = "KeiPilot (camera only)  green=expert  magenta=model"
        lines.append(f"t={env.elapsed:5.1f}s  RC={env.metrics()['RC'] * 100:4.1f}%")
        bev = env.render_bev(paths=paths_cam)
        frame = compose_dashboard(obs["rgb"], bev, seg_panel, lines, cam=cam, pose=e.pose, paths=paths_cam, title=title)
        vw.write(frame)
        if args.show:
            cv2.imshow("KeiSim", frame)
            if cv2.waitKey(1) == 27:
                break
        obs, r, done, info = env.step(action)
        step += 1
    vw.close()
    info = env.metrics()
    print(f"{info['status']} RC={info['RC']:.2f} DS={info['DS']:.2f} infractions={[i['type'] for i in info['infractions']]}"
          f"  ({step} steps, {step / (time.time() - t0):.1f} it/s) -> {args.out}")


if __name__ == "__main__":
    main()
