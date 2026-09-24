#!/usr/bin/env python3
"""Run one closed-loop episode and record a dashboard video.

  python scripts/demo.py --agent expert --out runs/demo_expert.mp4
  python scripts/demo.py --agent model --ckpt runs/keipilot/best.pt --town 1001 --episode 7
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keisim.config import EnvConfig  # noqa: E402
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
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cfg = EnvConfig()
    env = KeiEnv(cfg)
    obs = env.reset(town_seed=args.town, episode_seed=args.episode, route_length=args.route, weather=args.weather)
    agent = None
    if args.agent == "model":
        from keipilot.agent import KeiPilotAgent
        agent = KeiPilotAgent(args.ckpt)
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
