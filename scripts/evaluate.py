#!/usr/bin/env python3
"""Closed-loop benchmark (KeiBench).

Suites are fixed lists of (town_seed, episode_seed): a town seed fixes the map,
an episode seed fixes route, traffic, pedestrians, signal phase and weather.

  python scripts/evaluate.py --agent expert --suite test
  python scripts/evaluate.py --agent model --ckpt runs/keipilot_bc/best.pt --suite test --videos 3
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUITES = {
    # seen town layouts (towns used for data collection), new routes/traffic
    "train": [(t, 9000 + e) for t in (3, 57, 121, 250, 333) for e in range(4)],
    # unseen towns
    "test": [(t, 9000 + e) for t in (1000, 1001, 1002, 1003, 1004) for e in range(4)],
    # unseen towns, second set (for final reporting)
    "test2": [(t, 9100 + e) for t in (1005, 1006, 1007, 1008, 1009) for e in range(4)],
    # hand-picked unseen-town routes with varied weather, used for demo videos
    "showcase": [(1000, 9001), (1004, 9002), (1001, 9002), (1003, 9001), (1007, 9102), (1009, 9100)],
}


_ENV = None


def run_route(job):
    global _ENV
    (town, ep), args, idx = job
    import cv2
    cv2.setNumThreads(1)
    from keisim.config import EnvConfig
    from keisim.env import KeiEnv
    cfg = EnvConfig()
    cfg.render_seg = False
    cfg.render_rgb = args["agent"] == "model"
    cfg.renderer = args.get("renderer", "keisim")
    if args.get("dense"):
        cfg.traffic.vehicle_spacing = (20.0, 28.0)
        cfg.traffic.ped_spacing = (14.0, 20.0)
        cfg.traffic.max_vehicles = 130
        cfg.traffic.max_peds = 110
        cfg.traffic.ego_cross_rate = 0.3
    if cfg.renderer == "keiview":
        if _ENV is None:                     # keep one headless Chrome per worker process
            _ENV = KeiEnv(cfg)
        env = _ENV
        env.cfg = cfg
        env.expert.env = env
    else:
        env = KeiEnv(cfg)
    agent = None
    if args["agent"] == "model":
        global _AGENT
        if "_AGENT" not in globals():
            from keipilot.agent import KeiPilotAgent
            _AGENT = KeiPilotAgent(args["ckpt"])
        agent = _AGENT
        agent.reset()
    obs = env.reset(town_seed=town, episode_seed=ep, route_length=args["route_length"], weather=args["weather"])
    video = None
    events = []
    if idx < args["videos"]:
        from keisim.viz import VideoWriter, colorize_seg, compose_dashboard, info_lines
        os.makedirs(args["video_dir"], exist_ok=True)
        video = VideoWriter(os.path.join(args["video_dir"], f"{args['tag']}_{town}_{ep}.mp4"), fps=10)
        if agent is None:
            env.cfg.render_rgb = env.cfg.render_seg = True
            obs = env._obs()
    dt = cfg.dt * cfg.action_repeat
    done = False
    steps = 0
    t0 = time.time()
    speeds = []
    while not done and steps < args["max_steps"]:
        e = env.world.ego
        if agent is None:
            action = env.expert_action()
            mp_ = None
        else:
            action, mp_ = agent.act(obs["rgb"], obs["command"], obs["target_point"], e.v, dt, with_seg=video is not None)
        if video is not None:
            plan = env.plan
            paths = [(plan["path"], (80, 255, 80))]
            if mp_ is not None:
                paths.append((mp_["path"], (255, 80, 255)))
                seg_panel = colorize_seg(mp_["seg"])
                lines = info_lines(e.v, mp_["target_speed"], obs["command"], tl_pred=mp_["tl"], tl_gt=plan["tl_state"],
                                   extra=[f"expert target {plan['target_speed'] * 3.6:5.1f} km/h"])
                title = "KeiPilot (camera only)  green=expert  magenta=model"
            else:
                seg_panel = colorize_seg(obs["seg"])
                lines = info_lines(e.v, plan["target_speed"], obs["command"], tl_gt=plan["tl_state"], reason=plan["reason"])
                title = "privileged expert"
            lines.append(f"t={env.elapsed:5.1f}s  RC={env.metrics()['RC'] * 100:4.1f}%")
            frame = compose_dashboard(obs["rgb"], env.render_bev(paths=paths), seg_panel, lines,
                                      cam=env.assets.camera, pose=e.pose, paths=paths, title=title)
            video.write(frame)
            events.append({"t": round(env.elapsed, 1), "v": round(e.v, 2), "reason": plan["reason"],
                           "tl_gt": plan["tl_state"], "tl_pred": None if mp_ is None else mp_["tl"],
                           "cmd": obs["command"], "target_model": None if mp_ is None else round(mp_["target_speed"], 2),
                           "target_expert": round(plan["target_speed"], 2)})
        obs, r, done, info = env.step(action)
        speeds.append(env.world.ego.v)
        steps += 1
    if video is not None:
        video.close()
        with open(video.path[:-4] + ".json", "w") as f:
            json.dump(events, f)
    m = env.metrics()
    if not done:
        m["status"] = "timeout"
    m.update({"town": town, "episode": ep, "steps": steps, "wall": time.time() - t0,
              "avg_speed": float(np.mean(speeds)) if speeds else 0.0, "weather": env.weather.name})
    return m


def summarize(results):
    n = len(results)
    km = sum(r["distance"] for r in results) / 1000.0
    s = {
        "routes": n,
        "DS": float(np.mean([r["DS"] for r in results])),
        "RC": float(np.mean([r["RC"] for r in results])),
        "IS": float(np.mean([r["IS"] for r in results])),
        "success_rate": float(np.mean([r["status"] == "success" for r in results])),
        "km": km,
        "avg_speed_kmh": float(np.mean([r["avg_speed"] for r in results]) * 3.6),
    }
    kinds = {}
    for r in results:
        for inf in r["infractions"]:
            kinds[inf["type"]] = kinds.get(inf["type"], 0) + 1
    s["infractions_per_km"] = {k: v / max(km, 1e-6) for k, v in kinds.items()}
    s["infraction_counts"] = kinds
    st = {}
    for r in results:
        st[r["status"]] = st.get(r["status"], 0) + 1
    s["status"] = st
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="expert", choices=["expert", "model"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--suite", default="test", choices=list(SUITES))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--route_length", type=float, default=600.0)
    ap.add_argument("--weather", default="random")
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--videos", type=int, default=0)
    ap.add_argument("--video_dir", default="runs/videos")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dense", action="store_true", help="stress test: dense traffic and many crossing pedestrians")
    ap.add_argument("--renderer", default="keisim", choices=["keisim", "keiview"], help="camera renderer")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.tag:
        tag = args.tag
    elif args.agent == "expert":
        tag = "expert"
    elif os.path.basename(args.ckpt) in ("last.pt", "best.pt"):
        tag = os.path.basename(os.path.dirname(args.ckpt))      # runs/keipilot_dagger/last.pt -> keipilot_dagger
    else:
        tag = os.path.splitext(os.path.basename(args.ckpt))[0]  # runs/keipilot.pt -> keipilot
    jobs = SUITES[args.suite]
    a = {"agent": args.agent, "ckpt": args.ckpt, "route_length": args.route_length, "weather": args.weather,
         "max_steps": args.max_steps, "videos": args.videos, "video_dir": args.video_dir, "tag": tag,
         "dense": args.dense, "renderer": args.renderer}
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = []
        for m in pool.imap(run_route, [(j, a, i) for i, j in enumerate(jobs)]):
            results.append(m)
            print(f"town {m['town']} ep {m['episode']}: {m['status']:>20} DS={m['DS']:.2f} RC={m['RC']:.2f} "
                  f"inf={[i['type'] for i in m['infractions']]} ({m['wall']:.0f}s)", flush=True)
    s = summarize(results)
    print(json.dumps(s, indent=1))
    suffix = ("_dense" if args.dense else "") + ("" if args.weather == "random" else f"_{args.weather}") + \
        ("_keiview" if args.renderer == "keiview" else "")
    out = args.out or f"runs/eval/{tag}_{args.suite}{suffix}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": s, "results": results, "args": vars(args), "time": time.time() - t0}, f, indent=1)
    print("saved", out, f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
