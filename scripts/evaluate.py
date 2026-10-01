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
    # unseen towns not used by any other suite, 2.5 km routes (~20 signalised stop lines each): rare failures
    "long": [(t, 9200 + e) for t in range(1010, 1020) for e in range(2)],
}
# per-suite defaults for arguments left unset on the command line
SUITE_DEFAULTS = {"long": {"route_length": 2500.0, "max_steps": 20000}}


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
            _AGENT = KeiPilotAgent(args["ckpt"], brake_hold=args.get("brake_hold", False))
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
    log = []            # per-step decisions, saved around failures (runs/failures/)
    while not done and steps < args["max_steps"]:
        e = env.world.ego
        if agent is None:
            action = env.expert_action()
            mp_ = None
        else:
            action, mp_ = agent.act(obs["rgb"], obs["command"], obs["target_point"], e.v, dt, with_seg=video is not None)
        plan = env.plan
        log.append((round(env.elapsed, 1), round(float(env.s_ego), 1), round(float(e.v), 2), plan["reason"],
                    int(plan["tl_state"]), None if mp_ is None else int(mp_["tl"]),
                    None if mp_ is None else round(float(mp_["target_speed"]), 2), round(float(plan["target_speed"]), 2)))
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
              "avg_speed": float(np.mean(speeds)) if speeds else 0.0, "weather": env.weather.name,
              "lights_passed": int(sum(1 for s_stop, _ in env.route.stops if s_stop < env.s_ego))})
    if m["status"] in ("blocked", "timeout") and log:
        m["blocked_by"] = blocked_cause(log[-int(cfg.blocked_timeout / dt):], dt)
    if (m["status"] != "success" or m["infractions"]) and args.get("fail_dir"):
        keys = ("t", "s", "v", "reason", "tl_gt", "tl_pred", "target_model", "target_expert")
        spans = [(i["t"] - 12.0, i["t"] + 3.0) for i in m["infractions"]]
        if m["status"] != "success":
            spans.append((log[-1][0] - cfg.blocked_timeout - 20.0, log[-1][0] + 1.0))
        windows = [[dict(zip(keys, r)) for r in log if a <= r[0] <= b] for a, b in spans]
        os.makedirs(args["fail_dir"], exist_ok=True)
        with open(os.path.join(args["fail_dir"], f"{town}_{ep}.json"), "w") as f:
            json.dump({"result": {k: v for k, v in m.items()}, "windows": windows}, f)
    return m


def blocked_cause(win, dt, go_s=3.0):
    """Who kept the ego standing? "model" if, while it stood, the expert wanted to drive on (target > 1 m/s) for
    at least `go_s` seconds in a row - e.g. a green it did not take, even when the red around it was long -
    else "traffic" (the expert would have waited too). win: log rows (t, s, v, reason, ..., target_expert)."""
    run = best = 0
    reasons = {}
    for r in win:
        if r[2] < 0.1:
            reasons[r[3]] = reasons.get(r[3], 0) + 1
            run = run + 1 if r[-1] > 1.0 else 0
            best = max(best, run)
        else:
            run = 0
    return {"cause": "model" if best * dt >= go_s else "traffic", "expert_go_s": round(best * dt, 1),
            "expert_reasons": reasons}


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
        "lights_passed": int(sum(r.get("lights_passed", 0) for r in results)),
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
    causes = {}
    for r in results:
        if "blocked_by" in r:
            causes[r["blocked_by"]["cause"]] = causes.get(r["blocked_by"]["cause"], 0) + 1
    if causes:
        s["blocked_by"] = causes
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="expert", choices=["expert", "model"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--suite", default="test", choices=list(SUITES))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--route_length", type=float, default=None, help="default 600 m (long suite: 2500 m)")
    ap.add_argument("--weather", default="random")
    ap.add_argument("--max_steps", type=int, default=None, help="default 4000 (long suite: 20000)")
    ap.add_argument("--videos", type=int, default=0)
    ap.add_argument("--video_dir", default="runs/videos")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dense", action="store_true", help="stress test: dense traffic and many crossing pedestrians")
    ap.add_argument("--renderer", default="keisim", choices=["keisim", "keiview"], help="camera renderer")
    ap.add_argument("--out", default=None)
    ap.add_argument("--brake_hold", action="store_true", help="safety layer: no throttle burst right after firm braking")
    args = ap.parse_args()
    dflt = {"route_length": 600.0, "max_steps": 4000, **SUITE_DEFAULTS.get(args.suite, {})}
    for k, v in dflt.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.tag:
        tag = args.tag
    elif args.agent == "expert":
        tag = "expert"
    elif os.path.basename(args.ckpt) in ("last.pt", "best.pt"):
        tag = os.path.basename(os.path.dirname(args.ckpt))      # runs/keipilot_dagger/last.pt -> keipilot_dagger
    else:
        tag = os.path.splitext(os.path.basename(args.ckpt))[0]  # runs/keipilot.pt -> keipilot
    if args.brake_hold:
        tag += "_hold"
    jobs = SUITES[args.suite]
    suffix = ("_dense" if args.dense else "") + ("" if args.weather == "random" else f"_{args.weather}") + \
        ("_keiview" if args.renderer == "keiview" else "")
    fail_dir = os.path.join("runs", "failures", f"{tag}_{args.suite}{suffix}")
    a = {"agent": args.agent, "ckpt": args.ckpt, "route_length": args.route_length, "weather": args.weather,
         "max_steps": args.max_steps, "videos": args.videos, "video_dir": args.video_dir, "tag": tag,
         "dense": args.dense, "renderer": args.renderer, "fail_dir": fail_dir, "brake_hold": args.brake_hold}
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
    out = args.out or f"runs/eval/{tag}_{args.suite}{suffix}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": s, "results": results, "args": vars(args), "time": time.time() - t0}, f, indent=1)
    print("saved", out, f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
