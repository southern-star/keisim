#!/usr/bin/env python3
"""Closed-loop benchmark (KeiBench).

Suites are fixed lists of (town_seed, episode_seed): a town seed fixes the map,
an episode seed fixes route, traffic, pedestrians, signal phase and weather.

  python scripts/evaluate.py --agent expert --suite test
  python scripts/evaluate.py --agent model --ckpt runs/keipilot_bc/best.pt --suite test --videos 3

A real town imported from OpenStreetMap (keisim/realworld.py, scripts/import_osm.py) replaces the suite's town seeds:

  python scripts/evaluate.py --agent model --ckpt runs/keipilot.pt --osm_town osm_sapporo --routes 20

The town is rebuilt from the Overpass cache data/osm/<name>.json, routes get the episode seeds 100.. (the same as
import_osm.py's expert drives) and the results go to runs/eval/<name>/ (never over the suites' JSON).
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
from keisim.config import TOWN_STYLES  # noqa: E402

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
    # the same seeds as "varied" towns (block lengths 70-200 m), with the two jam fixes (box rule, hidden release)
    "long2": [(t, 9200 + e) for t in range(1010, 1020) for e in range(2)],
    # the same seeds again, now with two-phase signals: right turns wait inside the junction for oncoming gaps
    "long3": [(t, 9200 + e) for t in range(1010, 1020) for e in range(2)],
    # long3's towns with other episodes (new routes and traffic): twice the routes, as rare failures are noisy
    "long3b": [(t, 9300 + e) for t in range(1010, 1020) for e in range(2)],
    # confirmation sets: long3's towns with episodes no model was picked on (KeiPilot v0.7 was chosen as the best of
    # several runs on long3 + long3b, so its scores there are optimistic)
    "long3c": [(t, 9400 + e) for t in range(1010, 1020) for e in range(2)],
    "long3d": [(t, 9500 + e) for t in range(1010, 1020) for e in range(2)],
    # more confirmation sets (since KeiPilot v0.8 models are judged on long3c + long3d): v16 vs v17 differed there by
    # 3 vs 8 collisions in 160 routes, within the run-to-run noise of these chaotic long routes
    "long3e": [(t, 9600 + e) for t in range(1010, 1020) for e in range(2)],
    "long3f": [(t, 9700 + e) for t in range(1010, 1020) for e in range(2)],
}
CAP_SCALE = {"classic": 1, "varied": 2, "twophase": 2, "roadtypes": 2}      # NPC count caps per town style
# per-suite defaults for arguments left unset on the command line
SUITE_DEFAULTS = {
    "long": {"route_length": 2500.0, "max_steps": 20000},
    # long2: the jam fixes leave only congestion (exit queues crawling through several signal cycles), so a route
    # counts as blocked after 180 s without progress (about 4 cycles) instead of 90 s
    "long2": {"route_length": 2500.0, "max_steps": 20000, "town_style": "varied", "jam_fixes": True,
              "blocked_timeout": 180.0},
    "long3b": {"route_length": 2500.0, "max_steps": 20000, "town_style": "twophase", "jam_fixes": True,
               "blocked_timeout": 180.0},
    "long3": {"route_length": 2500.0, "max_steps": 20000, "town_style": "twophase", "jam_fixes": True,
              "blocked_timeout": 180.0},
}
SUITE_DEFAULTS["long3c"] = SUITE_DEFAULTS["long3d"] = SUITE_DEFAULTS["long3"]
SUITE_DEFAULTS["long3e"] = SUITE_DEFAULTS["long3f"] = SUITE_DEFAULTS["long3"]


_ENV = None


def run_route(job):
    global _ENV
    (town, ep), args, idx = job
    import cv2
    cv2.setNumThreads(1)
    from keisim.config import EnvConfig, town_config
    from keisim.env import KeiEnv
    cfg = EnvConfig()
    osm = None
    if isinstance(town, str):                        # a real town (keisim/realworld.py), built once per worker
        global _TOWN
        if "_TOWN" not in globals() or _TOWN.name != town:
            from keisim.realworld import osm_town
            _TOWN = osm_town(tuple(args["osm_bbox"]), cache=f"data/osm/{town}.json", name=town)
        osm = _TOWN
        cfg.town = osm.cfg
    else:
        cfg.town = town_config(args.get("town_style", "classic"))
    cfg.traffic.box_rule = cfg.traffic.release_hidden = bool(args.get("jam_fixes", False))
    cfg.blocked_timeout = float(args.get("blocked_timeout", 90.0))
    cfg.render_seg = False
    cfg.render_rgb = args["agent"] == "model"
    cfg.renderer = args.get("renderer", "keisim")
    cfg.keiview_roads = args.get("keiview_roads", "plain")
    if args.get("dense"):
        cfg.traffic.vehicle_spacing = (20.0, 28.0)
        cfg.traffic.ped_spacing = (14.0, 20.0)
        cfg.traffic.max_vehicles = 130
        cfg.traffic.max_peds = 110
        cfg.traffic.ego_cross_rate = 0.3
    # varied towns have ~1.75x the lane length: twice the caps keep the traffic density of the classic towns
    cap = CAP_SCALE.get(args.get("town_style", "classic"), 1)
    cfg.traffic.max_vehicles *= cap
    cfg.traffic.max_peds *= cap
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
            _AGENT = KeiPilotAgent(args["ckpt"], brake_hold=args.get("brake_hold", False), red_hold=args.get("red_hold", False))
        agent = _AGENT
        agent.reset()
    obs = env.reset(town_seed=osm if osm is not None else town, episode_seed=ep, route_length=args["route_length"],
                    weather=args["weather"])
    kv = env._keiview if cfg.renderer == "keiview" else None
    kv_ms = []                                  # KeiView: ego_server's render time per frame (ms)
    kv_town_ms = kv.town_ms if kv is not None else None
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
        if kv is not None and cfg.render_rgb:
            kv_ms.append(kv.last_ms)
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
    if kv is not None:
        m["keiview_ms"] = {"town": kv_town_ms, "frames": len(kv_ms),
                           "mean": float(np.mean(kv_ms)) if kv_ms else None,
                           "p90": float(np.percentile(kv_ms, 90)) if kv_ms else None}
    m["junction_passes"] = junction_passes(env, log, dt)
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


def junction_passes(env, log, dt, approach=15.0):
    """What happened at each junction of the route: its kind (signalised / priority crossing / bend), the ego's turn,
    whether the expert wanted a stop there ("yield" at a priority crossing, a red light, ...), whether the ego actually
    stood (v < 0.1), for how long, and for how long of that the expert would have driven on (a needless stop).
    s is relative to the route start, like the infractions' s. Routes the ego never reached are listed with
    reached=False."""
    s0 = env.route.s_start
    rows = [r for r in log]
    out = []
    for j in env.route.junctions:
        J = env.world.town.junctions[j["junction"]]
        a, b = j["s_in"] - approach, j["s_out"]
        win = [r for r in rows if a <= r[1] + s0 <= b]
        kind = "signalised" if j["signalized"] else ("priority" if len(J.arms) >= 3 else "bend")
        rec = {"junction": int(j["junction"]), "kind": kind, "arms": len(J.arms), "turn": j["turn"],
               "s_in": round(j["s_in"] - s0, 1), "s_out": round(j["s_out"] - s0, 1),
               "reached": bool(win), "yields": bool(env.world.town.lanes[j["lane"]].yields)}
        if win:
            reasons = {}
            for r in win:
                reasons[r[3]] = reasons.get(r[3], 0) + 1
            stood = [r for r in win if r[2] < 0.1]
            rec.update({"expert_reasons": reasons, "expert_stop": any(r[-1] < 0.3 for r in win),
                        "v_min": round(min(r[2] for r in win), 2), "stood_s": round(len(stood) * dt, 1),
                        "needless_s": round(sum(1 for r in stood if r[-1] > 1.0) * dt, 1),
                        "s_first_stop": None if not stood else round(stood[0][1], 1)})
        out.append(rec)
    return out


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
    ap.add_argument("--keiview_roads", default="plain", choices=["plain", "detail"],
                    help="KeiView road surface: plain (as trained) or detail (markings, signs, ...)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--brake_hold", action="store_true", help="safety layer: no throttle burst right after firm braking")
    ap.add_argument("--red_hold", action="store_true", help="safety layer: no creeping while the model sees red/yellow")
    ap.add_argument("--town_style", default=None, choices=list(TOWN_STYLES),
                    help="default: classic (long2: varied, long3: twophase)")
    ap.add_argument("--blocked_timeout", type=float, default=None, help="s without progress (default 90, long2 180)")
    ap.add_argument("--jam_fixes", type=int, default=None, choices=[0, 1],
                    help="box rule + release of NPCs stuck out of the ego camera's view (default: 0, long2: 1)")
    ap.add_argument("--osm_town", default=None,
                    help="a real town (keisim.realworld.osm_town, cached at data/osm/<name>.json) instead of --suite")
    ap.add_argument("--osm_bbox", type=float, nargs=4, metavar=("S", "W", "N", "E"),
                    default=[43.088, 141.360, 43.097, 141.372], help="--osm_town's box (default: Sapporo Higashi-ku)")
    ap.add_argument("--routes", type=int, default=20, help="--osm_town: number of routes (episode seeds 100..)")
    ap.add_argument("--episode0", type=int, default=100, help="--osm_town: first episode seed")
    args = ap.parse_args()
    if args.osm_town:                   # real towns: mixed road kinds, two-phase signals, the long3 traffic rules
        SUITE_DEFAULTS["osm"] = {"town_style": "twophase", "jam_fixes": True, "blocked_timeout": 180.0}
        args.suite = "osm"
    dflt = {"route_length": 600.0, "max_steps": 4000, "town_style": "classic", "jam_fixes": 0, "blocked_timeout": 90.0,
            **SUITE_DEFAULTS.get(args.suite, {})}
    suite_style, suite_fixes, suite_timeout = dflt["town_style"], int(dflt["jam_fixes"]), dflt["blocked_timeout"]
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
    if args.red_hold:
        tag += "_red"
    if args.osm_town:
        jobs = [(args.osm_town, args.episode0 + e) for e in range(args.routes)]
    else:
        jobs = SUITES[args.suite]
    suffix = ("_dense" if args.dense else "") + ("" if args.weather == "random" else f"_{args.weather}") + \
        ("_keiview" if args.renderer == "keiview" else "") + \
        ("_kvdetail" if args.renderer == "keiview" and args.keiview_roads == "detail" else "") + \
        ("" if args.town_style == suite_style else f"_{args.town_style}") + \
        ("" if int(args.jam_fixes) == suite_fixes else ("_jamfix" if args.jam_fixes else "_nojamfix")) + \
        ("" if args.blocked_timeout == suite_timeout else f"_wait{args.blocked_timeout:.0f}")
    if args.osm_town:
        suffix = suffix.replace("_twophase", "")
        fail_dir = os.path.join("runs", "failures", args.osm_town, f"{tag}{suffix}")
    else:
        fail_dir = os.path.join("runs", "failures", f"{tag}_{args.suite}{suffix}")
    a = {"agent": args.agent, "ckpt": args.ckpt, "route_length": args.route_length, "weather": args.weather,
         "max_steps": args.max_steps, "videos": args.videos, "video_dir": args.video_dir, "tag": tag,
         "dense": args.dense, "renderer": args.renderer, "keiview_roads": args.keiview_roads, "fail_dir": fail_dir,
         "brake_hold": args.brake_hold,
         "red_hold": args.red_hold, "town_style": args.town_style, "jam_fixes": bool(args.jam_fixes),
         "blocked_timeout": args.blocked_timeout, "osm_bbox": args.osm_bbox}
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
    out = args.out or (f"runs/eval/{args.osm_town}/{tag}{suffix}.json" if args.osm_town
                       else f"runs/eval/{tag}_{args.suite}{suffix}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": s, "results": results, "args": vars(args), "time": time.time() - t0}, f, indent=1)
    print("saved", out, f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
