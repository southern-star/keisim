"""Open-loop red-light false positives next to a vehicle (KeiPilot v0.8 check). The first checkpoint drives a suite
with dense traffic (safety layers on, as scripts/evaluate.py --dense); at every step where the true light is none and
a vehicle leads the expert's corridor within --gap m, the same camera frame goes through every checkpoint.

Per frame it also records the route's next signal: the expert labels a light only within 45 m of its stop line, so a
"none" frame can still show the ego's own signal further on (a red there is then read correctly, not a false
positive). And (KeiSim renderer) whenever a checkpoint says red, the frame is rendered again with every vehicle's
tail lamps dark and judged again: a red that goes away is a tail-lamp confusion.

Scores per checkpoint and frame group (all; no own signal within --sig_range m ahead in view; own signal 45 m+
ahead in view; lead partly outside the view or on a sharp curve, scripts/collect.py --focus_lead_edge): share of
frames classified red, mean P(red), and the share of the reds that disappear without tail lamps.

usage: red_fp.py OUT.json WORKERS --ckpts CKPT [CKPT ...] [--suite long3] [--gap 8] [--sig_range 110] [--routes N]
       [--renderer keisim|keiview]
"""
import argparse
import json
import math
import os
import sys
from contextlib import contextmanager

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")


@contextmanager
def tail_lamps_dark():
    """KeiSim renderer: every vehicle's tail lamps drawn as dark trim (not red, not emissive) while inside."""
    import numpy as np
    from keisim.render.meshes import CAR_T, R_DARK, R_TAIL, TRUCK_T
    saved = [(T, T["role"]) for T in (CAR_T, TRUCK_T)]
    for T, role in saved:
        T["role"] = np.where(role == R_TAIL, R_DARK, role)
    try:
        yield
    finally:
        for T, role in saved:
            T["role"] = role


def next_signal(env, rng):
    """The route's next stop line with a signal ahead of the ego's front: (distance m, state, stop line in the
    camera's horizontal FOV), or (inf, None, False) when there is none within rng m."""
    from collect import corners_outside_fov
    w = env.world
    s_front = env.s_ego + w.ego.LENGTH / 2
    for s_stop, lid in env.route.stops:
        d = s_stop - s_front
        if d < -3.0:
            continue
        if d > rng:
            break
        p = env.route.poly.interp(min(s_stop, env.route.poly.length))
        e = w.ego
        # a 1 m x 3 m "vehicle" at the stop line stands in for the signal heads beside it
        out = corners_outside_fov(env.cfg.camera, (e.x, e.y, e.yaw), p, e.yaw, 0.05, 1.5, height=3.0)
        return d, int(w.town.signal_state_for_lane(lid, w.t)), not out
    return math.inf, None, False


def run(job):
    (town, ep), a = job
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    from collect import lead_at_edge
    from keipilot.agent import KeiPilotAgent
    from keisim.config import TL_NONE, TL_RED, EnvConfig, town_config
    from keisim.env import KeiEnv
    sd = a["sd"]
    style = sd.get("town_style", "classic")
    cfg = EnvConfig()
    cfg.town = town_config(style)
    cfg.traffic.box_rule = cfg.traffic.release_hidden = bool(sd.get("jam_fixes", False))
    cfg.blocked_timeout = float(sd.get("blocked_timeout", 90.0))
    cfg.render_seg = False
    cfg.render_rgb = True
    cfg.renderer = a["renderer"]
    cfg.traffic.vehicle_spacing = (20.0, 28.0)
    cfg.traffic.ped_spacing = (14.0, 20.0)
    cfg.traffic.max_vehicles = 130
    cfg.traffic.max_peds = 110
    cfg.traffic.ego_cross_rate = 0.3
    cap = {"classic": 1, "varied": 2, "twophase": 2, "roadtypes": 2}[style]
    cfg.traffic.max_vehicles *= cap
    cfg.traffic.max_peds *= cap
    env = KeiEnv(cfg)
    driver = KeiPilotAgent(a["ckpts"][0], brake_hold=True, red_hold=True)
    scorers = [driver] + [KeiPilotAgent(c) for c in a["ckpts"][1:]]
    driver.reset()
    obs = env.reset(town_seed=town, episode_seed=ep, route_length=sd.get("route_length", 600.0), weather="random")
    dt = cfg.dt * cfg.action_repeat
    rows, done, steps = [], False, 0
    while not done and steps < sd.get("max_steps", 4000):
        e, plan = env.world.ego, env.plan
        if plan["tl_state"] == TL_NONE and plan.get("lead_owner", -1) >= 0 and plan["gap"] < a["gap"]:
            edge = lead_at_edge(env, plan, a["gap"])
            sig_d, sig_st, sig_view = next_signal(env, a["sig_range"])
            ps = []
            for ag in scorers:
                p = ag.plan(obs["rgb"], obs["command"], obs["target_point"], speed=e.v)
                ps.append([int(p["tl"] == TL_RED), round(float(p["tl_probs"][TL_RED]), 4), -1])
            if cfg.renderer == "keisim" and any(x[0] for x in ps):
                with tail_lamps_dark():
                    rgb_dark, _ = env.render_camera(want_seg=False)
                for ag, x in zip(scorers, ps):
                    if x[0]:
                        p = ag.plan(rgb_dark, obs["command"], obs["target_point"], speed=e.v)
                        x[2] = int(p["tl"] == TL_RED)
            rows.append([round(env.elapsed, 1), round(float(plan["gap"]), 2), int(edge),
                         None if not math.isfinite(sig_d) else round(sig_d, 1), sig_st, int(sig_view), ps])
        action, _ = driver.act(obs["rgb"], obs["command"], obs["target_point"], e.v, dt)
        obs, _, done, _ = env.step(action)
        steps += 1
    m = env.metrics()
    env.close()
    return {"town": town, "episode": ep, "status": m["status"], "rows": rows}


def summary(results, ckpts):
    import numpy as np
    groups = (
        ("all", lambda r: True),
        ("no own signal ahead in view", lambda r: not r[5]),
        ("own signal 45 m+ ahead in view", lambda r: bool(r[5])),
        ("  of which red / yellow", lambda r: bool(r[5]) and r[4] in (0, 1)),
        ("lead at the edge / curve", lambda r: r[2] == 1),
    )
    out = {}
    for name, sel in groups:
        R = [r for x in results for r in x["rows"] if sel(r)]
        out[name] = {"frames": len(R)}
        for i, c in enumerate(ckpts):
            red = np.array([r[6][i][0] for r in R], bool) if R else np.zeros(0, bool)
            pr = np.array([r[6][i][1] for r in R]) if R else np.zeros(0)
            dark = np.array([r[6][i][2] for r in R]) if R else np.zeros(0)
            gone = int(((dark == 0) & red).sum())
            out[name][c] = {"red_share": float(red.mean()) if len(R) else None, "mean_p_red": float(pr.mean()) if len(R) else None,
                            "red_frames": int(red.sum()), "red_gone_without_tail_lamps": gone}
    return out


if __name__ == "__main__":
    import evaluate as ev
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("workers", type=int)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--suite", default="long3")
    ap.add_argument("--gap", type=float, default=8.0)
    ap.add_argument("--sig_range", type=float, default=110.0, help="look this far ahead for the route's next signal (m)")
    ap.add_argument("--renderer", default="keisim", choices=["keisim", "keiview"])
    ap.add_argument("--routes", type=int, default=999, help="only the first N routes of the suite")
    args = ap.parse_args()
    import multiprocessing as mp
    a = {"sd": ev.SUITE_DEFAULTS.get(args.suite, {}), "ckpts": args.ckpts, "gap": args.gap, "renderer": args.renderer,
         "sig_range": args.sig_range}
    jobs = [(j, a) for j in ev.SUITES[args.suite][:args.routes]]
    results = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for r in pool.imap_unordered(run, jobs):
            results.append(r)
            print(r["town"], r["episode"], r["status"], len(r["rows"]), flush=True)
    s = summary(results, args.ckpts)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"args": vars(args), "summary": s, "routes": results}, open(args.out, "w"))
    print(json.dumps(s, indent=1))
