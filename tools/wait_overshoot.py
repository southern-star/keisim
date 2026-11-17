"""Where the expert stops when it gives way at a right turn, against the Town's wait point (Lane.wait_s), with its
throttle and brake late (evaluate.py --expert_delay) and its wait point moved back (--expert_wait_back): the stops
past the Town's point, whether an oncoming car then stands within 8 m of the ego's front (a standoff in the making),
and how long until the ego moves on.

    uv run python tools/wait_overshoot.py --suite suburban --delay 0.4 --wait_back 0 [--routes 20] [--workers 1]
"""
import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def run(job):
    town, ep, suite, delay, wb, max_steps = job
    from evaluate import CAP_SCALE, SUITE_DEFAULTS, ActionDelay
    from keisim.config import EnvConfig, town_config
    from keisim.env import KeiEnv
    d = SUITE_DEFAULTS[suite]
    cfg = EnvConfig()
    cfg.town = town_config(d["town_style"])
    cfg.traffic.box_rule = cfg.traffic.release_hidden = bool(d.get("jam_fixes"))
    cfg.blocked_timeout = float(d.get("blocked_timeout", 90.0))
    cfg.render_rgb = cfg.render_seg = False
    cfg.expert_wait_back = wb
    cap = CAP_SCALE.get(d["town_style"], 1)
    cfg.traffic.max_vehicles *= cap
    cfg.traffic.max_peds *= cap
    env = KeiEnv(cfg)
    env.reset(town_seed=town, episode_seed=ep, route_length=float(d["route_length"]), render=False,
              via_ramp=bool(d.get("via_ramp", False)))
    dt = cfg.dt * cfg.action_repeat
    act = ActionDelay(round(delay / dt))
    R, T, w = env.route, env.world.town, env.world
    stops = []            # (past the Town's wait point m, oncoming car standing within 8 m, s until moving on, reason)
    cur = None
    done, n = False, 0
    while not done and n < max_steps:
        n += 1
        k = R.lane_index_at(env.s_ego)
        L = T.lanes[R.lanes[k]]
        if L.kind == "conn" and L.yields and L.wait_s is not None:
            front = R.local_s(env.s_ego, k) + w.ego.LENGTH / 2
            if w.ego.v < 0.05 and cur is None:
                e = w.ego
                c, s_ = math.cos(e.yaw), math.sin(e.yaw)
                tr = w.traffic
                loc = (tr.xy - e.xy) @ np.array([[c, -s_], [s_, c]])
                facing = (loc[:, 0] > 0) & (loc[:, 0] < e.LENGTH / 2 + 8.0) & (np.abs(loc[:, 1]) < 3.0) & (tr.v < 0.3)
                cur = [front - L.wait_s, bool(facing.any()), w.t, env.plan["reason"]]
            elif w.ego.v > 0.5 and cur is not None:
                stops.append((cur[0], cur[1], w.t - cur[2], cur[3]))
                cur = None
        elif cur is not None:
            stops.append((cur[0], cur[1], w.t - cur[2], cur[3]))
            cur = None
        _, _, done, _ = env.step(act(env.expert_action()))
    if cur is not None:
        stops.append((cur[0], cur[1], w.t - cur[2], cur[3]))
    return env.status, stops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="suburban")
    ap.add_argument("--delay", type=float, default=0.4)
    ap.add_argument("--wait_back", type=float, default=0.0)
    ap.add_argument("--routes", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    from evaluate import SUITE_DEFAULTS, SUITES
    jobs = [(t, e, a.suite, a.delay, a.wait_back, int(SUITE_DEFAULTS[a.suite]["max_steps"]))
            for t, e in SUITES[a.suite][:a.routes]]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, jobs))
    allstops = [s for _, ss in res for s in ss]
    stops = [s for s in allstops if s[3] == "yield"]
    past = [s for s in stops if s[0] > 0.5]
    print(f"{a.suite} delay {a.delay:g} wait_back {a.wait_back:g}: {len(jobs)} routes, statuses "
          f"{sorted(set(r for r, _ in res))}")
    print(f"  stops on a yielding connector: {len(allstops)} (reasons {sorted(set(s[3] for s in allstops))}); "
          f"giving way (reason yield): {len(stops)}; past the Town's wait point by more than 0.5 m: {len(past)}")
    if stops:
        x = np.array([s[0] for s in stops])
        print(f"  stop position against the Town's wait point: median {np.median(x):+.2f} m, max {x.max():+.2f} m")
    for p in past:
        print(f"    past by {p[0]:.2f} m, oncoming car standing in front {p[1]}, moved on after {p[2]:.1f} s")


if __name__ == "__main__":
    main()
