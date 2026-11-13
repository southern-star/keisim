"""How often do the expert's bicycle rules follow a rider the camera does not see?

Drives the expert (EnvConfig.expert_bike_visible on, the Mc rig) and, at every step where bike_rules or the path's
move-over binds, asks the same rules again with only the riders the first camera sees: the frames where the answer
changes are labels that follow an unseen rider. Counted per part: the 巻き込み確認 (left turns), the corridor
(follow / pass slowly / crossing riders), and the path's move-over.

    uv run python tools/bike_unseen.py --style twophase --towns 1010 1020 --episodes 9400 9401 --steps 3000
"""
import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run(job):
    style, town, ep, steps, cw = job
    from keisim.config import EnvConfig, set_camera_rig, town_config
    from keisim.env import KeiEnv
    from keisim.expert import PATH_S
    cfg = set_camera_rig(EnvConfig(), "wide_narrow")
    cfg.town = town_config(style)
    cfg.traffic.box_rule = cfg.traffic.release_hidden = True
    cfg.traffic.bicycles, cfg.traffic.crosswalks = 8.0, cw
    cfg.traffic.max_vehicles *= 2
    cfg.traffic.max_peds *= 2
    cfg.expert_bike_visible = True
    cfg.render_rgb = cfg.render_seg = False
    env = KeiEnv(cfg)
    env.reset(town_seed=town, episode_seed=ep, route_length=2500.0, render=False)
    ex, w = env.expert, env.world
    c = {"steps": 0, "bike_near": 0, "reason_bicycle": 0, "corridor_unseen": 0, "left_unseen": 0,
         "path_unseen": 0, "behind_unseen": 0}
    done, n = False, 0
    while not done and n < steps:
        n += 1
        c["steps"] += 1
        bk = w.bikes
        scan = ex.last_bikes
        if bk is not None and bk.n and scan is not None and len(scan["along"]) == bk.n:
            route, s, pose, v = env.route, env.s_ego, w.ego.pose, w.ego.v
            near = np.hypot(*(bk.xy - w.ego.xy).T) < 40.0
            c["bike_near"] += int(near.any())
            c["reason_bicycle"] += int(env.plan["reason"] == "bicycle")
            c["behind_unseen"] += int(scan["behind_unseen"].any())
            # the same rules with the unseen riders taken away
            unseen = ~scan["visible"]
            if (unseen & near).any():
                saved = bk.xy.copy()
                pts_w = route.poly.interp(np.minimum(s + PATH_S, route.poly.length))
                cum = np.asarray(PATH_S, float)

                def target(scn):
                    p = ex.path_local(route, s, pose, scn)
                    cc, sn = math.cos(pose[2]), math.sin(pose[2])
                    p = np.stack([pose[0] + cc * p[:, 0] - sn * p[:, 1], pose[1] + sn * p[:, 0] + cc * p[:, 1]], -1)
                    wait = ex._bk_wait
                    t = ex.bike_rules(route, s, v, scn, p, cum, False)
                    ex._bk_wait = wait
                    return t, p
                t_all, p_all = target(scan)
                bk.xy[unseen] = 1e6
                scan2 = ex.bike_scan(route, s, pose, v)
                circles = bk.circles

                def seen_circles(predict=False):     # the predicted circles of the unseen riders go too
                    cxy, cr, own, csp, cyw = circles(predict=predict)
                    m = ~unseen[own]
                    return cxy[m], cr[m], own[m], csp[m], cyw[m]
                bk.circles = seen_circles
                try:
                    t_vis, p_vis = target(scan2)
                finally:
                    bk.circles = circles
                    bk.xy[:] = saved
                if min(t_all, 30.0) < min(t_vis, 30.0) - 0.3:
                    # which part: the 巻き込み確認 only stops at the stop line before a left turn
                    left = any(0 <= j["s_in"] - s < 30 and w.town.lanes[j["lane"]].turn == "left"
                               for j in route.junctions) and v < 3.0
                    c["left_unseen" if left else "corridor_unseen"] += 1
                if np.abs(p_all - p_vis).max() > 0.1:
                    c["path_unseen"] += 1
        _, _, done, _ = env.step(env.expert_action())
    c["status"] = env.status
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="twophase")
    ap.add_argument("--towns", type=int, nargs=2, default=(1010, 1020))
    ap.add_argument("--episodes", type=int, nargs="+", default=[9400])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--crosswalks", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(a.style, t, e, a.steps, a.crosswalks) for t in range(*a.towns) for e in a.episodes]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, jobs))
    tot = {k: sum(r[k] for r in res) for k in res[0] if k != "status"}
    st = max(tot["steps"], 1)
    print(f"{a.style}: {len(jobs)} episodes, {tot['steps']} steps; statuses {[r['status'] for r in res]}")
    for k, v in tot.items():
        print(f"  {k:16s} {v:6d}  ({v / st:.2%} of the steps)")


if __name__ == "__main__":
    main()
