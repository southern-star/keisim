"""Hash the towns, routes and expert episodes of a fixed set of cases, to check that a change leaves them as they
were (run it on main and on the branch, compare the lines).

    uv run python tools/episode_hash.py [--steps 600] [--only classic suburban ...] [--set key=value ...]

Each case hashes its town (every lane's points, kind, turn and links; the signals), the route (lanes, junction
commands) and, for --steps steps of the expert, the ego pose, speed, command, target point, the plan's path, target
speed, reason and light, and every NPC car, person and bicycle. --set k=v sets an EnvConfig field (e.g.
expert_bike_visible=True) for every case.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# name: (town style or OSM town, town seed, episode seed, route length, jam fixes, crosswalks, bicycles)
CASES = {
    "classic": ("classic", 1000, 0, 600.0, False, 0.0, 0.0),
    "varied": ("varied", 1010, 9200, 600.0, True, 0.0, 0.0),
    "twophase": ("twophase", 1010, 9400, 800.0, True, 0.0, 0.0),
    "roadtypes": ("roadtypes", 1105, 3, 800.0, True, 0.0, 0.0),
    "suburban": ("suburban", 2001, 9800, 800.0, True, 0.0, 0.0),
    "ramps": ("ramps", 2000, 9900, 800.0, True, 0.0, 0.0),
    "bigjunction": ("bigjunction", 3000, 9800, 800.0, True, 0.7, 0.0),
    "bigjunction_3009": ("bigjunction", 3009, 9801, 800.0, True, 0.7, 0.0),
    "twophase_cw_bk": ("twophase", 1012, 9401, 800.0, True, 0.7, 8.0),
    "suburban_cw_bk": ("suburban", 2003, 9850, 800.0, True, 0.7, 8.0),
    "twophase_bk": ("twophase", 1015, 9400, 800.0, True, 0.0, 8.0),
    "osm_nishioi": ("osm_nishioi", None, 100, 600.0, True, 0.0, 0.0),
    "osm_sapporo": ("osm_sapporo", None, 100, 600.0, True, 0.0, 0.0),
}


def h_arr(m, a, nd=5):
    m.update(np.round(np.asarray(a, float), nd).tobytes())


def town_hash(T):
    m = hashlib.sha1()
    for L in T.lanes:
        h_arr(m, L.pts)
        m.update(repr((L.kind, L.turn, list(L.succ), list(L.pred), getattr(L, "yields", None),
                       L.stop_s if L.stop_s is None else round(float(L.stop_s), 5))).encode())
    for J in T.junctions:
        m.update(repr((J.signalized, getattr(J, "phases", None), round(float(getattr(J, "green", 0.0)), 6),
                       round(float(getattr(J, "offset", 0.0)), 6), getattr(J, "groups", None))).encode())
    return m.hexdigest()[:12]


def run_case(name, steps, sets):
    from keisim.config import EnvConfig, set_camera_rig, town_config
    from keisim.env import KeiEnv
    style, seed, ep, length, fixes, cw, bk = CASES[name]
    cfg = set_camera_rig(EnvConfig(), "wide_narrow")
    town = seed
    if style.startswith("osm_"):
        from keisim.realworld import OSM_BOXES, osm_town
        town = osm_town(tuple(OSM_BOXES[style]), cache=f"data/osm/{style}.json", name=style)
        cfg.town = town.cfg
    else:
        cfg.town = town_config(style)
    cfg.traffic.box_rule = cfg.traffic.release_hidden = fixes
    cfg.traffic.crosswalks, cfg.traffic.bicycles = cw, bk
    cfg.render_rgb = cfg.render_seg = False
    if style != "classic":
        cfg.traffic.max_vehicles *= 2
        cfg.traffic.max_peds *= 2
    for k, v in sets.items():
        setattr(cfg, k, v)
    env = KeiEnv(cfg)
    obs = env.reset(town_seed=town, episode_seed=ep, route_length=length, render=False)
    th = town_hash(env.world.town)
    m = hashlib.sha1()
    R = env.route
    m.update(repr((R.lanes, [(j["turn"], j["lane"]) for j in R.junctions])).encode())
    w = env.world
    done, n = False, 0
    while not done and n < steps:
        p = obs["expert"]
        h_arr(m, w.ego.pose)
        h_arr(m, [w.ego.v, p["target_speed"], p["tl_state"]])
        h_arr(m, obs["target_point"])
        h_arr(m, p["path"])
        m.update(repr((obs["command"], p["reason"])).encode())
        h_arr(m, w.traffic.xy)
        h_arr(m, w.peds.xy)
        if getattr(w, "bikes", None) is not None and getattr(w.bikes, "n", 0):
            h_arr(m, w.bikes.xy)
        obs, _, done, info = env.step(env.expert_action())
        n += 1
    return th, m.hexdigest()[:12], n, env.status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="EnvConfig field=value (Python literal) for every case")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    import ast
    sets = {k: ast.literal_eval(v) for k, v in (s.split("=", 1) for s in args.set)}
    names = args.only or list(CASES)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(run_case, names, [args.steps] * len(names), [sets] * len(names)))
    for name, (th, eh, n, st) in zip(names, res):
        print(f"{name:18s} town {th}  episode {eh}  steps {n:5d}  {st}")


if __name__ == "__main__":
    main()
