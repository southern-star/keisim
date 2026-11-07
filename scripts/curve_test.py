"""Curve test: does a driving model cut the inside of curves where the road gives few cues (no white lines, a
guardrail)? Drives the curve courses of keisim/curve_course.py (one open road of 90 deg left and right curves of
radius 15-150 m) end to end, both ways, without other cars or pedestrians, and measures the car's lateral offset from
the centre of its lane on every curve.

    python scripts/curve_test.py --agent model --ckpt runs/keipilot_v10.pt --brake_hold --red_hold
    python scripts/curve_test.py --agent expert --variants lines nolines
    python scripts/curve_test.py --agent model --ckpt runs/keipilot_v10.pt --renderer keiview --variants nolines
    python scripts/curve_test.py ... --video runs/videos/curve        (one mp4 per run)

Offsets are + to the inside of the curve (left on a left curve). Per curve: the mean and the largest inside offset,
the offset at the curve's end minus the one at its start ("grows": cutting in more and more), and the time off the
road surface. Per run: the end (finished, route_deviation at 6 m, a collision) and the offsets on the straights.
Results: runs/eval/curve/<tag>.json (and a table on stdout).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from keisim.config import EnvConfig  # noqa: E402
from keisim.curve_course import COURSE_VERSION, VARIANT_NAMES, VARIANTS, course_route, curve_town  # noqa: E402
from keisim.env import KeiEnv  # noqa: E402
from keisim.geometry import world_to_local  # noqa: E402

TP_AHEAD = 50.0      # m: --tp ahead puts the model's target point this far ahead along the route


def run_one(job):
    variant, reverse, a = job
    cfg = EnvConfig()
    cfg.traffic.max_vehicles = 0
    cfg.traffic.max_peds = 0
    cfg.render_seg = False
    cfg.renderer = a["renderer"]
    cfg.camera.guardrail_light = a.get("guardrail_light") or "sun"
    cfg.render_rgb = a["agent"] == "model" or bool(a["video"])
    cfg.terminate_on_collision = True
    env = KeiEnv(cfg)
    town = curve_town(variant)
    route, curves = course_route(town, reverse=reverse, start=a.get("from_s"))
    obs = env.reset(town_seed=town, episode_seed=a["episode"], weather=a["weather"], route=route)
    agent = None
    if a["agent"] == "model":
        import torch
        torch.set_num_threads(a["threads"])       # the workers share the CPU (one thread each by default)
        from keipilot.agent import KeiPilotAgent
        agent = KeiPilotAgent(a["ckpt"], brake_hold=a["brake_hold"], red_hold=a["red_hold"])
        agent.reset()
    vw = None
    if a["video"]:
        from keisim.viz import VideoWriter, compose_dashboard, info_lines
        os.makedirs(a["video"], exist_ok=True)
        vw = VideoWriter(os.path.join(a["video"], f"{variant}_{'rev' if reverse else 'fwd'}.mp4"), fps=10)
    dt = cfg.dt * cfg.action_repeat
    log = []                                   # (route s, lateral + left, speed, on the road surface)
    done = False
    info = {"status": "running"}
    while not done:
        e = env.world.ego
        s = env.s_ego
        if agent is None:
            action = env.expert_action()
            plan_path, target = env.plan["path"], env.plan["target_speed"]
        else:
            tp = obs["target_point"]
            if a["tp"] == "ahead":
                tp = world_to_local(route.poly.interp(min(s + TP_AHEAD, route.s_end))[None], e.x, e.y, e.yaw)[0]
            action, pinfo = agent.act(obs["rgb"], obs["command"], tp, e.v, dt)
            plan_path, target = pinfo["path"], pinfo["target_speed"]
        p = route.poly.interp1(s)
        h = route.poly.heading1(s)
        lat = float(-(e.x - p[0]) * math.sin(h) + (e.y - p[1]) * math.cos(h))
        on_road = bool(env.world.town.is_drivable(np.array([[e.x, e.y]]))[0])
        log.append((float(s), lat, float(e.v), on_road))
        if vw is not None:
            img = obs["rgb"]
            paths = [(plan_path, (255, 80, 255))]
            lines = info_lines(e.v, target, obs["command"]) + [f"{variant} {'reverse' if reverse else 'forward'}",
                                                               f"offset {lat:+.2f} m (+ left)", f"s {s:6.1f} m"]
            vw.write(compose_dashboard(np.ascontiguousarray(img), env.render_bev(paths=paths),
                                       np.zeros((160, 320, 3), np.uint8), lines, cam=env.assets.camera, pose=e.pose,
                                       paths=paths, title=f"curve test  {variant}  {a['agent']}"))
        obs, _, done, info = env.step(action)
        if not done and env.s_ego >= route.s_end - 3.0:      # at the end (the car stops just short of it)
            done, info = True, {"status": "finished"}
        if not done and a.get("max_s") and env.s_ego >= a["max_s"]:
            done, info = True, {"status": "max_s"}
    if vw is not None:
        vw.close()
    status = info["status"]
    L = np.array(log)
    res = {"variant": variant, "reverse": reverse, "status": status, "s_end": float(L[-1, 0]),
           "route_end": float(route.s_end), "curves": [], "infractions": [{k: i[k] for k in ("type", "s")} for i in env.infractions]}
    in_curve = np.zeros(len(L), bool)
    for c in curves:
        m = (L[:, 0] >= c["s0"]) & (L[:, 0] <= c["s1"])
        in_curve |= m
        rec = {k: round(v, 2) if isinstance(v, float) else v for k, v in c.items()}
        if m.sum() < 3:
            rec["reached"] = False
            res["curves"].append(rec)
            continue
        sign = 1.0 if c["turn"] == "left" else -1.0
        inside = sign * L[m, 1]
        rec.update({"reached": True, "mean_in": round(float(inside.mean()), 3), "max_in": round(float(inside.max()), 3),
                    "min_in": round(float(inside.min()), 3), "grows": round(float(inside[-1] - inside[0]), 3),
                    "speed": round(float(L[m, 2].mean()), 2),
                    "off_road_s": round(float((~L[m, 3].astype(bool)).sum() * dt), 2)})
        res["curves"].append(rec)
    st = ~in_curve & (L[:, 2] > 1.0)
    res["straight"] = {"mean_left": round(float(L[st, 1].mean()), 3) if st.any() else None,
                       "p90_abs": round(float(np.percentile(np.abs(L[st, 1]), 90)), 3) if st.any() else None}
    res["trace"] = [[round(x, 2) for x in row[:3]] for row in L[::5].tolist()]
    env.close()
    return res


def table(results):
    rows = []
    for r in results:
        for c in r["curves"]:
            if not c.get("reached"):
                rows.append(f"{r['variant']:20s} {'rev' if r['reverse'] else 'fwd'} {c['turn']:5s} R{c['radius']:5.1f}  "
                            f"(not reached: {r['status']})")
                continue
            rows.append(f"{r['variant']:20s} {'rev' if r['reverse'] else 'fwd'} {c['turn']:5s} R{c['radius']:5.1f}  "
                        f"mean {c['mean_in']:+5.2f}  max {c['max_in']:+5.2f}  grows {c['grows']:+5.2f}  "
                        f"{c['speed'] * 3.6:4.1f} km/h  off road {c['off_road_s']:.1f} s")
        rows.append(f"{r['variant']:20s} {'rev' if r['reverse'] else 'fwd'} end: {r['status']} at {r['s_end']:.0f} / "
                    f"{r['route_end']:.0f} m, straights mean {r['straight']['mean_left']:+.2f} m (+ left)")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent", default="model", choices=["expert", "model"])
    ap.add_argument("--ckpt", default="runs/keipilot_v10.pt")
    ap.add_argument("--brake_hold", action="store_true")
    ap.add_argument("--red_hold", action="store_true")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANT_NAMES,
                    help="default: the five with buildings; <variant>_nobld: without (keisim/curve_course.py)")
    ap.add_argument("--directions", nargs="+", default=["fwd", "rev"], choices=["fwd", "rev"])
    ap.add_argument("--renderer", default="keisim", choices=["keisim", "keiview"])
    ap.add_argument("--weather", default="noon")
    ap.add_argument("--episode", type=int, default=0, help="episode seed (lighting of KeiView, weather 'random')")
    ap.add_argument("--tp", default="route", choices=["route", "ahead"],
                    help="model's target point: KeiSim's (the route's end: no junction on the course) or 50 m ahead")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=None, help="torch threads per worker (default: CPUs / workers)")
    ap.add_argument("--video", default=None, help="directory for one mp4 per run")
    ap.add_argument("--from_s", type=float, default=None, help="start at this route s (m) instead of the course's end")
    ap.add_argument("--max_s", type=float, default=None, help="stop after this many metres of the route (KeiView without a "
                    "GPU: about 2 s per frame)")
    ap.add_argument("--guardrail_light", default="sun", choices=["sun", "flat"],
                    help="KeiSim renderer: guardrails shaded by the sun (default) or the same whatever its azimuth")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    a = vars(args)
    tag = args.tag or (f"{args.agent}" + (f"_{os.path.splitext(os.path.basename(args.ckpt))[0]}" if args.agent == "model" else "")
                       + f"_{args.renderer}_{args.weather}" + (f"_tp{args.tp}" if args.tp != "route" else "")
                       + (f"_from{args.from_s:.0f}" if args.from_s else "") + (f"_{args.max_s:.0f}m" if args.max_s else "")
                       + (f"_ep{args.episode}" if args.episode else "")
                       + ("_grflat" if args.guardrail_light == "flat" else "") + f"_{COURSE_VERSION}")
    # every run is saved when it ends (runs/eval/curve/<tag>/), and a run already saved there is not run again
    part_dir = f"runs/eval/curve/{tag}"
    os.makedirs(part_dir, exist_ok=True)
    part = lambda v, rev: os.path.join(part_dir, f"{v}_{'rev' if rev else 'fwd'}.json")   # noqa: E731
    jobs = [(v, d == "rev", a) for v in args.variants for d in args.directions]
    todo = [j for j in jobs if not os.path.exists(part(j[0], j[1]))]
    workers = 1 if args.renderer == "keiview" else max(1, min(args.workers, len(todo)))
    a["threads"] = args.threads or max(1, (os.cpu_count() or 1) // workers)

    def save(r):
        with open(part(r["variant"], r["reverse"]), "w") as f:
            json.dump(r, f)
        print(f"done {r['variant']} {'rev' if r['reverse'] else 'fwd'}: {r['status']}", flush=True)

    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:      # (fork deadlocks torch's thread pool)
            for r in pool.imap_unordered(run_one, todo):
                save(r)
    else:
        for j in todo:
            save(run_one(j))
    results = []
    for v, rev, _ in jobs:
        with open(part(v, rev)) as f:
            results.append(json.load(f))
    print(table(results))
    out = f"runs/eval/curve/{tag}.json"
    with open(out, "w") as f:
        json.dump({"args": {k: v for k, v in a.items()}, "results": results}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
