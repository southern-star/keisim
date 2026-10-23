#!/usr/bin/env python3
"""Collect training data in KeiSim.

mode=expert : the privileged expert drives, with DART-style steering/throttle
              noise injection (labels stay the expert's corrective plan) and
              virtual camera-rig perturbations (render from a displaced rig,
              re-express labels in that frame).
mode=dagger : a trained KeiPilot drives (on-policy states), the expert labels.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keisim.config import TL_NONE, TL_RED, EnvConfig  # noqa: E402
from keisim.env import KeiEnv  # noqa: E402
from keipilot.data import ShardWriter, write_meta  # noqa: E402
from keisim.config import TOWN_STYLES, town_config  # noqa: E402
from keisim.expert import LABEL_VERSION  # noqa: E402

REASONS = {"cruise": 0, "curve": 1, "vehicle": 2, "pedestrian": 3, "red_light": 4, "junction_blocked": 5, "yield": 6,
           "crossing": 7, "stop_sign": 8}
WEATHERS = {"noon": 0, "cloudy": 1, "sunset": 2, "fog": 3}


class Perturbation:
    """Occasional smooth steering bumps and throttle/brake glitches."""

    def __init__(self, rng, steer_rate=1 / 5.0, lon_rate=1 / 14.0):
        self.rng = rng
        self.steer_rate, self.lon_rate = steer_rate, lon_rate
        self.s_t = self.s_T = self.s_A = 0.0
        self.s_on = False
        self.l_t = self.l_T = 0.0
        self.l_on = False
        self.l_mode = 0

    def __call__(self, a, dt):
        a = np.array(a, float)
        rng = self.rng
        if not self.s_on and rng.random() < self.steer_rate * dt:
            self.s_on, self.s_t = True, 0.0
            self.s_T = rng.uniform(0.6, 1.8)
            self.s_A = rng.uniform(0.08, 0.38) * rng.choice([-1, 1])
        if self.s_on:
            a[0] = np.clip(a[0] + self.s_A * math.sin(math.pi * self.s_t / self.s_T), -1, 1)
            self.s_t += dt
            self.s_on = self.s_t < self.s_T
        if not self.l_on and rng.random() < self.lon_rate * dt:
            self.l_on, self.l_t = True, 0.0
            self.l_T = rng.uniform(0.5, 1.5)
            self.l_mode = int(rng.integers(2))
        if self.l_on:
            if self.l_mode == 0:     # coast
                a[1], a[2] = 0.0, 0.0
            else:                    # extra throttle
                a[1], a[2] = max(a[1], 0.5), 0.0
            self.l_t += dt
            self.l_on = self.l_t < self.l_T
        return a


def random_offset(rng, virtual_prob):
    """Virtual camera displacement (lateral m, yaw rad, pitch) or None."""
    if rng.random() < virtual_prob:
        return (float(rng.uniform(-1.2, 1.2)), float(np.radians(rng.uniform(-9, 9))), 0.0)
    return None


LEAD_TURN_DEG = 30.0          # --focus_lead_edge: a route turning this much over the next 15 m is a curve


def route_turn_deg(path, ahead=15.0):
    """Heading change of the expert's path (local waypoints every 2 m from 2 m) over the next `ahead` m, measured as
    in tools/fov_log.py; the repeated last waypoint at the end of a route is skipped."""
    seg = np.diff(np.asarray(path, float)[:, :2], axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    seg, ln = seg[ln > 0.1], ln[ln > 0.1]
    if not len(seg):
        return 0.0
    hd = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    cum = np.concatenate([[0.0], np.cumsum(ln)])[:-1]
    return float(np.degrees(hd[min(int(np.searchsorted(cum, ahead)), len(hd) - 1)] - hd[0]))


def corners_outside_fov(cam, ego_pose, lead_xy, lead_yaw, half_l, half_w, height=0.7):
    """True when a corner of a vehicle's footprint, at `height` (tail lights), projects outside the camera's
    horizontal FOV or lies beside or behind the camera. Pinhole at the camera mount (cam.x, cam.y, cam.z), pitched
    down by cam.pitch_deg, as in tools/fov_log.py (it matches KeiSim's segmentation of the vehicle to about a pixel)."""
    x, y, yaw = ego_pose
    c, s = math.cos(yaw), math.sin(yaw)
    cx, cy = x + cam.x * c - cam.y * s, y + cam.x * s + cam.y * c
    sp, cp = math.sin(math.radians(cam.pitch_deg)), math.cos(math.radians(cam.pitch_deg))
    lim = math.tan(math.radians(cam.fov_deg) / 2)
    oc, os_ = math.cos(lead_yaw), math.sin(lead_yaw)
    for a, b in ((half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)):
        dx = lead_xy[0] + a * oc - b * os_ - cx
        dy = lead_xy[1] + a * os_ + b * oc - cy
        fwd, left = dx * c + dy * s, -dx * s + dy * c
        depth = fwd * cp - (height - cam.z) * sp
        if depth <= 0.05 or abs(left / depth) > lim:
            return True
    return False


def lead_at_edge(env, plan, max_gap):
    """--focus_lead_edge: a vehicle leads the expert within max_gap m, and it is partly outside the camera's view or
    the route turns LEAD_TURN_DEG or more over the next 15 m (where KeiPilot v0.7 followed too closely)."""
    o = plan.get("lead_owner", -1)
    if o < 0 or not plan["gap"] < max_gap:
        return False
    tr, e = env.world.traffic, env.world.ego
    if corners_outside_fov(env.cfg.camera, (e.x, e.y, e.yaw), tr.xy[o], tr.yaw[o], tr.dims[o, 0] / 2, tr.dims[o, 1] / 2):
        return True
    return abs(route_turn_deg(plan["path"])) >= LEAD_TURN_DEG


# real towns (keisim/realworld.py, scripts/import_osm.py): name -> box (south, west, north, east); the Overpass answer is
# cached at data/osm/<name>.json. osm_sapporo is the evaluation town (evaluate.py --osm_town, episodes 100-119).
OSM_BOXES = {
    "osm_sapporo": (43.088, 141.360, 43.097, 141.372),
    "osm_sapporo_n": (43.097, 141.360, 43.106, 141.372),
    "osm_sapporo_e": (43.088, 141.372, 43.097, 141.384),
    "osm_sapporo_s": (43.079, 141.360, 43.088, 141.372),
}


def osm_spec(spec):
    """--osm_towns entry: NAME (a box of OSM_BOXES) or NAME:S,W,N,E."""
    name, _, box = spec.partition(":")
    return name, (tuple(float(x) for x in box.split(",")) if box else OSM_BOXES[name])


# per-worker counts (stats_wNN.json, saved with every shard so a resumed run keeps them): every recording step
# (steps_*) and the kept frames
STAT_KEYS = ("steps", "steps_focus", "steps_lead_edge", "frames", "focus", "lead_edge", "lead_edge_vehicle",
             "lead_edge_tl_none", "lead_edge_tl_none_red")


def record(env, writer, rng, v_prob, weather_id, virtual_prob, offset=False, rgb=None, seg=None, prev=None):
    """offset=False: draw a random virtual camera for this frame; else use the given one (None = real camera).
    rgb/seg: an already rendered frame for that camera; prev: its frame `--history` s earlier."""
    plan = env.plan
    e = env.world.ego
    if offset is False:
        offset = random_offset(rng, virtual_prob)
    if rgb is None:
        rgb, seg = env.render_camera(offset=offset)
    if offset is None:
        path, tp = plan["path"], env.target_point()
    else:
        vp = env.virtual_pose(offset)
        path = env.expert.path_local(env.route, env.s_ego, vp)
        tp = env.target_point(vp)
    writer.add(rgb, seg, prev=prev, cmd=np.int8(env.command()), tp=np.asarray(tp, np.float32),
               path=np.asarray(path, np.float32), speed=np.float32(plan["target_speed"]),
               tl=np.int8(plan["tl_state"]), v=np.float32(e.v), reason=np.int8(REASONS.get(plan["reason"], 0)),
               town=np.int32(env.town_seed), episode=np.int64(env.episode_seed), step=np.int32(env._step),
               weather=np.int8(weather_id), virtual=np.bool_(offset is not None),
               # inputs of the expert's traffic-light decision -> labels for counterfactual ego speeds
               target_nolight=np.float32(plan["target_nolight"]), lt_over=np.bool_(plan["lt_over"]),
               lt_d=np.float32(plan["lt_d"]), lt_st=np.int8(plan["lt_st"]), lt_trem=np.float32(plan["lt_trem"]),
               lt_blocked=np.bool_(plan["lt_blocked"]),
               # auxiliary label: earliest arrival of a car with priority at the next right turn that gives way
               yield_tta=np.float32(plan.get("yield_tta", np.nan)),
               # 止まれ: the pending stop line ahead and the target without it (counterfactual standing frames)
               ss_d=np.float32(plan.get("ss_d", np.nan)), ss_go=np.float32(plan.get("ss_go", np.nan)))


def worker(wid, args, quota, out_dir, counter):
    import cv2
    cv2.setNumThreads(1)
    # resumable: a worker continues after the shards it already wrote (an interrupted shard is lost), on a fresh
    # random stream so it does not replay the episodes it has recorded
    prefix = f"{args.mode}_w{wid:02d}"
    have = sorted(glob.glob(os.path.join(out_dir, f"{prefix}_*.npz")))
    rng = np.random.default_rng([args.seed, wid, len(have)] if have else args.seed * 1000 + wid)
    n0 = sum(len(np.load(f)["jpg_off"]) - 1 for f in have)
    if n0 >= quota:                                     # this worker's share is complete
        return n0
    cfg = EnvConfig()
    cfg.town = town_config(args.town_style)
    cfg.traffic.box_rule = cfg.traffic.release_hidden = bool(args.jam_fixes)
    if args.town_style != "classic" or args.osm_towns:   # varied / real towns: twice the caps keep the classic density
        cfg.traffic.max_vehicles *= 2
        cfg.traffic.max_peds *= 2
    # real towns: each worker builds its own variant of every box once (seed: signal timings, trees, building colours),
    # named apart so KeiView's town files (web/.towns/town_<name>.json) never collide
    osm = {}

    def osm_town_of(k):
        if k not in osm:
            from keisim.realworld import osm_town
            name, box = osm_spec(args.osm_towns[k])
            seed = args.osm_seed0 + 100 * k + wid
            osm[k] = osm_town(box, cache=f"data/osm/{name}.json", name=f"{name}_s{seed}", seed=seed)
        return osm[k]
    cfg.route_length = (args.route_min, args.route_max)
    cfg.route_deviation = 4.0 if args.mode == "dagger" else cfg.route_deviation
    cfg.render_rgb = cfg.render_seg = False
    cfg.renderer = args.renderer
    cfg.keiview_roads = args.keiview_roads
    if args.ego_cross_rate is not None:
        cfg.traffic.ego_cross_rate = args.ego_cross_rate
    if args.ped_spacing is not None:
        cfg.traffic.ped_spacing = tuple(args.ped_spacing)
    if args.vehicle_spacing is not None:
        cfg.traffic.vehicle_spacing = tuple(args.vehicle_spacing)
    if args.mode == "dagger":
        cfg.blocked_timeout = 40.0
    env = KeiEnv(cfg)
    agent = None
    if args.mode == "dagger":
        from keipilot.agent import KeiPilotAgent
        agent = KeiPilotAgent(args.ckpt, device=args.device)
    writer = ShardWriter(out_dir, prefix, frames_per_shard=args.shard, start=len(have))
    n = n0
    stats_path = os.path.join(out_dir, f"stats_w{wid:02d}.json")
    stats = dict.fromkeys(STAT_KEYS, 0)
    if have and os.path.exists(stats_path):
        stats.update(json.load(open(stats_path)))

    def save_stats():
        with open(stats_path + ".tmp", "w") as f:
            json.dump(stats, f)
        os.replace(stats_path + ".tmp", stats_path)

    shards_saved = writer.k
    t0 = time.time()
    town, left = None, 0
    while n < quota:
        if left <= 0:
            if args.osm_towns:
                town = osm_town_of(int(rng.integers(len(args.osm_towns))))
                env.cfg.town = town.cfg
            else:
                town = int(rng.integers(args.town_lo, args.town_hi))
            left = args.episodes_per_town
        left -= 1
        ep_seed = int(rng.integers(1 << 40))
        env.reset(town_seed=town, episode_seed=ep_seed, render=False)
        env._step = 0
        weather_id = WEATHERS.get(env.weather.name, 0)
        pert = Perturbation(rng) if args.mode == "expert" else None
        beta_on = False
        beta_t = 0.0
        if agent is not None:
            agent.reset()
        done = False
        dt = cfg.dt * cfg.action_repeat
        # multi-frame data: every eligible step is rendered (same virtual camera for `--segment` s) and kept,
        # so a recorded frame can carry the frame `--history` s before it from the same camera
        hist = int(round(args.history / dt))
        assert hist % args.every == 0, "--history must be a multiple of the recording interval"
        frames, seg_off, seg_left = {}, None, 0
        while not done and n < quota:
            a_exp = env.expert_action()
            v_model = None                                  # the driving model's target speed this step (dagger)
            tl_model = None                                 # and its traffic-light class
            if agent is None:
                a = pert(a_exp, dt)
            else:
                rgb, _ = env.render_camera(want_seg=False)
                a, p_model = agent.act(rgb, env.command(), env.target_point(), env.world.ego.v, dt)
                v_model, tl_model = p_model["target_speed"], p_model["tl"]
                # beta-mixture: short expert takeovers keep episodes progressing
                if not beta_on and rng.random() < args.beta_rate * dt:
                    beta_on, beta_t = True, rng.uniform(1.0, 3.0)
                if beta_on:
                    a = a_exp
                    v_model = None
                    beta_t -= dt
                    beta_on = beta_t > 0
            if env._step % args.every == 0:
                plan = env.plan
                stopped = env.world.ego.v < 0.2 and plan["target_speed"] == 0.0
                keep = not stopped or rng.random() < args.stopped_keep
                edge = args.focus_lead_edge > 0 and lead_at_edge(env, plan, args.focus_lead_edge)
                focus = False
                if args.focus_yield < 1.0:          # mostly frames around junction decisions, and model mistakes
                    focus = np.isfinite(plan.get("yield_tta", np.nan)) or plan["reason"] in args.focus_reasons or \
                        (v_model is not None and abs(v_model - plan["target_speed"]) > args.focus_disagree) or edge
                    keep = keep and (focus or rng.random() < args.focus_yield)
                stats["steps"] += 1
                stats["steps_focus"] += int(focus)
                stats["steps_lead_edge"] += int(edge)
                if keep:
                    stats["frames"] += 1
                    stats["focus"] += int(focus)
                    if edge:
                        stats["lead_edge"] += 1
                        stats["lead_edge_vehicle"] += int(plan["reason"] == "vehicle")
                        if plan["tl_state"] == TL_NONE:
                            stats["lead_edge_tl_none"] += 1
                            stats["lead_edge_tl_none_red"] += int(tl_model == TL_RED)
                if hist:
                    if seg_left <= 0:                               # new camera segment: history restarts
                        seg_off, seg_left, frames = random_offset(rng, args.virtual_prob), int(args.segment / dt), {}
                    seg_left -= args.every
                    rgb, seg = env.render_camera(offset=seg_off, want_seg=keep)
                    frames[env._step] = rgb
                    frames.pop(env._step - hist - args.every, None)
                    if keep:
                        record(env, writer, rng, None, weather_id, args.virtual_prob, offset=seg_off, rgb=rgb, seg=seg,
                               prev=frames.get(env._step - hist))
                        n += 1
                elif keep:
                    record(env, writer, rng, None, weather_id, args.virtual_prob)
                    n += 1
                if writer.k != shards_saved:                        # a shard was written: its counts with it
                    shards_saved = writer.k
                    save_stats()
            _, _, done, info = env.step(a)
            env._step += 1
        with counter.get_lock():
            counter.value += 0
        if wid == 0:
            el = time.time() - t0
            share = f"  focus {stats['focus'] / max(stats['frames'], 1):.0%}" + (
                f"  lead at the edge {stats['lead_edge'] / max(stats['frames'], 1):.0%}" if args.focus_lead_edge > 0 else "")
            print(f"[w0] {n}/{quota} frames  {(n - n0) / max(el, 1e-6):.1f} fps{share}  last={info['status']}", flush=True)
    writer.flush()
    save_stats()
    env.close()
    return n


def report(out_dir):
    """Kept-frame counts of every worker (stats_wNN.json): focus shares and, for --focus_lead_edge, what those
    frames are (expert reason, true light none, the driving model's red there)."""
    tot = dict.fromkeys(STAT_KEYS, 0)
    for f in glob.glob(os.path.join(out_dir, "stats_w*.json")):
        for k, v in json.load(open(f)).items():
            tot[k] = tot.get(k, 0) + v
    st, fr, le = max(tot["steps"], 1), max(tot["frames"], 1), max(tot["lead_edge"], 1)
    print(f"{tot['steps']} recording steps: focus {tot['steps_focus'] / st:.1%}, lead at the edge / on a curve"
          f" {tot['steps_lead_edge'] / st:.1%}")
    print(f"{tot['frames']} frames kept: focus {tot['focus'] / fr:.1%}, lead at the edge / on a curve {tot['lead_edge'] / fr:.1%}"
          f" (reason vehicle {tot['lead_edge_vehicle'] / le:.0%}, true light none {tot['lead_edge_tl_none'] / le:.0%},"
          f" model red with no light {tot['lead_edge_tl_none_red'] / le:.1%} of them)")
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/expert")
    ap.add_argument("--mode", default="expert", choices=["expert", "dagger"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", type=int, default=120000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--town_lo", type=int, default=0)
    ap.add_argument("--town_hi", type=int, default=400)
    ap.add_argument("--route_min", type=float, default=300.0)
    ap.add_argument("--route_max", type=float, default=600.0)
    ap.add_argument("--every", type=int, default=2, help="record every N control steps (10 Hz / N)")
    ap.add_argument("--virtual_prob", type=float, default=0.3)
    ap.add_argument("--stopped_keep", type=float, default=0.25)
    ap.add_argument("--beta_rate", type=float, default=1 / 25.0)
    ap.add_argument("--shard", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ego_cross_rate", type=float, default=None, help="pedestrian crossing trigger rate ahead of ego")
    ap.add_argument("--ped_spacing", type=float, nargs=2, default=None)
    ap.add_argument("--vehicle_spacing", type=float, nargs=2, default=None, help="m of lane per NPC vehicle (default 32 75)")
    ap.add_argument("--focus_yield", type=float, default=1.0,
                    help="keep only this fraction of the frames that are not in focus: before a right turn that gives "
                         "way, an expert reason in --focus_reasons, (dagger mode) the driving model's target speed "
                         "more than --focus_disagree m/s off the expert's, or --focus_lead_edge")
    ap.add_argument("--focus_reasons", nargs="*", default=["yield", "junction_blocked"])
    ap.add_argument("--focus_disagree", type=float, default=1.5)
    ap.add_argument("--focus_lead_edge", type=float, default=0.0,
                    help="with --focus_yield < 1, frames are also in focus when a vehicle leads within this many m and "
                         "is partly outside the camera's view or the route turns 30 deg or more over the next 15 m")
    ap.add_argument("--renderer", default="keisim", choices=["keisim", "keiview"], help="camera renderer")
    ap.add_argument("--keiview_roads", default="plain", choices=["plain", "detail"],
                    help="KeiView road surface: plain (every KeiView dataset so far) or detail (markings, signs, ...)")
    ap.add_argument("--episodes_per_town", type=int, default=1, help="episodes before switching town")
    ap.add_argument("--town_style", default="classic", choices=list(TOWN_STYLES),
                    help="town style (keisim/config.py); twophase: right turns wait for gaps in the oncoming traffic; "
                         "roadtypes: twophase with main roads, residential roads without sidewalks and alleys")
    ap.add_argument("--jam_fixes", type=int, default=0, choices=[0, 1], help="box rule + hidden jam release (long2/3)")
    ap.add_argument("--osm_towns", nargs="*", default=[], metavar="NAME[:S,W,N,E]",
                    help="drive real towns (keisim/realworld.py; boxes of OSM_BOXES or given) instead of procedural "
                         "ones; a town is picked per --episodes_per_town episodes. Use --jam_fixes 1 as evaluate.py does")
    ap.add_argument("--osm_seed0", type=int, default=1000, help="seeds of the real towns' variants (per worker)")
    ap.add_argument("--allow_eval_town", action="store_true", help="let --osm_towns include osm_sapporo (tests only)")
    ap.add_argument("--history", type=float, default=0.0, help="also store the frame this many s earlier (multi-frame)")
    ap.add_argument("--segment", type=float, default=3.0, help="with --history: seconds a virtual camera is kept")
    args = ap.parse_args()
    if any(osm_spec(t)[0] == "osm_sapporo" for t in args.osm_towns) and not args.allow_eval_town:
        ap.error("osm_sapporo is the evaluation town (evaluate.py --osm_town): collect in other boxes")
    quota = [args.frames // args.workers + (1 if i < args.frames % args.workers else 0) for i in range(args.workers)]
    write_meta(args.out, label_version=LABEL_VERSION, mode=args.mode, ckpt=args.ckpt, frames=args.frames, history=args.history,
               renderer=args.renderer, town_style=args.town_style, jam_fixes=args.jam_fixes, late_yield=True,
               **({"osm_towns": args.osm_towns, "osm_seed0": args.osm_seed0} if args.osm_towns else {}),
               **({"keiview_roads": args.keiview_roads} if args.keiview_roads != "plain" else {}))
    ctx = mp.get_context("spawn" if args.mode == "dagger" else "fork")
    counter = ctx.Value("i", 0)
    t0 = time.time()
    procs = [ctx.Process(target=worker, args=(i, args, quota[i], args.out, counter)) for i in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")
    report(args.out)
    failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
    if failed:                  # e.g. KeiView without node on PATH: the workers die, the run must not look complete
        sys.exit(f"workers {failed} failed (exit codes {[procs[i].exitcode for i in failed]})")


if __name__ == "__main__":
    main()
