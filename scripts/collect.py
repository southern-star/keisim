#!/usr/bin/env python3
"""Collect training data in KeiSim.

mode=expert : the privileged expert drives, with DART-style steering/throttle
              noise injection (labels stay the expert's corrective plan) and
              virtual camera-rig perturbations (render from a displaced rig,
              re-express labels in that frame).
mode=dagger : a trained KeiPilot drives (on-policy states), the expert labels.

--curve_scenes P: a town is, with probability P, a random winding road with guardrails and (mostly) no paint
(keisim/curve_course.py scene_town: where KeiPilot is pulled towards a guardrail in tight curves). Its episodes drive
a random stretch either way with no other traffic, the target point 50 m ahead; every frame there is in focus.
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

from dataclasses import asdict  # noqa: E402

from keisim.config import CAMERA_RIGS, TL_NONE, TL_RED, EnvConfig, set_camera_rig  # noqa: E402
from keisim.env import KeiEnv  # noqa: E402
from keipilot.data import STAND_CAP, ShardWriter, stand_update, write_meta  # noqa: E402
from keisim.config import TOWN_STYLES, town_config  # noqa: E402
from keisim.expert import LABEL_VERSION  # noqa: E402
from keisim.realworld import OSM_BOXES  # real towns: name -> box (south, west, north, east)  # noqa: E402

REASONS = {"cruise": 0, "curve": 1, "vehicle": 2, "pedestrian": 3, "red_light": 4, "junction_blocked": 5, "yield": 6,
           "crossing": 7, "stop_sign": 8, "crosswalk": 9, "bicycle": 10}
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


class StandPert:
    """--stand_pert: at that share of the route's priority junctions the car brakes to a stand somewhere from
    `before` m ahead of the junction to `after` m past it, where the expert drives on slowly (a reason in REASONS,
    under v_max) and no car with priority is near, stands for --stand_hold s and is then handed back. In the data a
    car standing at a junction was nearly always waiting for a car with priority, so the standing inputs (speed 0,
    stand_m 0) themselves said "wait": KeiView's phantom stands never ended. These frames carry the expert's "go"."""

    REASONS = ("crossing", "cruise", "curve")

    def __init__(self, env, rng, p, hold, before=12.0, after=5.0, v_max=5.0):
        self.rng, self.hold, self.v_max = rng, hold, v_max
        town = env.world.town
        self.at = sorted(float(rng.uniform(j["s_in"] - before, j["s_out"] + after)) for j in env.route.junctions
                         if not j["signalized"] and len(town.junctions[j["junction"]].arms) >= 3 and rng.random() < p)
        self.on = self.tail = False
        self.left = 0.0                                     # s of standing still left

    def __call__(self, env, a, dt):
        """The action to apply instead of the driver's `a`, and whether this step belongs to a stand (the brake,
        the stand, and the start until the car is moving again)."""
        e, plan = env.world.ego, env.plan
        if not (self.on or self.tail) and self.at and env.s_ego >= self.at[0]:
            while self.at and env.s_ego >= self.at[0]:
                self.at.pop(0)
            tta = plan.get("yield_tta", np.nan)
            if plan["reason"] in self.REASONS and plan["target_speed"] > 2.0 and e.v < self.v_max and \
                    not (np.isfinite(tta) and tta < 6.0):
                self.on, self.left = True, float(self.rng.uniform(*self.hold))
                return self(env, a, dt)
        if self.on:
            a = np.array(a, float)
            a[1], a[2] = 0.0, (0.6 if e.v > 0.05 else 1.0)
            if e.v <= 0.05:
                self.left -= dt
                self.on = self.left > 0
                self.tail = not self.on
            return a, True
        if self.tail:
            self.left -= dt
            self.tail = e.v < 1.0 and self.left > -3.0
            return a, True
        return a, False


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


RAMP_LEAD = 40.0              # --focus_ramp / --ramp_early_drop: m before the route's change into a ramp
RAMP_EARLY = 0.5              # --ramp_early_drop: m toward the ramp ahead of the route's path into it


CW_FOCUS_BEHIND = 2.0         # --focus_crosswalk: m the ego's front may be past a zebra's start and still count
CW_FOCUS_REACH = 8.0          # --focus_crosswalk: s within which someone on the zebra (or setting off) reaches the ego's
#                               stretch of it (Pedestrians.crosswalk_blocked with this horizon instead of CW_REACH)


def crosswalk_near(env, ahead):
    """--focus_crosswalk: a zebra at the exit of a junction the route goes through (Route.zebras), from `ahead` m
    ahead of the ego's front to just past its start, where someone on it or setting off from its kerb gets to the
    ego's stretch of it within CW_FOCUS_REACH s: slowing down for them, waiting and setting off again (crosswalks at
    signals). Not the zebra before a junction, which people mostly cross while the ego waits at its red light."""
    pd, t, rt = env.world.peds, env.world.t, env.route
    if not pd.cw_on:
        return False
    if getattr(rt, "_zebra_kinds", None) is None:
        rt._zebra_kinds = [kind for _, kind, _ in rt.zebras(env.world.town)]
    s_front = env.s_ego + env.world.ego.LENGTH / 2
    for (s0c, _, ci, u0, u1, _), kind in zip(rt.crossings, rt._zebra_kinds):
        d = s0c - s_front
        if d > ahead:
            break
        if kind == "exit" and d >= -CW_FOCUS_BEHIND and pd.crosswalk_blocked(ci, u0, u1, t, CW_FOCUS_REACH):
            return True
    return False


BK_FOCUS_AHEAD = 20.0         # m: bicycle_near for the report without --focus_bicycle
BK_FOCUS_BEHIND = 3.0         # --focus_bicycle: m a rider riding along may be behind the ego's rear and still count


def bicycle_near(env, plan, ahead):
    """--focus_bicycle: the expert's reason is "bicycle" (following one, passing one slowly, the 巻き込み確認 before a
    left turn), or a rider rides along the route (Expert.bike_scan) from BK_FOCUS_BEHIND m behind the ego's rear to
    `ahead` m ahead of its front, one the camera sees (expert_bike_visible: the riders the expert's rules count)."""
    if plan["reason"] == "bicycle":
        return True
    bk = env.world.bikes
    scan = env.expert.last_bikes
    if bk is None or not bk.n or scan is None or len(scan["along"]) != bk.n:
        return False
    hl = env.world.ego.LENGTH / 2
    ds = scan["s"] - env.s_ego
    m = scan["along"] & scan["visible"] & (ds > -(hl + BK_FOCUS_BEHIND)) & (ds < hl + ahead)
    return bool(m.any())


def ramp_state(env):
    """(near, early) of the ego against the ramps its route takes: near, from RAMP_LEAD m before the change into a
    ramp to the end of the ramp's junction (the acceleration lane's merge included); early, ahead of the route's path
    into the ramp (RAMP_EARLY m or more toward the ramp before or during that change). The expert changes over where
    the deceleration lane runs beside the lane, after its taper: a model that moves over in the taper gets a label
    back out of the ramp it is meant to take, and mc9 learned from those to stay out of it (11-05)."""
    T, rt, s = env.world.town, env.route, env.s_ego
    near = early = False
    for c in rt.changes:
        if T.lanes[c["to"]].kind == "ramp" and c["s_a"] - RAMP_LEAD <= s <= c["s_b"]:
            side = next(rp["side"] for rp in T.ramps if rp["lane"] == c["to"])
            near, early = True, early or env.lat * side >= RAMP_EARLY
    for j in rt.junctions:
        if T.lanes[j["lane"]].kind == "ramp" and j["s_in"] - RAMP_LEAD <= s <= j["s_out"]:
            near = True
    return near, early


def osm_spec(spec):
    """--osm_towns entry: NAME (a box of OSM_BOXES) or NAME:S,W,N,E."""
    name, _, box = spec.partition(":")
    return name, (tuple(float(x) for x in box.split(",")) if box else OSM_BOXES[name])


# per-worker counts (stats_wNN.json, saved with every shard so a resumed run keeps them): every recording step
# (steps_*) and the kept frames
STAT_KEYS = ("steps", "steps_focus", "steps_lead_edge", "frames", "focus", "lead_edge", "lead_edge_vehicle",
             "lead_edge_tl_none", "lead_edge_tl_none_red", "steps_freeze", "freeze", "freezes", "stand_perts",
             "stand_pert", "steps_ramp", "ramp", "ramp_early_dropped",
             "scene", "crosswalk", "crosswalk_near", "bicycle", "bicycle_near")
FREEZE_REASONS = ("curve", "cruise")    # --freeze_reasons: the expert drives on along the road, no one to wait for


def record(env, writer, rng, v_prob, weather_id, virtual_prob, offset=False, rgb=None, seg=None, prev=None,
           stand_m=np.nan):
    """offset=False: draw a random virtual camera for this frame; else use the given one (None = real camera).
    rgb/seg: an already rendered frame for that camera; prev: its frame `--history` s earlier.
    stand_m: metres the ego has driven since it last stood (keipilot.data.stand_update).
    Camera rigs: every camera of the rig is rendered on the same (displaced) rig and stored."""
    plan = env.plan
    e = env.world.ego
    if offset is False:
        offset = random_offset(rng, virtual_prob)
    extra = ()
    if rgb is None:
        views = env.render_cameras(offset=offset)
        (rgb, seg), extra = views[0], views[1:]
    if offset is None:
        path, tp = plan["path"], env.target_point()
    else:
        vp = env.virtual_pose(offset)
        path = env.expert.path_local(env.route, env.s_ego, vp)
        tp = env.target_point(vp)
    writer.add(rgb, seg, prev=prev, extra=extra, cmd=np.int8(env.command()), tp=np.asarray(tp, np.float32),
               path=np.asarray(path, np.float32), speed=np.float32(plan["target_speed"]),
               tl=np.int8(plan["tl_state"]), v=np.float32(e.v), reason=np.int8(REASONS.get(plan["reason"], 0)),
               town=np.int32(env.town_seed), episode=np.int64(env.episode_seed), step=np.int32(env._step),
               weather=np.int8(weather_id), virtual=np.bool_(offset is not None),
               # inputs of the expert's traffic-light decision -> labels for counterfactual ego speeds
               target_nolight=np.float32(plan["target_nolight"]), lt_over=np.bool_(plan["lt_over"]),
               lt_d=np.float32(plan["lt_d"]), lt_st=np.int8(plan["lt_st"]), lt_trem=np.float32(plan["lt_trem"]),
               lt_blocked=np.bool_(plan["lt_blocked"]), lt_in=np.float32(plan.get("lt_in", np.nan)),
               # auxiliary label: earliest arrival of a car with priority at the next right turn that gives way
               yield_tta=np.float32(plan.get("yield_tta", np.nan)),
               # 止まれ: the pending stop line ahead and the target without it (counterfactual standing frames)
               ss_d=np.float32(plan.get("ss_d", np.nan)), ss_go=np.float32(plan.get("ss_go", np.nan)),
               # the ego's memory of its last stop (memory-input models)
               stand_m=np.float32(stand_m))


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
    cfg = set_camera_rig(EnvConfig(), args.rig)
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
    cfg.expert_yield_visible = bool(args.yield_visible)
    cfg.expert_bike_visible = bool(args.bike_visible)
    cfg.expert_bike_sweep = bool(args.bike_sweep)
    if args.ego_cross_rate is not None:
        cfg.traffic.ego_cross_rate = args.ego_cross_rate
    if args.ped_spacing is not None:
        cfg.traffic.ped_spacing = tuple(args.ped_spacing)
    if args.vehicle_spacing is not None:
        cfg.traffic.vehicle_spacing = tuple(args.vehicle_spacing)
    cfg.traffic.ego_patience_s = args.ego_patience
    cfg.traffic.light_past_line = bool(args.npc_past_line)
    cfg.traffic.crosswalks = args.crosswalks
    if args.crosswalk_spawn is not None:
        cfg.traffic.crosswalk_spawn = args.crosswalk_spawn
    cfg.traffic.bicycles = args.bicycles
    if args.mode == "dagger":
        cfg.blocked_timeout = 40.0
    env = KeiEnv(cfg)
    base_town_cfg, base_vehicles = cfg.town, cfg.traffic.max_vehicles
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
    town, left, in_scene = None, 0, False
    while n < quota:
        if left <= 0:
            in_scene = args.curve_scenes > 0 and rng.random() < args.curve_scenes
            if in_scene:
                from keisim.curve_course import scene_town
                town = scene_town(int(rng.integers(1 << 30)))
                env.cfg.town = town.cfg
            elif args.osm_towns:
                town = osm_town_of(int(rng.integers(len(args.osm_towns))))
                env.cfg.town = town.cfg
            else:
                town = int(rng.integers(args.town_lo, args.town_hi))
                env.cfg.town = base_town_cfg
            left = args.episodes_per_town
        left -= 1
        ep_seed = int(rng.integers(1 << 40))
        via = args.via_ramp > 0 and bool(rng.random() < args.via_ramp)     # (no draw at 0: the old random streams)
        route = None
        env.cfg.traffic.max_vehicles = 0 if in_scene else base_vehicles     # one road between dead ends: no traffic
        if in_scene:
            from keisim.curve_course import scene_route
            route = scene_route(town, rng, rng.uniform(args.route_min, args.route_max))
        env.reset(town_seed=town, episode_seed=ep_seed, render=False, via_ramp=via, route=route)
        env._step = 0
        weather_id = WEATHERS.get(env.weather.name, 0)
        pert = Perturbation(rng) if args.mode == "expert" else None
        beta_on = False
        beta_t = 0.0
        frozen_t = 0.0                                      # --freeze_takeover: s the model has stood against the expert
        stand_pert = StandPert(env, rng, args.stand_pert, args.stand_hold) if args.stand_pert > 0 else None
        if agent is not None:
            agent.reset()
        done = False
        dt = cfg.dt * cfg.action_repeat
        # multi-frame data: every eligible step is rendered (same virtual camera for `--segment` s) and kept,
        # so a recorded frame can carry the frame `--history` s before it from the same camera
        hist = int(round(args.history / dt))
        assert hist % args.every == 0, "--history must be a multiple of the recording interval"
        frames, seg_off, seg_left = {}, None, 0
        stand = STAND_CAP                                   # as KeiPilotAgent counts it (reset, then every step)
        while not done and n < quota:
            stand = stand_update(stand, env.world.ego.v, dt)
            a_exp = env.expert_action()
            v_model = None                                  # the driving model's target speed this step (dagger)
            tl_model = None                                 # and its traffic-light class
            if agent is None:
                a = pert(a_exp, dt)
            else:
                views = env.render_cameras(want_seg=False)
                rgb = [v[0] for v in views] if len(views) > 1 else views[0][0]
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
                # --freeze_takeover: the model asks to stand while the expert drives on (the look-based freezes in
                # curves and left turns, which never end by themselves). After that many s the expert drives for a
                # while: a freeze adds a few frames of the standing car with the expert's "go", not blocked_timeout's
                # worth of the same picture, and the episode goes on
                if args.freeze_takeover > 0:
                    frozen = v_model is not None and v_model < 0.5 and env.plan["target_speed"] > 2.0 and \
                        env.plan["reason"] in args.freeze_reasons
                    frozen_t = frozen_t + dt if frozen else 0.0
                    if frozen_t >= args.freeze_takeover:
                        beta_on, beta_t, frozen_t = True, rng.uniform(3.0, 5.0), 0.0
                        stats["freezes"] += 1
            standing = False                                # --stand_pert: braking to, holding or leaving a stand
            if stand_pert is not None:
                was = stand_pert.on
                a, standing = stand_pert(env, a, dt)
                stats["stand_perts"] += int(stand_pert.on and not was)
            if env._step % args.every == 0:
                plan = env.plan
                stopped = env.world.ego.v < 0.2 and plan["target_speed"] == 0.0
                keep = standing or not stopped or rng.random() < args.stopped_keep
                edge = args.focus_lead_edge > 0 and lead_at_edge(env, plan, args.focus_lead_edge)
                near_ramp, early = ramp_state(env) if (args.focus_ramp or args.ramp_early_drop) and \
                    env.world.town.ramps else (False, False)
                focus = in_scene                    # the guardrail scenes are all in focus
                cw_near = args.focus_crosswalk > 0 and crosswalk_near(env, args.focus_crosswalk)
                bk_near = args.focus_bicycle > 0 and bicycle_near(env, plan, args.focus_bicycle)
                if args.focus_yield < 1.0 and not focus:   # mostly frames around junction decisions, model mistakes
                    focus = np.isfinite(plan.get("yield_tta", np.nan)) or plan["reason"] in args.focus_reasons or \
                        (v_model is not None and abs(v_model - plan["target_speed"]) > args.focus_disagree) or edge \
                        or (args.focus_ramp and near_ramp) or cw_near or bk_near
                    keep = keep and (focus or standing or rng.random() < args.focus_yield)
                if args.ramp_early_drop and early and keep:
                    keep = False
                    stats["ramp_early_dropped"] += 1
                stats["steps"] += 1
                stats["steps_ramp"] += int(near_ramp)
                stats["steps_focus"] += int(focus)
                stats["steps_lead_edge"] += int(edge)
                stats["steps_freeze"] += int(frozen_t > 0)
                if keep:
                    stats["frames"] += 1
                    stats["focus"] += int(focus)
                    stats["freeze"] += int(frozen_t > 0)
                    stats["stand_pert"] += int(standing)
                    stats["ramp"] += int(near_ramp)
                    stats["scene"] += int(in_scene)
                    stats["crosswalk_near"] += int(cw_near)
                    stats["crosswalk"] += int(plan["reason"] == "crosswalk")
                    stats["bicycle"] += int(plan["reason"] == "bicycle")
                    stats["bicycle_near"] += int(bk_near if args.focus_bicycle > 0 else
                                                 env.world.bikes is not None and bicycle_near(env, plan, BK_FOCUS_AHEAD))
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
                               prev=frames.get(env._step - hist), stand_m=stand)
                        n += 1
                elif keep:
                    record(env, writer, rng, None, weather_id, args.virtual_prob, stand_m=stand)
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
                f"  lead at the edge {stats['lead_edge'] / max(stats['frames'], 1):.0%}" if args.focus_lead_edge > 0 else "") + (
                f"  freezes {stats['freezes']}" if args.freeze_takeover > 0 else "") + (
                f"  stands {stats['stand_perts']}" if args.stand_pert > 0 else "")
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
    if tot["freezes"] or tot["steps_freeze"]:
        print(f"freezes handed to the expert: {tot['freezes']}; frames kept while the model stood against the expert:"
              f" {tot['freeze']} ({tot['freeze'] / fr:.1%})")
    if tot["stand_perts"]:
        print(f"stands made (--stand_pert): {tot['stand_perts']}; frames kept from them: {tot['stand_pert']}"
              f" ({tot['stand_pert'] / fr:.1%})")
    if tot.get("steps_ramp") or tot.get("ramp_early_dropped"):
        print(f"recording steps by a ramp the route takes: {tot['steps_ramp']}; frames kept there: {tot['ramp']}"
              f" ({tot['ramp'] / fr:.1%}); dropped ahead of the route's path into a ramp: {tot['ramp_early_dropped']}")
    if tot.get("scene"):
        print(f"{tot['scene']} frames ({tot['scene'] / fr:.1%}) on the guardrail scenes (--curve_scenes)")
    if tot.get("crosswalk") or tot.get("crosswalk_near"):
        print(f"crosswalks: {tot['crosswalk']} frames ({tot['crosswalk'] / fr:.1%}) where the expert gives way at a zebra, "
              f"{tot['crosswalk_near']} ({tot['crosswalk_near'] / fr:.1%}) by a zebra with people in the way (--focus_crosswalk)")
    if tot.get("bicycle") or tot.get("bicycle_near"):
        print(f"bicycles: {tot['bicycle']} frames ({tot['bicycle'] / fr:.1%}) with the expert's reason bicycle, "
              f"{tot['bicycle_near']} ({tot['bicycle_near'] / fr:.1%}) with a rider about (bicycle_near: the reason, or "
              f"one riding along within --focus_bicycle m, {BK_FOCUS_AHEAD:g} m without it)")
    return tot


def parse_args(argv=None):
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
    ap.add_argument("--freeze_takeover", type=float, default=0.0,
                    help="dagger: once the driving model has asked to stand (target < 0.5 m/s) for this many s while the "
                         "expert drives on (> 2 m/s, a reason in --freeze_reasons), the expert drives for 3-5 s (0: off)")
    ap.add_argument("--freeze_reasons", nargs="*", default=list(FREEZE_REASONS),
                    help="--freeze_takeover's expert reasons; add crossing for the stands after a 止まれ, at the "
                         "give-way crossing nobody comes to")
    ap.add_argument("--stand_pert", type=float, default=0.0,
                    help="share of the priority junctions where the car is made to stand a while although the expert "
                         "drives on (StandPert; 0: off)")
    ap.add_argument("--stand_hold", type=float, nargs=2, default=[1.0, 4.0], help="--stand_pert: s standing (range)")
    ap.add_argument("--ego_patience", type=float, default=0.0,
                    help="s the ego may stand before the NPCs that give way to it stop waiting for it "
                         "(TrafficConfig.ego_patience_s; 0: they wait for ever)")
    ap.add_argument("--npc_past_line", type=int, default=0, choices=[0, 1],
                    help="1: NPCs standing a little past their stop line still obey its light "
                         "(TrafficConfig.light_past_line)")
    ap.add_argument("--crosswalks", type=float, default=0.0,
                    help="crosswalks at signals (TrafficConfig.crosswalks): the chance that a person at a two-phase "
                         "signal's road end crosses on the zebra with the pedestrian light; turning cars give way")
    ap.add_argument("--crosswalk_spawn", type=float, default=None,
                    help="with --crosswalks: corner people per zebra on average (TrafficConfig.crosswalk_spawn, 1)")
    ap.add_argument("--focus_crosswalk", type=float, default=0.0,
                    help="with --focus_yield < 1, frames are also in focus while the zebra at the exit of a junction "
                         "the route goes through, within this many m ahead, has someone on it or setting off who gets "
                         "to the ego's stretch of it within CW_FOCUS_REACH s (crosswalk_near)")
    ap.add_argument("--bicycles", type=float, default=0.0,
                    help="bicycles per km of road (TrafficConfig.bicycles; 0: none)")
    ap.add_argument("--bike_visible", type=int, default=0, choices=[0, 1],
                    help="the expert's 巻き込み確認 and give-way counts only the riders the camera sees, and riders "
                         "coming unseen from behind do not move its path (EnvConfig.expert_bike_visible)")
    ap.add_argument("--bike_sweep", type=int, default=0, choices=[0, 1],
                    help="the expert's room beside a rider ahead counts its front corner off the path when it points "
                         "off it (EnvConfig.expert_bike_sweep)")
    ap.add_argument("--focus_bicycle", type=float, default=0.0,
                    help="with --focus_yield < 1, frames are also in focus where the expert's reason is bicycle or a "
                         "rider the camera sees rides along the route within this many m ahead (bicycle_near)")
    ap.add_argument("--renderer", default="keisim", choices=["keisim", "keiview"], help="camera renderer")
    ap.add_argument("--keiview_roads", default="plain", choices=["plain", "detail"],
                    help="KeiView road surface: plain (every KeiView dataset so far) or detail (markings, signs, ...)")
    ap.add_argument("--episodes_per_town", type=int, default=1, help="episodes before switching town")
    ap.add_argument("--town_style", default="classic", choices=list(TOWN_STYLES),
                    help="town style (keisim/config.py); twophase: right turns wait for gaps in the oncoming traffic; "
                         "roadtypes: twophase with main roads, residential roads without sidewalks and alleys")
    ap.add_argument("--jam_fixes", type=int, default=0, choices=[0, 1], help="box rule + hidden jam release (long2/3)")
    ap.add_argument("--yield_visible", type=int, default=0, choices=[0, 1],
                    help="the expert gives way only to cars the camera sees (EnvConfig.expert_yield_visible)")
    ap.add_argument("--osm_towns", nargs="*", default=[], metavar="NAME[:S,W,N,E]",
                    help="drive real towns (keisim/realworld.py; boxes of OSM_BOXES or given) instead of procedural "
                         "ones; a town is picked per --episodes_per_town episodes. Use --jam_fixes 1 as evaluate.py does")
    ap.add_argument("--via_ramp", type=float, default=0.0,
                    help="share of the episodes whose route takes one of the town's ramps (route.ramp_route, "
                         "evaluate.py --via_ramp; ramps towns: the random routes rarely do)")
    ap.add_argument("--focus_ramp", type=int, default=0, choices=[0, 1],
                    help="with --focus_yield < 1, frames by a ramp the route takes are in focus (ramp_state: from "
                         "RAMP_LEAD m before the change into it to the end of its junction)")
    ap.add_argument("--ramp_early_drop", type=int, default=0, choices=[0, 1],
                    help="drop the frames where the ego is ahead of the route's path into a ramp (moved over in the "
                         "taper): their label leads back out of the ramp (ramp_state)")
    ap.add_argument("--osm_seed0", type=int, default=1000, help="seeds of the real towns' variants (per worker)")
    ap.add_argument("--allow_eval_town", action="store_true", help="let --osm_towns include osm_sapporo (tests only)")
    ap.add_argument("--history", type=float, default=0.0, help="also store the frame this many s earlier (multi-frame)")
    ap.add_argument("--segment", type=float, default=3.0, help="with --history: seconds a virtual camera is kept")
    ap.add_argument("--rig", default="front", choices=list(CAMERA_RIGS),
                    help="camera rig (keisim CAMERA_RIGS): every camera's image is stored (jpg_c1, ...)")
    ap.add_argument("--curve_scenes", type=float, default=0.0,
                    help="probability that a town is a random winding road with guardrails and (mostly) no paint "
                         "(keisim/curve_course.py scene_town; 1: only those)")
    args = ap.parse_args(argv)
    if any(osm_spec(t)[0] == "osm_sapporo" for t in args.osm_towns) and not args.allow_eval_town:
        ap.error("osm_sapporo is the evaluation town (evaluate.py --osm_town): collect in other boxes")
    if args.rig != "front" and args.history:
        ap.error("camera rigs: no --history")
    if args.mode == "dagger":
        from keipilot.agent import checkpoint_rig
        if checkpoint_rig(args.ckpt) != args.rig:
            ap.error(f"--ckpt drives with the camera rig {checkpoint_rig(args.ckpt)}: pass it as --rig")
    if not 0.0 <= args.curve_scenes <= 1.0:
        ap.error("--curve_scenes is a probability")
    return args


def main():
    args = parse_args()
    quota = [args.frames // args.workers + (1 if i < args.frames % args.workers else 0) for i in range(args.workers)]
    write_meta(args.out, label_version=LABEL_VERSION, mode=args.mode, ckpt=args.ckpt, frames=args.frames, history=args.history,
               renderer=args.renderer, town_style=args.town_style, jam_fixes=args.jam_fixes, late_yield=True,
               **({"osm_towns": args.osm_towns, "osm_seed0": args.osm_seed0} if args.osm_towns else {}),
               **({"keiview_roads": args.keiview_roads} if args.keiview_roads != "plain" else {}),
               **({"yield_visible": 1} if args.yield_visible else {}),
               **({"ego_patience": args.ego_patience} if args.ego_patience else {}),
               **({"npc_past_line": 1} if args.npc_past_line else {}),
               **({"crosswalks": args.crosswalks} if args.crosswalks > 0 else {}),
               **({"crosswalk_spawn": args.crosswalk_spawn} if args.crosswalk_spawn is not None else {}),
               **({"focus_crosswalk": args.focus_crosswalk} if args.focus_crosswalk > 0 else {}),
               **({"bicycles": args.bicycles} if args.bicycles > 0 else {}),
               **({"bike_visible": 1} if args.bike_visible else {}),
               **({"bike_sweep": 1} if args.bike_sweep else {}),
               **({"focus_bicycle": args.focus_bicycle} if args.focus_bicycle > 0 else {}),
               **({"freeze_reasons": args.freeze_reasons} if args.freeze_takeover > 0 else {}),
               **({"stand_pert": args.stand_pert, "stand_hold": args.stand_hold} if args.stand_pert > 0 else {}),
               **({"via_ramp": args.via_ramp} if args.via_ramp > 0 else {}),
               **({"focus_ramp": 1} if args.focus_ramp else {}),
               **({"ramp_early_drop": 1} if args.ramp_early_drop else {}),
               **({"rig": args.rig, "cameras": [asdict(c) for c in set_camera_rig(EnvConfig(), args.rig).cameras]}
                  if args.rig != "front" else {}),
               **({"curve_scenes": args.curve_scenes} if args.curve_scenes else {}))
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
