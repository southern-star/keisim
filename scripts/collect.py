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
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keisim.config import EnvConfig  # noqa: E402
from keisim.env import KeiEnv  # noqa: E402
from keipilot.data import ShardWriter, write_meta  # noqa: E402
from keisim.config import town_config  # noqa: E402
from keisim.expert import LABEL_VERSION  # noqa: E402

REASONS = {"cruise": 0, "curve": 1, "vehicle": 2, "pedestrian": 3, "red_light": 4, "junction_blocked": 5}
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
               lt_blocked=np.bool_(plan["lt_blocked"]))


def worker(wid, args, quota, out_dir, counter):
    import cv2
    cv2.setNumThreads(1)
    rng = np.random.default_rng(args.seed * 1000 + wid)
    cfg = EnvConfig()
    cfg.town = town_config(args.town_style)
    cfg.traffic.box_rule = cfg.traffic.release_hidden = bool(args.jam_fixes)
    if args.town_style != "classic":                   # varied towns: twice the caps keep the classic density
        cfg.traffic.max_vehicles *= 2
        cfg.traffic.max_peds *= 2
    cfg.route_length = (args.route_min, args.route_max)
    cfg.route_deviation = 4.0 if args.mode == "dagger" else cfg.route_deviation
    cfg.render_rgb = cfg.render_seg = False
    cfg.renderer = args.renderer
    if args.ego_cross_rate is not None:
        cfg.traffic.ego_cross_rate = args.ego_cross_rate
    if args.ped_spacing is not None:
        cfg.traffic.ped_spacing = tuple(args.ped_spacing)
    if args.mode == "dagger":
        cfg.blocked_timeout = 40.0
    env = KeiEnv(cfg)
    agent = None
    if args.mode == "dagger":
        from keipilot.agent import KeiPilotAgent
        agent = KeiPilotAgent(args.ckpt, device=args.device)
    writer = ShardWriter(out_dir, f"{args.mode}_w{wid:02d}", frames_per_shard=args.shard)
    n = 0
    t0 = time.time()
    town, left = None, 0
    while n < quota:
        if left <= 0:
            town, left = int(rng.integers(args.town_lo, args.town_hi)), args.episodes_per_town
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
            if agent is None:
                a = pert(a_exp, dt)
            else:
                rgb, _ = env.render_camera(want_seg=False)
                a, _ = agent.act(rgb, env.command(), env.target_point(), env.world.ego.v, dt)
                # beta-mixture: short expert takeovers keep episodes progressing
                if not beta_on and rng.random() < args.beta_rate * dt:
                    beta_on, beta_t = True, rng.uniform(1.0, 3.0)
                if beta_on:
                    a = a_exp
                    beta_t -= dt
                    beta_on = beta_t > 0
            if env._step % args.every == 0:
                stopped = env.world.ego.v < 0.2 and env.plan["target_speed"] == 0.0
                keep = not stopped or rng.random() < args.stopped_keep
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
            _, _, done, info = env.step(a)
            env._step += 1
        with counter.get_lock():
            counter.value += 0
        if wid == 0:
            el = time.time() - t0
            print(f"[w0] {n}/{quota} frames  {n / max(el, 1e-6):.1f} fps  last={info['status']}", flush=True)
    writer.flush()
    env.close()
    return n


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
    ap.add_argument("--renderer", default="keisim", choices=["keisim", "keiview"], help="camera renderer")
    ap.add_argument("--episodes_per_town", type=int, default=1, help="episodes before switching town")
    ap.add_argument("--town_style", default="classic", choices=["classic", "varied", "twophase"],
                    help="town style (keisim/config.py); twophase: right turns wait for gaps in the oncoming traffic")
    ap.add_argument("--jam_fixes", type=int, default=0, choices=[0, 1], help="box rule + hidden jam release (long2/3)")
    ap.add_argument("--history", type=float, default=0.0, help="also store the frame this many s earlier (multi-frame)")
    ap.add_argument("--segment", type=float, default=3.0, help="with --history: seconds a virtual camera is kept")
    args = ap.parse_args()
    quota = [args.frames // args.workers + (1 if i < args.frames % args.workers else 0) for i in range(args.workers)]
    write_meta(args.out, label_version=LABEL_VERSION, mode=args.mode, ckpt=args.ckpt, frames=args.frames, history=args.history,
               renderer=args.renderer, town_style=args.town_style, jam_fixes=args.jam_fixes)
    ctx = mp.get_context("spawn" if args.mode == "dagger" else "fork")
    counter = ctx.Value("i", 0)
    t0 = time.time()
    procs = [ctx.Process(target=worker, args=(i, args, quota[i], args.out, counter)) for i in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
