#!/usr/bin/env python3
"""Measure KeiSim throughput (single process) and determinism."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keisim.config import EnvConfig  # noqa: E402
from keisim.env import KeiEnv, get_assets  # noqa: E402


def run(env, n, render):
    env.reset(town_seed=1000, episode_seed=1, render=render)
    t = time.time()
    k = 0
    for _ in range(n):
        _, _, done, _ = env.step(env.expert_action())
        k += 1
        if done:
            env.reset(town_seed=1000, episode_seed=k, render=render)
    return k / (time.time() - t)


def main():
    cfg = EnvConfig()
    env = KeiEnv(cfg)
    t = time.time()
    a = get_assets(1000, cfg)
    _ = a.town
    t_town = time.time() - t
    t = time.time()
    _ = a.camera
    t_assets = time.time() - t
    print(f"town generation {t_town:.2f}s, textures+meshes {t_assets:.2f}s")
    sim = run(env, 600, render=False)
    print(f"physics+traffic+expert, no rendering : {sim:7.1f} control steps/s  ({sim * 0.1:.1f}x real time)")
    cfg.render_seg = False
    full = run(env, 300, render=True)
    print(f"with RGB camera 320x160 (SSAA x2)    : {full:7.1f} control steps/s  ({full * 0.1:.1f}x real time)")
    cfg.render_seg = True
    full2 = run(env, 300, render=True)
    print(f"with RGB + semantic labels           : {full2:7.1f} control steps/s")
    # determinism
    trajs = []
    for _ in range(2):
        env.reset(town_seed=1001, episode_seed=5, render=False)
        xs = []
        for _ in range(300):
            _, _, d, _ = env.step(env.expert_action())
            xs.append(env.world.ego.pose)
            if d:
                break
        trajs.append(np.array(xs))
    det = bool(np.allclose(trajs[0], trajs[1]))
    print("deterministic replay:", det)
    out = {"town_s": t_town, "asset_s": t_assets, "sim_sps": sim, "sim_x": sim * 0.1, "cam_sps": full,
           "cam_x": full * 0.1, "cam_seg_sps": full2, "deterministic": det}
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    if ckpt and os.path.exists(ckpt):
        import torch
        from keipilot.agent import KeiPilotAgent
        agent = KeiPilotAgent(ckpt)
        obs = env.reset(town_seed=1000, episode_seed=1)
        for _ in range(10):
            agent.plan(obs["rgb"], obs["command"], obs["target_point"])
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(100):
            agent.plan(obs["rgb"], obs["command"], obs["target_point"])
        torch.cuda.synchronize()
        out["inf_ms"] = (time.time() - t) / 100 * 1000
        print(f"KeiPilot inference (bs=1, incl. transfer): {out['inf_ms']:.1f} ms")
    os.makedirs("runs/eval", exist_ok=True)
    json.dump(out, open("runs/eval/speed.json", "w"), indent=1)


if __name__ == "__main__":
    main()
