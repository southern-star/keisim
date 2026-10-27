"""Replay one Sapporo route of a rig model (KeiSim, the evaluate.py --osm_town settings) and save both cameras'
frames in a window before the collision, with the colliding car's bearing / distance from the camera.
  python replay_rig.py ckpt episode t_collision out_prefix"""
import math, sys
import cv2
import numpy as np
from keisim.config import EnvConfig, set_camera_rig
from keisim.env import KeiEnv
from keisim.realworld import osm_town
from keipilot.agent import KeiPilotAgent

ckpt, ep, tc, prefix = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
cfg = set_camera_rig(EnvConfig(), "wide_narrow")
town = osm_town((43.088, 141.36, 43.097, 141.372), cache="data/osm/osm_sapporo.json", name="osm_sapporo")
cfg.town = town.cfg
cfg.traffic.box_rule = cfg.traffic.release_hidden = True
cfg.blocked_timeout = 180.0
cfg.traffic.max_vehicles *= 2      # evaluate.py --osm_town: town_style twophase, CAP_SCALE 2
cfg.traffic.max_peds *= 2
cfg.render_seg = False
env = KeiEnv(cfg)
agent = KeiPilotAgent(ckpt, brake_hold=True, red_hold=True)
agent.reset()
obs = env.reset(town_seed=town, episode_seed=ep, route_length=600.0, weather="random")
dt = cfg.dt * cfg.action_repeat
rows, frames, done = [], [], False
while not done and env.elapsed < tc + 1.0:
    e = env.world.ego
    action, mp = agent.act(obs.get("rgbs", obs["rgb"]), obs["command"], obs["target_point"], e.v, dt)
    plan = env.plan
    if env.elapsed >= tc - 7.0:
        frames.append((round(env.elapsed, 1), [im.copy() for im in obs["rgbs"]], e.pose, e.v, mp["target_speed"],
                       plan["target_speed"], plan["reason"], plan.get("yield_tta")))
    obs, r, done, info = env.step(action)
hits = [i for i in env.infractions if i["type"] == "vehicle"]
print(f"ep {ep}: status {env.status}, collisions {hits}")
tr = env.world.traffic
vid = hits[0]["id"] if hits else None
for k, (t, ims, (x, y, yaw), v, tm, te, why, tta) in enumerate(frames):
    note = ""
    if vid is not None and vid < tr.n:
        # the car's position now (for the frames, its trajectory is unknown: give the final bearing only)
        pass
    if k % 5 == 0 or k == len(frames) - 1:
        print(f"t={t:5.1f} v={v:4.1f} model={tm:4.1f} expert={te:4.1f} {why:10s} tta={tta:5.1f}")
# frames every 1 s: wide over narrow, side by side in time
sel = [f for f in frames if abs((f[0] * 10) % 10) < 1e-6][-7:]
tiles = []
for t, ims, pose, v, tm, te, why, tta in sel:
    w = ims[0].copy()
    cv2.putText(w, f"t={t:.0f} v={v:.1f} m={tm:.1f} e={te:.1f} {why}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(np.concatenate([w, ims[1]], 0))
cv2.imwrite(f"{prefix}.png", np.concatenate(tiles, 1) if len(tiles) < 4 else
            np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:] + [np.zeros_like(tiles[0])] * (8 - len(tiles)), 1)[:, :4 * 320]], 0))
