"""Replay a Sapporo route of a rig model (KeiSim, evaluate.py --osm_town settings) and save both cameras and the
model's segmentation when the ego first reaches each of the given route positions s (m).
  python replay_at_s.py ckpt episode out.png s1 s2 ..."""
import sys
import cv2
import numpy as np
from keisim.config import EnvConfig, set_camera_rig
from keisim.env import KeiEnv
from keisim.realworld import osm_town
from keisim.viz import colorize_seg
from keipilot.agent import KeiPilotAgent

ckpt, ep, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
marks = sorted(float(a) for a in sys.argv[4:])
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
tiles, done = [], False
while not done and marks:
    e = env.world.ego
    action, mp = agent.act(obs.get("rgbs", obs["rgb"]), obs["command"], obs["target_point"], e.v, dt, with_seg=True)
    plan = env.plan
    if env.s_ego >= marks[0]:
        marks.pop(0)
        w = obs["rgbs"][0].copy()
        cv2.putText(w, f"t={env.elapsed:.1f} s={env.s_ego:.0f} v={e.v:.1f} m={mp['target_speed']:.1f} e={plan['target_speed']:.1f} "
                       f"{plan['reason']} cmd={obs['command']}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
        segc = cv2.resize(colorize_seg(mp["seg"]), (w.shape[1], w.shape[0]), interpolation=cv2.INTER_NEAREST)[..., ::-1]
        tiles.append(np.concatenate([w, obs["rgbs"][1], segc], 0))
        print(f"s={env.s_ego:.1f} t={env.elapsed:.1f} pose={tuple(round(float(a), 2) for a in e.pose)} cmd={obs['command']} "
              f"tp={np.round(obs['target_point'], 1)} model={mp['target_speed']:.1f} expert={plan['target_speed']:.1f} {plan['reason']}")
    obs, r, done, info = env.step(action)
print(f"ep {ep}: status {env.status} t={env.elapsed:.1f}")
cv2.imwrite(out, np.concatenate(tiles, 1))
