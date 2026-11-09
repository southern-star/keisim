"""Replay a Sapporo route of a rig model (KeiSim, the evaluate.py --osm_town settings) up to its first vehicle collision
and print who hit whom: the other car's place in the ego's frame (ahead / behind, left / right), its heading against the
ego's, and both speeds over the last 3 s.
  python replay_hit.py ckpt episode [t_max]"""
import math
import sys
import numpy as np
from keisim.config import EnvConfig, set_camera_rig
from keisim.env import KeiEnv
from keisim.realworld import osm_town
from keipilot.agent import KeiPilotAgent

ckpt, ep = sys.argv[1], int(sys.argv[2])
t_max = float(sys.argv[3]) if len(sys.argv) > 3 else 200.0
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
hist, done = [], False
while not done and env.elapsed < t_max:
    e = env.world.ego
    action, mp = agent.act(obs.get("rgbs", obs["rgb"]), obs["command"], obs["target_point"], e.v, dt)
    tr = env.world.traffic
    hist.append((round(env.elapsed, 1), e.xy.copy(), float(e.yaw), e.v, mp["target_speed"], env.plan["target_speed"],
                 env.plan["reason"], tr.xy.copy(), tr.yaw.copy(), tr.v.copy()))
    hist = hist[-40:]
    obs, r, done, info = env.step(action)
    if any(i["type"] == "vehicle" for i in env.infractions):
        break
print(f"ep {ep}: status {env.status} t={env.elapsed:.1f} infractions {env.infractions}")
hits = [i for i in env.infractions if i["type"] == "vehicle"]
if not hits:
    sys.exit("no vehicle collision in the replay")
k = hits[0]["id"]
e = env.world.ego
tr = env.world.traffic


def rel(exy, eyaw, oxy, oyaw):
    d = oxy - exy
    c, s = math.cos(eyaw), math.sin(eyaw)
    fwd, left = d[0] * c + d[1] * s, -d[0] * s + d[1] * c
    dyaw = (math.degrees(oyaw - eyaw) + 180) % 360 - 180
    return fwd, left, dyaw


f, l, dy = rel(e.xy, e.yaw, tr.xy[k], tr.yaw[k])
print(f"at the hit: other car {k} at {f:+.1f} m ahead, {l:+.1f} m left of the ego, heading {dy:+.0f} deg from the ego's;"
      f" ego v {e.v * 3.6:.1f} km/h, other v {tr.v[k] * 3.6:.1f} km/h")
print("   t    ego km/h  model  expert  reason      other: ahead  left  dyaw  km/h")
for t, exy, eyaw, ev, tm, te, why, oxy, oyaw, ov in hist[::3] + [hist[-1]]:
    if k < len(oxy):
        f, l, dy = rel(exy, eyaw, oxy[k], oyaw[k])
        print(f"{t:6.1f} {ev * 3.6:8.1f} {tm * 3.6:6.1f} {te * 3.6:7.1f}  {why:10s} {f:+12.1f} {l:+5.1f} {dy:+5.0f} {ov[k] * 3.6:5.1f}")
