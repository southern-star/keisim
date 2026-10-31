"""Replay a grid-suite route of a rig model (KeiSim, evaluate.py's long3* settings) up to its first vehicle collision;
print the other car's place against the ego and both speeds, every 2 s over the last 90 s and every 0.2 s over the
last 3 s, with the model's raw target and yield_tta and stand_m. Saves the model inputs at the given times
(out_prefix_T.npz, as tools/replay_kv_freeze.py's: kv and ks both KeiSim's) for tools/cmp_ckpts_npz.py,
tools/swap_region.py and tools/occlude_kv.py.
  python replay_hit_grid.py ckpt town episode [dense 0/1 [t_max [out_prefix T ...]]]   (fork dir)"""
import math
import sys
import numpy as np
from keisim.config import EnvConfig, set_camera_rig, town_config
from keisim.env import KeiEnv
from keipilot.agent import KeiPilotAgent

ckpt, town, ep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dense = len(sys.argv) > 4 and sys.argv[4] == "1"
t_max = float(sys.argv[5]) if len(sys.argv) > 5 else 2000.0
out = sys.argv[6] if len(sys.argv) > 6 else None
snaps = [float(x) for x in sys.argv[7:]]
cfg = set_camera_rig(EnvConfig(), "wide_narrow")
cfg.town = town_config("twophase")
cfg.traffic.box_rule = cfg.traffic.release_hidden = True
cfg.blocked_timeout = 180.0
cfg.render_seg = False
if dense:
    cfg.traffic.vehicle_spacing = (20.0, 28.0)
    cfg.traffic.ped_spacing = (14.0, 20.0)
    cfg.traffic.max_vehicles = 130
    cfg.traffic.max_peds = 110
    cfg.traffic.ego_cross_rate = 0.3
cfg.traffic.max_vehicles *= 2
cfg.traffic.max_peds *= 2
env = KeiEnv(cfg)
agent = KeiPilotAgent(ckpt, brake_hold=True, red_hold=True)
agent.reset()
obs = env.reset(town_seed=town, episode_seed=ep, route_length=2500.0, weather="random")
dt = cfg.dt * cfg.action_repeat
hist, done = [], False
while not done and env.elapsed < t_max:
    e = env.world.ego
    action, mp = agent.act(obs.get("rgbs", obs["rgb"]), obs["command"], obs["target_point"], e.v, dt)
    tr = env.world.traffic
    for T in snaps:
        if abs(env.elapsed - T) < 0.05:
            np.savez_compressed(f"{out}_{T:g}.npz", kv=np.stack(obs["rgbs"]), ks=np.stack(obs["rgbs"]),
                                command=int(obs["command"]), target_point=np.asarray(obs["target_point"], np.float32),
                                speed=e.v, stand_m=agent.stand_m)
    hist.append((round(env.elapsed, 1), e.xy.copy(), float(e.yaw), e.v, mp["target_speed"], mp.get("target_speed_raw", np.nan),
                 mp.get("yield_tta", np.nan), getattr(agent, "stand_m", np.nan), env.plan["target_speed"],
                 env.plan["reason"], tr.xy.copy(), tr.yaw.copy(), tr.v.copy()))
    hist = hist[-900:]
    obs, r, done, info = env.step(action)
    if any(i["type"] == "vehicle" for i in env.infractions):
        break
print(f"town {town} ep {ep}: status {env.status} t={env.elapsed:.1f} infractions {env.infractions}")
hits = [i for i in env.infractions if i["type"] == "vehicle"]
if not hits:
    sys.exit("no vehicle collision in the replay")
k = hits[0]["id"]


def rel(exy, eyaw, oxy, oyaw):
    d = oxy - exy
    c, s = math.cos(eyaw), math.sin(eyaw)
    return d[0] * c + d[1] * s, -d[0] * s + d[1] * c, (math.degrees(oyaw - eyaw) + 180) % 360 - 180


print("   t    ego km/h  model   raw  tta  stand_m  expert  reason      other: ahead  left  dyaw  km/h")
tc = hist[-1][0]
for h in hist:
    t, exy, eyaw, ev, tm, traw, tta, sm, te, why, oxy, oyaw, ov = h
    if not (round(t * 10) % 20 == 0 or t >= tc - 3 and round(t * 10) % 2 == 0 or h is hist[-1]):
        continue
    if k < len(oxy):
        f, l, dy = rel(exy, eyaw, oxy[k], oyaw[k])
        print(f"{t:6.1f} {ev * 3.6:8.1f} {tm * 3.6:6.1f} {traw * 3.6:5.1f} {tta:4.1f} {sm:7.1f} {te * 3.6:7.1f}  {why:10s} "
              f"{f:+12.1f} {l:+5.1f} {dy:+5.0f} {ov[k] * 3.6:5.1f}")
