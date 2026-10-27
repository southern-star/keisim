"""replay_stop.py with the KeiView renderer: replay a Sapporo route of a rig model (the evaluate.py --osm_town settings)
and save both cameras' frames and the model's segmentation around the first moment the model asks to stop while the
expert wants to go (curve / cruise), as the freezes in KeiView's left turns. KeiView draws its own lighting per
episode (sun azimuth and elevation, exposure, haze: env.reset's stream [episode, 7707]); KeiSim's weather name does not
change KeiView's images. light: 'episode' (as evaluate.py), 'default' (KeiView's default look) or a number (another
random lighting, stream [episode, number]); the simulation stays the same.
  python replay_kv_stop.py ckpt episode out.png [weather] [t_max] [light]"""
import sys
import cv2
import numpy as np
from keisim.config import EnvConfig, set_camera_rig
from keisim.env import KeiEnv
from keisim.realworld import osm_town
from keisim.viz import colorize_seg
from keipilot.agent import KeiPilotAgent

ckpt, ep, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
weather = sys.argv[4] if len(sys.argv) > 4 else "random"
t_max = float(sys.argv[5]) if len(sys.argv) > 5 else 120.0
light = sys.argv[6] if len(sys.argv) > 6 else "episode"
cfg = set_camera_rig(EnvConfig(), "wide_narrow")
town = osm_town((43.088, 141.36, 43.097, 141.372), cache="data/osm/osm_sapporo.json", name="osm_sapporo")
cfg.town = town.cfg
cfg.traffic.box_rule = cfg.traffic.release_hidden = True
cfg.blocked_timeout = 180.0
cfg.traffic.max_vehicles *= 2      # evaluate.py --osm_town: town_style twophase, CAP_SCALE 2
cfg.traffic.max_peds *= 2
cfg.render_seg = False
cfg.render_rgb = True
cfg.renderer = "keiview"
cfg.keiview_roads = "plain"
env = KeiEnv(cfg)
agent = KeiPilotAgent(ckpt, brake_hold=True, red_hold=True)
agent.reset()
obs = env.reset(town_seed=town, episode_seed=ep, route_length=600.0, weather=weather)
if light != "episode":           # from the second frame on (new_episode only resets the actor pools besides)
    env._keiview.new_episode(None if light == "default" else np.random.default_rng([ep, int(light)]))
print("light", light, env._keiview._light)
print(f"ep {ep}: weather {env.weather.name} sun {np.round(env.weather.sun_dir, 2) if hasattr(env.weather, 'sun_dir') else ''}")
dt = cfg.dt * cfg.action_repeat
hist, t_stop, done = [], None, False
while not done and env.elapsed < t_max:
    e = env.world.ego
    action, mp = agent.act(obs.get("rgbs", obs["rgb"]), obs["command"], obs["target_point"], e.v, dt, with_seg=True)
    plan = env.plan
    hist.append((round(env.elapsed, 1), [im.copy() for im in obs["rgbs"]], mp["seg"], round(float(env.s_ego), 1), e.v,
                 mp["target_speed"], plan["target_speed"], plan["reason"], mp["tl"]))
    hist = hist[-60:]
    if t_stop is None and mp["target_speed"] < 0.5 and plan["target_speed"] > 2.0 and plan["reason"] in ("curve", "cruise") \
            and env.elapsed > 3.0:
        t_stop = env.elapsed
        print(f"model stops at t={t_stop:.1f} s={env.s_ego:.1f} pose={tuple(round(float(a), 2) for a in e.pose)}")
    if t_stop is not None and env.elapsed >= t_stop + 2.0:
        break
    obs, r, done, info = env.step(action)
print(f"ep {ep}: status {env.status} t={env.elapsed:.1f} s={env.s_ego:.1f} infractions {env.infractions}")
env.close()
if t_stop is None:
    sys.exit("no stop found")
sel = [h for h in hist if h[0] in {round(t_stop - 3.0, 1), round(t_stop - 1.5, 1), round(t_stop - 0.5, 1), round(t_stop, 1),
                                    round(t_stop + 2.0, 1)}]
tiles = []
for t, ims, seg, s, v, tm, te, why, tl in sel:
    w = ims[0].copy()
    cv2.putText(w, f"t={t:.1f} s={s:.0f} v={v:.1f} m={tm:.1f} e={te:.1f} {why} tl={tl}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, (255, 255, 255), 1, cv2.LINE_AA)
    segc = colorize_seg(seg)
    segc = cv2.resize(segc, (w.shape[1], w.shape[0]), interpolation=cv2.INTER_NEAREST)
    tiles.append(np.concatenate([w, ims[1], segc[..., ::-1] if segc.shape[2] == 3 else segc], 0))
cv2.imwrite(out, np.concatenate(tiles, 1))
print("saved", out, [h[0] for h in sel])
