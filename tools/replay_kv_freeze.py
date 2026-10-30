"""Replay a Sapporo route of a rig model in KeiView (evaluate.py --osm_town settings, --ego_patience P) with a KeiSim
shadow (tools/replay_kv_shadow.py), printing a 1 s timeline from t_from to t_to: ego v, the expert's target / reason,
both models' targets (yield_tta; KeiView's also before the safety layers, and its light head's red+yellow), and the
NPCs within 30 m (bearing from the ego, + left; distance; speed).
Saves the wide+narrow frames of both renderers at the given times (out_prefix_T.png: KeiView left, KeiSim right) and
the model inputs there (out_prefix_T.npz: both renderers' images, command, target point, speed, stand_m), for
tools/occlude_kv.py.
For a model that stands where the expert would go: KeiView's model at 0 with a short yield_tta while the KeiSim shadow
drives on, with no NPC near, is a phantom in KeiView's picture.
Each KeiView renderer holds about 2.2 GB of the GPU: while KeiView jobs run, KV_GL=soft CUDA_VISIBLE_DEVICES= renders
on the CPU instead (SwiftShader; far too slow next to a busy CPU, though).
  python replay_kv_freeze.py ckpt episode patience t_from t_to out_prefix [T ...]   (fork dir)"""
import math, os, sys
import cv2
import numpy as np
from keisim.config import EnvConfig, set_camera_rig
from keisim.env import KeiEnv
from keisim.realworld import osm_town
from keipilot.agent import KeiPilotAgent

ckpt, ep, pat, t_from, t_to, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
snaps = sorted(float(a) for a in sys.argv[7:])
cfg = set_camera_rig(EnvConfig(), "wide_narrow")
town = osm_town((43.088, 141.36, 43.097, 141.372), cache="data/osm/osm_sapporo.json", name="osm_sapporo")
cfg.town = town.cfg
cfg.traffic.box_rule = cfg.traffic.release_hidden = True
cfg.blocked_timeout = 180.0
cfg.traffic.max_vehicles *= 2
cfg.traffic.max_peds *= 2
cfg.traffic.ego_patience_s = pat
cfg.render_seg = False
cfg.render_rgb = True
cfg.renderer = "keiview"
cfg.keiview_roads = "plain"
cfg.keiview_gl = os.environ.get("KV_GL", cfg.keiview_gl)
env = KeiEnv(cfg)
agent = KeiPilotAgent(ckpt, brake_hold=True, red_hold=True)
shadow = KeiPilotAgent(ckpt, brake_hold=True, red_hold=True)
agent.reset(); shadow.reset()


def tta_of(a):
    box, fwd = {"tta": math.nan}, a.model.forward
    def forward(*args, **kw):
        o = fwd(*args, **kw)
        box["tta"] = float(o["yield_tta"][0]) if "yield_tta" in o else math.nan
        return o
    a.model.forward = forward
    return box


tta_kv, tta_ks = tta_of(agent), tta_of(shadow)
obs = env.reset(town_seed=town, episode_seed=ep, route_length=600.0, weather="random")
dt = cfg.dt * cfg.action_repeat
done = False
print(f"ep {ep} patience {pat}")
print("     t      s    v | expert       reason      tta | KeiView raw (tta) r+y | KeiSim (tta) | NPCs within 30 m: bearing dist v")
while not done and env.elapsed < t_to + 0.05:
    e = env.world.ego
    state = env.world.scene_state()
    win = env.elapsed >= t_from - 1e-6
    ks = [cam.render(e.pose, state, env.weather, want_rgb=True, want_seg=False)[0] for cam in env.assets.cameras]
    action, mp = agent.act(obs["rgbs"], obs["command"], obs["target_point"], e.v, dt)
    _, ms = shadow.act(ks, obs["command"], obs["target_point"], e.v, dt)
    t = env.elapsed
    if win and abs(t - round(t)) < 0.05:
        tr = env.world.traffic
        x, y, yaw = e.pose
        near = []
        for i in range(tr.n):
            dx, dy = tr.xy[i, 0] - x, tr.xy[i, 1] - y
            d = math.hypot(dx, dy)
            if d < 30:
                b = (math.degrees(math.atan2(dy, dx) - yaw) + 180) % 360 - 180
                near.append((d, f"{b:+.0f}°/{d:.0f}m/{tr.v[i] * 3.6:.0f}"))
        near = " ".join(n for _, n in sorted(near)[:5])
        print(f"{t:6.1f} {env.s_ego:6.1f} {e.v * 3.6:4.1f} | {env.plan['target_speed'] * 3.6:5.1f} {env.plan['reason']:10s} "
              f"{env.plan['yield_tta']:5.1f} | {mp['target_speed'] * 3.6:5.1f} {mp['target_speed_raw'] * 3.6:4.1f} ({tta_kv['tta']:4.1f}) "
              f"{mp['tl_probs'][0] + mp['tl_probs'][1]:.2f} | "
              f"{ms['target_speed'] * 3.6:5.1f} ({tta_ks['tta']:4.1f}) | {near}", flush=True)
    if snaps and t >= snaps[0] - 1e-6:
        T = snaps.pop(0)
        im = np.concatenate([np.concatenate(obs["rgbs"], 0), np.concatenate(ks, 0)], 1)
        cv2.putText(im, f"t={t:.1f} v={e.v * 3.6:.1f} ex={env.plan['target_speed'] * 3.6:.1f} {env.plan['reason']} "
                    f"KV {mp['target_speed'] * 3.6:.1f} KS {ms['target_speed'] * 3.6:.1f}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(f"{out}_{T:g}.png", im)
        np.savez_compressed(f"{out}_{T:g}.npz", kv=np.stack(obs["rgbs"]), ks=np.stack(ks), command=int(obs["command"]),
                            target_point=np.asarray(obs["target_point"], np.float32), speed=e.v, stand_m=agent.stand_m)
    obs, r, done, info = env.step(action)
print(f"ep {ep}: status {env.status} t={env.elapsed:.1f} s={env.s_ego:.1f} infractions {env.infractions}")
env.close()
