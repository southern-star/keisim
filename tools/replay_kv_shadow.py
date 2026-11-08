"""Replay a Sapporo route of a rig model in KeiView (the evaluate.py --osm_town settings) with a shadow: a second copy
of the model gets KeiSim's frames of the same scene at every step (same ego, same route; its actions are not applied),
so the two targets differ only by the renderer. Prints both targets (with the models' yield_tta: the auxiliary head's
arrival time of the first car with priority) and the expert's from t_from to t_to, with the place of the car the ego
hits (or of `car`, an NPC index) and the vehicle pixels in its direction in each renderer's wide
segmentation, and saves both renderers' frames at six times in that window (KeiView over KeiSim, wide over narrow; a
red mark over the wide images at the car's bearing) and, enlarged, the wide images around the car while the expert
gives way (out_zoom.png).
KeiView's lighting: as replay_kv_stop.py ('episode', 'default' or a number).
  python replay_kv_shadow.py ckpt episode t_from t_to out.png [car|-] [light]"""
import math, sys
import cv2
import numpy as np
from keisim.config import SEM, EnvConfig, set_camera_rig
from keisim.env import KeiEnv
from keisim.realworld import osm_town
from keipilot.agent import KeiPilotAgent

ckpt, ep, t_from, t_to, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
car = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != "-" else None
light = sys.argv[7] if len(sys.argv) > 7 else "episode"
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
shadow = KeiPilotAgent(ckpt, brake_hold=True, red_hold=True)
agent.reset()
shadow.reset()


def tta_of(a):
    """Keep the model's yield_tta (its auxiliary head: when the first car with priority arrives) of every call."""
    box, fwd = {"tta": math.nan}, a.model.forward

    def forward(*args, **kw):
        out = fwd(*args, **kw)
        box["tta"] = float(out["yield_tta"][0]) if "yield_tta" in out else math.nan
        return out
    a.model.forward = forward
    return box


tta_kv, tta_ks = tta_of(agent), tta_of(shadow)
obs = env.reset(town_seed=town, episode_seed=ep, route_length=600.0, weather="random")
if light != "episode":           # from the second frame on (new_episode only resets the actor pools besides)
    env._keiview.new_episode(None if light == "default" else np.random.default_rng([ep, int(light)]))
print(f"ep {ep}: KeiSim weather {env.weather.name}, KeiView light {light} {env._keiview._light}")
dt = cfg.dt * cfg.action_repeat
wide = cfg.camera
rows, done = [], False
while not done and env.elapsed < t_to + 0.05:
    e = env.world.ego
    state = env.world.scene_state()
    win = env.elapsed >= t_from - 1e-6
    ksv = [cam.render(e.pose, state, env.weather, want_rgb=True, want_seg=win) for cam in env.assets.cameras]
    ks = [v[0] for v in ksv]
    action, mp = agent.act(obs["rgbs"], obs["command"], obs["target_point"], e.v, dt)
    _, ms = shadow.act(ks, obs["command"], obs["target_point"], e.v, dt)
    if win:
        tr = env.world.traffic
        rows.append(dict(t=round(env.elapsed, 1), s=float(env.s_ego), v=e.v, kv=mp["target_speed"], ks=ms["target_speed"],
                         ex=env.plan["target_speed"], why=env.plan["reason"], tta=env.plan["yield_tta"],
                         tkv=tta_kv["tta"], tks=tta_ks["tta"],
                         pose=tuple(float(a) for a in e.pose), oxy=tr.xy.copy(), oyaw=tr.yaw.copy(), ov=tr.v.copy(),
                         ims=[im.copy() for im in obs["rgbs"]] + [im.copy() for im in ks],
                         segs=[obs["segs"][0] if obs.get("segs") else None, ksv[0][1]]))
    env.cfg.render_seg = env.elapsed + dt >= t_from - 1e-6     # both renderers' wide segmentation in the window
    obs, r, done, info = env.step(action)
print(f"ep {ep}: status {env.status} t={env.elapsed:.1f} s={env.s_ego:.1f} infractions {env.infractions}")
env.close()
hits = [i["id"] for i in env.infractions if i["type"] == "vehicle"]
k = car if car is not None else (hits[0] if hits else None)


def place(r):
    """The car's bearing from the wide camera (deg, + left), distance (m), heading against the ego's, speed."""
    x, y, yaw = r["pose"]
    cx, cy = x + wide.x * math.cos(yaw), y + wide.x * math.sin(yaw)
    d = r["oxy"][k] - (cx, cy)
    fwd, left = d[0] * math.cos(yaw) + d[1] * math.sin(yaw), -d[0] * math.sin(yaw) + d[1] * math.cos(yaw)
    return (math.degrees(math.atan2(left, fwd)), math.hypot(fwd, left),
            (math.degrees(r["oyaw"][k] - yaw) + 180) % 360 - 180, float(r["ov"][k]))


def car_px(r):
    """Vehicle pixels of each renderer's wide segmentation in the car's columns (its bearing +- its half size + 1.5
    deg; another vehicle in the same direction counts too)."""
    b, dist, _, _ = place(r)
    half = math.degrees(math.atan2(2.5, max(dist, 1.0))) + 1.5
    u0, u1 = (int(round(wide.width / 2 - a / 180.0 * wide.width)) for a in (b + half, b - half))
    u0, u1 = max(u0, 0), min(u1 + 1, wide.width)
    return [-1 if s is None or u1 <= u0 else int((s[:, u0:u1] == SEM["vehicle"]).sum()) for s in r["segs"]]


print(f"car {k}" + ("" if k is not None else ": none (no collision, no car given)"))
print("    t      s    v   expert (reason, tta)   KeiView (tta)  KeiSim (tta)" +
      ("   car: bearing  dist  dyaw   v   pixels KV / KS" if k is not None else ""))
for r in rows:
    if round(r["t"] * 10) % 2 and r is not rows[-1]:
        continue
    line = (f"{r['t']:6.1f} {r['s']:6.1f} {r['v']:4.1f}  {r['ex']:4.1f} {r['why']:10s} {r['tta']:5.1f}   "
            f"{r['kv']:5.2f} ({r['tkv']:4.1f})  {r['ks']:5.2f} ({r['tks']:4.1f})")
    if k is not None and k < len(r["oxy"]):
        b, dist, dy, ov = place(r)
        line += f"     {b:+7.0f} {dist:5.1f} {dy:+5.0f} {ov:4.1f}" + ("   %4d / %4d" % tuple(car_px(r)) if abs(b) <= 90 else "")
    print(line)
sel = [rows[int(round(i))] for i in np.linspace(0, len(rows) - 1, 6)]
cols = []
for r in sel:
    ims = [im.copy() for im in r["ims"]]
    if k is not None and k < len(r["oxy"]):
        b, dist, _, _ = place(r)
        if abs(b) <= 90:
            u = int(round(wide.width / 2 - b / 180.0 * wide.width))
            for i in (0, 2):
                cv2.drawMarker(ims[i], (u, 6), (0, 0, 255), cv2.MARKER_TRIANGLE_DOWN, 8, 2)
    cv2.putText(ims[0], f"t={r['t']:.1f} v={r['v']:.1f} ex={r['ex']:.1f} {r['why']}", (4, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(ims[0], f"KeiView model {r['kv']:.1f}", (4, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(ims[2], f"KeiSim model {r['ks']:.1f}", (4, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1,
                cv2.LINE_AA)
    cols.append(np.concatenate(ims, 0))
cv2.imwrite(out, np.concatenate([np.concatenate(cols[:3], 1), np.concatenate(cols[3:], 1)], 0))
print("saved", out, [r["t"] for r in sel])
# zoom: the wide images around the car while the expert gives way (else at the six times), 3x, KeiView over KeiSim
if k is not None:
    ys = [r for r in rows if r["why"] == "yield" and k < len(r["oxy"]) and abs(place(r)[0]) <= 84]
    zs = [ys[int(round(i))] for i in np.linspace(0, len(ys) - 1, min(6, len(ys)))] if ys else sel
    tiles = []
    for r in zs:
        b = place(r)[0]
        u = int(round(wide.width / 2 - b / 180.0 * wide.width))
        u0 = min(max(u - 48, 0), wide.width - 96)
        pair = []
        for i, name in ((0, "KeiView"), (2, "KeiSim"), (0, "KeiView seg")):
            im = r["ims"][i].copy()
            seg = r["segs"][0]
            if name == "KeiView seg" and seg is not None:      # its vehicle pixels in red over the picture
                m = seg == SEM["vehicle"]
                im[m] = (0.4 * im[m] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
            crop = cv2.resize(im[36:100, u0:u0 + 96], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            tm = r["ks"] if name == "KeiSim" else r["kv"]
            cv2.putText(crop, f"{name} t={r['t']:.1f} {tm:.1f} (ex {r['ex']:.1f})", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            pair.append(crop)
        tiles.append(np.concatenate(pair, 0))
    zout = out[:-4] + "_zoom.png"
    cv2.imwrite(zout, np.concatenate(tiles, 1))
    print("saved", zout, [r["t"] for r in zs])
