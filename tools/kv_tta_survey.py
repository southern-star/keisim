"""Does a rig model see the cars with priority in KeiView? The model drives a Sapporo box in KeiView (evaluate.py's
--osm_town settings); at every step where the expert has a car with priority arriving (its yield_tta < 8 s), a second
copy of the model also gets KeiSim's frames of the same scene, and both auxiliary yield_tta outputs are kept with the
expert's. town: osm_sapporo (the evaluation box) or a training box variant as collect.py builds it, NAME_sSEED (e.g.
osm_sapporo_n_s1000). Prints, for the steps with the expert's yield_tta < 5 s, the share each renderer's copy predicts
>= 6 s ('nobody coming'), the medians and the mean absolute error, and saves every kept step as JSON.
  python kv_tta_survey.py ckpt town episode0 episodes out.json"""
import json, math, sys
import numpy as np
from keisim.config import EnvConfig, set_camera_rig
from keisim.env import KeiEnv
from keisim.realworld import osm_town
from keipilot.agent import KeiPilotAgent

ckpt, town_name, ep0, n_ep, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
BOXES = {"osm_sapporo": (43.088, 141.360, 43.097, 141.372), "osm_sapporo_n": (43.097, 141.360, 43.106, 141.372),
         "osm_sapporo_e": (43.088, 141.372, 43.097, 141.384), "osm_sapporo_s": (43.079, 141.360, 43.088, 141.372)}
base, seed = (town_name, "0") if town_name == "osm_sapporo" else town_name.rsplit("_s", 1)
town = osm_town(BOXES[base], cache=f"data/osm/{base}.json", name=town_name, seed=int(seed))
cfg = set_camera_rig(EnvConfig(), "wide_narrow")
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
shadow = KeiPilotAgent(ckpt)


def tta_of(a):
    """Keep the model's yield_tta (its auxiliary head) of every call."""
    box, fwd = {"tta": math.nan}, a.model.forward

    def forward(*args, **kw):
        res = fwd(*args, **kw)
        box["tta"] = float(res["yield_tta"][0]) if "yield_tta" in res else math.nan
        return res
    a.model.forward = forward
    return box


tta_kv, tta_ks = tta_of(agent), tta_of(shadow)
dt = cfg.dt * cfg.action_repeat
rows = []
for ep in range(ep0, ep0 + n_ep):
    agent.reset()
    obs = env.reset(town_seed=town, episode_seed=ep, route_length=600.0, weather="random")
    done = False
    while not done:
        e = env.world.ego
        action, mp = agent.act(obs["rgbs"], obs["command"], obs["target_point"], e.v, dt)
        tta = env.plan["yield_tta"]
        if tta < 8.0:               # a car with priority arrives within 8 s: the same scene in KeiSim
            ks = [cam.render(e.pose, env.world.scene_state(), env.weather, want_rgb=True, want_seg=False)[0]
                  for cam in env.assets.cameras]
            shadow.plan(ks, obs["command"], obs["target_point"], speed=e.v, stand_m=agent.stand_m)
            rows.append(dict(ep=ep, t=round(env.elapsed, 1), v=round(e.v, 2), reason=env.plan["reason"], tta=round(tta, 2),
                             kv=round(tta_kv["tta"], 2), ks=round(tta_ks["tta"], 2), m_kv=round(mp["target_speed"], 2),
                             expert=round(env.plan["target_speed"], 2)))
        obs, r, done, info = env.step(action)
    print(f"ep {ep}: {env.status} t={env.elapsed:.0f} infractions {[i['type'] for i in env.infractions]}, "
          f"{sum(r['ep'] == ep for r in rows)} steps with a car coming", flush=True)
env.close()
json.dump(rows, open(out, "w"))
c = [r for r in rows if r["tta"] < 5.0]
if c:
    kv, ks, lab = (np.array([r[k] for r in c]) for k in ("kv", "ks", "tta"))
    print(f"{town_name}: {len(c)} steps with the expert's yield_tta < 5 s ({len({r['ep'] for r in c})} episodes); "
          f"label median {np.median(lab):.1f}")
    for name, p in (("KeiView", kv), ("KeiSim ", ks)):
        print(f"  {name}: predicted median {np.median(p):.1f}, >= 6 s {100 * (p >= 6).mean():.1f} %, MAE {np.abs(p - lab).mean():.2f}")
