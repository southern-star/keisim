"""Is the vehicle ahead inside the camera's horizontal field of view? Expert driving, suite settings as
scripts/evaluate.py run_route (--dense). Every step with a vehicle lead (corridor search of the expert) records the
lead's 4 footprint corners (at 0.7 m height, tail-light height) projected into the ego camera (pinhole with the
mount offset and 8 deg pitch of CameraConfig): normalized image x = right/depth for each corner (+-inf when the
corner is beside/behind the camera plane), the bearing of the lead's centre and the turn of the route over the
next 15 m. FOVs are compared offline (visible iff |x| <= tan(fov/2)). Occlusion is not modelled.

usage: fov_log.py OUT.jsonl WORKERS [--suite test|long3]
"""
import json
import math
import multiprocessing as mp
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")


def run(job):
    (town, ep), sd = job
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    import keisim.expert as kx
    from keisim.config import EnvConfig, town_config
    from keisim.env import KeiEnv

    last = {}
    orig = kx.corridor_gaps

    def wrapped(*a, **k):
        g, lv, lo = orig(*a, **k)
        last["v"], last["o"] = float(lv[0]), int(lo[0])
        return g, lv, lo
    kx.corridor_gaps = wrapped

    style = sd.get("town_style", "classic")
    cfg = EnvConfig()
    cfg.town = town_config(style)
    cfg.traffic.box_rule = cfg.traffic.release_hidden = bool(sd.get("jam_fixes", False))
    cfg.blocked_timeout = float(sd.get("blocked_timeout", 90.0))
    cfg.render_seg = cfg.render_rgb = False
    cfg.traffic.vehicle_spacing = (20.0, 28.0)
    cfg.traffic.ped_spacing = (14.0, 20.0)
    cfg.traffic.max_vehicles = 130
    cfg.traffic.max_peds = 110
    cfg.traffic.ego_cross_rate = 0.3
    cap = {"classic": 1, "varied": 2, "twophase": 2, "roadtypes": 2}[style]
    cfg.traffic.max_vehicles *= cap
    cfg.traffic.max_peds *= cap
    cam = cfg.camera
    sp, cp = math.sin(math.radians(cam.pitch_deg)), math.cos(math.radians(cam.pitch_deg))
    env = KeiEnv(cfg)
    env.reset(town_seed=town, episode_seed=ep, route_length=sd.get("route_length", 600.0), weather="random")
    rows, done, steps = [], False, 0
    while not done and steps < sd.get("max_steps", 4000):
        e = env.world.ego
        plan = env.plan
        o, lv = last.get("o", -1), last.get("v", 0.0)
        g = float(plan["gap"])
        if o >= 0 and np.isfinite(g) and g < 40 and lv > -0.5:
            tr = env.world.traffic
            c, s = math.cos(e.yaw), math.sin(e.yaw)
            cx, cy = e.x + cam.x * c - cam.y * s, e.y + cam.x * s + cam.y * c
            L, W = tr.dims[o, 0] / 2, tr.dims[o, 1] / 2
            oc, os_ = math.cos(tr.yaw[o]), math.sin(tr.yaw[o])
            xs = []
            for a, b in ((L, W), (L, -W), (-L, W), (-L, -W)):
                px, py = tr.xy[o, 0] + a * oc - b * os_, tr.xy[o, 1] + a * os_ + b * oc
                dx, dy = px - cx, py - cy
                fwd, left, up = dx * c + dy * s, -dx * s + dy * c, 0.7 - cam.z
                zc = fwd * cp - up * sp
                xs.append(round(-left / zc, 4) if zc > 0.05 else (math.inf if left < 0 else -math.inf))
            dx, dy = tr.xy[o, 0] - cx, tr.xy[o, 1] - cy
            bearing = math.degrees(math.atan2(-dx * s + dy * c, dx * c + dy * s))   # + = left
            path = np.asarray(plan["path"], float)
            seg = np.diff(path[:, :2], axis=0)
            hd = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
            cum = np.concatenate([[0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])[:-1]
            k = np.searchsorted(cum, 15.0)
            turn = float(np.degrees(hd[min(k, len(hd) - 1)] - hd[0])) if len(hd) else 0.0
            rows.append([round(env.elapsed, 1), round(float(e.v), 2), round(g, 2), round(lv, 2), o,
                         [x if math.isfinite(x) else (1e9 if x > 0 else -1e9) for x in xs],
                         round(bearing, 1), round(turn, 1)])
        _, _, done, _ = env.step(env.expert_action())
        steps += 1
    m = env.metrics()
    env.close()
    return {"town": town, "episode": ep, "status": m["status"], "steps": steps,
            "cols": ["t", "v", "gap", "lead_v", "lead_o", "corner_x", "bearing_deg", "route_turn_15m_deg"],
            "rows": rows}


if __name__ == "__main__":
    import evaluate as ev
    out, workers = sys.argv[1], int(sys.argv[2])
    suite = sys.argv[sys.argv.index("--suite") + 1] if "--suite" in sys.argv else "test"
    sd = ev.SUITE_DEFAULTS.get(suite, {})
    have = set()
    if os.path.exists(out):
        have = {(r["town"], r["episode"]) for r in map(json.loads, open(out))}
    jobs = [(j, sd) for j in ev.SUITES[suite] if tuple(j) not in have]
    print(f"{suite} dense expert fov: {len(have)} done, {len(jobs)} to go", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool, open(out, "a") as f:
        for r in pool.imap_unordered(run, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["town"], r["episode"], r["status"], r["steps"], len(r["rows"]), flush=True)
