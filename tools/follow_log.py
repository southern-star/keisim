"""Per-step car-following log for the model (v0.7, safety layers) or the expert, test --dense suite settings
(same environment as scripts/evaluate.py run_route). The expert's corridor search is wrapped to also keep the
lead obstacle's speed and owner, so every step records gap / lead speed / who leads.

usage: follow_log.py CKPT|expert OUT.jsonl WORKERS [--suite test] [--routes N] [--no_safety]
--no_safety: the model drives without BrakeHold / RedHold (can it stop for the vehicle without a red light?)
"""
import json
import multiprocessing as mp
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")


def run(job):
    (town, ep), who, ckpt, sd, safety = job
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

    cfg = EnvConfig()
    style = sd.get("town_style", "classic")
    cfg.town = town_config(style)
    cfg.traffic.box_rule = cfg.traffic.release_hidden = bool(sd.get("jam_fixes", False))
    cfg.blocked_timeout = float(sd.get("blocked_timeout", 90.0))
    cfg.render_seg = False
    cfg.render_rgb = who == "model"
    cfg.traffic.vehicle_spacing = (20.0, 28.0)
    cfg.traffic.ped_spacing = (14.0, 20.0)
    cfg.traffic.max_vehicles = 130
    cfg.traffic.max_peds = 110
    cfg.traffic.ego_cross_rate = 0.3
    cap = {"classic": 1, "varied": 2, "twophase": 2, "roadtypes": 2}[style]
    cfg.traffic.max_vehicles *= cap
    cfg.traffic.max_peds *= cap
    env = KeiEnv(cfg)
    agent = None
    if who == "model":
        from keipilot.agent import KeiPilotAgent
        agent = KeiPilotAgent(ckpt, brake_hold=safety, red_hold=safety)
        agent.reset()
    obs = env.reset(town_seed=town, episode_seed=ep, route_length=sd.get("route_length", 600.0), weather="random")
    dt = cfg.dt * cfg.action_repeat
    rows, done, steps = [], False, 0
    while not done and steps < sd.get("max_steps", 4000):
        e = env.world.ego
        plan = env.plan                       # expert plan for the current state (computed in step/reset)
        lead = dict(last)
        if agent is None:
            action, mt = env.expert_action(), None
        else:
            action, mp_ = agent.act(obs["rgb"], obs["command"], obs["target_point"], e.v, dt)
            mt = float(mp_["target_speed"])
        rows.append([round(env.elapsed, 1), round(float(e.v), 3), plan["reason"], round(float(plan["gap"]), 2)
                     if np.isfinite(plan["gap"]) else None, round(lead.get("v", 0.0), 3), lead.get("o", -1),
                     None if mt is None else round(mt, 3), round(float(plan["target_speed"]), 3),
                     [round(float(x), 3) for x in action]])
        obs, _, done, _ = env.step(action)
        steps += 1
    m = env.metrics()
    return {"town": town, "episode": ep, "who": who, "status": m["status"], "DS": m["DS"],
            "infractions": m["infractions"], "dt": dt,
            "cols": ["t", "v", "reason", "gap", "lead_v", "lead_o", "target_model", "target_expert", "action"],
            "rows": rows}


if __name__ == "__main__":
    import evaluate as ev
    who_arg, out, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    suite = sys.argv[sys.argv.index("--suite") + 1] if "--suite" in sys.argv else "test"
    n = int(sys.argv[sys.argv.index("--routes") + 1]) if "--routes" in sys.argv else 999
    who = "expert" if who_arg == "expert" else "model"
    import os
    sd = ev.SUITE_DEFAULTS.get(suite, {})
    have = set()
    if os.path.exists(out):                   # resumable: skip routes already logged
        have = {(r["town"], r["episode"]) for r in map(json.loads, open(out))}
    safety = "--no_safety" not in sys.argv
    jobs = [(j, who, who_arg, sd, safety) for j in ev.SUITES[suite][:n] if tuple(j) not in have]
    print(f"{suite} dense {who}: {len(have)} done, {len(jobs)} to go", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool, open(out, "a") as f:
        for r in pool.imap_unordered(run, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["who"], r["town"], r["episode"], r["status"], round(r["DS"], 3), len(r["rows"]), flush=True)
