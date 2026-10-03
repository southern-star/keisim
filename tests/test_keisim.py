"""Basic invariants of the simulator (run: python -m pytest tests -q)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keisim.config import EnvConfig  # noqa: E402
from keisim.env import KeiEnv  # noqa: E402
from keisim.geometry import normals_of  # noqa: E402
from keisim.roadnet import Town  # noqa: E402


def test_lanes_on_road_and_poles_off_road():
    for seed in range(5):
        town = Town(seed)
        for lane in town.lanes:
            n = normals_of(lane.pts)
            pts = np.concatenate([lane.pts, lane.pts + n * 0.95, lane.pts - n * 0.95])
            assert (town.surface(pts) == 1).all(), f"lane {lane.id} leaves the road in town {seed}"
            if lane.kind == "road":
                assert lane.succ, "every road lane must continue into a junction"
        for sh in town.signal_heads:
            assert town.surface(sh["pole"][None])[0] != 1


def test_signal_phases_are_exclusive():
    town = Town(3)
    for J in town.junctions:
        if not J.signalized:
            continue
        for t in np.linspace(0, 200, 400):
            greens = sum(J.signal_state(p, t) != 0 for p in range(J.phases))  # not red
            assert greens <= 1


def test_deterministic_replay():
    env = KeiEnv(EnvConfig())
    poses = []
    for _ in range(2):
        env.reset(town_seed=7, episode_seed=11, render=False)
        xs = []
        for _ in range(150):
            _, _, done, _ = env.step(env.expert_action())
            xs.append(env.world.ego.pose)
            if done:
                break
        poses.append(np.array(xs))
    assert np.allclose(poses[0], poses[1])


def test_expert_completes_short_route():
    env = KeiEnv(EnvConfig())
    env.reset(town_seed=1003, episode_seed=2, route_length=250, render=False)
    done = False
    while not done:
        _, _, done, info = env.step(env.expert_action())
    assert info["status"] == "success", info
    assert info["DS"] > 0.99, info


def test_counterfactual_speed_labels_match_expert():
    """target_for_speed(v, plan) must reproduce the expert's own target at its actual speed, including the
    red / yellow / overshoot / don't-block-the-box cases, so relabelling with other speeds is exact."""
    from keisim.expert import target_for_speed

    env = KeiEnv(EnvConfig())
    seen = set()
    for town, ep in ((1003, 2), (1001, 5), (7, 11)):
        env.reset(town_seed=town, episode_seed=ep, route_length=500, render=False)
        for _ in range(3000):
            plan = env.plan
            assert abs(target_for_speed(env.world.ego.v, plan) - plan["target_speed"]) < 1e-9, plan
            seen.add(plan["reason"])
            for v in (0.0, 2.5, 6.0, 11.0):           # every counterfactual speed gives a valid label
                assert 0.0 <= target_for_speed(v, plan) <= env.world.town.cfg.speed_limit + 1e-6
            _, _, done, _ = env.step(env.expert_action())
            if done:
                break
    assert "red_light" in seen


def test_light_rules():
    """Red and yellow: stop at the line unless even a hard stop (6 m/s^2) is impossible, whatever the remaining
    yellow time; green: go (unless the junction exit is full)."""
    from keisim.config import TL_GREEN, TL_RED, TL_YELLOW
    from keisim.expert import light_stop

    assert light_stop(5.0, 10.0, TL_YELLOW, 2.4, False) == "red_light"      # 3.9 m needed: stop, time left or not
    assert light_stop(10.0, 8.0, TL_YELLOW, 0.5, False) is None             # 8.6 m needed: too close, go on
    assert light_stop(10.0, 12.0, TL_YELLOW, 0.5, False) == "red_light"     # still possible -> stop (v3 went on)
    assert light_stop(10.0, 9.0, TL_RED, 0.0, False) == "red_light"         # red: 8.6 m needed at 6 m/s^2
    assert light_stop(11.0, 8.0, TL_RED, 0.0, False) is None                # 10.4 m needed: cannot stop
    assert light_stop(8.0, 6.0, TL_GREEN, 0.0, False) is None
    assert light_stop(3.0, 8.0, TL_GREEN, 0.0, True) == "junction_blocked"  # don't block the box
    assert light_stop(0.5, 0.1, TL_RED, 0.0, False) == "red_light"          # v5: creeping with the bumper at the
    assert light_stop(0.5, 0.1, TL_YELLOW, 2.0, False) == "red_light"       # line, it stays (no follow across)
    assert light_stop(5.0, 0.1, TL_YELLOW, 2.0, False) is None              # at speed it still goes on


def test_camera_render_shapes():
    env = KeiEnv(EnvConfig())
    obs = env.reset(town_seed=5, episode_seed=1)
    assert obs["rgb"].shape == (160, 320, 3) and obs["rgb"].dtype == np.uint8
    assert obs["seg"].shape == (160, 320)
    assert obs["expert"]["path"].shape == (10, 2)


def test_corridor_search_grid_matches_dense():
    """The grid prefilter for many NPCs must find exactly what the all-pairs test finds."""
    from keisim import traffic

    rng = np.random.default_rng(3)
    K, P, M = 150, 48, 900
    origin = rng.uniform(-300, 300, (K, 2))
    heading = rng.uniform(-np.pi, np.pi, K)
    step = rng.uniform(0.5, 1.0, K)
    s = np.arange(P)[None] * step[:, None]
    curve = rng.normal(0, 0.02, K)[:, None] * s
    paths = origin[:, None] + s[..., None] * np.stack([np.cos(heading[:, None] + curve), np.sin(heading[:, None] + curve)], -1)
    valid = s <= rng.uniform(20, 45, K)[:, None]
    cxy = np.concatenate([rng.uniform(-320, 320, (M, 2)), paths[:20, 10] + 0.5])   # some right on paths
    M = len(cxy)
    args = (paths, s, valid, origin, heading, np.full(K, 2.2), np.full(K, 0.9), cxy, np.full(M, 0.9),
            rng.integers(-3, K, M), rng.uniform(0, 10, M), rng.uniform(-np.pi, np.pi, M), np.arange(K),
            np.where(rng.random(M) < 0.2, 0.9, 0.35)[None])
    keep = traffic.GRID_MIN_PAIRS
    try:
        traffic.GRID_MIN_PAIRS = 10 ** 15
        dense = traffic.corridor_gaps(*args)
        traffic.GRID_MIN_PAIRS = 0
        grid = traffic.corridor_gaps(*args)
    finally:
        traffic.GRID_MIN_PAIRS = keep
    assert np.isfinite(dense[0]).sum() >= 20
    for a, b in zip(dense, grid):
        assert np.array_equal(a, b)


def test_varied_town_style():
    """'varied' towns mix block lengths of 70-200 m; the classic towns keep their settings and file names."""
    from keisim.config import TownConfig, town_config, town_key

    assert town_key(1010, town_config("classic")) == "1010" and town_key(1010, TownConfig()) == "1010"
    key = town_key(1010, town_config("varied"))
    assert key.startswith("1010-") and len(key) == 13
    lengths = []
    for seed in (1010, 1011, 1012):
        T = Town(seed, town_config("varied"))
        lengths += [float(np.linalg.norm(np.diff(r.center, axis=0), axis=1).sum()) for r in T.roads]
        assert T.total_lane_length() > Town(seed).total_lane_length()
    assert min(lengths) > 40.0 and max(lengths) > 150.0 and np.percentile(lengths, 10) < 90.0


def test_box_rule():
    """A vehicle from another approach inside the junction makes the box busy; one from our own approach does not."""
    cfg = EnvConfig()
    cfg.render_rgb = cfg.render_seg = False
    cfg.traffic.box_rule = True
    env = KeiEnv(cfg)
    env.reset(town_seed=5, episode_seed=2)
    tr, town = env.world.traffic, env.world.town
    J = next(j for j in town.junctions if j.signalized)
    conns = [town.lanes[c] for c in J.connectors]
    a = conns[0]
    other = next(c for c in conns if c.pred[0] != a.pred[0])
    same = [c for c in conns if c.pred[0] == a.pred[0] and c.id != a.id]
    assert not tr.box_busy(a.id, tr.box_occupants({}))
    assert tr.box_busy(a.id, tr.box_occupants({other.id: [(1.0, 0.0, 4.5)]}))
    if same:
        assert not tr.box_busy(a.id, tr.box_occupants({same[0].id: [(1.0, 0.0, 4.5)]}))
    assert tr.box_busy(a.id, tr.box_occupants({}, ego_xy=J.pos))          # the ego inside the junction
    # a car past its stop line and still moving is entering (e.g. at the end of a yellow): busy as well
    lo = town.lanes[other.pred[0]]
    past = lo.stop_s + 1.5 + 2.25                                            # its front 1.5 m past the line
    assert tr.box_busy(a.id, tr.box_occupants({lo.id: [(past, 5.0, 4.5)]}))
    assert not tr.box_busy(a.id, tr.box_occupants({lo.id: [(past, 0.0, 4.5)]}))           # standing: not counted
    assert not tr.box_busy(a.id, tr.box_occupants({lo.id: [(lo.stop_s - 10.0, 5.0, 4.5)]}))  # not at the line yet
    # with the box rule, NPCs are never placed right before a stop line (they could not stop at a red light)
    for i in range(tr.n):
        lane = town.lanes[tr.route[i][0]]
        if lane.stop_s is not None and tr.ri[i] == 0:
            assert tr.s[i] <= lane.stop_s - 19.0 or tr.s[i] > lane.stop_s


def test_two_phase_signals_and_yields():
    """Two-phase towns: opposite approaches share a green; right turns across oncoming traffic yield and wait
    where they block nobody. Split-signal (classic) towns have no yields."""
    from keisim.config import town_config
    from keisim.roadnet import YIELD_CLEAR

    assert not any(getattr(l, "yields", None) for l in Town(1012).lanes)
    T = Town(1012, town_config("twophase"))
    n_right = 0
    for J in T.junctions:
        if not J.signalized:
            continue
        assert J.phases == 2
        groups = [T.lanes[a["in"]].signal[1] for a in J.arms]
        assert sorted(set(groups)) == [0, 1]
        for c in J.connectors:
            L = T.lanes[c]
            if not L.yields:
                continue
            assert L.turn != "straight"
            n_right += L.turn == "right"
            assert 0.0 <= L.wait_s <= L.yields[0][1]
            for b, _, _ in L.yields:          # yields only to the other approach of the same signal group
                B = T.lanes[b]
                assert B.pred[0] != L.pred[0] and T.lanes[B.pred[0]].signal[1] == T.lanes[L.pred[0]].signal[1]
            # the waiting car's front circle stays out of the paths it yields to
            c0 = L.poly.interp1(max(L.wait_s - 0.9, 0.0))
            others = np.concatenate([T.lanes[b].poly.pts for b, _, _ in L.yields])
            assert np.min(np.hypot(*(others - np.asarray(c0)).T)) >= YIELD_CLEAR - 0.05
    assert n_right >= 4


def test_right_turn_gap_acceptance():
    """must_yield: an oncoming car close to the crossing makes a right turn wait; a far one, or one stopping at its
    red light, does not."""
    from keisim.config import TL_GREEN, town_config

    cfg = EnvConfig()
    cfg.render_rgb = cfg.render_seg = False
    cfg.town = town_config("twophase")
    env = KeiEnv(cfg)
    env.reset(town_seed=1012, episode_seed=1)
    tr, town = env.world.traffic, env.world.town
    conn = next(l for l in town.lanes if l.kind == "conn" and l.turn == "right" and l.yields
                and len(town.junctions[l.junction].arms) == 4)
    b, _, s2 = next(y for y in conn.yields if town.lanes[y[0]].turn == "straight")
    pred = town.lanes[b].pred[0]
    L = town.lanes[pred].length
    J = town.junctions[conn.junction]
    t_green = next(t for t in np.arange(0, J.cycle, 0.1) if town.signal_state_for_lane(pred, t) == TL_GREEN)

    def tables(s_center, v, ego=False):
        return {pred: [(s_center, v, 4.5, ego, b)]}
    assert not tr.must_yield(conn, t_green, {})
    assert tr.must_yield(conn, t_green, tables(L - 10.0, 8.0))            # about 2 s away
    assert not tr.must_yield(conn, t_green, tables(L - 80.0, 8.0))        # about 11 s away
    assert tr.must_yield(conn, t_green, {b: [(s2 - 2.0, 0.0, 4.5, False, None)]})  # standing in the crossing
    # an oncoming car queued behind one that stands still to turn right itself cannot come: no deadlock
    targets = {y[0] for y in conn.yields}
    other = next(c for c in town.lanes[pred].succ if c not in targets)            # the oncoming right turn
    queue = {pred: [(L - 4.0, 0.0, 4.5, False, other), (L - 11.0, 0.0, 4.5, False, b)]}
    assert not tr.must_yield(conn, t_green, queue)
    assert not tr.must_yield(conn, t_green, {other: [(1.0, 0.0, 4.5, False, None)], pred: [(L - 4.0, 0.0, 4.5, False, b)]})
    t_red = next(t for t in np.arange(0, J.cycle, 0.1) if town.signal_state_for_lane(pred, t) != TL_GREEN
                 and town.signal_state_for_lane(pred, t + 3.0) != TL_GREEN)
    assert not tr.must_yield(conn, t_red, tables(L - 25.0, 6.0))          # stops at its red line
    stop_front = town.lanes[pred].stop_s - 0.5                               # creeping at the line: the NPC rule
    yellow = [t for t in np.arange(0, J.cycle, 0.1) if town.signal_state_for_lane(pred, t) == 1]
    if yellow:                                                               # lets it go at yellow, so it counts
        assert tr.must_yield(conn, yellow[0], tables(stop_front - 2.25, 0.5))
    assert tr.must_yield(conn, t_red, tables(L - 25.0, 6.0, ego=True))     # the ego might take it anyway
    # arrival estimates: a standing NPC reacts for 1 s, then accelerates; the ego could go at once
    assert 4.8 < tr._arrival(15.5, 0.0, 10.5) < 5.1 and tr._arrival(15.5, 0.0, 10.5, ego=True) < 3.1
    assert 3.8 < tr._arrival(40.0, 8.0, 10.5) < 4.1


def test_release_hidden_stuck_vehicles():
    """With release_hidden, long-stuck NPCs the ego camera cannot see are moved away; visible ones stay."""
    cfg = EnvConfig()
    cfg.render_rgb = cfg.render_seg = False
    cfg.traffic.release_hidden = True
    env = KeiEnv(cfg)
    env.reset(town_seed=5, episode_seed=2)
    env.step(env.expert_action())
    tr, e = env.world.traffic, env.world.ego
    assert tr.n >= 4
    fwd, left = np.array([np.cos(e.yaw), np.sin(e.yaw)]), np.array([-np.sin(e.yaw), np.cos(e.yaw)])
    spots = {0: e.xy + 30 * fwd, 1: e.xy - 30 * fwd, 2: e.xy + 25 * left, 3: e.xy + 80 * fwd}   # ahead, behind, side, far
    tr.stuck[:] = 0.0
    for i, p in spots.items():
        tr.xy[i] = p
        tr.stuck[i] = 50.0                     # past stuck_hidden_s (45 s), not yet stuck_far_s (60 s)
    tr.release_stuck(e.xy, e.yaw)
    moved = {i: not np.allclose(tr.xy[i], p) for i, p in spots.items()}
    assert moved == {0: False, 1: True, 2: True, 3: False}
    tr.stuck[3] = 61.0                          # far from the ego: the old rule
    tr.release_stuck(e.xy, e.yaw)
    assert not np.allclose(tr.xy[3], spots[3])


def test_speed_input_starts_as_camera_only_model():
    """A camera-only checkpoint loaded into a speed-input KeiPilot must behave exactly as before."""
    import torch

    from keipilot.model import KeiPilot

    torch.manual_seed(0)
    base = KeiPilot(pretrained=False).eval()
    spd = KeiPilot(pretrained=False, speed_input=True).eval()
    missing, unexpected = spd.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("speed_") for k in missing)
    img = torch.randint(0, 256, (2, 3, 160, 320), dtype=torch.uint8)
    cmd, tp = torch.tensor([0, 1]), torch.tensor([[20.0, 3.0], [15.0, -2.0]])
    with torch.no_grad():
        a = base(img, cmd, tp, with_seg=False)
        b = spd(img, cmd, tp, with_seg=False, speed=torch.tensor([0.0, 8.0]))
        c = spd(img, cmd, tp, with_seg=False)
    for k in ("path", "speed_logits", "tl_logits"):
        assert torch.allclose(a[k], b[k], atol=1e-5) and torch.allclose(a[k], c[k], atol=1e-5)


def test_brake_hold_blocks_throttle_burst():
    """A model that loses a pedestrian for a few frames mid-stop must keep braking; braking harder is never
    limited, and the hold ends on its own."""
    from keipilot.agent import BrakeHold

    h = BrakeHold()
    assert h(3.75, 7.5, 0.1) == 3.75                       # firm braking starts a hold
    burst = [h(9.98, 7.0, 0.1) for _ in range(5)]          # pedestrian lost: the model asks for 10 m/s
    assert max(burst) < 4.1                                # target rises only slowly -> still braking
    assert h(0.0, 6.0, 0.1) == 0.0                         # braking harder is never limited
    out = [h(8.0, 0.0, 0.1) for _ in range(20)]            # pedestrian gone, standing: hold runs out
    assert out[-1] == 8.0 and out[0] < 1.0


def test_history_model_starts_as_single_frame_model():
    """A single-frame checkpoint loaded into a history model must behave exactly as before, with or without a
    previous frame."""
    import torch

    from keipilot.model import KeiPilot

    for mode in ("frame", "diff"):
        _check_history_start(mode)


def _check_history_start(mode):
    import torch

    from keipilot.model import KeiPilot

    torch.manual_seed(0)
    base = KeiPilot(pretrained=False, speed_input=True).eval()
    hist = KeiPilot(pretrained=False, speed_input=True, history=True, history_mode=mode).eval()
    assert hist.load_compatible(base.state_dict()) == []
    img = torch.randint(0, 256, (2, 3, 160, 320), dtype=torch.uint8)
    prev = torch.randint(0, 256, (2, 3, 160, 320), dtype=torch.uint8)
    cmd, tp, v = torch.tensor([0, 1]), torch.tensor([[20.0, 3.0], [15.0, -2.0]]), torch.tensor([0.0, 8.0])
    with torch.no_grad():
        a = base(img, cmd, tp, with_seg=False, speed=v)
        b = hist(img, cmd, tp, with_seg=False, speed=v, img_prev=prev, has_prev=torch.tensor([True, False]))
        c = hist(img, cmd, tp, with_seg=False, speed=v)
    for k in ("path", "speed_logits", "tl_logits"):
        assert torch.allclose(a[k], b[k], atol=1e-5) and torch.allclose(a[k], c[k], atol=1e-5)


def test_agent_drives_with_history_and_brake_hold(tmp_path):
    """End to end through KeiPilotAgent.act (CPU): plain numbers out, frame history filled after history_dt."""
    import torch

    from keipilot.agent import KeiPilotAgent
    from keipilot.model import KeiPilot

    ck = tmp_path / "m.pt"
    model = KeiPilot(pretrained=False, speed_input=True, history=True, history_dt=0.2)
    torch.save({"model": model.state_dict(), "model_cfg": {"speed_input": True, "history": True, "history_dt": 0.2}}, ck)
    agent = KeiPilotAgent(str(ck), device="cpu", brake_hold=True)
    agent.reset()
    rgb = np.zeros((160, 320, 3), np.uint8)
    for _ in range(4):
        action, p = agent.act(rgb, 1, (20.0, 0.0), 3.0, 0.1)
    assert len(agent.frames) == 3 and np.isfinite(action).all() and isinstance(p["target_speed"], float)


def test_red_hold_blocks_creeping():
    """Standing with the light head sure of red/yellow: no creeping; moving, or not sure: untouched."""
    from keipilot.agent import RedHold

    h = RedHold()
    red = np.array([0.9, 0.05, 0.03, 0.02])
    assert h(0.6, 0.3, red) == 0.0                                # creeping at the line on red: held
    assert h(0.6, 0.3, np.array([0.3, 0.1, 0.5, 0.1])) == 0.6     # not sure it is red: untouched
    assert h(5.0, 6.0, red) == 5.0                                # moving: untouched
    assert h(8.0, 0.0, red) == 8.0                                # standing far from the light: may pull up
