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
    yellow time; green: go (unless the junction exit is full or the box busy). light_stop -> (reason, metres to the
    stopping point) or None."""
    from keisim.config import TL_GREEN, TL_RED, TL_YELLOW
    from keisim.expert import TL_MARGIN, light_stop

    def why(*a):
        r = light_stop(*a)
        return None if r is None else r[0]

    assert why(5.0, 10.0, TL_YELLOW, 2.4, False) == "red_light"      # 3.9 m needed: stop, time left or not
    assert why(10.0, 8.0, TL_YELLOW, 0.5, False) is None             # 8.6 m needed: too close, go on
    assert why(10.0, 12.0, TL_YELLOW, 0.5, False) == "red_light"     # still possible -> stop (v3 went on)
    assert why(10.0, 9.0, TL_RED, 0.0, False) == "red_light"         # red: 8.6 m needed at 6 m/s^2
    assert why(11.0, 8.0, TL_RED, 0.0, False) is None                # 10.4 m needed: cannot stop
    assert why(8.0, 6.0, TL_GREEN, 0.0, False) is None
    assert light_stop(3.0, 8.0, TL_GREEN, 0.0, True) == ("junction_blocked", 8.0 - TL_MARGIN)   # don't block the box
    assert why(0.5, 0.1, TL_RED, 0.0, False) == "red_light"          # v5: creeping with the bumper at the
    assert why(0.5, 0.1, TL_YELLOW, 2.0, False) == "red_light"       # line, it stays (no follow across)
    assert why(5.0, 0.1, TL_YELLOW, 2.0, False) is None              # at speed it still goes on
    # v7: slow, too close to stop at the line for a busy box (it crept up to it), it stops short of the entry
    assert light_stop(1.25, 0.35, TL_GREEN, 0.0, True) == ("junction_blocked", 0.0)
    assert light_stop(0.0, -2.0, TL_GREEN, 0.0, True, 5.0) == ("junction_blocked", 0.0)    # over the line too
    assert why(1.25, 0.35, TL_GREEN, 0.0, False) is None             # a free box: go on
    assert why(4.0, 0.35, TL_GREEN, 0.0, True) is None               # at BOX_V or faster: committed, as before
    assert why(2.0, -3.5, TL_GREEN, 0.0, True, 5.0) is None          # too close to the entry to stop short of it


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

    def car(s_center, v, nxt=None, ego=False, waits=False):     # a row of Traffic.yield_tables
        return (s_center, v, 4.5, ego, nxt, None, waits)

    def tables(s_center, v, ego=False):
        return {pred: [car(s_center, v, b, ego)]}
    assert not tr.must_yield(conn, t_green, {})
    assert tr.must_yield(conn, t_green, tables(L - 10.0, 8.0))            # about 2 s away
    assert not tr.must_yield(conn, t_green, tables(L - 80.0, 8.0))        # about 11 s away
    assert tr.must_yield(conn, t_green, {b: [car(s2 - 2.0, 0.0)]})        # standing in the crossing
    # an oncoming car queued behind one that stands still to turn right itself cannot come: no deadlock
    targets = {y[0] for y in conn.yields}
    other = next(c for c in town.lanes[pred].succ if c not in targets)            # the oncoming right turn
    queue = {pred: [car(L - 4.0, 0.0, other), car(L - 11.0, 0.0, b)]}
    assert not tr.must_yield(conn, t_green, queue)
    assert not tr.must_yield(conn, t_green, {other: [car(1.0, 0.0)], pred: [car(L - 4.0, 0.0, b)]})
    # an ego that has stood still for TrafficConfig.ego_patience_s (waits) does not count before the crossing, nor
    # do the cars queued behind it; inside the crossing it does
    assert tr.must_yield(conn, t_green, {pred: [car(L - 8.0, 0.0, b, ego=True)]})
    assert not tr.must_yield(conn, t_green, {pred: [car(L - 8.0, 0.0, b, ego=True, waits=True), car(L - 16.0, 0.0, b)]})
    assert not tr.must_yield(conn, t_green, {b: [car(s2 - 8.0, 0.0, ego=True, waits=True), car(s2 - 15.0, 0.0)]})
    assert tr.must_yield(conn, t_green, {b: [car(s2 - 2.0, 0.0, ego=True, waits=True)]})
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
    # moving_only (the late check past the wait point): a standing car with priority waits for the ego
    assert np.isfinite(tr.min_arrival(conn, t_green, tables(L - 8.0, 0.0)))
    assert np.isinf(tr.min_arrival(conn, t_green, tables(L - 8.0, 0.0), moving_only=True))
    assert tr.min_arrival(conn, t_green, tables(L - 10.0, 8.0), moving_only=True) < 2.5


def test_ego_patience_marks_a_standing_ego():
    """World.step hands the NPCs' yield tables an ego that has stood still for TrafficConfig.ego_patience_s as
    waiting, and never one with the default 0."""
    from keisim.config import town_config

    for patience in (0.0, 3.0):
        cfg = EnvConfig()
        cfg.render_rgb = cfg.render_seg = False
        cfg.town = town_config("twophase")
        cfg.traffic.ego_patience_s = patience
        env = KeiEnv(cfg)
        env.reset(town_seed=1012, episode_seed=1)
        w, waits = env.world, []
        step = w.traffic.step
        w.traffic.step = lambda *a, **k: waits.append(a[10] and a[10][-1]) or step(*a, **k)
        for _ in range(round(8.0 / (cfg.dt * cfg.action_repeat))):
            env.step([0.0, 0.0, 1.0])                                   # full brake
        assert w.ego_stood >= 4.0
        assert waits[-1] is (patience > 0.0) and not any(waits[:round(2.0 / cfg.dt)])


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


def test_mem_input_starts_as_speed_model():
    """A speed-input checkpoint loaded into a memory-input KeiPilot must behave exactly as before, whatever the
    distance since the last stop, given or not."""
    import torch

    from keipilot.model import KeiPilot

    torch.manual_seed(0)
    base = KeiPilot(pretrained=False, speed_input=True).eval()
    mem = KeiPilot(pretrained=False, speed_input=True, mem_input=True).eval()
    missing, unexpected = mem.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("mem_") for k in missing)
    img = torch.randint(0, 256, (2, 3, 160, 320), dtype=torch.uint8)
    cmd, tp, v = torch.tensor([0, 1]), torch.tensor([[20.0, 3.0], [15.0, -2.0]]), torch.tensor([0.0, 3.0])
    with torch.no_grad():
        a = base(img, cmd, tp, with_seg=False, speed=v)
        b = mem(img, cmd, tp, with_seg=False, speed=v, stand_m=torch.tensor([0.0, 2.5]))
        c = mem(img, cmd, tp, with_seg=False, speed=v)
    for k in ("path", "speed_logits", "tl_logits"):
        assert torch.allclose(a[k], b[k], atol=1e-5) and torch.allclose(a[k], c[k], atol=1e-5)


def test_stand_memory_of_old_shards(tmp_path):
    """stand_m (metres since the ego last stood) is kept where recorded; shards without it get 0 / STAND_CAP from
    their speed where no label depends on it, and stay unknown in a directory with 止まれ frames (reason 8), where
    the expert's labels do depend on it. A hidden speed hides it too."""
    from keipilot.data import LABEL_KEYS, STAND_CAP, DrivingDataset, ShardWriter, load_shards, stand_update

    assert stand_update(STAND_CAP, 0.0, 0.1) == 0.0 and stand_update(0.0, 2.0, 0.1) == 0.2
    assert stand_update(9.9, 5.0, 0.1) == STAND_CAP
    v = [0.0, 5.0, 3.0]
    for d, reasons, stand in (("grid", [0, 1, 0], None), ("osm", [0, 8, 0], None), ("new", [0, 8, 0], [0.0, 1.5, 3.0])):
        w = ShardWriter(str(tmp_path / d), "dagger", frames_per_shard=10)
        for i in range(3):
            lab = {k: (np.zeros(2, np.float32) if k == "tp" else np.zeros((10, 2), np.float32) if k == "path" else 0)
                   for k in LABEL_KEYS}
            lab.update(v=np.float32(v[i]), reason=np.int8(reasons[i]), episode=i)
            if stand:
                lab["stand_m"] = np.float32(stand[i])
            w.add(np.zeros((16, 32, 3), np.uint8), np.zeros((16, 32), np.uint8), **lab)
        w.flush()
    data, _ = load_shards([str(tmp_path / d) for d in ("grid", "osm", "new")])
    assert np.array_equal(data["stand_m"][:3], [0.0, STAND_CAP, STAND_CAP])
    assert np.isnan(data["stand_m"][3:6]).all()
    assert np.array_equal(data["stand_m"][6:], [0.0, 1.5, 3.0])
    ds = DrivingDataset(data, np.arange(9), train=False)
    assert [bool(ds[i]["stand_known"]) for i in range(9)] == [True] * 3 + [False] * 3 + [True] * 3
    hide = DrivingDataset(data, np.arange(9), train=True, speed_drop=1.0)
    assert not any(bool(hide[i]["stand_known"]) for i in range(9))


def test_stale_creep_of_old_shards(tmp_path):
    """Label version 6 (徐行 from a 止まれ line to the wait point) on older shards: in the stop zone the counterfactual
    ss_go takes the cap; past a stop, every frame with a crossing ahead that gives way is dropped (stale_creep),
    the waits as well as the faster labels, so what remains does not lean towards waiting. Version 6 shards and
    directories without 止まれ lines keep everything."""
    from keipilot.data import LABEL_KEYS, STAND_CAP, ShardWriter, load_shards, write_meta
    from keisim.expert import CROSSING_V

    nan = np.float32(np.nan)
    # (reason, stand_m, yield_tta, ss_d, ss_go, target_nolight): fast after a stop, waiting after a stop, after a stop
    # with no crossing ahead, not stood lately, in the stop zone
    frames = [(7, 2.0, 3.0, nan, nan, 5.0), (6, 2.0, 1.0, nan, nan, 0.0), (0, 2.0, nan, nan, nan, 8.0),
              (7, STAND_CAP, 3.0, nan, nan, 5.0), (8, STAND_CAP, 3.0, 1.0, 5.0, 2.0)]
    for d, version, reasons in (("osm5", 5, None), ("osm6", 6, None), ("grid5", 5, (7, 6, 0, 7, 0))):
        w = ShardWriter(str(tmp_path / d), "dagger", frames_per_shard=10)
        for i, (reason, stand, tta, ss_d, ss_go, target) in enumerate(frames):
            lab = {k: (np.zeros(2, np.float32) if k == "tp" else np.zeros((10, 2), np.float32) if k == "path" else 0)
                   for k in LABEL_KEYS}
            lab.update(v=np.float32(3.0), reason=np.int8(reasons[i] if reasons else reason), episode=0, step=i,
                       stand_m=np.float32(stand), yield_tta=np.float32(tta), ss_d=np.float32(ss_d),
                       ss_go=np.float32(ss_go), target_nolight=np.float32(target), lt_over=False, lt_d=nan,
                       lt_st=np.int8(3), lt_trem=nan, lt_blocked=False)
            w.add(np.zeros((16, 32, 3), np.uint8), np.zeros((16, 32), np.uint8), **lab)
        w.flush()
        write_meta(str(tmp_path / d), label_version=version)
    data, _ = load_shards([str(tmp_path / d) for d in ("osm5", "osm6", "grid5")])
    assert data["stale_creep"].tolist() == [True, True, False, False, False] + [False] * 10
    assert data["ss_go"][4] == np.float32(CROSSING_V) and data["ss_go"][9] == 5.0


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


def test_aux_yield_head_and_labels():
    """The auxiliary yield head starts from an older checkpoint, its loss ignores frames without a label, and the
    expert reports the earliest oncoming arrival before right turns that give way (NaN elsewhere)."""
    import torch

    from keipilot.model import KeiPilot, compute_loss
    from keisim.config import town_config

    torch.manual_seed(0)
    base = KeiPilot(pretrained=False, speed_input=True)
    aux = KeiPilot(pretrained=False, speed_input=True, aux_yield=True)
    missing = aux.load_compatible(base.state_dict())
    assert missing and all(k.startswith("aux_") for k in missing)
    img = torch.randint(0, 256, (2, 3, 160, 320), dtype=torch.uint8)
    out = aux(img, torch.tensor([2, 1]), torch.tensor([[14.0, -14.0], [20.0, 0.0]]), speed=torch.tensor([3.0, 5.0]))
    assert out["yield_tta"].shape == (2,)
    batch = {"path": torch.zeros(2, 10, 2), "speed": torch.zeros(2), "tl": torch.zeros(2, dtype=torch.long),
             "yield_tta": torch.tensor([2.5, float("nan")])}
    _, logs = compute_loss(aux, out, batch, w_aux=1.0)
    assert "aux" in logs and np.isfinite(logs["aux"])
    cfg = EnvConfig()
    cfg.render_rgb = cfg.render_seg = False
    cfg.town = town_config("twophase")
    cfg.traffic.box_rule = True
    env = KeiEnv(cfg)
    seen = []
    for ep in range(12):                        # until a route has passed a right turn that gives way
        env.reset(town_seed=1012, episode_seed=ep, route_length=600.0)
        done = False
        while not done:
            seen.append(env.plan["yield_tta"])
            _, _, done, _ = env.step(env.expert_action())
        if np.isfinite(seen).any():
            break
    seen = np.array(seen)
    assert np.isnan(seen).any() and np.isfinite(seen).any() and np.nanmax(seen) <= 8.0


def test_load_shards_fills_buffers_in_place(tmp_path):
    """load_shards preallocates the byte buffers from the shard headers: same frames and offsets as concatenating
    shard by shard, earlier frames where recorded, and none at all with with_prev=False."""
    import cv2
    from keipilot.data import LABEL_KEYS, ShardWriter, load_shards
    lab = lambda i: {k: (np.zeros(2, np.float32) if k == "tp" else np.zeros((10, 2), np.float32) if k == "path" else i)
                     for k in LABEL_KEYS}
    frames = []
    for d, with_prev in (("a", True), ("b", False)):
        w = ShardWriter(str(tmp_path / d), "expert", frames_per_shard=3)
        for i in range(5):
            rgb = np.full((16, 32, 3), 10 + 22 * len(frames), np.uint8)          # every frame its own grey
            prev = np.full((16, 32, 3), 5, np.uint8) if with_prev and i % 2 == 0 else None
            w.add(rgb, np.zeros((16, 32), np.uint8), prev=prev, **lab(i))
            frames.append((rgb, prev))
        w.flush()
    dirs = [str(tmp_path / "a"), str(tmp_path / "b")]
    data, files = load_shards(dirs)
    assert len(files) == 4 and len(data["cmd"]) == 10
    for i, (rgb, prev) in enumerate(frames):
        img = cv2.imdecode(data["jpg"][data["jpg_off"][i]:data["jpg_off"][i + 1]], cv2.IMREAD_COLOR)
        assert img.shape == rgb.shape and np.abs(img.astype(int) - rgb).max() <= 2
        a, b = data["jpg_prev_off"][i], data["jpg_prev_off"][i + 1]
        assert (b > a) == (prev is not None)
    no_prev, _ = load_shards(dirs, with_prev=False)
    assert len(no_prev["jpg_prev"]) == 0 and not np.any(np.diff(no_prev["jpg_prev_off"]))
    assert np.array_equal(no_prev["jpg"], data["jpg"]) and np.array_equal(no_prev["jpg_off"], data["jpg_off"])


def test_late_yield_after_overshoot():
    """A right turn past its wait point (a driving model came in too fast) still stops short of the oncoming lane
    when a moving car with priority would arrive before it clears the crossing; closer than LATE_COMMIT to the
    crossing it goes on. Standing cars with priority do not count there (they wait for the ego). At signals each
    crossing still ahead counts (label version 7)."""
    from keisim.config import town_config
    from keisim.expert import LATE_COMMIT

    cfg = EnvConfig()
    cfg.render_rgb = cfg.render_seg = False
    cfg.town = town_config("twophase")
    cfg.traffic.max_vehicles = cfg.traffic.max_peds = 0
    env = KeiEnv(cfg)
    for ep in range(40):                        # a route through a right turn that gives way
        env.reset(town_seed=1012, episode_seed=ep, route_length=600.0)
        jn = next((j for j in env.route.junctions if env.world.town.lanes[j["lane"]].yields), None)
        if jn is not None:
            break
    conn = env.world.town.lanes[jn["lane"]]
    tr, ego, hl = env.world.traffic, env.world.ego, env.world.ego.LENGTH / 2
    calls = []

    def plan_at(s_front, tta, v=1.0, later=float("inf")):
        # tta: the first crossing's car (or the earliest of all, before the wait point); later: the crossings after it
        tr.min_arrival = lambda c, t, on_lane, moving_only=False, yields=None: calls.append(moving_only) or (
            tta if yields is None or yields[0] == c.yields[0] else later)
        ego.v = v
        s = s_front - hl
        p0, p1 = env.route.poly.interp(np.array([s, s + 0.5]))
        ego.x, ego.y, ego.yaw = float(p0[0]), float(p0[1]), float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        return env.expert.plan(pose=(ego.x, ego.y, ego.yaw), s=s)

    s_wait, s_cross = jn["s_in"] + conn.wait_s, jn["s_in"] + conn.yields[0][1]
    assert s_cross - s_wait > LATE_COMMIT + 1.0
    p = plan_at(s_wait + 0.2, 3.0)                          # at the wait point: the usual 5 s rule
    assert p["reason"] == "yield" and p["target_speed"] == 0.0 and calls[-1] is False
    p = plan_at(s_wait + 1.0, 1.5)                          # overshot by 1 m, a car 1.5 s away: stop now
    assert p["reason"] == "yield" and p["target_speed"] == 0.0 and calls[-1] is True and p["yield_tta"] == 1.5
    p = plan_at(s_wait + 1.0, 9.0)                          # nobody coming soon: go on
    assert p["reason"] != "yield" and p["target_speed"] > 1.0
    p = plan_at(s_cross - LATE_COMMIT + 0.3, 1.5)          # almost at the crossing: clear it rather than stop in it
    assert p["reason"] != "yield" and p["target_speed"] > 1.0
    # label version 7 (signals): past the first crossing, a car about to reach a later one (the oncoming left turn
    # merging into the same exit) still stops the turn, while it is LATE_COMMIT or more short of that crossing
    assert jn["signalized"] and len(conn.yields) > 1
    s_last = jn["s_in"] + conn.yields[-1][1]
    assert s_last - LATE_COMMIT > s_cross - LATE_COMMIT + 0.3
    p = plan_at(s_cross - LATE_COMMIT + 0.3, 9.0, later=1.0)
    assert p["reason"] == "yield" and p["target_speed"] == 0.0 and p["yield_tta"] == 1.0
    p = plan_at(s_last - LATE_COMMIT + 0.3, 9.0, later=1.0)  # almost at that one too: go on
    assert p["reason"] != "yield" and p["target_speed"] > 1.0


def test_road_types_town_style():
    """'roadtypes' towns: main roads, streets, residential roads without sidewalks and narrow alleys on the
    twophase layout. The existing styles keep their settings and exported town names."""
    from keisim.config import ROAD_KINDS, town_config, town_tag
    from keisim.roadnet import NARROW_LANE

    assert town_tag(town_config("classic")) == ""
    assert town_tag(town_config("varied")) == "645f29b1" and town_tag(town_config("twophase")) == "98782ab7"
    kinds = set()
    for seed in (1000, 1010, 1012):
        T = Town(seed, town_config("roadtypes"))
        ref = Town(seed, town_config("twophase"))
        assert np.allclose([J.pos for J in T.junctions], [J.pos for J in ref.junctions])   # same layout
        assert [(r.a, r.b) for r in T.roads] == [(r.a, r.b) for r in ref.roads]
        kinds |= {r.kind for r in T.roads}
        for r in T.roads:
            spec = ROAD_KINDS.get(r.kind)
            if spec:
                assert r.spec.lane_width == spec["lane_width"] and r.spec.sidewalk == spec["sidewalk"]
            for lid in (r.lane_ab, r.lane_ba):
                L = T.lanes[lid]
                assert L.width == r.spec.lane_width and L.speed_limit == r.spec.speed_limit
                n = normals_of(L.pts)
                pts = np.concatenate([L.pts, L.pts + n * 0.9, L.pts - n * 0.9])
                assert (T.surface(pts) == 1).all(), f"lane {lid} leaves the road in town {seed}"
            if r.spec.sidewalk == 0:      # no street trees without a sidewalk
                d = [np.min(np.hypot(*(r.center - [t["x"], t["y"]]).T)) for t in T.trees]
                assert not d or min(d) > r.half_width + 0.5
        for w in T.walkways:              # sidewalk, or the 路側帯 of residential roads; alleys have none
            r = T.roads[w["road"]]
            assert r.kind != "alley" and w["in_road"] == (r.spec.sidewalk == 0)
        assert T.markings_yellow and T.speed_varies and T.in_road_walkways
        for c in (l for l in T.lanes if l.kind == "conn"):
            assert c.speed_limit == min(T.lanes[c.pred[0]].speed_limit, T.lanes[c.succ[0]].speed_limit)
        if T.truck_lanes:                 # trucks keep to wide lanes they can always drive on from
            assert all(T.lanes[l].width >= NARROW_LANE for l in T.truck_lanes)
            assert all(any(T.lanes[c].succ[0] in T.truck_lanes for c in T.lanes[l].succ) for l in T.truck_lanes)
    assert kinds == {"main", "street", "residential", "alley"}


def test_equal_turns_that_cross_give_way():
    """Two-phase junctions: of two equal movements of the same green whose paths cross (right turns at a skewed
    junction), exactly one gives way; one that already waits for the other's approach is the one."""
    from keisim.config import town_config

    n = 0
    for style, seeds in (("twophase", range(1010, 1020)), ("roadtypes", (1227, 1310))):
        for seed in seeds:
            T = Town(seed, town_config(style))
            for J in (J for J in T.junctions if J.signalized):
                conns = [T.lanes[c] for c in J.connectors]
                for i, a in enumerate(conns):
                    for b in conns[i + 1:]:
                        if a.pred[0] == b.pred[0] or a.turn != b.turn or \
                                T.lanes[a.pred[0]].signal[1] != T.lanes[b.pred[0]].signal[1]:
                            continue
                        d = np.hypot(*(a.poly.pts[:, None, :] - b.poly.pts[None, :, :]).transpose(2, 0, 1)).min()
                        if d >= 2.2:
                            continue
                        n += 1
                        a_gives = b.id in [y[0] for y in a.yields]
                        assert a_gives != (a.id in [y[0] for y in b.yields])
                        waits = lambda x, y: any(T.lanes[c].pred[0] == y.pred[0] and T.lanes[c].turn != x.turn
                                                 for c, _, _ in x.yields)
                        if waits(a, b) != waits(b, a):
                            assert a_gives == waits(a, b)
    assert n >= 4


def test_road_types_narrow_bends_and_expert():
    """Narrow bends are taken one at a time (the outer turn waits clear of the inner one); the expert completes a
    route through residential roads and alleys, keeping each road's speed limit."""
    from keisim.config import town_config
    from keisim.roadnet import NARROW_CLEAR

    T = Town(1105, town_config("roadtypes"))
    n = 0
    for J in T.junctions:
        if J.signalized:
            continue
        for c in J.connectors:
            L = T.lanes[c]
            if not L.yields:
                continue
            n += 1
            assert L.turn == "right" and -12.0 <= L.wait_s <= L.yields[0][1]
            sp = L.wait_s
            front = L.poly.interp1(sp) if sp >= 0 else L.poly.pts[0] + sp * L.poly.seg_dir[0]
            others = np.concatenate([T.lanes[b].poly.pts for b, _, _ in L.yields])
            assert np.min(np.hypot(*(others - np.asarray(front)).T)) >= NARROW_CLEAR
    assert n >= 2
    cfg = EnvConfig()
    cfg.town = town_config("roadtypes")
    cfg.traffic.box_rule = cfg.traffic.release_hidden = True
    env = KeiEnv(cfg)
    env.reset(town_seed=1105, episode_seed=55, render=False)
    done, over, kinds = False, 0, set()
    while not done:
        _, _, done, info = env.step(env.expert_action())
        lane = env.world.town.lanes[env.route.lane_at(env.s_ego)]
        over += env.world.ego.v > lane.speed_limit + 0.5
        if lane.kind == "road":
            kinds.add(env.world.town.roads[lane.road].kind)
    assert info["status"] == "success", info
    assert over == 0 and {"residential", "alley"} <= kinds


def test_lead_edge_focus_geometry():
    """--focus_lead_edge (scripts/collect.py): a car ahead in the lane is in view; one close at the front left, or
    beside / behind the camera, is partly outside its 100 deg; a quarter circle of radius 10 m turns about 92 deg over
    the path's next 15 m, a straight path 0 (also when the last waypoints repeat at the end of a route)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from collect import corners_outside_fov, route_turn_deg
    cam = EnvConfig().camera
    ego = (0.0, 0.0, 0.0)
    assert not corners_outside_fov(cam, ego, (10.0, 0.0), 0.0, 2.2, 0.85)
    assert not corners_outside_fov(cam, ego, (30.0, -3.5), 0.0, 2.2, 0.85)
    assert corners_outside_fov(cam, ego, (6.0, 5.0), 0.0, 2.2, 0.85)        # front corner in view, rear one out
    assert corners_outside_fov(cam, ego, (-2.0, 3.5), 0.0, 2.2, 0.85)
    s = np.arange(1, 11) * 2.0
    arc = np.stack([10 * np.sin(s / 10), 10 * (1 - np.cos(s / 10))], 1)
    assert abs(route_turn_deg(arc) - np.degrees(1.6)) < 0.5
    line = np.stack([np.minimum(s, 12.0), np.zeros_like(s)], 1)
    assert route_turn_deg(line) == 0.0


def test_visible_from_field_of_view_and_buildings():
    """Traffic.visible_from: a car counts when a corner is inside the camera's horizontal field of view, within
    range, and (given a town) the line to that corner does not pass through a building."""
    import math
    from keisim.geometry import rays_clear

    sq = np.array([[[9, -1], [11, -1]], [[11, -1], [11, 1]], [[11, 1], [9, 1]], [[9, 1], [9, -1]]], float)
    assert rays_clear((0, 0), [[20, 0], [20, 5], [9, 0]], sq).tolist() == [False, True, True]   # an edge at the end point

    cfg = EnvConfig()
    cfg.render_rgb = cfg.render_seg = False
    env = KeiEnv(cfg)
    for ep in range(20):
        env.reset(town_seed=5, episode_seed=ep, route_length=500.0)
        if env.world.traffic.n >= 5:
            break
    tr = env.world.traffic
    tr.xy[:] = 1000.0                                           # everyone far away, then five cars around (0, 0)
    tr.xy[:5] = [[20, 0], [-20, 0], [20, 30], [200, 0], [30, 0]]
    tr.yaw[:5] = 0.0
    tr.dims[:5, :2] = [4.0, 1.7]

    class Walls:                                                # one building between the camera and car 4
        buildings = [{"x": 25.0, "y": 0.0, "yaw": 0.0, "l": 2.0, "w": 6.0}]
        _building_edges = None
        building_edges = Town.building_edges

    narrow = tr.visible_from((0, 0), 0.0, math.radians(50), 110.0)
    assert narrow[:5].tolist() == [True, False, False, False, True]      # behind, 56 deg off, too far
    assert tr.visible_from((0, 0), 0.0, math.radians(75), 110.0)[2]      # a 150 deg camera sees the side car
    walls = tr.visible_from((0, 0), 0.0, math.radians(50), 110.0, Walls())
    assert walls[:5].tolist() == [True, False, False, False, False]      # car 4 is behind the building
    assert not narrow[5:].any()


def test_expert_yield_visible_counts_only_cars_in_view():
    """EnvConfig.expert_yield_visible: the give-way rule's tables hold only the cars Traffic.visible_from passes,
    seen from the camera on the planned pose; without it, every car."""
    import math
    from keisim.config import town_config

    cfg = EnvConfig()
    cfg.render_rgb = cfg.render_seg = False
    cfg.town = town_config("twophase")
    env = KeiEnv(cfg)
    for ep in range(40):                        # a route through a turn that gives way, with traffic
        env.reset(town_seed=1012, episode_seed=ep, route_length=600.0)
        jn = next((j for j in env.route.junctions if env.world.town.lanes[j["lane"]].yields), None)
        if jn is not None and env.world.traffic.n >= 10:
            break
    conn = env.world.town.lanes[jn["lane"]]
    tr, ego, hl = env.world.traffic, env.world.ego, env.world.ego.LENGTH / 2
    seen = []
    tr.min_arrival = lambda c, t, on_lane, moving_only=False: seen.append(sum(map(len, on_lane.values()))) or 9.0
    s = jn["s_in"] + conn.wait_s - 3.0 - hl                    # 3 m before the wait point
    p0, p1 = env.route.poly.interp(np.array([s, s + 0.5]))
    pose = (float(p0[0]), float(p0[1]), float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
    env.expert.plan(pose=pose, s=s)
    assert seen[-1] == tr.n
    env.cfg.expert_yield_visible = True
    env.expert.plan(pose=pose, s=s)
    c = cfg.camera
    cam = (pose[0] + c.x * math.cos(pose[2]), pose[1] + c.x * math.sin(pose[2]))
    vis = tr.visible_from(cam, pose[2], math.radians(c.fov_deg) / 2, c.max_dist, env.world.town)
    assert seen[-1] == int(vis.sum()) < tr.n


def test_suburban_two_lane_main_roads():
    """'suburban' towns: the main roads have two lanes each way (片側2車線), kerb lane first. The cars keep their
    lane: kerb-side turns go from kerb lane to kerb lane, turns across the oncoming traffic from the innermost lane
    to the innermost lane, straights in the same lane; every lane has somewhere to go and no two lanes of a road
    merge. The other styles keep their settings and exported town names; the expert completes a route on both lanes."""
    from keisim.config import town_config, town_tag

    assert town_tag(town_config("roadtypes")) == "dae67d73"
    for seed in (2000, 2001, 2002, 2003):
        T = Town(seed, town_config("suburban"))
        n2 = 0
        for r in T.roads:
            n = r.spec.lanes
            assert n == 1 or r.kind == "main"
            n2 += n == 2
            assert len(r.lanes_ab) == len(r.lanes_ba) == n and r.lane_ab == r.lanes_ab[0]
            for k, lid in enumerate(r.lanes_ab + r.lanes_ba):
                L = T.lanes[lid]
                assert L.lane_index == k % n and L.nlanes == n and L.succ
                mid = L.pts[len(L.pts) // 2]
                d = np.min(np.hypot(*(r.center - mid).T))
                assert abs(d - (n - L.lane_index - 0.5) * r.spec.lane_width) < 0.3
                nrm = normals_of(L.pts)
                pts = np.concatenate([L.pts, L.pts + nrm * 0.9, L.pts - nrm * 0.9])
                assert (T.surface(pts) == 1).all(), f"lane {lid} leaves the road in town {seed}"
        assert n2 >= 3
        for J in T.junctions:
            into = {}
            for c in (T.lanes[c] for c in J.connectors):
                a, b = T.lanes[c.pred[0]], T.lanes[c.succ[0]]
                assert into.setdefault((b.id, a.road), c.id) == c.id           # no merges of one road's lanes
                if len(J.arms) >= 3 and c.turn == "left":                       # left-hand traffic: kerb side
                    assert a.lane_index == b.lane_index == 0
                elif len(J.arms) >= 3 and c.turn == "right":
                    assert (a.lane_index, b.lane_index) == (a.nlanes - 1, b.nlanes - 1)
                else:
                    assert b.lane_index == a.lane_index
    cfg = EnvConfig()
    cfg.town = town_config("suburban")
    cfg.traffic.box_rule = cfg.traffic.release_hidden = True
    env = KeiEnv(cfg)
    env.reset(town_seed=2001, episode_seed=69, render=False)
    T = env.world.town
    done, used = False, set()
    while not done:
        _, _, done, info = env.step(env.expert_action())
        lane = T.lanes[env.route.lane_at(env.s_ego)]
        if lane.kind == "road" and lane.nlanes == 2:
            used.add(lane.lane_index)
    assert info["status"] == "success", info
    assert used == {0, 1}


def test_lane_change_routes():
    """Suburban routes change lanes where the next movement needs it (now and then where it does not): the path
    blends from one lane into its sibling inside the lane's window, route positions map exactly onto each lane
    outside the blend, the stop lines stay where they are, and a change moved later (Route.retime_change) leaves the
    path before it as it was. Routes in towns without multi-lane roads are those of the code before lane changes."""
    import hashlib

    from keisim.config import town_config
    from keisim.route import LC_ENTRY, LC_EXIT, LC_MIN, random_route

    m = hashlib.sha1()
    for style, seed in (("classic", 1000), ("roadtypes", 1105)):
        T = Town(seed, town_config(style))
        for ep in range(5):
            r = random_route(T, np.random.default_rng(ep), 600.0)
            assert not r.changes
            m.update(np.round(r.poly.pts, 6).tobytes())
            m.update(np.round(np.asarray(r.offsets), 6).tobytes())
            m.update(repr((r.lanes, round(r.s_start, 6), round(r.s_end, 6),
                           [(round(float(a), 6), b) for a, b in r.stops])).encode())
    assert m.hexdigest()[:16] == "1533b9616db393e2"
    T = Town(2000, town_config("suburban"))
    n = 0
    for ep in range(30):
        r = random_route(T, np.random.default_rng(ep), 1500.0)
        assert np.hypot(*np.diff(r.poly.pts, axis=0).T).max() < 0.55          # no jumps
        for s_stop, lid in r.stops:
            L = T.lanes[lid]
            assert np.allclose(r.poly.interp(s_stop), L.poly.interp(L.stop_s), atol=1e-6)
        for ch in r.changes:
            n += 1
            A, B = T.lanes[ch["from"]], T.lanes[ch["to"]]
            assert B.id in A.siblings and A.road == B.road and ch["lc"] >= LC_MIN - 1e-9
            assert LC_ENTRY - 1e-9 <= ch["a_min"] <= ch["a0"] <= ch["a_max"]
            assert ch["a_max"] + ch["lc"] <= A.length - LC_EXIT + 1e-6
            k = ch["k"]
            for s in np.linspace(r.offsets[k - 1] + 0.1, ch["s_a"], 10):
                assert np.allclose(r.poly.interp(s), A.poly.interp(r.local_s(s, k - 1)), atol=1e-6)
            for s in np.linspace(ch["s_b"], r.offsets[k + 1] - 0.1, 10):
                assert np.allclose(r.poly.interp(s), B.poly.interp(r.local_s(s, k)), atol=1e-6)
            assert r.lane_at(ch["s_a"] + 1.0) == A.id and r.lane_at(ch["s_b"] - 1.0) == B.id
        if r.changes and r.changes[0]["a0"] + 5.0 < r.changes[0]["a_max"]:
            ch = r.changes[0]
            before = r.poly.interp(np.linspace(0.0, ch["s_a"], 50))
            end = r.poly.interp(r.s_end)
            r.retime_change(0, ch["a0"] + 5.0)
            assert abs(r.changes[0]["s_a"] - ch["s_a"] - 5.0) < 1e-6
            assert np.allclose(r.poly.interp(np.linspace(0.0, ch["s_a"], 50)), before, atol=1e-9)
            assert np.allclose(r.poly.interp(r.s_end), end, atol=1e-6)
    assert n >= 20


def test_expert_lane_change_waits_for_room():
    """The expert changes lanes only with room in the target lane: on this route a car there makes it move the
    change on along its window (and slow down to let the car pass) before it changes, with room behind and ahead,
    and completes the route without touching the cars in the target lane."""
    from keisim.config import town_config
    from keisim.geometry import box_corners, obb_overlap

    cfg = EnvConfig()
    cfg.town = town_config("suburban")
    cfg.traffic.box_rule = cfg.traffic.release_hidden = True
    env = KeiEnv(cfg)
    env.reset(town_seed=2001, episode_seed=113, render=False)
    R, tr, e = env.route, env.world.traffic, env.world.ego
    assert len(R.changes) == 1
    a0, moved, waited, started, last_bad = R.changes[0]["a0"], 0, 0, False, None
    done = False
    while not done:
        if not started:
            _, last_bad = env.expert.lane_change_gap(R, env.s_ego, e.v)
        _, _, done, info = env.step(env.expert_action())
        ch = R.changes[0]
        moved += ch["a0"] != a0
        a0 = ch["a0"]
        waited += env.plan["reason"] == "lane_change"
        if ch["s_a"] - 0.3 <= env.s_ego <= ch["s_b"]:
            if not started:                             # it started with room in the target lane
                assert last_bad == [], last_bad
                started = True
            on = [i for i in range(tr.n) if tr.route[i][tr.ri[i]] == ch["to"]]
            if on:
                bc = box_corners(tr.xy[on, 0], tr.xy[on, 1], tr.yaw[on], tr.dims[on, 0], tr.dims[on, 1])
                assert not obb_overlap(e.corners(), bc).any()
    assert info["status"] == "success", info
    assert started and env.s_ego > R.changes[0]["s_b"] and moved > 0 and waited > 0


def test_npc_lane_changes():
    """Suburban NPCs change lanes too (the blend of route.py, with room in the target lane): over a minute of an
    episode many do, smoothly (no jumps beyond their speed, heading along their motion) and without ever overlapping
    another car. Towns without multi-lane roads have no lane changes (Traffic.has_lc)."""
    from keisim.config import town_config
    from keisim.geometry import box_corners, obb_overlap

    cfg = EnvConfig()
    cfg.town = town_config("roadtypes")
    env = KeiEnv(cfg)
    env.reset(town_seed=1105, episode_seed=55, render=False)
    assert not env.world.traffic.has_lc
    cfg = EnvConfig()
    cfg.town = town_config("suburban")
    cfg.traffic.box_rule = cfg.traffic.release_hidden = True
    cfg.traffic.max_vehicles *= 2
    env = KeiEnv(cfg)
    env.reset(town_seed=2001, episode_seed=113, route_length=600.0, render=False)
    T, tr = env.world.town, env.world.traffic
    assert tr.has_lc
    lane0 = [tr.route[i][tr.ri[i]] for i in range(tr.n)]
    xy0 = tr.xy.copy()
    changes = 0
    for _ in range(600):
        _, _, done, info = env.step(env.expert_action())
        assert not done or info["status"] == "success", info
        lane1 = [tr.route[i][tr.ri[i]] for i in range(tr.n)]
        changes += sum(lane1[i] in T.lanes[lane0[i]].siblings for i in range(tr.n))
        d = tr.xy - xy0
        dist = np.hypot(*d.T)
        moved = (dist > 0.2) & (dist < 30.0)                     # (respawned cars jump far)
        assert (dist[dist < 30.0] <= tr.v[dist < 30.0] * 0.1 + 0.5).all()
        ang = np.abs((np.arctan2(d[:, 1], d[:, 0]) - tr.yaw + np.pi) % (2 * np.pi) - np.pi)
        assert (ang[moved] < np.radians(12.0)).all()
        bc = box_corners(tr.xy[:, 0], tr.xy[:, 1], tr.yaw, tr.dims[:, 0], tr.dims[:, 1])
        near = np.hypot(*(tr.xy[:, None] - tr.xy[None]).transpose(2, 0, 1)) < 7.0
        for i, j in zip(*np.nonzero(np.triu(near, 1))):
            assert not obb_overlap(bc[i], bc[j][None])[0], (i, j)
        lane0, xy0 = lane1, tr.xy.copy()
        if done:
            break
    assert changes >= 10


def test_ramp_town_style():
    """'ramps' towns: at-grade left-turn ramps at corners of the main roads' signalised junctions. Each ramp lane
    runs one lane width outside the approach lane A (deceleration lane) and the lane C it turns into (acceleration
    lane), linked to both (Lane.links); the junction's own left turn from A into C goes. The ramp is road surface, with
    no tree, building or signal pole on it and no sidewalk path across it. Suburban towns have no ramps."""
    from keisim.config import town_config
    from keisim.route import random_route

    assert not Town(2002, town_config("suburban")).ramps
    n_ramps = n_routes = 0
    for seed in (2000, 2002, 2003):
        T = Town(seed, town_config("ramps"))
        n_ramps += len(T.ramps)
        for rp in T.ramps:
            R, A, C = T.lanes[rp["lane"]], T.lanes[rp["from"]], T.lanes[rp["to"]]
            assert R.kind == "ramp" and not R.pred and not R.succ and R.id not in T.road_lanes
            assert R.id in A.links and C.id in R.links and A.lane_index == 0 and C.lane_index == 0
            assert not any(T.lanes[c].succ[0] == C.id for c in A.succ)           # no left turn at the junction
            a_lo, a_hi, r_lo, sc = A.links[R.id]
            for sa in np.linspace(a_lo, a_hi, 5):                             # one lane width beside A and C
                d = np.hypot(*(R.poly.interp(r_lo + (sa - a_lo) * sc) - A.poly.interp(sa)))
                assert abs(d - A.width) < 0.05
            r_lo, r_hi, c_lo, sc = R.links[C.id]
            for sr in np.linspace(r_lo, rp["r_spans"][3], 5):
                d = np.hypot(*(R.poly.interp(sr) - C.poly.interp(c_lo + (sr - r_lo) * sc)))
                assert abs(d - C.width) < 0.05
            assert T.is_drivable(R.pts).all()
            nrm = normals_of(R.pts)
            assert T.is_drivable(np.concatenate([R.pts + nrm * 1.5, R.pts - nrm * 1.5])).all()
        for t in T.trees:
            assert not T.on_ramp([t["x"], t["y"]], 0.5)[0]
        for sh in T.signal_heads:
            assert not T.on_ramp(sh["pole"], 0.3)[0]
        for w in T.walkways:
            pts = w["poly"].pts
            assert not T.on_ramp(np.concatenate([pts, (pts[1:] + pts[:-1]) / 2])).any()
        ramp_ids = {rp["lane"] for rp in T.ramps}
        for ep in range(40):                                                   # routes take the ramps
            r = random_route(T, np.random.default_rng(ep), 1500.0)
            for j in r.junctions:
                if j["lane"] in ramp_ids:
                    n_routes += 1
                    assert j["turn"] == "left" and r.command(j["s_in"] - 3.0) == 0
            assert np.hypot(*np.diff(r.poly.pts, axis=0).T).max() < 0.55
    assert n_ramps >= 4 and n_routes >= 5


def test_ramp_routes_take_a_ramp():
    """route.ramp_route (the ramp_routes suite): every route takes one of the town's ramps, starting on the road
    before it, on its approach lane or beside it (then changing into the approach lane first)."""
    from keisim.config import town_config
    from keisim.route import ramp_route

    starts = set()
    for seed in (2000, 2003):
        T = Town(seed, town_config("ramps"))
        ids = {rp["lane"] for rp in T.ramps}
        approach = {rp["from"] for rp in T.ramps}
        for ep in range(12):
            r = ramp_route(T, np.random.default_rng(ep), 600.0)
            j = [j for j in r.junctions if j["lane"] in ids]
            assert j and j[0]["s_out"] < r.s_end and j[0]["turn"] == "left"
            first = T.lanes[r.lanes[0]]
            starts.add("approach" if first.id in approach else
                       "beside" if set(first.siblings) & approach else "before")
            assert np.hypot(*np.diff(r.poly.pts, axis=0).T).max() < 0.55
    assert starts == {"approach", "beside", "before"}


def test_export_lane_graph():
    """scripts/export_town.py lane_graph, the lanes of the browser demo for 片側2車線 roads and ramps (web/src/lanes.js):
    none for towns without them; else every road and ramp lane once with its siblings and links (siblings: scale
    None, both ways; a ramp: from its approach lane, and on to the lane it merges into), every junction connector with
    KeiSim's give-way table and signal phases. The bundled towns 2000 and 2002 are the current exports."""
    import json
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from export_town import export, lane_graph
    from keisim.config import town_config, town_key

    assert lane_graph(Town(1015, town_config("twophase"))) is None
    T = Town(2000, town_config("ramps"))
    g = lane_graph(T)
    road = [l for l in T.lanes if l.kind in ("road", "ramp")]
    assert len(g["lanes"]) == len(road) and len(g["conns"]) == sum(l.kind == "conn" for l in T.lanes)
    ramps = [l for l in g["lanes"] if l["kind"] == "ramp"]
    assert len(ramps) == len(T.ramps) > 0
    for l in g["lanes"]:
        for b, lo, hi, blo, scale in l["links"]:
            if scale is None:
                assert b in l["sib"] and l["id"] in g["lanes"][b]["sib"]
            else:
                assert lo < hi and (l["kind"] == "ramp") != (g["lanes"][b]["kind"] == "ramp")
    for R in ramps:
        assert len(R["links"]) == 1 and sum(any(k[0] == R["id"] for k in l["links"]) for l in g["lanes"]) == 1
    assert any(len(l["sib"]) == 1 for l in g["lanes"]) and any("signal" in l for l in g["lanes"])
    for c in g["conns"]:
        assert g["lanes"][c["from"]]["kind"] == g["lanes"][c["to"]]["kind"] == "road"
        assert all(0 <= y[0] < len(g["conns"]) for y in c.get("yields", ())) and (("yields" in c) == ("wait" in c))
    assert any("yields" in c for c in g["conns"]) and g["truck_lanes"]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for seed in (2000, 2002):
        cfg = town_config("ramps")
        with open(os.path.join(root, "web", "towns", f"town_{town_key(seed, cfg)}.json")) as f:
            ref = json.load(f)
        assert json.loads(json.dumps(export(seed, cfg))) == ref, seed


def test_ramps_expert_and_npcs():
    """The expert takes a ramp (it changes into the deceleration lane, goes round the corner and merges from the
    acceleration lane) and completes its route; NPCs take ramps too, without any two cars overlapping."""
    from keisim.config import town_config
    from keisim.geometry import box_corners, obb_overlap

    cfg = EnvConfig()
    cfg.town = town_config("ramps")
    cfg.traffic.box_rule = cfg.traffic.release_hidden = True
    cfg.traffic.max_vehicles *= 2
    cfg.traffic.max_peds *= 2
    env = KeiEnv(cfg)
    env.reset(town_seed=2000, episode_seed=432, route_length=800.0, weather="noon", render=False)
    T, tr, R = env.world.town, env.world.traffic, env.route
    ramp_ids = {rp["lane"] for rp in T.ramps}
    ramp_k = [k for k, l in enumerate(R.lanes) if l in ramp_ids]
    assert ramp_k
    lane0 = [tr.route[i][tr.ri[i]] for i in range(tr.n)]
    npc_ramps = 0
    done = False
    while not done:
        _, _, done, info = env.step(env.expert_action())
        lane1 = [tr.route[i][tr.ri[i]] for i in range(tr.n)]
        npc_ramps += sum(lane1[i] in ramp_ids and lane0[i] not in ramp_ids for i in range(tr.n))
        lane0 = lane1
        bc = box_corners(tr.xy[:, 0], tr.xy[:, 1], tr.yaw, tr.dims[:, 0], tr.dims[:, 1])
        near = np.hypot(*(tr.xy[:, None] - tr.xy[None]).transpose(2, 0, 1)) < 7.0
        for i, j in zip(*np.nonzero(np.triu(near, 1))):
            assert not obb_overlap(bc[i], bc[j][None])[0], (i, j)
    assert info["status"] == "success", info
    assert env.route.lane_index_at(env.s_ego) > ramp_k[0] + 1                    # past the ramp, merged
    assert npc_ramps >= 1


def test_guardrail_light():
    """CameraConfig.guardrail_light: "sun" shades the guardrails by the sun's azimuth (the rail facing the sun is
    lit, the other one dark); "flat" lights them the same for any azimuth, so on a course with nothing else standing
    (nolines_guard_nobld) the camera image does not change when the sun moves to the other side."""
    import math

    from keisim.curve_course import course_route, curve_town

    T = curve_town("nolines_guard_nobld", radii=(15.0,))
    route, _ = course_route(T)
    imgs = {}
    for light in ("sun", "flat"):
        cfg = EnvConfig()
        cfg.traffic.max_vehicles = 0
        cfg.traffic.max_peds = 0
        cfg.render_seg = False
        cfg.camera.guardrail_light = light
        env = KeiEnv(cfg)
        env.reset(town_seed=T, episode_seed=0, weather="noon", route=route)
        for az in (90.0, -90.0):                        # the sun on the car's left, then on its right
            e = math.radians(55.0)
            env.weather.sun_dir = np.array([math.cos(e) * math.cos(math.radians(az)),
                                            math.cos(e) * math.sin(math.radians(az)), math.sin(e)])
            imgs[light, az] = env._obs()["rgb"].astype(int)
    assert np.abs(imgs["sun", 90.0] - imgs["sun", -90.0]).max() > 40
    assert np.abs(imgs["flat", 90.0] - imgs["flat", -90.0]).max() == 0


def test_lots_off_the_road():
    """scripts/export_town.py: KeiView's lots cover no road, sidewalk or guardrail. A KeiSim building's lot (its box
    and the strip in front of it, `setback` deep with a straight front line) is cut back where that strip would reach
    onto them (the inside of a curve), and an extra parcel that would is not placed."""
    import math

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from export_town import (LOT_CLEAR_TOL, building_frontage, export_town, lot_overlap_depth, occupied_mask,
                             road_samples, surface_masks)
    from keisim.curve_course import curve_town
    from keisim.geometry import box_corners

    T = curve_town("lines_guard", radii=(15.0, 25.0))
    d = export_town(T)
    R, road, walk = surface_masks(T)
    occ = occupied_mask(T, R, road, walk)
    samples = road_samples(T)
    cut = 0
    for b, kb in zip(d["buildings"], T.buildings):
        n = np.array([-math.sin(b["yaw"]) * b["side"], math.cos(b["yaw"]) * b["side"]])
        c = np.array([b["x"], b["y"]])
        mid = c - n * (b["setback"] / 2)
        lot = box_corners(mid[0], mid[1], b["yaw"], b["l"], b["w"] + b["setback"])
        assert lot_overlap_depth(R, occ, lot, c - n * (b["w"] / 2 + b["setback"]), n) == 0.0, b
        cut += b["setback"] < building_frontage(T, kb, samples)["setback"] - 0.05
    assert cut >= 3
    for p in d["parcels"]:
        f = np.array([math.cos(p["face"]), math.sin(p["face"])])
        c = np.array([p["fx"], p["fy"]]) - f * p["d"] / 2
        lot = box_corners(c[0], c[1], p["face"] + math.pi / 2, p["w"], p["d"])
        assert lot_overlap_depth(R, occ, lot, np.array([p["fx"], p["fy"]]), -f) == 0.0, p
    assert LOT_CLEAR_TOL < 0.2 and len(d["parcels"]) > 20


def test_curve_scenes():
    """keisim/curve_course.py training scenes (scripts/collect.py --curve_scenes): random winding roads that never
    come back near themselves, of tight curves mostly, with the guardrail's side, height and the paint drawn per scene;
    the rail's height reaches KeiSim's mesh and the export; the route's target point stays 50 m ahead; the expert
    drives them in its lane; and a collection on them writes only scene frames, all in focus."""
    import json
    import math
    import multiprocessing as mp
    import tempfile

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from collect import parse_args, worker
    from export_town import export_town

    from keipilot.data import load_shards
    from keisim.curve_course import SCENE_TP_AHEAD, scene_route, scene_town
    from keisim.render.meshes import StaticScene

    seen = set()
    for seed in range(12):
        T = scene_town(seed)
        sp, curves = T.course["spec"], T.course["curves"]
        seen.add(sp["guardrail"])
        assert len(curves) >= 4 and min(c["radius"] for c in curves) >= 10.0
        P = T.course["centre"][::10]
        d = np.hypot(*(P[:, None] - P[None, :]).transpose(2, 0, 1))
        far = np.abs(np.arange(len(P))[:, None] - np.arange(len(P))[None, :]) * 5.0 > 80.0
        assert d[far].min() > 29.0
        assert scene_town(seed).course["spec"] == sp                       # a seed is one scene
        if sp["guardrail"]:
            assert all(g["beam"] == sp["beam"] for g in T.guardrails) and 0.35 <= sp["beam"][0] < sp["beam"][1] <= 0.95
    assert {"left", "right", "both"} <= seen
    T = scene_town(5)                                                       # left rail, no paint, tight curves
    assert T.course["spec"]["guardrail"] == "left" and not T.markings
    z = StaticScene(T).faces["verts"][..., 2]
    assert abs(z.max() - T.course["spec"]["beam"][1]) < 1e-4
    assert export_town(T)["guardrails"][0]["beam"] == [round(v, 3) for v in T.course["spec"]["beam"]]
    cfg = EnvConfig()
    cfg.traffic.max_vehicles = 0
    cfg.traffic.max_peds = 0
    cfg.render_rgb = cfg.render_seg = False
    env = KeiEnv(cfg)
    route = scene_route(T, np.random.default_rng(1), 300)
    env.reset(town_seed=T, episode_seed=1, route=route)
    worst, done = 0.0, False
    while not done and env.s_ego < route.s_end - 3.0:
        e, s = env.world.ego, env.s_ego
        p, h = route.poly.interp1(s), route.poly.heading1(s)
        worst = max(worst, abs(-(e.x - p[0]) * math.sin(h) + (e.y - p[1]) * math.cos(h)))
        if s < route.s_end - SCENE_TP_AHEAD - 5:
            tp = route.target_point_world(s)
            assert abs(route.poly.project(tp)[0] - (s + SCENE_TP_AHEAD)) < 1.0
        _, _, done, _ = env.step(env.expert_action())
    assert not done and worst < 0.45, (env.status, worst)
    with tempfile.TemporaryDirectory() as out:
        args = parse_args(["--out", out, "--frames", "40", "--workers", "1", "--curve_scenes", "1", "--shard", "20",
                           "--route_min", "120", "--route_max", "150"])
        assert worker(0, args, 40, out, mp.Value("i", 0)) >= 40
        with open(os.path.join(out, "stats_w00.json")) as f:
            st = json.load(f)
        assert st["frames"] >= 40 and st["scene"] == st["frames"] == st["focus"]
        data, _ = load_shards([out], with_prev=False)
        assert len(data["v"]) >= 40 and np.median(data["tp"][:, 0]) > 30.0


def test_weak_course():
    """keisim/curve_course.py weak_town: one road of the curves KeiPilot got wrong, never near itself; the guardrails
    (RoadSpec.guardrail_spans) stand only along their pieces, on their side, at their height, KeiSim's collision
    boxes with them; either way, a piece's rails are named from the driver's seat (the road's left is the car's
    right in reverse); the expert drives it both ways in its lane, and curve_test.py reports every piece. The bundled
    browser copies (web/towns/town_curve_*.json, scripts/export_town.py --curve) are the current exports."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from curve_test import run_one

    from keisim.curve_course import (RAIL_PAD, WEAK_LOW_BEAM, WEAK_PIECES, course_pieces, course_route, weak_line,
                                     weak_town)
    from keisim.geometry import Polyline
    from keisim.roadnet import GUARDRAIL_OFFSET

    pts, pieces = weak_line()
    P = pts[::10]
    far = np.abs(np.arange(len(P))[:, None] - np.arange(len(P))[None, :]) * 5.0 > 80.0
    assert np.hypot(*(P[:, None] - P[None, :]).transpose(2, 0, 1))[far].min() > 45.0
    T = weak_town("weak")
    assert T.name == "curve_weak_w1" and not T.markings and not T.buildings and len(weak_town("weak_lines").markings) > 50
    road, centre = T.roads[0], Polyline(T.course["centre"])
    assert len(T.guardrails) == sum(len(p["rails"]) for p in pieces)
    for g in T.guardrails:                       # each rail: beside its piece (and RAIL_PAD on), on its side
        s_on = [centre.project(q)[0] for q in g["pts"][[0, len(g["pts"]) // 2, -1]]]
        p = next(p for p in pieces if p["s0"] - RAIL_PAD - 3 <= s_on[1] <= p["s1"] + RAIL_PAD + 3)
        assert abs(s_on[0] - (p["s0"] - RAIL_PAD)) < 3 and abs(s_on[-1] - (p["s1"] + RAIL_PAD)) < 3, (p["id"], s_on)
        side = "left" if g["side"] > 0 else "right"
        beam = dict(p["rails"])[side]
        assert (beam is None) == ("beam" not in g) and (beam is None or tuple(g["beam"]) == WEAK_LOW_BEAM)
        d = np.hypot(*(g["pts"][::5, None, :] - road.center[None, ::2]).transpose(2, 0, 1)).min(axis=1)
        assert np.all(np.abs(d - (road.half_width + GUARDRAIL_OFFSET)) < 0.05)
    assert len(T.static_boxes) >= sum(len(g["pts"]) - 1 for g in T.guardrails)
    fwd = {c["id"]: c for c in course_pieces(T, course_route(T)[0])}
    rev = {c["id"]: c for c in course_pieces(T, course_route(T, reverse=True)[0], reverse=True)}
    assert len(fwd) == len(rev) == len(WEAK_PIECES)
    assert (fwd["L15_left"]["turn"], fwd["L15_left"]["rails"]) == ("left", "car_left")      # inner rail, car's left
    assert (rev["R15_right"]["turn"], rev["R15_right"]["rails"]) == ("left", "car_left")    # its mirror, reversed
    assert (fwd["R15_left"]["turn"], fwd["R15_left"]["rails"]) == ("right", "car_left")     # outer rail, car's left
    assert (rev["L15_right"]["turn"], rev["L15_right"]["rails"]) == ("right", "car_left")
    assert fwd["L15_left_low"]["rails"] == "car_left_low" and rev["S_LR_both"]["rails"] == "both"
    assert abs(fwd["L15_left"]["radius"] - 13.5) < 1e-6 and abs(fwd["R15_right"]["radius"] - 16.5) < 1e-6
    import json

    from export_town import export_town
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for v in ("weak", "nolines_guard_nobld"):            # the browser's copies (web/towns) are the current exports
        from keisim.curve_course import curve_town
        town = curve_town(v)
        with open(os.path.join(root, "web", "towns", f"town_{town.name}.json")) as f:
            assert json.loads(json.dumps(export_town(town))) == json.load(f), town.name
    a = {"renderer": "keisim", "agent": "expert", "video": None, "episode": 0, "weather": "noon", "tp": "ahead"}
    for reverse in (False, True):
        res = run_one(("weak", reverse, a))
        assert res["status"] == "finished" and len(res["pieces"]) == len(WEAK_PIECES), res["status"]
        assert all(c["reached"] and max(abs(c["max_in"]), abs(c["min_in"])) < 0.3 for c in res["pieces"])


def test_curve_course():
    """keisim/curve_course.py: one open two-way road (dead ends at both ends) of 90 deg left and right curves; the
    variants differ only in paint and guardrails (<variant>_nobld: also no buildings, no KeiView lots). A guardrail
    runs GUARDRAIL_OFFSET outside each road edge, is a static obstacle and is drawn (pole class); the expert drives the
    course end to end within 0.35 m of its lane."""
    from keisim.config import SEM, EnvConfig
    from keisim.curve_course import RADII, course_route, curve_town
    from keisim.render.meshes import StaticScene
    from keisim.roadnet import GUARDRAIL_OFFSET

    T = curve_town("nolines_guard")
    assert len(T.roads) == 1 and [len(J.arms) for J in T.junctions] == [1, 1] and not T.markings
    assert len(curve_town("lines").markings) > 50 and not curve_town("lines").guardrails
    assert T.buildings and T.extra_lots
    N = curve_town("nolines_guard_nobld")      # without buildings: no KeiSim boxes, no KeiView lots, same guardrails
    assert not N.buildings and not N.extra_lots and N.name != T.name
    assert [g["side"] for g in N.guardrails] == [g["side"] for g in T.guardrails]
    assert all(np.allclose(a["pts"], b["pts"]) for a, b in zip(N.guardrails, T.guardrails))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from export_town import export_town
    d = export_town(N)
    assert not d["buildings"] and not d["parcels"] and len(d["guardrails"]) == 2 and d["far_town"] is False
    road = T.roads[0]
    assert sorted(g["side"] for g in T.guardrails) == [-1.0, 1.0]
    for g in T.guardrails:
        d = np.hypot(*(g["pts"][::20, None, :] - road.center[None, ::2]).transpose(2, 0, 1)).min(axis=1)
        assert np.all(np.abs(d - (road.half_width + GUARDRAIL_OFFSET)) < 0.05)
    assert len(T.static_boxes) > sum(len(g["pts"]) - 1 for g in T.guardrails) - 1
    sc = StaticScene(T)
    assert (sc.faces["sem"] == SEM["pole"]).sum() > 1000
    E = curve_town("nolines_nobld", radii=(15.0,))     # nothing standing at all (no building, rail, tree, signal)
    assert len(StaticScene(E).faces["verts"]) == 0
    cfg = EnvConfig()
    cfg.traffic.max_vehicles = 0
    cfg.traffic.max_peds = 0
    obs = KeiEnv(cfg).reset(town_seed=E, episode_seed=0, route=course_route(E)[0])
    assert obs["rgb"].shape == (cfg.camera.height, cfg.camera.width, 3) and obs["seg"].max() > 0
    for rev in (False, True):                # every curve's span on the route has its turn and radius
        r, curves = course_route(T, reverse=rev)
        assert len(curves) == 2 * len(RADII) and {c["turn"] for c in curves} == {"left", "right"}
        for c in curves:
            k = float(np.median(r.curv[(r.poly.s > c["s0"] + 3) & (r.poly.s < c["s1"] - 3)]))
            assert (k > 0) == (c["turn"] == "left") and abs(1 / abs(k) - c["radius"]) < 0.05 * c["radius"], (rev, c)
    cfg = EnvConfig()
    cfg.traffic.max_vehicles = 0
    cfg.traffic.max_peds = 0
    cfg.render_rgb = cfg.render_seg = False
    env = KeiEnv(cfg)
    route, curves = course_route(T, reverse=True)
    env.reset(town_seed=T, episode_seed=0, route=route)
    assert env.world.traffic.n == 0 and env.world.peds.n == 0
    worst = 0.0
    done = False
    while not done and env.s_ego < route.s_end - 3.0:
        e, s = env.world.ego, env.s_ego
        p, h = route.poly.interp1(s), route.poly.heading1(s)
        worst = max(worst, abs(-(e.x - p[0]) * np.sin(h) + (e.y - p[1]) * np.cos(h)))
        _, _, done, info = env.step(env.expert_action())
    assert not done and env.s_ego >= route.s_end - 3.0 and worst < 0.35, (env.status, worst)


def test_crosswalks_at_signals():
    """Zebras at two-phase signals (Town.ped_crossings): where they are, their pedestrian light, and people crossing
    on them with it while the cars give way (TrafficConfig.crosswalks); off, nobody uses them."""
    from keisim.config import TL_GREEN, town_config
    from keisim.roadnet import PED_FLASH, PED_GREEN, PED_RED
    assert not Town(1000).ped_crossings                          # split signals (classic): no pedestrian light
    town = Town(1012, town_config("twophase"))
    pc = town.ped_crossings
    assert len(pc) >= 10
    for ci, c in enumerate(pc):
        J = town.junctions[c["junction"]]
        assert J.signalized and J.phases == 2 and c["phase"] == 1 - J.groups[c["arm"]]
        assert 6.0 < c["width"] < 25.0 and abs(np.hypot(*(c["pts"][1] - c["pts"][0])) - c["width"]) < 1e-6
        road = town.roads[c["road"]]
        for lid in list(road.lanes_ab) + list(road.lanes_ba):
            mine = [x for x in town.lane_crossings[lid] if x[0] == ci]
            assert len(mine) == 1
            _, s0, s1, u0, u1 = mine[0]
            assert 0.0 <= s0 < s1 <= town.lanes[lid].length and 0.0 < u0 < u1 < c["width"]
        # the light: green / flashing only within the green of the cars running along the zebra
        for t in np.arange(0.0, J.cycle, 0.25):
            st, left = J.ped_signal(c["phase"], t, 4.0)
            if st != PED_RED:
                assert J.signal_state(c["phase"], t) == TL_GREEN
                assert J.signal_state(J.groups[c["arm"]], t) != TL_GREEN and left > 4.0
            if st == PED_FLASH:
                assert J.remaining(c["phase"], t) <= 4.0 + 1e-6
            if st == PED_GREEN:
                assert J.remaining(c["phase"], t) > 4.0 - 1e-6

    def run(crosswalks, steps):
        cfg = EnvConfig()
        cfg.town = town_config("twophase")
        cfg.traffic.box_rule = cfg.traffic.release_hidden = True
        cfg.traffic.crosswalks = crosswalks
        cfg.render_rgb = cfg.render_seg = False
        env = KeiEnv(cfg)
        env.reset(1012, 7, route_length=500, render=False)
        pd = env.world.peds
        seen = {"set_off": 0, "set_off_red": 0, "on_zebra": 0, "crosswalk": 0}
        for _ in range(steps):
            prev = pd.mode.copy()
            _, _, done, info = env.step(env.expert_action())
            for i in np.nonzero((prev == pd.WAIT) & (pd.mode == pd.CROSS))[0]:
                c = pc[pd.cw[i]]
                seen["set_off"] += 1
                J = town.junctions[c["junction"]]          # (one env step: action_repeat world steps of dt)
                seen["set_off_red"] += all(J.ped_signal(c["phase"], env.world.t - k * cfg.dt, cfg.traffic.ped_flash)[0]
                                           == PED_RED for k in range(1, cfg.action_repeat + 1))
            seen["on_zebra"] += int(((pd.mode == pd.CROSS) & (pd.cw >= 0)).sum())
            seen["crosswalk"] += env.plan["reason"] == "crosswalk"
            if done:
                break
        return env, info, seen

    env, info, seen = run(0.0, 200)
    assert not env.world.peds.cw_on and seen["on_zebra"] == 0 and (env.world.peds.mode != env.world.peds.WAIT).all()
    env, info, seen = run(0.7, 1200)
    assert env.world.peds.cw_on and seen["set_off"] >= 5 and seen["on_zebra"] > 0
    assert seen["set_off_red"] == 0
    assert not [i for i in info["infractions"] if i["type"] in ("pedestrian", "crosswalk")], info["infractions"]
    # the expert's route crosses zebras, and the exits of its turns are among them
    kinds = [k for _, k, _ in env.route.zebras(env.world.town)]
    assert "exit" in kinds and "entry" in kinds


def test_crosswalk_blocked():
    """Pedestrians.crosswalk_blocked: a car's stretch of a zebra is taken by someone on it who has not passed it yet,
    or who gets there within the horizon, and by someone waiting at the kerb while their light is green."""
    from keisim.config import EnvConfig, town_config
    from keisim.roadnet import PED_GREEN
    cfg = EnvConfig()
    cfg.town = town_config("twophase")
    cfg.traffic.crosswalks = 1.0
    cfg.traffic.crosswalk_spawn = 0.0
    cfg.render_rgb = cfg.render_seg = False
    env = KeiEnv(cfg)
    env.reset(1012, 3, route_length=300, render=False)
    w = env.world
    pd = w.peds
    town = w.town
    c = town.ped_crossings[0]
    J = town.junctions[c["junction"]]
    t_green = next(t for t in np.arange(0.0, J.cycle, 0.1) if J.ped_signal(c["phase"], t, 4.0)[0] == PED_GREEN)
    t_red = next(t for t in np.arange(0.0, J.cycle, 0.1) if J.signal_state(c["phase"], t) == 0)
    width = c["width"]
    u0, u1 = width - 4.0, width - 1.0                   # a lane's stretch near kerb 1
    pd.cw[:] = -1
    pd.mode[:] = pd.WALK

    def put(u, side, mode, t):
        pd.cw[0], pd.cw_from[0], pd.mode[0], pd.speed[0] = 0, side, mode, 1.3
        pd.xy[0] = c["pts"][0] + c["dir"] * u
        pd._cw_t = None
        return pd.crosswalk_blocked(0, u0, u1, t)
    assert not pd.crosswalk_blocked(0, u0, u1, t_green)          # nobody there
    assert put(width - 2.0, 0, pd.CROSS, t_green)                # on the stretch
    assert put(u1 + 0.3, 0, pd.CROSS, t_green)                   # just past it, within CW_PAST: still taken
    assert not put(u1 + 0.8, 0, pd.CROSS, t_green)               # further past it: free
    assert put(u0 - 3.0, 0, pd.CROSS, t_green)                   # 3 m short at 1.3 m/s: there within 4 s
    assert not put(u0 - 7.0, 0, pd.CROSS, t_green)               # 7 m short: not within 4 s
    assert not put(u0 - 2.0, 1, pd.CROSS, t_green)               # walking away from it, past it
    assert put(u1 + 2.0, 1, pd.CROSS, t_green)                   # coming back from kerb 1 towards it
    pd.cw[0], pd.cw_from[0], pd.mode[0] = 0, 1, pd.WAIT
    pd._cw_t = None
    assert pd.crosswalk_blocked(0, u0, u1, t_green)              # waiting at kerb 1 (1 m away) at green: taken
    pd._cw_t = None
    assert not pd.crosswalk_blocked(0, u0, u1, t_red)            # ... at red: they stay put


def test_crosswalk_step_back():
    """Someone on a zebra held up by a car standing in front of them (CW_STALL) walks back to their kerb and waits
    there again: a car stuck on the zebra and the people it waits for do not stand for ever. People waiting at a
    zebra of a road without a sidewalk stand off the carriageway."""
    from keisim.config import TrafficConfig, town_config
    from keisim.traffic import CW_STALL, Pedestrians
    town = Town(1105, town_config("roadtypes"))
    cfg = TrafficConfig()
    cfg.crosswalks, cfg.crosswalk_spawn = 1.0, 0.0
    pd = Pedestrians(town, cfg, np.random.default_rng(0), 3)
    for c in town.ped_crossings:                              # waiting spots off the carriageway
        rs = town.roads[c["road"]].spec
        assert c["width"] >= 2 * rs.half_width + 0.6 - 1e-6
    c = town.ped_crossings[0]
    J = town.junctions[c["junction"]]
    t = next(t for t in np.arange(0.0, J.cycle, 0.1) if J.ped_signal(c["phase"], t, cfg.ped_flash)[0] == 1
             and J.ped_signal(c["phase"], t - 3.5, cfg.ped_flash)[0] == 1)
    pd.mode[:] = pd.WALK
    pd._start_wait(0, 0, 0, there=True)
    start = pd.xy[0].copy()
    pd.step(0.05, np.zeros((0, 2)), np.zeros(0), np.zeros(0), t=t)
    assert pd.mode[0] == pd.CROSS                             # green (and past the reaction time): off they go
    for _ in range(20):                                      # 1 m onto the zebra
        pd.step(0.05, np.zeros((0, 2)), np.zeros(0), np.zeros(0), t=t)
    car = pd.xy[0] + c["dir"] * 2.0                           # a car standing 2 m ahead of them on the zebra
    k = 0
    while pd.mode[0] == pd.CROSS and not pd.cw_retreat[0] and k < 200:
        pd.step(0.05, car[None], np.zeros(1), np.zeros(1), t=t)
        k += 1
    assert pd.cw_retreat[0] and abs(k * 0.05 - CW_STALL) < 0.2
    for _ in range(100):
        pd.step(0.05, car[None], np.zeros(1), np.zeros(1), t=t)
    assert pd.mode[0] == pd.WAIT and pd.cw[0] == 0 and pd.cw_from[0] == 0 and np.hypot(*(pd.xy[0] - start)) < 0.2


def test_scramble_crossings():
    """TownConfig.scramble: a third phase with every car light red, in which people cross on the four zebras and the
    two diagonals (Town.ped_crossings, painted inside the junction); the zebras are red in the cars' phases. With
    scramble 0 the towns are as before (two phases everywhere)."""
    from keisim.config import TL_GREEN, town_config
    from keisim.roadnet import PED_RED
    cfg = town_config("suburban")
    cfg.main_roads, cfg.scramble = 3, 1.0
    town = Town(3000, cfg)
    plain = Town(3000, town_config("suburban"))
    assert not any(J.scramble for J in plain.junctions) and all(J.phases in (0, 2) for J in plain.junctions)
    sc = [J for J in town.junctions if J.scramble]
    assert sc and all(J.phases == 3 and cfg.scramble_green[0] <= J.green <= cfg.scramble_green[1] for J in sc)
    diag = [c for c in town.ped_crossings if c.get("diagonal")]
    assert len(diag) >= 2 and len(town.crosswalks) > len(plain.crosswalks)
    for J in sc:
        zs = [c for c in town.ped_crossings if c["junction"] == J.id]
        assert zs and all(c["phase"] == 2 for c in zs)
        for t in np.arange(0.0, J.cycle, 0.5):
            cars_green = any(J.signal_state(p, t) == TL_GREEN for p in (0, 1))
            people = J.ped_signal(2, t, 4.0)[0] != PED_RED
            assert not (cars_green and people)
            if people:
                assert all(J.signal_state(p, t) != TL_GREEN for p in (0, 1))
    for c in diag:
        assert c["width"] > 20.0 and abs(float(np.hypot(*(c["pts"][1] - c["pts"][0]))) - c["width"]) < 1e-6
    # people walk over the diagonals
    ecfg = EnvConfig()
    ecfg.town = cfg
    ecfg.traffic.crosswalks, ecfg.traffic.crosswalk_spawn = 0.7, 2.0
    ecfg.traffic.box_rule = ecfg.traffic.release_hidden = True
    ecfg.render_rgb = ecfg.render_seg = False
    env = KeiEnv(ecfg)
    env.reset(3000, 5, route_length=300, render=False)
    pd = env.world.peds
    on_diag = set(i for i, c in enumerate(town.ped_crossings) if c.get("diagonal"))
    seen = 0
    for _ in range(1500):
        env.step([0.0, 0.0, 1.0])
        seen += int(((pd.mode == pd.CROSS) & np.isin(pd.cw, list(on_diag))).sum())
    assert seen > 0


def test_bigjunction_town():
    """bigjunction towns: main roads with a centre strip (RoadSpec.median) and right-turn lanes (lane kind 'pocket')
    before their signals; only the right-turn lane turns right there, cars change into it from the inner lane, and
    the expert drives one. Towns of the other styles have neither."""
    from keisim.config import town_config
    from keisim.route import Route, change_window, random_route
    plain = Town(2000, town_config("suburban"))
    assert not any(l.kind == "pocket" for l in plain.lanes) and all(r.spec.median == 0.0 for r in plain.roads)
    cfg = town_config("bigjunction")
    t = Town(3007, cfg)
    mains = [r for r in t.roads if r.spec.kind == "main"]
    assert mains and all(r.spec.median == cfg.main_median for r in mains)
    pockets = [l for l in t.lanes if l.kind == "pocket"]
    assert len(pockets) >= 4 and any(J.scramble for J in t.junctions)
    for P in pockets:
        J = t.junctions[P.end_junction]
        arm = next(a for a in J.arms if P.id in a["ins"])
        inner = t.lanes[arm["ins"][-2]]
        assert P.succ and all(t.lanes[c].turn == "right" for c in P.succ)
        assert not any(t.lanes[c].turn == "right" for c in inner.succ)      # the inner lane goes straight on
        assert change_window(inner, 0.0, P) is not None
        assert P.signal == inner.signal and P.stop_s is not None and not P.pred
        assert np.hypot(*(P.pts[-1] - inner.pts[-1])) > 2.5                  # beside it, in the centre strip
        assert (t.surface(P.pts) == 1).all()
    # random routes take them: a change from the inner lane, then its right turn
    rng = np.random.default_rng(1)
    used = 0
    for _ in range(40):
        r = random_route(t, rng, 800, turn_weights={"right": 4.0})
        for k, x in enumerate(r.lanes):
            if t.lanes[x].kind == "pocket":
                assert any(c["to"] == x for c in r.changes) and t.lanes[r.lanes[k + 1]].turn == "right"
                used += 1
    assert used > 0
    # the expert drives a right-turn lane: from 90 m before the junction on the inner lane
    P = max(pockets, key=lambda l: t.lanes[next(a for a in t.junctions[l.end_junction].arms
                                                if l.id in a["ins"])["ins"][-2]].length)
    J = t.junctions[P.end_junction]
    inner = t.lanes[next(a for a in J.arms if P.id in a["ins"])["ins"][-2]]
    conn = P.succ[0]
    route = Route(t, [inner.id, P.id, conn, t.lanes[conn].succ[0]], max(0.0, inner.length - 90.0))
    ecfg = EnvConfig()
    ecfg.town = cfg
    ecfg.blocked_timeout = 180.0           # (a scramble crossing's red lasts up to 75 s; right turns queue behind it)
    ecfg.render_rgb = ecfg.render_seg = False
    env = KeiEnv(ecfg)
    env.reset(3007, 4, render=False, route=route)
    on_pocket = 0
    for _ in range(3000):
        _, _, done, info = env.step(env.expert_action())
        on_pocket += env.route.lane_at(env.s_ego) == P.id
        if done:
            break
    assert info["status"] == "success" and not info["infractions"] and on_pocket > 0


def test_turn_rule_ordered():
    """TownConfig.turn_rule "ordered" (bigjunction towns): where two exits of a junction approach share "left" (or
    "right") by the heading alone, the one closest to straight ahead becomes "straight", so every exit of an approach
    has its own command; approaches without such a pair keep their turns. The other styles keep "heading"."""
    from keisim.config import TOWN_STYLES, town_config
    from keisim.roadnet import turn_labels
    assert turn_labels([0.71, 1.89]) == ["left", "left"]
    assert turn_labels([0.71, 1.89], "ordered") == ["straight", "left"]
    assert turn_labels([-1.97, -0.71], "ordered") == ["right", "straight"]
    assert turn_labels([-1.72, -0.51, 1.24], "ordered") == ["right", "straight", "left"]
    assert turn_labels([-1.6, 0.7], "ordered") == ["right", "left"]          # no pair: as by the heading
    assert turn_labels([0.2, 0.7, 1.5], "ordered") == ["straight", "left", "left"]   # a straight one already
    assert turn_labels([1.1, 1.9], "ordered") == ["left", "left"]            # nothing close to straight ahead
    assert [s for s, d in TOWN_STYLES.items() if d.get("turn_rule", "heading") != "heading"] == ["bigjunction"]

    def approaches(T):
        out = {}
        for L in T.lanes:
            for c in L.succ:
                C = T.lanes[c]
                if L.kind != "conn" and C.kind == "conn" and len(T.junctions[C.junction].arms) >= 3:
                    out.setdefault((C.junction, L.road), set()).add((C.turn, T.lanes[C.succ[0]].road))
        return out

    def n_ambiguous(T):
        return sum(len({t for t, _ in v}) < len({r for _, r in v}) for v in approaches(T).values())

    old = town_config("bigjunction")
    old.turn_rule = "heading"
    for seed in (3005, 3009):                    # 2 and 4 ambiguous approaches by the heading alone
        a, b = Town(seed, old), Town(seed, town_config("bigjunction"))
        assert n_ambiguous(a) > 0 and n_ambiguous(b) == 0
        same = sum(approaches(a)[k] == v for k, v in approaches(b).items())
        assert same >= len(approaches(b)) - n_ambiguous(a)


def _bike_env(density=10.0, style=None):
    from keisim.config import town_config
    cfg = EnvConfig()
    if style:
        cfg.town = town_config(style)
    cfg.traffic.bicycles = density
    cfg.render_rgb = cfg.render_seg = False
    return KeiEnv(cfg)


def test_bicycles_off_by_default_and_replay():
    """TrafficConfig.bicycles is 0 unless asked for (no rider, nothing for the renderers); with it on, the same
    seeds give the same riders."""
    assert EnvConfig().traffic.bicycles == 0.0
    env = KeiEnv(EnvConfig())
    env.reset(town_seed=3, episode_seed=7, render=False)
    assert env.world.bikes is None or env.world.bikes.n == 0
    assert getattr(env.world.scene_state(), "bike_xy", None) is None
    runs = []
    for _ in range(2):
        env = _bike_env()
        env.reset(town_seed=3, episode_seed=7, render=False)
        for _ in range(120):
            env.step(env.expert_action())
        bk = env.world.bikes
        runs.append(np.concatenate([bk.xy.ravel(), bk.v, bk.yaw, bk.roll, bk.crank]))
    assert bk.n >= 5 and np.array_equal(runs[0], runs[1])


def test_bicycles_ride_the_left_edge_and_keep_clear():
    """Riders stay on the road surface left of their lane's centre line, at a bicycle's speed, lean into their
    turns, stop for red lights, and neither they nor the cars (the expert's included) touch each other."""
    from keisim.bicycles import BIKE_LEN, BIKE_W
    from keisim.config import TL_RED
    from keisim.geometry import box_corners, obb_overlap
    for town_seed, style in ((5, None), (1015, "twophase")):
        env = _bike_env(12.0, style)
        env.reset(town_seed=town_seed, episode_seed=100 + town_seed, render=False)
        w = env.world
        bk, tr, town = w.bikes, w.traffic, w.town
        assert bk.n >= 8
        vs, ran_red, lean = [], 0, []
        for _ in range(500):
            lane0, s0 = bk.lane.copy(), bk.lane_s.copy()
            _, _, done, info = env.step(env.expert_action())
            assert not done, info
            assert np.isfinite(bk.xy).all() and np.isfinite(bk.yaw).all()
            assert town.is_drivable(bk.xy).all()
            assert (bk.v >= 0).all() and bk.v.max() < 9.5 and np.abs(bk.roll).max() < 0.6
            assert bk.side.min() > -0.6                              # never far over the centre line of their lane
            bc = box_corners(bk.xy[:, 0], bk.xy[:, 1], bk.yaw, np.full(bk.n, BIKE_LEN), np.full(bk.n, BIKE_W))
            cc = box_corners(tr.xy[:, 0], tr.xy[:, 1], tr.yaw, tr.dims[:, 0], tr.dims[:, 1])
            for j in range(bk.n):
                near = np.nonzero(np.hypot(*(tr.xy - bk.xy[j]).T) < 6.0)[0]
                assert not len(near) or not obb_overlap(bc[j], cc[near]).any()
                lane = town.lanes[bk.lane[j]]
                if (lane.stop_s is not None and bk.lane[j] == lane0[j] and s0[j] < lane.stop_s + 0.3 <= bk.lane_s[j]
                        and bk.lane_s[j] - s0[j] < 1.0 and town.signal_state_for_lane(lane.id, w.t) == TL_RED):
                    ran_red += 1
            vs.append(bk.v.mean())
            lean.append(np.corrcoef(bk.roll, bk.yawrate * bk.v)[0, 1] if bk.v.std() > 0 and np.abs(bk.yawrate).max() > 0.2
                        else np.nan)
        assert 1.5 < np.mean(vs) < 6.5                               # 5 .. 23 km/h with the waits at lights
        assert ran_red == 0
        assert np.nanmean(lean) > 0.5                                # leaning to the inside of the turn


def test_bicycles_in_the_renderers():
    """A rider in front of the camera is drawn as a vehicle (the bike) and a pedestrian (the rider); KeiView gets one
    row for each bike with its type, size, colours and pose."""
    import math
    from keisim.config import SEM
    from keisim.render.camera import CameraRenderer
    from keisim.render.keiview import KeiViewRenderer
    cfg = EnvConfig()
    cfg.traffic.bicycles = 10.0
    env = KeiEnv(cfg)
    env.reset(town_seed=1, episode_seed=3, render=False)
    w = env.world
    st = w.scene_state()
    e = w.ego
    for k in ("veh_xy", "veh_yaw", "veh_dims", "veh_kind", "veh_color", "veh_brake", "ped_xy"):
        setattr(st, k, getattr(st, k)[:0])
    st.bike_xy = np.array([[e.x + 7 * math.cos(e.yaw), e.y + 7 * math.sin(e.yaw)]])
    st.bike_yaw = np.array([e.yaw + math.pi / 2])
    st.bike_pose = np.array([[0.1, 0.1, 1.0, 2.0, 0.0, -1.0, 0.5]])
    st.bike_kind, st.bike_height, st.bike_colors = np.array([0]), np.array([1.7]), w.bikes.colors[:1]
    a = env.assets
    a.cameras
    _, seg = CameraRenderer(a.town, a.tex, a._static, cfg.cameras[0]).render(e.pose, st, env.weather)
    assert (seg == SEM["vehicle"]).sum() > 20 and (seg == SEM["pedestrian"]).sum() > 20
    veh, ped, bike = KeiViewRenderer._actors(type("R", (), {"max_dist": 150.0})(), np.array([e.x, e.y, 1.2]), st)
    assert not veh and not ped and len(bike) == 1 and len(bike[0]) == 25
    assert bike[0][4] == 0 and abs(bike[0][5] - 1.7) < 1e-6 and bike[0][18:] == [0.1, 0.1, 1.0, 2.0, 0.0, -1.0, 0.5]


def test_expert_moves_over_for_a_bicycle_only_on_its_road():
    """The expert's path moves over to the right for a bicycle it will pass, but not while the path is still in the
    junction before the bicycle's road (a right turn's wait point moved 0.5 m towards the oncoming lane: the ego
    and an oncoming car blocked each other for good, long3c 1012/9400 with bicycles)."""
    import math
    from keisim.expert import BIKE_START, PATH_S
    env = _bike_env(10.0, "twophase")
    env.reset(town_seed=1012, episode_seed=9400, route_length=600.0)
    w, ex, route = env.world, env.expert, env.route
    bk = w.bikes
    jn = next(j for j in route.junctions
              if j["s_in"] > 10.0 and w.town.lanes[route.lanes[route.lane_index_at(j["s_out"] + 1.0)]].length > 60.0)
    s_b = jn["s_out"] + 12.0                                    # one rider 12 m into the road after the junction
    h = float(route.poly.heading(s_b))
    bk.xy[:] = 1e6
    bk.xy[0] = route.poly.interp(s_b) + 1.5 * np.array([-math.sin(h), math.cos(h)])
    bk.yaw[0], bk.v[0] = h, 3.0

    def offsets(s):
        p = route.poly.interp(s)
        pose = (float(p[0]), float(p[1]), float(route.poly.heading(s)))
        scan = ex.bike_scan(route, s, pose, 5.0)
        assert scan["want"][0] > 0.3                            # there is room to move over beside this one
        none = {k: np.zeros_like(v) for k, v in scan.items()}
        d = ex.path_local(route, s, pose, scan) - ex.path_local(route, s, pose, none)
        return s + PATH_S, np.hypot(d[:, 0], d[:, 1])
    sw, off = offsets(jn["s_in"] - 3.0)                         # the ego about to enter the junction
    assert (sw < jn["s_out"]).sum() >= 3
    assert np.abs(off[sw <= jn["s_out"]]).max() < 1e-9
    sw, off = offsets(jn["s_out"] - 4.0)                        # ... and about to leave it
    assert np.abs(off[sw <= jn["s_out"]]).max() < 1e-9
    assert off[sw >= jn["s_out"] + BIKE_START].min() > 0.3
    assert np.all(np.diff(off) >= -1e-9)


def test_a_car_moved_over_for_a_bicycle_checks_where_it_drives():
    """A car that has moved over to the right for a bicycle looks for obstacles along the line it will drive, not
    along its lane's centre line: all of the shift while it is held beside the bicycle, the rest of it on the way
    back (the car waiting to turn right in the middle of a junction was hidden from it)."""
    from keisim.traffic import BIKE_BACK, corridor_gaps
    env = _bike_env(12.0, "twophase")
    env.reset(town_seed=1015, episode_seed=1115, render=False)
    tr = env.world.traffic
    n, P = tr.n, 48
    d = np.tile(np.arange(P) * 1.0, (n, 1))
    paths = np.stack([d, np.zeros((n, P))], -1)             # every car: straight along +x from the origin
    tr.yaw = np.zeros(n)
    tr.v = np.full(n, 5.0)
    tr.shift = np.full(n, 0.45)
    tr.shift_tgt = np.full(n, 0.45)                         # held beside the bicycle
    held = tr._shifted_paths(paths, d)
    assert np.allclose(held[..., 1], -0.45) and np.allclose(held[..., 0], d)
    tr.shift_tgt = np.zeros(n)                              # on the way back to the centre line
    back = tr._shifted_paths(paths, d)
    assert np.allclose(back[:, 0, 1], -0.45) and (np.diff(back[..., 1], axis=1) >= -1e-9).all()
    assert (back[:, 2, 1] < -0.45 + BIKE_BACK * 2.0 + 1e-9).all() and np.allclose(back[:, 12:, 1], 0.0, atol=0.01)
    # something standing just right of the lane: clear of the lane's own line, in the way of the car moved over
    hl, hw = np.full(n, 2.2), np.full(n, 0.85)
    obs = (np.array([[9.0, -(0.85 + 0.35 + 0.95 + 0.25)]]), np.array([0.95]), np.array([-1]), np.zeros(1), np.zeros(1))
    xy0 = np.zeros((n, 2))
    valid = np.ones((n, P), bool)
    g0, _, _ = corridor_gaps(paths, d, valid, xy0, tr.yaw, hl, hw, *obs, np.arange(n), 0.35)
    g1, _, _ = corridor_gaps(held, d, valid, xy0 + [0.0, -0.45], tr.yaw, hl, hw, *obs, np.arange(n), 0.35)
    assert np.isinf(g0).all() and (g1 < 9.0).all()


def test_expert_waits_only_so_long_for_a_bicycle_standing_beside_its_left_turn():
    """巻き込み確認: the ego stopped before a left turn lets a rider standing beside it go first, for BIKE_LEFT_WAIT s.
    After that it goes, and keeps going while the rider still stands there (the wait used to start again as soon
    as the rule let the ego move: one step every 6 s, and both stood for good, long3 1014/9201 with bicycles)."""
    import math
    from keisim.expert import BIKE_LEFT_WAIT, PATH_S
    env = _bike_env(10.0, "twophase")
    env.reset(town_seed=1014, episode_seed=9201, route_length=600.0, render=False)
    w, ex, route = env.world, env.expert, env.route
    bk, ego, L = w.bikes, w.ego, w.town.lanes
    jn = next(j for j in route.junctions if L[j["lane"]].kind == "conn" and L[j["lane"]].turn == "left")
    lin = L[L[jn["lane"]].pred[0]]
    s = jn["s_in"] - (lin.length - lin.stop_s if lin.stop_s is not None else 0.0) - ego.LENGTH / 2 - 0.3
    p, h = route.poly.interp(s), float(route.poly.heading(s))
    ego.x, ego.y, ego.yaw = float(p[0]), float(p[1]), h
    bk.xy[:] = 1e6
    bk.xy[0] = route.poly.interp(s - 1.0) + 1.5 * np.array([-math.sin(h), math.cos(h)])    # beside the rear door
    bk.yaw[0], bk.v[0] = h, 0.0
    pose = (ego.x, ego.y, h)
    step = env.cfg.dt * env.cfg.action_repeat

    def rule(v):
        scan = ex.bike_scan(route, s, pose, v)
        assert scan["along"][0]
        pts = ex.path_local(route, s, pose, scan)
        c, sn = math.cos(h), math.sin(h)
        pts = np.stack([ego.x + c * pts[:, 0] - sn * pts[:, 1], ego.y + sn * pts[:, 0] + c * pts[:, 1]], -1)
        return ex.bike_rules(route, s, v, scan, pts, np.asarray(PATH_S, float), True)
    ex._bk_wait = 0.0
    n_wait = 0
    while rule(0.0) < 0.3 and n_wait < 200:
        n_wait += 1
    assert abs(n_wait * step - BIKE_LEFT_WAIT) <= 2 * step
    for v in (0.0, 0.4, 1.2, 2.5, 0.0):                         # setting off, the rider still standing there
        assert rule(v) > 1.0
    bk.v[0] = 3.0                                               # the rider rides on: the next one is waited for again
    rule(0.0)
    assert ex._bk_wait == 0.0
