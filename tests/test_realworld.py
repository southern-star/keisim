"""keisim/realworld.py on a hand-made Overpass-style answer (offline), and the refactored Town against the bundled
export (the procedural towns must not change)."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keisim.config import EnvConfig  # noqa: E402
from keisim.realworld import Projection, osm_town, road_kind, town_config  # noqa: E402

LAT0, LON0 = 43.09, 141.366
BBOX = (LAT0 - 0.0025, LON0 - 0.0035, LAT0 + 0.0025, LON0 + 0.0035)


def mini_osm():
    """A 3 x 3 grid of 90 m blocks (residential, the middle row a tertiary street with signals at its centre), a
    bend road around the north-east corner, a side road leaving the x = 0 road 20 m from a junction (the two
    junctions get contracted into one), and what the importer must drop: a one-way link, a dead-end stub, a service
    road. Three buildings, one of them on a road."""
    proj = Projection(LAT0, LON0)
    nodes, ways, nid, wid = {}, [], [1], [100]

    def node(x, y, tags=None):
        lat, lon = proj.latlon(x, y)
        nodes[nid[0]] = {"type": "node", "id": nid[0], "lat": float(lat), "lon": float(lon), **({"tags": tags} if tags else {})}
        nid[0] += 1
        return nid[0] - 1

    def way(ids, tags):
        ways.append({"type": "way", "id": wid[0], "nodes": ids, "tags": tags})
        wid[0] += 1

    grid = {(i, j): node(90.0 * (i - 1), 90.0 * (j - 1), {"highway": "traffic_signals"} if (i, j) == (1, 1) else None)
            for i in range(3) for j in range(3)}
    for j in range(3):                                    # east-west roads (one way each, through the junctions)
        way([grid[(i, j)] for i in range(3)], {"highway": "tertiary" if j == 1 else "residential"})
    near = node(0.0, -70.0)                                # 20 m north of grid (1, 0): too close for two junctions
    for i in range(3):                                    # north-south roads
        ids = [grid[(i, j)] for j in range(3)]
        way(ids[:1] + [near] + ids[1:] if i == 1 else ids, {"highway": "residential"})
    way([near, grid[(0, 1)]], {"highway": "residential"})
    # extras: a bend road around the north-east corner, with a shape vertex
    bend = node(130.0, 50.0)
    way([grid[(2, 2)], node(130.0, 90.0), bend, grid[(2, 1)]], {"highway": "residential"})
    way([grid[(0, 0)], node(-90.0, -140.0)], {"highway": "residential"})                   # dead-end stub
    way([grid[(1, 0)], node(0.0, -140.0), grid[(2, 0)]], {"highway": "residential", "oneway": "yes"})
    way([grid[(0, 1)], node(-140.0, 0.0)], {"highway": "service"})
    for cx, cy in ((45.0, 45.0), (-45.0, 45.0), (0.0, 30.0)):                             # the last one is on a road
        ring = [node(cx + dx, cy + dy) for dx, dy in ((-5, -4), (5, -4), (5, 4), (-5, 4))]
        way(ring + [ring[0]], {"building": "house", "building:levels": "2"})
    return {"elements": list(nodes.values()) + ways}


def test_road_kind_filters():
    assert road_kind({"highway": "residential"}) == "residential"
    assert road_kind({"highway": "tertiary", "lanes": "2"}) == "street"
    assert road_kind({"highway": "secondary", "lanes": "4"}) is None
    assert road_kind({"highway": "residential", "oneway": "yes"}) is None
    assert road_kind({"highway": "service"}) is None
    assert road_kind({"highway": "primary"}, drop_kinds=("primary",)) is None


def test_mini_town_builds_and_expert_drives():
    town = osm_town(BBOX, osm=mini_osm(), name="mini")
    # the grid (its south-middle junction moved 10 m north: contracted with the side road's node) and the two bend
    # vertices; the dead end, the one-way link and the service road are gone
    pos = sorted((round(float(J.pos[0])), round(float(J.pos[1]))) for J in town.junctions)
    want = sorted([(90 * (i - 1), 90 * (j - 1)) for i in range(3) for j in range(3) if (i, j) != (1, 0)]
                  + [(0, -80), (130, 90), (130, 50)])
    assert pos == want, pos
    arms = {(round(float(J.pos[0])), round(float(J.pos[1]))): len(J.arms) for J in town.junctions}
    assert arms[(0, -80)] == 4 and arms[(90, 0)] == 4 and arms[(0, 0)] == 4 and arms[(130, 90)] == 2, arms
    assert sum(J.signalized for J in town.junctions) == 1
    centre = next(J for J in town.junctions if J.signalized)
    assert tuple(centre.pos) == (0.0, 0.0) and centre.phases == 2
    assert dict(town.graph.dropped) == {"dead end": 1, "short edge (contracted)": 1}, dict(town.graph.dropped)   # one-way, service: never added
    assert all(len(J.arms) >= 2 for J in town.junctions)
    assert all(l.succ for l in town.lanes if l.kind == "road")
    kinds = {r.kind for r in town.roads}
    assert kinds == {"residential", "street"}
    # unsignalised crossings are priority crossings: minor approaches give way
    prio = [J for J in town.junctions if not J.signalized and len(J.arms) >= 3]
    assert prio and all(any(town.lanes[c].yields for c in J.connectors) for J in prio)
    # buildings: two kept (6 m deep -> KeiSim boxes), the one on the road dropped
    assert len(town.buildings) == 2 and all(abs(b["h"] - 6.0) < 1e-6 for b in town.buildings)
    assert all(math.isfinite(v) for b in town.buildings for v in (b["x"], b["y"], b["yaw"]))
    # the expert completes a route
    from keisim.env import KeiEnv
    env = KeiEnv(EnvConfig(town=town.cfg, render_rgb=False, render_seg=False))
    env.reset(town, episode_seed=3, route_length=300, render=False)
    done = False
    while not done:
        _, _, done, info = env.step(env.expert_action())
    assert info["status"] == "success" and info["DS"] > 0.99, info


def test_graph_junction_radius_follows_road_width():
    """Town.from_graph sizes its 3+-arm junctions from the arms' cross-sections (residential 8 m, streets 8.25 m, a
    skew raises the floor) and every turn through them stays drivable at the car's minimum turning radius; the
    procedural _build keeps its 9.5 m (test_procedural_export_unchanged)."""
    from keisim.geometry import curvature_of
    from keisim.roadnet import RoadSpec, Town
    from keisim.vehicle import EgoVehicle
    cfg = town_config()
    res, st = RoadSpec.of_kind("residential", cfg), RoadSpec.of_kind("street", cfg)
    assert abs(Town.graph_min_radius([res]) - 8.0) < 1e-9 and abs(Town.graph_min_radius([st]) - 8.25) < 1e-9
    assert abs(Town.graph_min_radius([res, st]) - 8.25) < 1e-9 and Town.graph_min_radius([res], math.radians(70)) > 8.5
    town = osm_town(BBOX, osm=mini_osm(), name="mini")
    radius = {(round(float(J.pos[0])), round(float(J.pos[1]))): J.radius for J in town.junctions}
    assert abs(radius[(90, 90)] - 8.0) < 1e-6 and abs(radius[(0, 0)] - 8.25) < 1e-6, radius
    assert town.max_vehicle_dims == (6.0, 2.0) and town.mirrored_turn_yield
    for J in town.junctions:
        if len(J.arms) >= 3:
            assert J.radius >= Town.graph_min_radius([town.roads[a["road"]].spec for a in J.arms]) - 1e-9
    r_car = EgoVehicle.WHEELBASE / math.tan(EgoVehicle.MAX_STEER)
    for lane in town.lanes:
        if lane.kind == "conn" and lane.turn != "straight" and len(town.junctions[lane.junction].arms) >= 3:
            assert 1.0 / np.abs(curvature_of(lane.pts)).max() >= r_car, (lane.junction, lane.turn)


def test_procedural_export_unchanged():
    """Town 1000's export must equal the bundled web/towns/town_1000.json (the grid layout and _build_from_graph)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from export_town import export
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "towns", "town_1000.json")) as f:
        ref = json.load(f)
    assert export(1000) == ref


def test_graph_duplicate_and_coincident_arms():
    """Two edges between the same nodes along the same line (OSM maps some streets twice; contractions can make such
    pairs too) are one road: drop_duplicates keeps one, and est_radius sizes a junction as if arms closer than
    drop_acute_arms' angle were one (before, a zero gap divided by tan(0), a 1 deg gap made a 300 m circle and the
    contraction swallowed the town: the boxes north and south of osm_sapporo)."""
    from keisim.realworld import Graph
    G = Graph()
    G.nodes.update({1: (0.0, 0.0), 2: (60.0, 0.0), 3: (0.0, 60.0), 4: (-60.0, 0.0), 5: (60.0, 1.0)})
    G.add_edge(1, 2, [(0.0, 0.0), (60.0, 0.0)], "residential", 10)
    G.add_edge(2, 1, [(60.0, 0.0), (30.0, 0.0), (0.0, 0.0)], "residential", 11)     # the same line, other way, other id
    G.add_edge(1, 3, [(0.0, 0.0), (0.0, 60.0)], "residential", 12)
    G.add_edge(1, 4, [(0.0, 0.0), (-60.0, 0.0)], "residential", 13)
    r = G.est_radius(1)
    assert math.isfinite(r) and r < 12.0, r
    G.drop_duplicates()
    assert sorted(e["way"] for e in G.edges.values()) == [10, 12, 13] and G.dropped["duplicate way"] == 1
    G.add_edge(1, 5, [(0.0, 0.0), (60.0, 1.0)], "residential", 14)                   # 1 deg from the edge to node 2
    r = G.est_radius(1)
    assert math.isfinite(r) and r < 12.0, r


def test_stop_signs():
    """止まれ (Town.from_graph): every approach of the minor road to a priority crossing, and no other lane, gets a
    stop line, the painted 止まれ (markings) and a sign beside the road; the expert stands still at each one it passes, with its front bumper about
    STOP_SIGN_MARGIN before the line, and still completes the route. Signalised junctions keep their own lines."""
    from keisim.expert import STOP_SIGN_MARGIN, STOP_SIGN_V
    town = osm_town(BBOX, osm=mini_osm(), name="mini")
    assert town.stop_signs and town.yield_lookback
    want = set()
    for J in town.junctions:
        if not J.signalized and len(J.arms) >= 3:
            want |= {lid for lid, g in town._priority_groups(J).items() if g == 1}
    have = {l.id for l in town.lanes if l.stop_sign_s is not None}
    assert want and have == want, (want, have)
    assert all(town.lanes[l].stop_s is None for l in have)                    # never a signal's approach
    signal_lines = sum(l.stop_s is not None for l in town.lanes)
    town2 = osm_town(BBOX, osm=mini_osm(), name="mini2")
    town2.stop_signs = False
    town2._make_yields()
    town2.markings = []
    town2._make_markings()
    extra = len(town.markings) - len(town2.markings)                          # a stop line + the painted 止まれ each
    assert extra > 10 * len(have) and signal_lines > 0, extra
    assert sorted(sp["lane"] for sp in town.stop_sign_posts) == sorted(have)  # one sign per stop line, off the road
    assert all(town.surface(sp["pole"][None])[0] != 1 for sp in town.stop_sign_posts)
    from keisim.env import KeiEnv
    env = KeiEnv(EnvConfig(town=town.cfg, render_rgb=False, render_seg=False))
    passed = 0
    for ep in range(3):
        env.reset(town, episode_seed=ep, route_length=300, render=False)
        lines = [s for s, _ in env.route.stop_signs]
        vmin = {s: 99.0 for s in lines}
        hl = env.world.ego.LENGTH / 2
        done = False
        while not done:
            front = env.s_ego + hl
            for s in lines:
                if s - STOP_SIGN_MARGIN - 2.0 <= front <= s + 0.5:
                    vmin[s] = min(vmin[s], env.world.ego.v)
            _, _, done, info = env.step(env.expert_action())
        assert info["status"] == "success", info
        gone = [s for s in lines if s < env.s_ego]
        assert all(vmin[s] < STOP_SIGN_V for s in gone), vmin
        passed += len(gone)
    assert passed > 0


def test_stop_sign_counterfactuals():
    """The expert's recorded 止まれ inputs (ss_d, ss_go) let target_for_speed rebuild its label at the recorded speed,
    and give "go on" (ss_go) for the same frame standing at the line: the pairs that teach a single-frame model when
    the stop is made."""
    from keisim.expert import STOP_SIGN_MARGIN, STOP_SIGN_ZONE, target_for_speed
    town = osm_town(BBOX, osm=mini_osm(), name="mini")
    from keisim.env import KeiEnv
    env = KeiEnv(EnvConfig(town=town.cfg, render_rgb=False, render_seg=False))
    keys = ("target_nolight", "lt_over", "lt_d", "lt_st", "lt_trem", "lt_blocked", "ss_d", "ss_go")
    pending = near = 0
    for ep in range(3):
        env.reset(town, episode_seed=ep, route_length=300, render=False)
        done = False
        while not done:
            p = env.plan
            lab = {k: p[k] for k in keys}
            assert abs(target_for_speed(env.world.ego.v, lab) - p["target_speed"]) < 1e-6, (lab, p["target_speed"])
            if math.isfinite(p["ss_d"]):
                pending += 1
                if p["ss_d"] <= STOP_SIGN_MARGIN + STOP_SIGN_ZONE:
                    near += 1
                    go = target_for_speed(0.0, lab)
                    assert go == (0.0 if p["ss_go"] < 0.3 else p["ss_go"]) or not math.isnan(p["lt_d"])
            _, _, done, _ = env.step(env.expert_action())
    assert pending > 0 and near > 0, (pending, near)


def test_stand_memory_tells_the_stop_apart():
    """Just before and just after its stop at a 止まれ line the car sees nearly the same; the distance since it last
    stood (stand_m, counted as KeiPilotAgent and collect.py count it) tells the two apart. Near a line the expert
    counts the stop as made exactly when the car stood no farther back than STOP_SIGN_MARGIN + STOP_SIGN_ZONE, i.e.
    when stand_m <= STOP_SIGN_MARGIN + STOP_SIGN_ZONE - d (d: the line ahead of the front bumper)."""
    from keipilot.data import STAND_CAP, stand_update
    from keisim.env import KeiEnv
    from keisim.expert import STOP_SIGN_MARGIN, STOP_SIGN_OVER, STOP_SIGN_ZONE
    town = osm_town(BBOX, osm=mini_osm(), name="mini")
    env = KeiEnv(EnvConfig(town=town.cfg, render_rgb=False, render_seg=False))
    dt = env.cfg.dt * env.cfg.action_repeat
    seen = {True: 0, False: 0}
    for ep in range(3):
        env.reset(town, episode_seed=ep, route_length=300, render=False)
        hl = env.world.ego.LENGTH / 2
        m, done = STAND_CAP, False
        while not done:
            m = stand_update(m, env.world.ego.v, dt)
            for s, _ in env.route.stop_signs:
                d = s - (env.s_ego + hl)
                if -STOP_SIGN_OVER < d <= STOP_SIGN_MARGIN + STOP_SIGN_ZONE:
                    made = s in env.expert._ss_done
                    room = STOP_SIGN_MARGIN + STOP_SIGN_ZONE - d
                    assert (m <= room + 0.3) if made else (m > room - 0.3), (ep, d, m, made)
                    seen[made] += 1
            _, _, done, _ = env.step(env.expert_action())
    assert seen[True] > 0 and seen[False] > 0, seen
