"""keisim/realworld.py on a hand-made Overpass-style answer (offline), and the refactored Town against the bundled
export (the procedural towns must not change)."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keisim.config import EnvConfig  # noqa: E402
from keisim.realworld import Graph, Projection, oneway_of, osm_town, road_kind, town_config  # noqa: E402

LAT0, LON0 = 43.09, 141.366
BBOX = (LAT0 - 0.0025, LON0 - 0.0035, LAT0 + 0.0025, LON0 + 0.0035)


def mini_osm():
    """A 3 x 3 grid of 90 m blocks (residential, the middle row a tertiary street with signals at its centre), a
    bend road around the north-east corner, a side road leaving the x = 0 road 20 m from a junction (the two
    junctions get contracted into one), a one-way link around the south-east corner (mapped against its direction
    of travel, oneway=-1: it is driven from grid (1, 0) to grid (2, 0)), and what the importer must drop: a dead-end
    stub, a service road. Three buildings, one of them on a road."""
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
    way([grid[(2, 0)], node(90.0, -140.0), node(0.0, -140.0), grid[(1, 0)]], {"highway": "residential", "oneway": "-1"})
    way([grid[(0, 1)], node(-140.0, 0.0)], {"highway": "service"})
    for cx, cy in ((45.0, 45.0), (-45.0, 45.0), (0.0, 30.0)):                             # the last one is on a road
        ring = [node(cx + dx, cy + dy) for dx, dy in ((-5, -4), (5, -4), (5, 4), (-5, 4))]
        way(ring + [ring[0]], {"building": "house", "building:levels": "2"})
    return {"elements": list(nodes.values()) + ways}


def test_road_kind_filters():
    assert road_kind({"highway": "residential"}) == "residential"
    assert road_kind({"highway": "tertiary", "lanes": "2"}) == "street"
    assert road_kind({"highway": "secondary", "lanes": "4"}) is None
    assert road_kind({"highway": "residential", "oneway": "yes"}) == "residential"      # one-way: a single lane
    assert road_kind({"highway": "residential", "oneway": "yes"}, oneway=False) is None   # (the old Sapporo rules)
    assert road_kind({"highway": "tertiary", "oneway": "yes", "lanes": "2"}) is None      # two lanes one way: no
    assert road_kind({"highway": "tertiary", "oneway": "-1", "lanes": "1"}) == "street"
    assert oneway_of({"oneway": "-1"}) == -1 and oneway_of({"oneway": "yes"}) == 1 and oneway_of({}) == 0
    assert road_kind({"highway": "service"}) is None
    assert road_kind({"highway": "primary"}, drop_kinds=("primary",)) is None


def test_mini_town_builds_and_expert_drives():
    town = osm_town(BBOX, osm=mini_osm(), name="mini")
    # the grid (its south-middle junction moved 10 m north: contracted with the side road's node), the two bend
    # vertices and the one-way link's two corners; the dead end and the service road are gone
    pos = sorted((round(float(J.pos[0])), round(float(J.pos[1]))) for J in town.junctions)
    want = sorted([(90 * (i - 1), 90 * (j - 1)) for i in range(3) for j in range(3) if (i, j) != (1, 0)]
                  + [(0, -80), (130, 90), (130, 50), (0, -140), (90, -140)])
    assert pos == want, pos
    arms = {(round(float(J.pos[0])), round(float(J.pos[1]))): len(J.arms) for J in town.junctions}
    assert arms[(0, -80)] == 5 and arms[(90, 0)] == 4 and arms[(0, 0)] == 4 and arms[(130, 90)] == 2 \
        and arms[(90, -90)] == 3, arms
    assert sum(J.signalized for J in town.junctions) == 1
    centre = next(J for J in town.junctions if J.signalized)
    assert tuple(centre.pos) == (0.0, 0.0) and centre.phases == 2
    assert dict(town.graph.dropped) == {"dead end": 1, "short edge (contracted)": 1}, dict(town.graph.dropped)   # service: never added
    assert all(len(J.arms) >= 2 for J in town.junctions)
    assert all(l.succ and l.pred for l in town.lanes if l.kind == "road")
    # the one-way link: three roads (two bends) of a single lane on the centre line, driven from (0, -80) round to
    # (90, -90) (oneway=-1 reversed the way); the junctions at its ends have an arm without an approach / an exit
    one = [r for r in town.roads if r.spec.oneway]
    assert len(one) == 3 and all(r.lane_ba == -1 and r.lane_ab >= 0 for r in one) and abs(sum(r.length for r in one) - 150) < 5
    assert all(abs(r.half_width - (3.0 / 2 + 0.75)) < 1e-9 and r.spec.center == "none" for r in one)
    for r in one:
        lane = town.lanes[r.lane_ab]
        assert np.abs(lane.pts[: len(r.center)] - r.center).max() < 1e-6 and lane.succ and lane.pred
    by_pos = {(round(float(J.pos[0])), round(float(J.pos[1]))): J for J in town.junctions}
    assert [a["in"] for a in by_pos[(0, -80)].arms].count(-1) == 1 and [a["out"] for a in by_pos[(90, -90)].arms].count(-1) == 1
    assert len(by_pos[(0, -80)].incoming) == 4 and len(by_pos[(0, -80)].outgoing) == 5
    assert any(r.a == by_pos[(0, -80)].id for r in one) and any(r.b == by_pos[(90, -90)].id for r in one)
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
    # ... and one through the one-way link
    one_lanes = {r.lane_ab for r in one}
    for ep in range(40):
        env.reset(town, episode_seed=100 + ep, route_length=300, render=False)
        if one_lanes & set(env.route.lanes):
            break
    else:
        raise AssertionError("no route through the one-way link")
    done = False
    while not done:
        _, _, done, info = env.step(env.expert_action())
    assert info["status"] == "success" and info["DS"] > 0.99, info


def test_graph_one_way_rules():
    """The clean-up on one-way edges: two opposite one-way edges along the same line (a road with a median) become
    one two-way edge, a one-way duplicate of a two-way edge goes, a one-way edge nobody can leave by (into a sink) is
    a dead end, and a one-way edge nobody can reach (from a source) falls out of the strongly connected component;
    a one-way lane out of a two-way road's end stays, although nothing feeds that road's far lane."""
    def square(G):
        for k, (x, y) in enumerate([(0, 0), (100, 0), (100, 100), (0, 100)]):
            G.nodes[k] = (float(x), float(y))
        for k in range(4):
            a, b = k, (k + 1) % 4
            G.add_edge(a, b, [G.nodes[a], G.nodes[b]], "residential", k)
    G = Graph()
    square(G)
    G.nodes[4], G.nodes[5] = (200.0, 0.0), (200.0, 100.0)
    G.add_edge(1, 4, [(100, 0), (200, 0.5)], "residential", 10, oneway=True)       # east, one way
    G.add_edge(4, 1, [(200, 0), (100, -0.5)], "residential", 11, oneway=True)      # ... and back: a median
    G.add_edge(2, 5, [(100, 100), (200, 100)], "residential", 12)
    G.add_edge(5, 2, [(200, 100.5), (100, 100.5)], "residential", 13, oneway=True)  # a one-way copy of a two-way edge
    G.add_edge(4, 5, [(200, 0), (200, 100)], "residential", 14)
    G.drop_duplicates()
    assert dict(G.dropped) == {"carriageway (merged)": 1, "duplicate way": 1}, dict(G.dropped)
    assert not any(e["oneway"] for e in G.edges.values()) and len(G.edges) == 7
    G = Graph()
    square(G)
    G.nodes[4], G.nodes[5], G.nodes[6] = (200.0, 0.0), (-100.0, 0.0), (200.0, 100.0)
    G.add_edge(1, 4, [(100, 0), (200, 0)], "residential", 10, oneway=True)         # into a sink: nowhere to go
    G.add_edge(5, 0, [(-100, 0), (0, 0)], "residential", 11, oneway=True)          # from a source: unreachable
    G.add_edge(2, 6, [(100, 100), (200, 100)], "residential", 12)                  # a two-way road ...
    G.add_edge(6, 1, [(200, 100), (100, 0)], "residential", 13, oneway=True)       # ... with a one-way back: kept
    G.prune_dead_ends()
    assert dict(G.dropped) == {"dead end": 1} and 4 not in G.adj, dict(G.dropped)
    G.largest_component()
    assert dict(G.dropped) == {"dead end": 1, "other component": 1} and 5 not in G.adj, dict(G.dropped)
    assert sorted(G.adj) == [0, 1, 2, 3, 6] and len(G.edges) == 6
    # merge_straight joins one-way edges only along their direction of travel
    G = Graph()
    for k, x in enumerate([0.0, 50.0, 100.0]):
        G.nodes[k] = (x, 0.0)
    G.add_edge(0, 1, [(0, 0), (50, 0)], "residential", 1, oneway=True)
    G.add_edge(2, 1, [(100, 0), (50, 0)], "residential", 2, oneway=True)           # head on: two one-ways meet
    G.merge_straight()
    assert len(G.edges) == 2
    G = Graph()
    for k, x in enumerate([0.0, 50.0, 100.0]):
        G.nodes[k] = (x, 0.0)
    G.add_edge(2, 1, [(100, 0), (50, 0)], "residential", 2, oneway=True)
    G.add_edge(1, 0, [(50, 0), (0, 0)], "residential", 1, oneway=True)
    G.merge_straight()
    (e,) = G.edges.values()
    assert e["n"] == [2, 0] and e["oneway"] and e["pts"][0][0] == 100 and e["pts"][-1][0] == 0


def test_graph_carriageways_fold():
    """A road with a median, mapped as one one-way way per carriageway 10 m apart and cut at different side streets
    (one joins each carriageway), folds into one two-way road along the midline, with both side streets meeting
    it; then merge_straight joins its pieces into a single edge between the side streets."""
    G = Graph()
    pts = {"a": (0.0, -5.0), "b": (120.0, -5.0), "c": (200.0, -5.0), "d": (200.0, 5.0), "e": (80.0, 5.0), "f": (0.0, 5.0),
           "s": (120.0, -100.0), "n": (80.0, 100.0)}
    for k, v in pts.items():
        G.nodes[k] = v
    G.add_edge("a", "b", [pts["a"], pts["b"]], "street", 1, oneway=True)          # eastbound carriageway, two pieces
    G.add_edge("b", "c", [pts["b"], pts["c"]], "street", 1, oneway=True)
    G.add_edge("d", "e", [pts["d"], pts["e"]], "street", 2, oneway=True)          # westbound carriageway, two pieces
    G.add_edge("e", "f", [pts["e"], pts["f"]], "street", 2, oneway=True)
    G.add_edge("b", "s", [pts["b"], pts["s"]], "residential", 3)                  # side streets
    G.add_edge("e", "n", [pts["e"], pts["n"]], "residential", 4)
    G.merge_carriageways()
    assert not any(e["oneway"] for e in G.edges.values()), [e["n"] for e in G.edges.values()]
    assert G.dropped["carriageway (merged)"] == 3 and len(G.edges) == 5, dict(G.dropped)   # piece by piece
    ends = sorted(G.nodes[n] for n in G.adj if G.degree(n) == 3)
    assert [round(x) for x, _ in ends] == [80, 120] and all(abs(y) < 2.0 for _, y in ends), ends   # pieces fold one after another: within the median
    assert all(abs(p[1]) < 2.0 for e in G.edges.values() if e["kind"] == "street" for p in e["pts"])
    G.merge_straight()
    assert len(G.edges) == 5 and len(G.adj) == 6


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


def test_bundled_sapporo_export_unchanged():
    """The bundled real town web/towns/town_osm_sapporo.json (the browser demo's 札幌) must equal the export of the
    bundled OpenStreetMap extract data/osm/osm_sapporo.json with the current importer and rules, so the browser's
    town is the one evaluate.py --osm_town osm_sapporo drives. Regenerate with
    `uv run scripts/import_osm.py --name osm_sapporo --out web/towns` after changing either."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "scripts"))
    from keisim.realworld import OSM_BOXES
    from export_town import export_town
    with open(os.path.join(root, "web", "towns", "town_osm_sapporo.json")) as f:
        ref = json.load(f)
    town = osm_town(OSM_BOXES["osm_sapporo"], cache=os.path.join(root, "data", "osm", "osm_sapporo.json"), name="osm_sapporo")
    assert town.max_vehicle_dims == (6.0, 2.0) and town.mirrored_turn_yield
    assert not any(r.spec.oneway for r in town.roads)                   # built without one-way roads (TWO_WAY_TOWNS)
    out = json.loads(json.dumps(export_town(town)))          # the file's float rounding
    assert out["attribution"].startswith("Map data") and out["source"]["osm_bbox"] == list(OSM_BOXES["osm_sapporo"])
    assert out == ref


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
