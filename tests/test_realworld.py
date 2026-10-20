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
