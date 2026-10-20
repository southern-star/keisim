"""Real-world towns: an OpenStreetMap road graph -> keisim.roadnet.Town (Town.from_graph).

    from keisim.realworld import osm_town
    town = osm_town((43.088, 141.360, 43.097, 141.372), cache="data/osm/sapporo_higashi.json")

Pipeline (see road_graph): the OSM `highway` ways of a lat/lon box (Overpass, cached as JSON; no osmnx needed) are
projected onto a local tangent plane (metres, +x east, +y north, box centre at the origin), cut into edges at the
nodes shared by several ways, and cleaned up into something KeiSim can represent, i.e. two-way roads of one lane
per direction meeting at junctions: one-way roads, roads of 3+ lanes, service roads and footways are dropped, dead
ends pruned, straight degree-2 nodes merged into their roads (sharp ones become bends), junctions closer than
~21 m contracted into one (KeiSim's junctions are circles: 8 m for residential roads, 8.25 m for streets,
Town.graph_min_radius), and the largest connected component kept. Junctions with 3+ arms that have a `highway=traffic_signals` node within SIGNAL_RADIUS get KeiSim signals;
the others are priority crossings (roadnet._make_yields). OSM highway classes map to config.ROAD_KINDS
(HIGHWAY_KINDS). Building outlines become oriented boxes (cv2.minAreaRect) with `height`, `building:levels` x 3 m,
or 7 m; the Town drops those over a road or sidewalk.
"""
from __future__ import annotations

import json
import math
import os
import urllib.parse
import urllib.request
from collections import defaultdict

import cv2
import numpy as np

from .config import TownConfig
from .roadnet import RoadSpec, Town

# OSM highway class -> KeiSim road kind (config.ROAD_KINDS and TownConfig's "street"). Everything else (service,
# footway, path, cycleway, pedestrian, track, motorway / trunk, ...) is left out.
HIGHWAY_KINDS = {
    "primary": "main", "secondary": "main",
    "tertiary": "street", "unclassified": "street",
    "residential": "residential",
    "living_street": "alley",
}
OVERPASS_SERVERS = ("https://overpass-api.de/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
                    "https://overpass.private.coffee/api/interpreter")
USER_AGENT = "keisim-import/0.1 (research; github.com/southern-star/keisim)"   # Overpass answers 406 without one
SIGNAL_RADIUS = 20.0     # m: a traffic_signals node this close to a junction signalises it
BEND_DEG = 25.0          # degree-2 nodes / interior vertices turning more than this become bend junctions
EDGE_MARGIN = 30.0       # m outside the box: edges ending further out are dropped (then the dead ends pruned)
LEVEL_HEIGHT = 3.0       # m per building:levels
MIN_SIDE = 3.0           # m: smaller building outlines (sheds, bike shelters) are left out
DEFAULT_HEIGHT = 7.0     # m, buildings without height / levels


# ----------------------------------------------------------------------------- OSM data
def overpass_query(bbox):
    s, w, n, e = bbox
    box = f"({s},{w},{n},{e})"
    return (f"[out:json][timeout:240];(way[\"highway\"]{box};node[\"highway\"=\"traffic_signals\"]{box};"
            f"way[\"building\"]{box};);(._;>;);out body;")


def fetch_osm(bbox, cache=None, servers=OVERPASS_SERVERS):
    """Overpass JSON of the highways, traffic signals and buildings in bbox = (south, west, north, east), cached in
    `cache` (a JSON file) when given."""
    if cache and os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    data = urllib.parse.urlencode({"data": overpass_query(bbox)}).encode()
    err = None
    for url in servers:
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=400) as r:
                osm = json.load(r)
            break
        except Exception as ex:                                  # noqa: BLE001 - try the next mirror
            err = ex
    else:
        raise RuntimeError(f"Overpass failed on every server: {err}")
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        with open(cache, "w") as f:
            json.dump(osm, f)
    return osm


class Projection:
    """Local tangent plane at (lat0, lon0): x east, y north in metres (WGS84 radii of curvature; the error over a
    few km is centimetres)."""

    def __init__(self, lat0, lon0):
        a, e2 = 6378137.0, 0.00669437999014
        phi = math.radians(lat0)
        s2 = math.sin(phi) ** 2
        self.lat0, self.lon0 = lat0, lon0
        self.kx = math.radians(1.0) * a / math.sqrt(1 - e2 * s2) * math.cos(phi)
        self.ky = math.radians(1.0) * a * (1 - e2) / (1 - e2 * s2) ** 1.5

    def xy(self, lat, lon):
        return np.stack([(np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky], -1)

    def latlon(self, x, y):
        return np.asarray(y) / self.ky + self.lat0, np.asarray(x) / self.kx + self.lon0


def _lanes_of(tags):
    try:
        return max(int(float(v)) for v in str(tags.get("lanes", "")).replace(";", " ").split())
    except ValueError:
        return None


def road_kind(tags, drop_kinds=()):
    """KeiSim road kind of an OSM way, or None if KeiSim cannot represent it (not a mapped class, one-way,
    3+ lanes, an area, a roundabout)."""
    kind = HIGHWAY_KINDS.get(tags.get("highway"))
    if kind is None or tags.get("highway") in drop_kinds:
        return None
    if tags.get("oneway") in ("yes", "-1", "1", "true") or tags.get("junction") == "roundabout":
        return None
    if tags.get("area") == "yes":
        return None
    n = _lanes_of(tags)
    if n is not None and n >= 3:
        return None
    return kind


# ----------------------------------------------------------------------------- the graph
class Graph:
    """Undirected multigraph of road edges with polyline geometry (metres). nodes: id -> (x, y); edges: id ->
    {"n": [node a, node b], "pts": (k, 2) vertices from a to b, "kind", "way"}; adj: node -> set of edge ids."""

    def __init__(self, cfg=None):
        self.cfg = cfg or town_config()
        self.nodes, self.edges, self.adj = {}, {}, defaultdict(set)
        self.signals = set()           # node ids tagged traffic_signals (also of merged / contracted nodes)
        self.dropped = defaultdict(int)
        self._next = 0

    def add_edge(self, a, b, pts, kind, way):
        eid = self._next
        self._next += 1
        self.edges[eid] = {"n": [a, b], "pts": np.asarray(pts, float), "kind": kind, "way": way}
        self.adj[a].add(eid)
        self.adj[b].add(eid)
        return eid

    def remove_edge(self, eid, why=None):
        """why: what the clean-up dropped it for (counted in self.dropped; None when it is replaced)."""
        e = self.edges.pop(eid)
        for n in e["n"]:
            self.adj[n].discard(eid)
        if why is not None:
            self.dropped[why] += 1

    def degree(self, n):
        return len(self.adj[n])

    def length(self, eid):
        p = self.edges[eid]["pts"]
        return float(np.hypot(*np.diff(p, axis=0).T).sum())

    def other(self, eid, n):
        a, b = self.edges[eid]["n"]
        return b if a == n else a

    def oriented(self, eid, start):
        """Vertices of the edge starting at node `start`."""
        e = self.edges[eid]
        return e["pts"] if e["n"][0] == start else e["pts"][::-1]

    def arm_angles(self, n, probe=6.0):
        """[(angle, edge id)] of the edges at node n, sorted; the angle is that of the point `probe` m along the
        edge (as Town.from_graph sets the arms)."""
        out = []
        for eid in self.adj[n]:
            pts = self.oriented(eid, n)
            s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
            q = np.array([np.interp(min(probe, s[-1] / 2), s, pts[:, 0]), np.interp(min(probe, s[-1] / 2), s, pts[:, 1])])
            q -= self.nodes[n]
            out.append((math.atan2(q[1], q[0]), eid))
        return sorted(out)

    def est_radius(self, n):
        """The junction radius Town._size_junction will give node n."""
        arms = self.arm_angles(n)
        if len(arms) < 2:
            return 3.0
        angs = np.array([a for a, _ in arms])
        gaps = np.diff(np.concatenate([angs, [angs[0] + 2 * math.pi]]))
        gmin = float(gaps.min())
        specs = [RoadSpec.of_kind(self.edges[e]["kind"], self.cfg) for _, e in arms]
        wo = max(rs.walk_outer for rs in specs)
        if len(arms) >= 3:
            return max(Town.graph_min_radius(specs, gmin), wo / math.tan(gmin / 2.0) + 1.5)
        wo = max(wo, self.cfg.walk_outer)
        return max(3.0, wo / math.tan(min(gmin, 2 * math.pi - gmin) / 2.0) + 1.0)

    # -- clean-up passes -------------------------------------------------------------------------------------------
    def drop_acute_arms(self, min_deg=35.0):
        """Two arms of a junction closer than min_deg (a road crossing another at a shallow angle) would need a
        huge junction circle: the lower-ranked (then shorter) of the two edges is dropped (its road then ends at
        its other junctions, or is pruned)."""
        rank = {"alley": 0, "residential": 1, "street": 2, "main": 3}
        changed = True
        while changed:
            changed = False
            for n in list(self.adj):
                if self.degree(n) < 2:
                    continue
                arms = self.arm_angles(n)
                for k in range(len(arms)):
                    a1, e1 = arms[k]
                    a2, e2 = arms[(k + 1) % len(arms)]
                    gap = (a2 - a1) % (2 * math.pi)
                    if gap >= math.radians(min_deg) or e1 not in self.edges or e2 not in self.edges or e1 == e2:
                        continue
                    key = lambda e: (rank[self.edges[e]["kind"]], self.length(e))   # noqa: E731
                    self.remove_edge(min((e1, e2), key=key), "acute arm")
                    changed = True
                    break
    def clip(self, x0, y0, x1, y1):
        for eid in list(self.edges):
            if any(not (x0 <= self.nodes[n][0] <= x1 and y0 <= self.nodes[n][1] <= y1) for n in self.edges[eid]["n"]):
                self.remove_edge(eid, "outside the box")

    def split_sharp(self, bend_deg=BEND_DEG):
        """Interior vertices turning by more than bend_deg become (degree-2) nodes."""
        for eid in list(self.edges):
            e = self.edges[eid]
            pts = e["pts"]
            if len(pts) < 3:
                continue
            d = np.diff(pts, axis=0)
            h = np.arctan2(d[:, 1], d[:, 0])
            turn = np.abs((np.diff(h) + math.pi) % (2 * math.pi) - math.pi)
            cuts = [i + 1 for i in np.nonzero(turn > math.radians(bend_deg))[0]]
            if not cuts:
                continue
            a, b = e["n"]
            nodes = [a]
            for i in cuts:
                nid = f"v{self._next}_{i}"
                self.nodes[nid] = tuple(pts[i])
                nodes.append(nid)
            nodes.append(b)
            bounds = [0] + cuts + [len(pts) - 1]
            self.remove_edge(eid)
            for k in range(len(nodes) - 1):
                self.add_edge(nodes[k], nodes[k + 1], pts[bounds[k]:bounds[k + 1] + 1], e["kind"], e["way"])

    def prune_dead_ends(self):
        """Remove degree-1 nodes (and their edge) until none is left: KeiSim has no cul-de-sacs."""
        changed = True
        while changed:
            changed = False
            for n in list(self.adj):
                if self.degree(n) == 1:
                    self.remove_edge(next(iter(self.adj[n])), "dead end")
                    changed = True
                if self.degree(n) == 0:
                    del self.adj[n]

    def merge_straight(self, bend_deg=BEND_DEG):
        """Degree-2 nodes where two edges of the same kind continue almost straight on are absorbed into one edge
        (the sharper ones stay as bend junctions)."""
        for n in list(self.adj):
            if self.degree(n) != 2:
                continue
            e1, e2 = sorted(self.adj[n])
            if self.edges[e1]["kind"] != self.edges[e2]["kind"] or self.other(e1, n) == self.other(e2, n) \
                    or self.other(e1, n) == n:
                continue
            p1 = self.oriented(e1, self.other(e1, n))      # ... -> n
            p2 = self.oriented(e2, n)                      # n -> ...
            h1 = math.atan2(*(p1[-1] - p1[-2])[::-1])
            h2 = math.atan2(*(p2[1] - p2[0])[::-1])
            if abs((h2 - h1 + math.pi) % (2 * math.pi) - math.pi) > math.radians(bend_deg):
                continue
            a, b = self.other(e1, n), self.other(e2, n)
            way = self.edges[e1]["way"]
            kind = self.edges[e1]["kind"]
            self.remove_edge(e1)
            self.remove_edge(e2)
            self.add_edge(a, b, np.concatenate([p1, p2[1:]]), kind, way)
            del self.adj[n]

    def contract_short(self, gap=5.0):
        """Edges too short for KeiSim's junction circles (est_radius at both ends plus `gap` m of road between them,
        the stop line's setback) are
        contracted: the two end nodes become one (at the junction's position if only one is a junction, else
        midway) and the edges of the other move over. Self-loops that result are dropped."""
        changed = True
        while changed:
            changed = False
            for eid in sorted(self.edges, key=self.length):
                if eid not in self.edges:
                    continue
                a, b = self.edges[eid]["n"]
                if a == b:
                    self.remove_edge(eid, "self-loop")
                    changed = True
                    continue
                if self.length(eid) >= self.est_radius(a) + self.est_radius(b) + gap:
                    continue
                da, db = self.degree(a), self.degree(b)
                if da >= 3 and db < 3:
                    keep, drop, pos = a, b, self.nodes[a]
                elif db >= 3 and da < 3:
                    keep, drop, pos = b, a, self.nodes[b]
                else:
                    keep, drop = a, b
                    pos = tuple((np.asarray(self.nodes[a]) + np.asarray(self.nodes[b])) / 2)
                self.remove_edge(eid, "short edge (contracted)")
                self.nodes[keep] = pos
                if drop in self.signals:
                    self.signals.add(keep)
                for e2 in list(self.adj[drop]):
                    e = self.edges[e2]
                    e["n"] = [keep if n == drop else n for n in e["n"]]
                    self.adj[drop].discard(e2)
                    self.adj[keep].add(e2)
                del self.adj[drop]
                for e2 in self.adj[keep]:         # end vertices follow the moved node
                    e = self.edges[e2]
                    if e["n"][0] == keep:
                        e["pts"][0] = pos
                    if e["n"][1] == keep:
                        e["pts"][-1] = pos
                changed = True
                break

    def largest_component(self):
        seen, best = set(), []
        for start in self.adj:
            if start in seen or not self.adj[start]:
                continue
            comp, stack = [], [start]
            seen.add(start)
            while stack:
                u = stack.pop()
                comp.append(u)
                for eid in self.adj[u]:
                    v = self.other(eid, u)
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            if len(comp) > len(best):
                best = comp
        keep = set(best)
        for eid in list(self.edges):
            if self.edges[eid]["n"][0] not in keep:
                self.remove_edge(eid, "other component")
        for n in list(self.adj):
            if n not in keep:
                del self.adj[n]


def road_graph(osm, bbox, drop_kinds=("primary",), bend_deg=BEND_DEG, signal_radius=SIGNAL_RADIUS, cfg=None):
    """OSM elements -> (Graph, Projection). drop_kinds: OSM classes to leave out besides the unmapped ones (the
    4-lane primary of the Sapporo test box, at its edge)."""
    s, w, n, e = bbox
    proj = Projection((s + n) / 2, (w + e) / 2)
    nodes = {el["id"]: el for el in osm["elements"] if el["type"] == "node"}
    ways = [el for el in osm["elements"] if el["type"] == "way" and road_kind(el.get("tags", {}), drop_kinds)]
    use = defaultdict(int)
    for wy in ways:
        for nid in wy["nodes"]:
            use[nid] += 1
    G = Graph(cfg)
    for wy in ways:
        kind = road_kind(wy["tags"], drop_kinds)
        ids = wy["nodes"]
        cut = [0] + [i for i in range(1, len(ids) - 1) if use[ids[i]] > 1] + [len(ids) - 1]
        for k in range(len(cut) - 1):
            seg = ids[cut[k]:cut[k + 1] + 1]
            pts = proj.xy([nodes[i]["lat"] for i in seg], [nodes[i]["lon"] for i in seg])
            for nid, p in ((seg[0], pts[0]), (seg[-1], pts[-1])):
                G.nodes[nid] = tuple(p)
            G.add_edge(seg[0], seg[-1], pts, kind, wy["id"])
    x0, y0 = proj.xy(s, w) - EDGE_MARGIN
    x1, y1 = proj.xy(n, e) + EDGE_MARGIN
    G.clip(x0, y0, x1, y1)
    G.split_sharp(bend_deg)
    for _ in range(6):
        before = len(G.edges)
        G.prune_dead_ends()
        G.merge_straight(bend_deg)
        G.drop_acute_arms()
        G.contract_short()
        if len(G.edges) == before:
            break
    G.prune_dead_ends()
    G.largest_component()
    # signals: 3+-arm junctions with a traffic_signals node nearby
    sig = np.array([proj.xy(el["lat"], el["lon"]) for el in nodes.values()
                    if el.get("tags", {}).get("highway") == "traffic_signals"]).reshape(-1, 2)
    for nid in G.adj:
        if G.degree(nid) >= 3 and len(sig) and np.min(np.hypot(*(sig - G.nodes[nid]).T)) <= signal_radius:
            G.signals.add(nid)
    G.signals = {nid for nid in G.signals if nid in G.adj and G.degree(nid) >= 3}
    return G, proj


def building_boxes(osm, proj, bbox):
    """Closed building ways -> boxes {x, y, yaw, l, w, h} (minimum-area rectangles) with their centre in the box."""
    s, w, n, e = bbox
    nodes = {el["id"]: el for el in osm["elements"] if el["type"] == "node"}
    out = []
    for el in osm["elements"]:
        if el["type"] != "way" or "building" not in el.get("tags", {}) or len(el["nodes"]) < 4:
            continue
        ids = el["nodes"][:-1] if el["nodes"][0] == el["nodes"][-1] else el["nodes"]
        if any(i not in nodes for i in ids):
            continue
        pts = proj.xy([nodes[i]["lat"] for i in ids], [nodes[i]["lon"] for i in ids]).astype(np.float32)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
        if min(rw, rh) < MIN_SIDE or rw * rh < 12.0:
            continue
        lat, lon = proj.latlon(cx, cy)
        if not (s <= lat <= n and w <= lon <= e):
            continue
        tags = el["tags"]
        h = None
        for key, scale in (("height", 1.0), ("building:levels", LEVEL_HEIGHT)):
            try:
                h = float(str(tags[key]).split()[0].rstrip("m")) * scale
                break
            except (KeyError, ValueError):
                continue
        out.append({"x": float(cx), "y": float(cy), "yaw": math.radians(ang), "l": float(rw), "w": float(rh),
                    "h": float(h) if h and h > 2.0 else DEFAULT_HEIGHT, "osm": el["id"]})
    return out


def align_boxes(boxes, G, step=2.0):
    """Orient every box like a KeiSim building: yaw along the nearest road (l along it, w deep), so KeiView's lot
    frames (scripts/export_town.py building_frontage) face the street. In place."""
    if not boxes or not G.edges:
        return boxes
    pts, hs = [], []
    for e in G.edges.values():
        p = e["pts"]
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
        si = np.arange(0.0, s[-1] + 1e-9, step)
        q = np.stack([np.interp(si, s, p[:, 0]), np.interp(si, s, p[:, 1])], 1)
        d = np.gradient(q, axis=0) if len(q) > 1 else p[-1:] - p[:1]
        pts.append(q)
        hs.append(np.arctan2(d[:, 1], d[:, 0]))
    pts, hs = np.concatenate(pts), np.concatenate(hs)
    for b in boxes:
        k = int(np.argmin(np.hypot(pts[:, 0] - b["x"], pts[:, 1] - b["y"])))
        rel = (b["yaw"] - hs[k]) % math.pi
        if math.pi / 4 < rel < 3 * math.pi / 4:                  # the rect's width runs across the road: swap axes
            b["yaw"], b["l"], b["w"] = b["yaw"] + math.pi / 2, b["w"], b["l"]
        b["yaw"] = float((b["yaw"] + math.pi) % (2 * math.pi) - math.pi)
    return boxes


def town_config():
    """TownConfig of real-world towns: mixed road kinds (per OSM class) with Japan's two-phase signals."""
    cfg = TownConfig()
    cfg.road_types = "mixed"
    cfg.signal_mode = "two_phase"
    return cfg


def graph_town(G, cfg, seed=0, buildings=None, name="osm"):
    """Graph -> Town.from_graph (junction ids in a fixed order; roads sorted by their end junctions)."""
    order = sorted(G.adj, key=lambda n: (G.nodes[n][1], G.nodes[n][0]))
    index = {n: k for k, n in enumerate(order)}
    positions = [G.nodes[n] for n in order]
    roads = []
    for eid in sorted(G.edges, key=lambda k: (index[G.edges[k]["n"][0]], index[G.edges[k]["n"][1]], k)):
        e = G.edges[eid]
        a, b = (index[n] for n in e["n"])
        pts = e["pts"]
        if a > b:
            a, b, pts = b, a, pts[::-1]
        roads.append((a, b, pts, RoadSpec.of_kind(e["kind"], cfg)))
    signalized = {index[n] for n in G.signals}
    town = Town.from_graph(seed, cfg, positions, roads, signalized, buildings, name=name)
    town.osm = {"node_ids": order, "edge_ways": [G.edges[k]["way"] for k in sorted(G.edges)]}
    # the NPC fleet of 6 m residential roads: vans and 2 t trucks (6.0 x 2.0 m), not the 7.6 m trucks of the
    # procedural towns, whose front corner sweeps into the next lane on the tight turns of these junctions
    town.max_vehicle_dims = (6.0, 2.0)
    return town


def osm_town(bbox, cache=None, osm=None, cfg=None, seed=0, name="osm", drop_kinds=("primary",), buildings=True,
             verbose=False):
    """The KeiSim town of an OSM box (south, west, north, east). osm: Overpass JSON (else fetched / cached)."""
    if osm is None:
        osm = fetch_osm(bbox, cache)
    cfg = cfg or town_config()
    G, proj = road_graph(osm, bbox, drop_kinds, cfg=cfg)
    boxes = align_boxes(building_boxes(osm, proj, bbox), G) if buildings else None
    town = graph_town(G, cfg, seed, boxes, name)
    town.proj, town.graph = proj, G
    town.n_boxes = len(boxes) if boxes is not None else None
    if verbose:
        print(graph_report(town))
    return town


def graph_report(town):
    """Counts of an osm_town: junctions by arms, roads by kind, what the clean-up dropped, buildings kept."""
    from collections import Counter
    G = town.graph
    arms = Counter(len(J.arms) for J in town.junctions)
    kinds = Counter(r.kind for r in town.roads)
    lines = [f"junctions by arms: {dict(sorted(arms.items()))}  signalised: {sum(J.signalized for J in town.junctions)}",
             f"roads by kind: {dict(kinds)}  total {sum(r.length for r in town.roads) / 1000:.2f} km",
             f"dropped: {dict(G.dropped)}"]
    if town.n_boxes is not None:
        lines.append(f"buildings: {town.n_boxes} OSM boxes -> {len(town.buildings)} kept off the roads and sidewalks")
    return "\n".join(lines)
