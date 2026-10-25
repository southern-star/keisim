"""Procedural town generator.

A town is a jittered grid of junctions connected by (optionally curved) two-lane
roads. Every junction with three or more arms is signalised with a conflict-free
split-phase controller (one approach green at a time). Lanes form a directed
graph (road lanes -> junction connector lanes -> road lanes) which is used for
routing, traffic and the privileged expert. With TownConfig.road_types="mixed"
the roads differ in cross-section (main roads, streets, residential roads
without sidewalks, alleys; see RoadSpec and config.ROAD_KINDS).
"""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from .config import ROAD_KINDS, TL_GREEN, TL_RED, TL_YELLOW, TownConfig
from .geometry import (Polyline, arc_bezier, box_corners, cubic_bezier, curvature_of,
                       headings_of, offset_polyline, resample, unit, wrap)
from .glyphs import TOMARE

LANE_DS = 0.5
TURN_RADIUS_MIN = 6.5     # m, the lane-centre arc of a right-angle turn in a Town.from_graph junction (graph_min_radius)
TURN_RADIUS_DRIVABLE = 4.5   # m, the tightest arc any junction connector may ask of the car (EgoVehicle: 3.95 m)
NARROW_LANE = 3.0        # m: lanes narrower than this keep trucks out, and their bends are taken one at a time
NARROW_PASS = 2.9        # m: two turns in an unsignalised bend whose centre lines come closer than this give way
MIRROR_PASS = 3.6        # m: a right turn whose centre line runs this close to the oncoming left turn's gives way to it
STOP_SIGN_SETBACK = 2.0  # m: the 止まれ stop line of a minor road's approach to a priority crossing, before its lane's end
TOMARE_GAP, TOMARE_LEN = 0.8, 2.6   # m: the painted 止まれ ends this far before the stop line and is this long
TOMARE_CHAR_W, TOMARE_SPACE = 0.72, 0.12   # m: width of each of its three characters across the lane, and between them
NARROW_CLEAR = 1.9       # m: ... and the outer one waits with its front corners and sides this far from the inner
                         # one's centre line (its half width + the swing of its body in the turn + a margin)


class Lane:
    def __init__(self, lid, pts, kind, speed_limit, width=3.5):
        pts = resample(pts, LANE_DS)
        self.id = lid
        self.pts = pts
        self.poly = Polyline(pts)
        self.length = self.poly.length
        self.s = self.poly.s
        self.heading = headings_of(pts)
        k = curvature_of(pts)
        if len(k) > 7:
            k = np.convolve(np.pad(k, 3, mode="edge"), np.ones(7) / 7.0, mode="valid")
        self.curv = k
        self.kind = kind                  # 'road' or 'conn'
        self.road = -1
        self.junction = -1                # connectors: junction id
        self.start_junction = -1
        self.end_junction = -1
        self.turn = "straight"
        self.succ: list[int] = []
        self.pred: list[int] = []
        self.speed_limit = speed_limit
        self.width = width                # lane width (connectors: the wider of the two roads' lanes)
        self.stop_s = None                # arc length of the stop line
        self.stop_sign_s = None           # arc length of a 止まれ stop line (minor road before a priority crossing)
        self.signal = None                # (junction id, phase index)

    def __repr__(self):
        return f"Lane({self.id},{self.kind},L={self.length:.1f},turn={self.turn})"


class RoadSpec:
    """Cross-section of a road. kind: street | main | residential | alley. shoulder: from the lane's outer edge to
    the edge of the road surface (the 路側帯 where people walk on roads without a sidewalk); sidewalk: width of the
    raised sidewalk on each side (0: none). center / edge: paint (see config.ROAD_KINDS). oneway: a single lane on
    the centre line, driven from the road's junction a to b (Town.from_graph towns), no centre line paint."""

    def __init__(self, kind, lane_width, shoulder, sidewalk, speed_limit, center="dashed", edge="classic",
                 oneway=False):
        self.kind = kind
        self.lane_width, self.shoulder, self.sidewalk = lane_width, shoulder, sidewalk
        self.speed_limit = speed_limit
        self.center, self.edge = (center, edge) if not oneway else ("none", edge)
        self.oneway = bool(oneway)
        self.lane_outer = lane_width / 2.0 if oneway else lane_width   # the lanes' outer edge, from the centre line
        self.half_width = self.lane_outer + shoulder  # road surface, centre line to edge
        self.walk_outer = self.half_width + sidewalk  # outer edge of the sidewalk (= half_width without one)

    @classmethod
    def street(cls, cfg, oneway=False):
        """TownConfig's own road: every road of a uniform town, the streets of a mixed one."""
        return cls("street", cfg.lane_width, cfg.shoulder, cfg.sidewalk_width, cfg.speed_limit, oneway=oneway)

    @classmethod
    def of_kind(cls, kind, cfg, oneway=False):
        return cls.street(cfg, oneway) if kind == "street" else cls(kind, **ROAD_KINDS[kind], oneway=oneway)

    def walkway(self):
        """Where pedestrians walk along this road: (offset from the centre line, lateral spread factor, on the road
        surface) or None. On the sidewalk's middle; on roads without one in the 路側帯 (if it is >= 0.75 m wide)."""
        if self.sidewalk > 0:
            return self.half_width + self.sidewalk / 2, 1.0, False
        if self.shoulder >= 0.75:
            return self.half_width - 0.4, 0.25, True
        return None


class Road:
    def __init__(self, rid, a, b, center, spec: RoadSpec):
        self.id = rid
        self.a, self.b = a, b
        self.center = resample(center, LANE_DS)
        self.poly = Polyline(self.center)
        self.length = self.poly.length
        self.spec = spec
        self.kind = spec.kind
        self.half_width = spec.half_width
        self.walk_outer = spec.walk_outer
        self.lane_ab = -1
        self.lane_ba = -1


YIELD_CLEAR = 2.4        # clearance of a waiting car's front circle from the paths it yields to


def _two_phase_groups(angles):
    """Signal group of each arm (angles sorted counter-clockwise): opposite arms share a green.
    4 arms: 0+2 and 1+3. 3 arms: the most nearly opposite pair (the main road), then the stem."""
    n = len(angles)
    if n == 4:
        return [0, 1, 0, 1]
    if n == 3:
        best, pair = -1.0, (0, 1)
        for a in range(3):
            for b in range(a + 1, 3):
                d = abs(math.atan2(math.sin(angles[a] - angles[b]), math.cos(angles[a] - angles[b])))
                if d > best:
                    best, pair = d, (a, b)
        return [0 if k in pair else 1 for k in range(3)]
    return list(range(n)) if n <= 2 else [k % 2 for k in range(n)]


class Junction:
    def __init__(self, jid, pos):
        self.id = jid
        self.pos = np.asarray(pos, float)
        self.arms: list[dict] = []
        self.radius = 0.0
        self.signalized = False
        self.incoming: list[int] = []
        self.outgoing: list[int] = []
        self.connectors: list[int] = []
        self.road_poly = None
        self.walk_poly = None
        self.phases = 0
        self.green = 8.0
        self.yellow = 2.5
        self.allred = 1.5
        self.offset = 0.0

    @property
    def cycle(self):
        return self.phases * (self.green + self.yellow + self.allred)

    def signal_state(self, phase, t):
        if not self.signalized:
            return TL_GREEN
        per = self.green + self.yellow + self.allred
        tau = (t + self.offset) % (per * self.phases)
        k = int(tau // per)
        if k != phase:
            return TL_RED
        r = tau - k * per
        if r < self.green:
            return TL_GREEN
        if r < self.green + self.yellow:
            return TL_YELLOW
        return TL_RED

    def remaining(self, phase, t):
        """Seconds until the state of `phase` changes."""
        per = self.green + self.yellow + self.allred
        T = per * self.phases
        tau = (t + self.offset) % T
        start = phase * per
        r = (tau - start) % T          # time since this phase's green started
        if r < self.green:
            return self.green - r
        if r < self.green + self.yellow:
            return self.green + self.yellow - r
        return T - r


class Raster:
    """World <-> pixel helper for axis aligned top-down rasters."""

    def __init__(self, bbox, res):
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.res = res
        self.W = int(math.ceil((self.x1 - self.x0) / res))
        self.H = int(math.ceil((self.y1 - self.y0) / res))

    def to_px(self, pts):
        pts = np.asarray(pts, float)
        return np.stack([(pts[..., 0] - self.x0) / self.res, (self.y1 - pts[..., 1]) / self.res], -1)

    def fill(self, img, poly, value):
        px = np.round(self.to_px(poly) * 8.0).astype(np.int32)
        cv2.fillPoly(img, [px.reshape(-1, 1, 2)], value, lineType=cv2.LINE_8, shift=3)

    def sample(self, img, pts, default=0):
        px = self.to_px(pts)
        c = np.floor(px[..., 0]).astype(int)
        r = np.floor(px[..., 1]).astype(int)
        ok = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
        out = np.full(c.shape, default, dtype=img.dtype)
        out[ok] = img[r[ok], c[ok]]
        return out


def _connected(nodes, edges):
    adj = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    start = next(iter(nodes))
    seen = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return len(seen) == len(nodes)


def _hull(points):
    pts = np.asarray(points, np.float64)
    h = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    return h.astype(np.float64)


def _ray_poly_exit(p, d, poly):
    """Largest t>=0 such that p + t d lies on the boundary of convex poly."""
    best = 0.0
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        den = d[0] * (-e[1]) - d[1] * (-e[0])
        if abs(den) < 1e-9:
            continue
        rhs = a - p
        t = (rhs[0] * (-e[1]) - rhs[1] * (-e[0])) / den
        u = (d[0] * rhs[1] - d[1] * rhs[0]) / den
        if t >= 0 and 0 <= u <= 1:
            best = max(best, t)
    return best


class Town:
    """A procedurally generated town. Deterministic given (seed, cfg).

    Town.from_graph builds a town from an explicit road graph instead (keisim/realworld.py: OpenStreetMap). Both
    share _build_from_graph, which derives everything from the junctions (position, arms, radius, signalised) and
    the roads (centre lines, cross-sections): lanes, connectors, signals, markings, rasters, buildings, trees."""

    def __init__(self, seed: int, cfg: TownConfig | None = None):
        self._init(seed, cfg)
        self._build()

    @classmethod
    def from_graph(cls, seed, cfg, positions, roads, signalized=(), buildings=None, name=None):
        """A town from a road graph. positions: junction centres [(x, y), ...]; roads: (a, b, centre polyline,
        RoadSpec) with the polyline running from junction a's centre to junction b's (both ends are cut back by the
        junction radius here); signalized: ids of the junctions with traffic lights (only those with 3+ arms count;
        the other 3+-arm junctions are priority crossings, see _make_yields); buildings: boxes {x, y, yaw, l, w, h}
        instead of the procedural ones (those over a road or sidewalk are dropped). The random stream (seed) only
        draws the signal timings, the trees and the building colours."""
        self = cls.__new__(cls)
        self._init(seed, cfg)
        if name is not None:
            self.name = str(name)
        signalized = set(signalized)
        for jid, pos in enumerate(positions):
            self.junctions.append(Junction(jid, pos))
        probe = 6.0                                   # m along the centre line that sets an arm's direction
        for rid, (a, b, pts, spec) in enumerate(roads):
            pl = Polyline(pts)
            if pl.length < 2 * probe:
                raise ValueError(f"road {rid} ({a}-{b}) is only {pl.length:.1f} m long")
            for jid, end, s_probe in ((a, "a", probe), (b, "b", pl.length - probe)):
                q = pl.interp(s_probe) - self.junctions[jid].pos
                self.junctions[jid].arms.append({"spec": rid, "end": end, "angle": float(math.atan2(q[1], q[0]))})
        for J in self.junctions:
            if not J.arms:
                raise ValueError(f"junction {J.id} has no roads")
            specs = [roads[arm["spec"]][3] for arm in J.arms]
            wo = max(rs.walk_outer for rs in specs)
            if len(J.arms) < 3:
                wo = max(wo, self.cfg.walk_outer)
            self._size_junction(J, wo, specs)
            J.signalized = J.id in signalized and len(J.arms) >= 3
        for rid, (a, b, pts, spec) in enumerate(roads):
            Ja, Jb = self.junctions[a], self.junctions[b]
            pl = Polyline(pts)
            s0 = self._circle_exit(pl, Ja.pos, Ja.radius, forward=True)
            s1 = self._circle_exit(pl, Jb.pos, Jb.radius, forward=False)
            if s0 is None or s1 is None or s1 - s0 < 4.0:
                raise ValueError(f"road {rid} ({a}-{b}, {pl.length:.1f} m) does not reach past both junction radii "
                                 f"({Ja.radius:.1f} m, {Jb.radius:.1f} m)")
            s = np.concatenate([[s0], pl.s[(pl.s > s0) & (pl.s < s1)], [s1]])
            center = pl.interp(s)
            self.roads.append(Road(rid, a, b, center, spec))
            for J, p in ((Ja, center[0]), (Jb, center[-1])):     # arms point at the trimmed ends (radial)
                arm = next(x for x in J.arms if x["spec"] == rid and x["end"] == ("a" if J is Ja else "b"))
                arm["angle"] = float(math.atan2(p[1] - J.pos[1], p[0] - J.pos[0]))
        for J in self.junctions:
            J.arms.sort(key=lambda x: x["angle"])
        self.mirrored_turn_yield = True
        self.stop_signs = True
        self.yield_lookback = True
        self._build_from_graph(buildings)
        return self

    @staticmethod
    def _circle_exit(pl, c, r, forward=True):
        """Arc length where polyline `pl` leaves the circle (c, r) for good when walked from its start (forward) or
        from its end (backward); None if it never does."""
        d = np.hypot(*(pl.pts - np.asarray(c)).T)
        inside = d <= r
        idx = range(len(d) - 1) if forward else range(len(d) - 1, 0, -1)
        for i in idx:
            j = i + 1 if forward else i - 1
            if inside[i] and not inside[j]:
                t = (r - d[i]) / max(d[j] - d[i], 1e-9)
                return float(pl.s[i] + (pl.s[j] - pl.s[i]) * t)
        return None if inside[0 if forward else -1] else float(pl.s[0 if forward else -1])

    def _init(self, seed, cfg):
        self.seed = int(seed)
        self.name = str(self.seed)
        self.cfg = cfg or TownConfig()
        self.rng = np.random.default_rng(self.seed)
        self.lanes: list[Lane] = []
        self.roads: list[Road] = []
        self.junctions: list[Junction] = []
        self.buildings: list[dict] = []
        self.trees: list[dict] = []
        self.signal_heads: list[dict] = []
        self.markings: list[np.ndarray] = []      # white paint polygons
        self.markings_yellow: list[np.ndarray] = []   # yellow paint (centre lines of main roads)
        self.crosswalks: list[np.ndarray] = []
        self.mirrored_turn_yield = False     # Town.from_graph: right turns give way to the oncoming left turn
        self.stop_signs = False              # Town.from_graph: 止まれ on the minor roads' approaches to priority crossings
        self.yield_lookback = False          # Town.from_graph: Traffic.min_arrival looks into the junction before
        self.stop_sign_posts: list[dict] = []   # 止まれ signs: {lane, pole (x, y), yaw (facing the driver), z}
        self._building_edges = None          # building_edges(), built on first use
        self.max_vehicle_dims = None         # (length, width) cap of the NPC cars (keisim.realworld: narrow roads)

    # ------------------------------------------------------------------ build
    def _build(self):
        cfg, rng = self.cfg, self.rng
        nx = int(rng.integers(cfg.grid_min, cfg.grid_max + 1))
        ny = int(rng.integers(cfg.grid_min, cfg.grid_max + 1))
        sp = float(rng.uniform(cfg.spacing_min, cfg.spacing_max))
        self.spacing = sp
        if cfg.block_mode == "varied":     # every grid column and row gets its own block length
            gx = np.concatenate([[0.0], np.cumsum(rng.uniform(cfg.spacing_min, cfg.spacing_max, nx - 1))])
            gy = np.concatenate([[0.0], np.cumsum(rng.uniform(cfg.spacing_min, cfg.spacing_max, ny - 1))])
            jit = cfg.jitter * cfg.spacing_min
        nodes = {}
        for i in range(nx):
            for j in range(ny):
                if cfg.block_mode == "varied":
                    nodes[(i, j)] = np.array([gx[i], gy[j]]) + rng.uniform(-1, 1, 2) * jit
                else:
                    nodes[(i, j)] = np.array([i * sp, j * sp]) + rng.uniform(-1, 1, 2) * cfg.jitter * sp
        edges = []
        for i in range(nx):
            for j in range(ny):
                if i + 1 < nx:
                    edges.append(((i, j), (i + 1, j)))
                if j + 1 < ny:
                    edges.append(((i, j), (i, j + 1)))
        # randomly drop edges but keep the graph connected and every degree >= 2
        order = rng.permutation(len(edges))
        kept = list(edges)
        for k in order:
            e = edges[k]
            if rng.random() >= cfg.edge_drop:
                continue
            trial = [x for x in kept if x != e]
            deg = {n: 0 for n in nodes}
            for a, b in trial:
                deg[a] += 1
                deg[b] += 1
            if min(deg.values()) < 2 or not _connected(nodes, trial):
                continue
            kept = trial
        keys = sorted(nodes.keys())
        index = {k: n for n, k in enumerate(keys)}
        for k in keys:
            self.junctions.append(Junction(index[k], nodes[k]))

        # ---- road end directions (with optional curvature)
        road_specs = self._road_specs(kept, nx, ny)
        specs = []
        for (ka, kb), rs in zip(kept, road_specs):
            a, b = index[ka], index[kb]
            pa, pb = self.junctions[a].pos, self.junctions[b].pos
            th = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
            phi_a = phi_b = 0.0
            if rng.random() < cfg.curve_prob:
                phi = math.radians(rng.uniform(8.0, cfg.curve_max_deg)) * rng.choice([-1, 1])
                if rng.random() < 0.55:
                    phi_a, phi_b = phi, -phi      # arc
                else:
                    phi_a, phi_b = phi, phi       # S-curve
                if rs.kind == "main":
                    phi_a = phi_b = 0.0           # main roads run straight (the draws stay, for the same layout)
            specs.append((a, b, th, phi_a, phi_b))
            self.junctions[a].arms.append({"spec": len(specs) - 1, "end": "a", "angle": wrap(th + phi_a)})
            self.junctions[b].arms.append({"spec": len(specs) - 1, "end": "b", "angle": wrap(th + phi_b + math.pi)})

        # ---- junction radius from the tightest angle between neighbouring arms (and the widest arm)
        for J in self.junctions:
            wo = max(road_specs[a["spec"]].walk_outer for a in J.arms)
            if cfg.road_types != "uniform" and len(J.arms) < 3:
                wo = max(wo, cfg.walk_outer)      # bends of narrow roads as wide as a street's: room to swing out
            self._size_junction(J, wo)
            J.signalized = len(J.arms) >= 3

        # ---- road centre lines
        for sid, (a, b, th, phi_a, phi_b) in enumerate(specs):
            Ja, Jb = self.junctions[a], self.junctions[b]
            p0 = Ja.pos + Ja.radius * unit(th + phi_a)
            p3 = Jb.pos + Jb.radius * unit(th + phi_b + math.pi)
            d = np.linalg.norm(p3 - p0) / 3.0
            pts = cubic_bezier(p0, p0 + d * unit(th + phi_a), p3 - d * unit(th + phi_b), p3, 200)
            road = Road(sid, a, b, pts, road_specs[sid])
            self.roads.append(road)
        self._build_from_graph()

    @staticmethod
    def _size_junction(J, wo, specs=None):
        """Sort the arms counter-clockwise and set the junction radius from the tightest angle between neighbouring
        arms and the widest arm's walk_outer `wo` (3+ arms: at least 9.5 m, or graph_min_radius of the arms'
        cross-sections `specs` when those are given, i.e. in Town.from_graph; bends: at least 3 m)."""
        J.arms.sort(key=lambda a: a["angle"])
        angs = np.array([a["angle"] for a in J.arms])
        gaps = np.diff(np.concatenate([angs, [angs[0] + 2 * math.pi]]))
        gmin = float(gaps.min())
        if len(J.arms) >= 3:
            floor = 9.5 if specs is None else Town.graph_min_radius(specs, gmin)
            J.radius = max(floor, wo / math.tan(gmin / 2.0) + 1.5)
        else:
            J.radius = max(3.0, wo / math.tan(min(gmin, 2 * math.pi - gmin) / 2.0) + 1.0)

    @staticmethod
    def graph_min_radius(specs, gmin=math.pi / 2):
        """Floor of a 3+-arm junction's radius in Town.from_graph towns, from the cross-sections of its arms and the
        tightest angle `gmin` between neighbouring arms: the procedural 9.5 m fits 7-8 m streets with sidewalks but
        makes the crossings of 6 m residential roads (about 8 m across in reality) look like blobs. The circle must
        clear the widest arm's sidewalk corners at a right angle (walk_outer + 1.5 m, the formula of _size_junction
        at 90 deg); give the right-angle turn between lane centres (radius - lane_width / 2) the arc TURN_RADIUS_MIN,
        which the pure pursuit controller sets: leaving a left turn it drifts outwards and still points outwards,
        so the car's body reaches 1.74 m right of its lane centre on the procedural 8 m arcs, 1.87 m on 6.5 m
        arcs, 1.92 m on 5.5 m arcs and 1.98 m on 4.5 m arcs (scripts in the hand-over note); against a 2 m wide
        van entering its own turn from the road being entered, 3 m lane spacing leaves 0.13 m at 6.5 m and nothing
        at 4.5 m (two expert collisions per 40 routes in the Sapporo town); and keep the sharpest turn, an arc of
        radius * tan(gmin / 2) - lane_width / 2 (both lanes sit on the inside of the wedge), drivable at
        TURN_RADIUS_DRIVABLE (the car's minimum is 2.7 m / tan 0.6 = 3.95 m). Right-angle crossings: 8.0 m for
        residential roads, 8.25 m for streets, 9.25 m for main roads; a 70 deg skew raises the residential floor
        to 8.6 m, 45 deg to 14.5 m."""
        wo = max(rs.walk_outer for rs in specs)
        lane = max(rs.lane_width for rs in specs)
        return max(wo + 1.5, lane / 2.0 + TURN_RADIUS_MIN,
                   (TURN_RADIUS_DRIVABLE + lane / 2.0) / math.tan(min(gmin, math.pi / 2) / 2.0))

    def _build_from_graph(self, buildings=None):
        """Everything derived from self.junctions (pos, arms [spec = road id, end, angle; sorted], radius,
        signalized) and self.roads (centre lines between the junction circles, specs): see the class docstring."""
        cfg, rng = self.cfg, self.rng
        # ---- lanes on roads
        side = 1.0 if cfg.left_hand_traffic else -1.0
        for road in self.roads:
            rs = road.spec
            off = 0.0 if rs.oneway else side * rs.lane_width / 2.0     # one-way roads: a single lane on the centre line
            ab = Lane(len(self.lanes), offset_polyline(road.center, off), "road", rs.speed_limit, rs.lane_width)
            ab.road, ab.start_junction, ab.end_junction = road.id, road.a, road.b
            self.lanes.append(ab)
            road.lane_ab, road.lane_ba = ab.id, -1
            if rs.oneway:
                continue
            ba = Lane(len(self.lanes), offset_polyline(road.center, -off)[::-1].copy(), "road", rs.speed_limit,
                      rs.lane_width)
            ba.road, ba.start_junction, ba.end_junction = road.id, road.b, road.a
            self.lanes.append(ba)
            road.lane_ba = ba.id

        # arms: which lanes come in / go out (-1: none, the far end of a one-way road)
        for J in self.junctions:
            for arm in J.arms:
                road = self.roads[arm["spec"]]
                arm["road"] = road.id
                if arm["end"] == "a":
                    arm["in"], arm["out"] = road.lane_ba, road.lane_ab
                    arm["pt"], arm["dir"] = road.center[0], unit(arm["angle"])
                else:
                    arm["in"], arm["out"] = road.lane_ab, road.lane_ba
                    arm["pt"], arm["dir"] = road.center[-1], unit(arm["angle"])
            J.incoming = [a["in"] for a in J.arms if a["in"] >= 0]
            J.outgoing = [a["out"] for a in J.arms if a["out"] >= 0]

        # ---- junction connectors + polygons + signals
        for J in self.junctions:
            for ia, arm_in in enumerate(J.arms):
                if arm_in["in"] < 0:
                    continue
                lin = self.lanes[arm_in["in"]]
                for ib, arm_out in enumerate(J.arms):
                    if ia == ib or arm_out["out"] < 0:
                        continue
                    lout = self.lanes[arm_out["out"]]
                    h0, h1 = lin.heading[-1], lout.heading[0]
                    pts = arc_bezier(lin.pts[-1], h0, lout.pts[0], h1, 64)
                    c = Lane(len(self.lanes), pts, "conn", min(lin.speed_limit, lout.speed_limit),
                             max(lin.width, lout.width))
                    c.junction = J.id
                    c.start_junction = c.end_junction = J.id
                    dh = float(wrap(h1 - h0))
                    c.turn = "left" if dh > 0.5 else ("right" if dh < -0.5 else "straight")
                    c.pred, c.succ = [lin.id], [lout.id]
                    lin.succ.append(c.id)
                    lout.pred.append(c.id)
                    self.lanes.append(c)
                    J.connectors.append(c.id)
            corners_r, corners_w = [], []
            for arm in J.arms:
                n = np.array([-arm["dir"][1], arm["dir"][0]])
                rs = self.roads[arm["road"]].spec
                corners_r += [arm["pt"] + n * rs.half_width, arm["pt"] - n * rs.half_width]
                corners_w += [arm["pt"] + n * rs.walk_outer, arm["pt"] - n * rs.walk_outer]
            J.road_poly = _hull(corners_r)
            J.walk_poly = _hull(corners_w)
            if J.signalized:
                if cfg.signal_mode == "two_phase":
                    group = _two_phase_groups([a["angle"] for a in J.arms])
                    J.phases = 2
                    J.green = float(rng.uniform(*cfg.two_phase_green))
                    J.yellow = cfg.two_phase_yellow
                    J.allred = cfg.two_phase_allred
                else:
                    group = list(range(len(J.arms)))
                    J.phases = len(J.arms)
                    J.green = float(rng.uniform(*cfg.signal_green))
                    J.yellow = cfg.signal_yellow
                    J.allred = cfg.signal_allred
                J.offset = float(rng.uniform(0, J.cycle))
                for k, arm in enumerate(J.arms):
                    if arm["in"] < 0:
                        continue
                    lin = self.lanes[arm["in"]]
                    lin.signal = (J.id, group[k])
                    lin.stop_s = max(1.0, lin.length - cfg.stop_line_setback)
        self._make_yields()

        # swept strips of junction connectors (bends / skewed junctions bulge past the hull)
        # (lane + shoulder of the wider road; sidewalk only as wide as the narrower one: none if either has none)
        self.conn_road_polys, self.conn_walk_polys, self.conn_curb_polys = [], [], []
        for lane in self.lanes:
            if lane.kind != "conn":
                continue
            ra = self.roads[self.lanes[lane.pred[0]].road].spec
            rb = self.roads[self.lanes[lane.succ[0]].road].spec
            hl = max(ra.lane_width / 2 + ra.shoulder, rb.lane_width / 2 + rb.shoulder)
            self.conn_road_polys.append(self.strip(lane.pts, hl))
            self.conn_curb_polys.append(self.strip(lane.pts, hl + 0.2))
            self.conn_walk_polys.append(self.strip(lane.pts, hl + min(ra.sidewalk, rb.sidewalk)))
        self._make_bbox()
        self._make_raster()
        self._make_markings()
        self._make_signal_heads()
        self._make_stop_sign_posts()
        self._make_buildings_and_trees(buildings)
        self._finalize_static()

    # ------------------------------------------------------------------ helpers
    def _road_specs(self, edges, nx, ny):
        """RoadSpec of every kept grid edge ((i, j), (i', j')). uniform: all TownConfig streets. mixed: the edges
        of `main_roads` interior grid lines (a row and a column first) are main roads; the others are streets,
        residential roads or alleys (road_mix). Own random stream: the layout draws stay those of the uniform
        town with the same seed."""
        cfg = self.cfg
        if cfg.road_types == "uniform":
            street = RoadSpec.street(cfg)
            return [street] * len(edges)
        if cfg.road_types != "mixed":
            raise ValueError(f"road_types must be uniform or mixed (got {cfg.road_types!r})")
        rng = np.random.default_rng([self.seed, 0x726F6164])
        rows = [("row", j) for j in range(1, ny - 1)]
        cols = [("col", i) for i in range(1, nx - 1)]
        lines = []
        for group in (rows, cols):
            if group and len(lines) < cfg.main_roads:
                lines.append(group[int(rng.integers(len(group)))])
        rest = [g for g in rows + cols if g not in lines]
        for k in rng.permutation(len(rest))[:max(0, cfg.main_roads - len(lines))]:
            lines.append(rest[k])
        mix = np.asarray(cfg.road_mix, float)
        kinds = ["street", "residential", "alley"]
        out = []
        for (i0, j0), (i1, j1) in edges:
            on_main = any((t == "row" and j0 == j1 == v) or (t == "col" and i0 == i1 == v) for t, v in lines)
            kind = "main" if on_main else kinds[int(rng.choice(3, p=mix / mix.sum()))]
            out.append(RoadSpec.of_kind(kind, cfg))
        return out

    def _make_bbox(self):
        pts = np.concatenate([r.center for r in self.roads])
        m = 45.0
        self.bbox = (pts[:, 0].min() - m, pts[:, 1].min() - m, pts[:, 0].max() + m, pts[:, 1].max() + m)

    @staticmethod
    def strip(pts, half):
        left = offset_polyline(pts, half)
        right = offset_polyline(pts, -half)
        return np.concatenate([left, right[::-1]])

    def road_polygon(self, road, half):
        left = offset_polyline(road.center, half)
        right = offset_polyline(road.center, -half)
        return np.concatenate([left, right[::-1]])

    def _make_raster(self):
        """Coarse drivable raster: 0 off-road, 1 road, 2 sidewalk."""
        R = Raster(self.bbox, 0.2)
        img = np.zeros((R.H, R.W), np.uint8)
        for road in self.roads:
            R.fill(img, self.road_polygon(road, road.walk_outer), 2)
        for J in self.junctions:
            R.fill(img, J.walk_poly, 2)
        for poly in self.conn_walk_polys:
            R.fill(img, poly, 2)
        for road in self.roads:
            R.fill(img, self.road_polygon(road, road.half_width), 1)
        for J in self.junctions:
            R.fill(img, J.road_poly, 1)
        for poly in self.conn_road_polys:
            R.fill(img, poly, 1)
        self.drv_raster = R
        self.drivable = img

    def is_drivable(self, pts):
        return self.drv_raster.sample(self.drivable, pts) == 1

    def surface(self, pts):
        return self.drv_raster.sample(self.drivable, pts)

    def _make_markings(self):
        cfg = self.cfg
        side = 1.0 if cfg.left_hand_traffic else -1.0
        mw = 0.15

        def strip(pts, off, width):
            a = offset_polyline(pts, off + width / 2)
            b = offset_polyline(pts, off - width / 2)
            return np.concatenate([a, b[::-1]])

        def stop_line(lane, s, rs):
            """Stop line across incoming `lane` at arc length `s` (centre-line side to edge; one-way roads: edge to
            edge), 0.45 m deep."""
            p = lane.poly.interp(s)
            h = float(lane.poly.heading(s))
            u, n = unit(h), unit(h + math.pi / 2)
            inner = -side * (rs.lane_width / 2.0)   # towards road centre
            if rs.edge == "classic":
                outer = side * (rs.lane_width / 2.0 + rs.shoulder - 0.35)
            else:                                    # up to the edge line
                outer = side * (rs.lane_width / 2.0 + 0.2)
            if rs.oneway:
                inner = -outer
            a0 = p + n * inner
            a1 = p + n * outer
            w = 0.45
            return np.array([a0 - u * w / 2, a1 - u * w / 2, a1 + u * w / 2, a0 + u * w / 2])

        def tomare_text(lane, s_line, lane_width):
            """The 止まれ painted before a stop line: its three characters side by side across the lane (left to
            right as the driver reads them), stretched along it, row 0 of the glyphs at the far end. Rectangles
            from runs of paint, merged down the rows (no holes, so both renderers fill them alike)."""
            cw = min(TOMARE_CHAR_W, (lane_width - 0.5 - 2 * TOMARE_SPACE) / 3)
            total = 3 * cw + 2 * TOMARE_SPACE
            s_far = s_line - TOMARE_GAP
            if s_far - TOMARE_LEN < 0.5:
                return []
            rects = []
            for k, glyph in enumerate(TOMARE):
                rows = [np.frombuffer(r.encode(), np.uint8) == ord("#") for r in glyph]
                nr, nc = len(rows), len(rows[0])
                left = total / 2 - k * (cw + TOMARE_SPACE)          # lateral of the character's left edge (+ = left)
                open_runs = {}                                       # (c0, c1) -> first row
                for r in range(nr + 1):
                    runs = set()
                    if r < nr:
                        m = np.concatenate([[False], rows[r], [False]])
                        d = np.diff(m.astype(np.int8))
                        runs = set(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))
                    for run in list(open_runs):
                        if run not in runs:                          # the rectangle ends above row r
                            r0 = open_runs.pop(run)
                            sa, sb = s_far - r0 * TOMARE_LEN / nr, s_far - r * TOMARE_LEN / nr
                            la, lb = left - run[0] * cw / nc, left - run[1] * cw / nc
                            pa, pb = lane.poly.interp(sa), lane.poly.interp(sb)
                            na = unit(float(lane.poly.heading(sa)) + math.pi / 2)
                            nb = unit(float(lane.poly.heading(sb)) + math.pi / 2)
                            rects.append(np.array([pa + na * la, pa + na * lb, pb + nb * lb, pb + nb * la]))
                    for run in runs:
                        open_runs.setdefault(run, r)
            return rects

        for road in self.roads:
            rs = road.spec
            hw = rs.half_width
            c = road.center
            L = road.length
            s = road.poly.s
            # edge lines (solid)
            if rs.edge == "classic":
                for sgn in (1, -1):
                    self.markings.append(strip(c, sgn * (hw - 0.35), mw))
            elif rs.edge == "lane":                 # just outside the lane (the 路側帯 line on residential roads)
                for sgn in (1, -1):
                    self.markings.append(strip(c, sgn * (rs.lane_outer + mw / 2 + 0.05), mw))
            # centre line: solid near junctions, dashed elsewhere (main roads: solid yellow, no overtaking)
            if rs.center == "yellow":
                self.markings_yellow.append(strip(c, 0.0, mw))
            sig_a = self.junctions[road.a].signalized
            sig_b = self.junctions[road.b].signalized
            solid_a = 18.0 if sig_a else 0.0
            solid_b = 18.0 if sig_b else 0.0
            seg = []
            if solid_a > 0:
                seg.append((0.0, min(solid_a, L)))
            t = solid_a + 2.0
            while t < L - solid_b - 2.0:
                seg.append((t, min(t + 5.0, L - solid_b - 2.0)))
                t += 10.0
            if solid_b > 0:
                seg.append((max(0.0, L - solid_b), L))
            for s0, s1 in seg if rs.center == "dashed" else ():
                m = (s >= s0) & (s <= s1)
                if m.sum() >= 2:
                    self.markings.append(strip(c[m], 0.0, mw))
            # stop lines and crosswalks at signalised ends; a stop line only at the minor road's 止まれ
            # (one-way roads: no lane arrives at their start, so no stop line there; the crosswalk stays)
            for end, J in (("a", self.junctions[road.a]), ("b", self.junctions[road.b])):
                if end == "b":
                    lane = self.lanes[road.lane_ab]
                    s_end = L
                    direction = 1.0
                else:
                    lane = self.lanes[road.lane_ba] if road.lane_ba >= 0 else None
                    s_end = 0.0
                    direction = -1.0
                if not J.signalized:
                    if lane is not None and lane.stop_sign_s is not None:
                        self.markings.append(stop_line(lane, lane.stop_sign_s, rs))
                        self.markings.extend(tomare_text(lane, lane.stop_sign_s, rs.lane_width))
                    continue
                if lane is not None:
                    self.markings.append(stop_line(lane, lane.stop_s, rs))
                # crosswalk (zebra) between stop line and junction
                cs0 = s_end - direction * 4.2
                cs1 = s_end - direction * 0.8
                lo, hi = min(cs0, cs1), max(cs0, cs1)
                m = (s >= lo) & (s <= hi)
                if m.sum() < 2:
                    continue
                cc = c[m]
                k = -hw + 0.5
                while k + 0.45 <= hw - 0.3:
                    self.crosswalks.append(strip(cc, k + 0.225, 0.45))
                    k += 0.9

    def _make_yields(self):
        """Who gives way inside a junction. Connectors of approaches that are green at the same time (opposite
        approaches with two-phase signals) may cross; the lower-ranked movement (right < left < straight) yields.
        Each yielding connector gets yields = [(other connector, s here, s there)], the closest points of the
        two centre lines, and wait_s: where its front bumper waits, short of the first crossing.
        Unsignalised junctions with 3+ arms (Town.from_graph) are priority crossings: the approaches of the minor road
        (_priority_groups) give way to every movement of the priority road they would touch, and movements of the
        same road are ranked as above.
        Two equal movements that cross (the right turns of the paired approaches at a skewed junction) are resolved
        too, by _equal_gives_way: if one of them already gives way to the other's approach, that one (cars queue
        behind each other in the single lane: the other order deadlocks), else the one from the minor road (road
        types: main > street > residential > alley), else the one with the other approach on its left (左方優先).
        web/src/traffic.js makeYields() makes the same decisions for the browser demo.
        In mixed towns the bends (unsignalised) whose two turns pass too closely are taken one at a time the same
        way: the outer turn waits clear of the inner one (_narrow_wait), possibly before the bend.
        Towns from a graph (mirrored_turn_yield): a right turn also gives way, at its entry, to the oncoming left turn
        around the same corner when their centre lines run closer than MIRROR_PASS (道路交通法: 右折車は対向の
        左折車の進行も妨げない). On the 3 m lanes of real residential roads the two bodies pass within decimetres
        (concentric arcs 3 m apart; the car's corners and its tracking error eat the rest), which cost the expert
        two collisions in the Sapporo town at junction radius 6 m; the procedural towns' 3.5 m streets and 9.5 m
        junctions leave more room and keep the old rules.
        Towns from a graph (stop_signs) also put a 止まれ stop line on every approach of the minor road to a priority
        crossing, STOP_SIGN_SETBACK before the end of its lane (Lane.stop_sign_s; _make_markings paints it): the
        expert stops there once, then goes on to its connector's wait point and gives way from there."""
        rank = {"right": 0, "left": 1, "straight": 2}
        for lane in self.lanes:
            lane.yields, lane.wait_s, lane.stop_sign_s = [], None, None
        for J in self.junctions:
            narrow = not J.signalized and self.cfg.road_types == "mixed" and len(J.arms) < 3
            prio = not J.signalized and len(J.arms) >= 3
            if not (J.signalized or narrow or prio):
                continue
            major = self._priority_groups(J) if prio else None
            if prio and self.stop_signs:
                for arm in J.arms:
                    if arm["in"] >= 0 and major[arm["in"]] == 1:
                        lin = self.lanes[arm["in"]]
                        lin.stop_sign_s = max(0.5, lin.length - STOP_SIGN_SETBACK)
            conns = [self.lanes[c] for c in J.connectors]
            equals = []
            for a in conns:
                ga = self.lanes[a.pred[0]].signal
                for b in conns:
                    gb = self.lanes[b.pred[0]].signal
                    if b.pred[0] == a.pred[0]:
                        continue
                    # priority crossing, a and b from different roads: the minor one gives way whatever the turns
                    minor = prio and major[a.pred[0]] != major[b.pred[0]]
                    if minor and major[a.pred[0]] == 0:
                        continue
                    if not (narrow or prio) and (ga is None or gb is None or ga[1] != gb[1]):
                        continue
                    equal = rank[a.turn] == rank[b.turn]       # equal pairs once each, in a fixed order
                    if not minor and (rank[a.turn] > rank[b.turn] or
                                      (equal and (narrow or (a.pred[0], a.succ[0]) > (b.pred[0], b.succ[0])))):
                        continue
                    pa, pb = a.poly.pts, b.poly.pts
                    d = np.hypot(*(pa[:, None, :] - pb[None, :, :]).transpose(2, 0, 1))
                    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
                    # signalised: the two cars would touch (a crossing or a merge). Unsignalised bends of a mixed
                    # town: the turns pass too closely (narrow lanes; the body of a turning car swings out)
                    if d[i, j] >= (NARROW_PASS if narrow else 2.2):
                        continue
                    if equal and not minor:                  # decided below, after the ranked ones
                        equals.append((a, b, float(a.poly.s[i]), float(b.poly.s[j])))
                        continue
                    a.yields.append((b.id, float(a.poly.s[i]), float(b.poly.s[j])))
            # approaches each connector gives way to by rank (taken before the equal pairs: order-independent)
            ranked = {c.id: {self.lanes[y[0]].pred[0] for y in c.yields} for c in conns}
            for a, b, sa, sb in equals:
                if self._equal_gives_way(J, a, b, ranked):
                    a.yields.append((b.id, sa, sb))
                else:
                    b.yields.append((a.id, sb, sa))
            if self.mirrored_turn_yield and len(J.arms) >= 3:
                for a in conns:
                    if a.turn != "right":
                        continue
                    for b in conns:
                        if b.turn != "left" or b.pred[0] == a.pred[0] or any(y[0] == b.id for y in a.yields):
                            continue
                        if (self.lanes[b.pred[0]].road != self.lanes[a.succ[0]].road or
                                self.lanes[b.succ[0]].road != self.lanes[a.pred[0]].road):
                            continue                         # not the same corner the other way round
                        d = np.hypot(*(a.poly.pts[:, None, :] - b.poly.pts[None, :, :]).transpose(2, 0, 1))
                        if d.min() < MIRROR_PASS:            # wait at the entry until the left turn has cleared
                            a.yields.append((b.id, 0.0, float(b.length)))
            for a in conns:
                if not a.yields:
                    continue
                a.yields.sort(key=lambda y: y[1])
                # the furthest front-bumper position whose front circle (half a car width behind the bumper) stays
                # out of the corridor that every car it yields to checks along its path (NPCs: half width + circle
                # radius + 0.35 m, about 2.25 m): waiting there blocks nobody
                others = np.concatenate([self.lanes[b].poly.pts for b, _, _ in a.yields])
                if narrow:
                    # the inner car is still straightening out over the first metres of its next lane
                    tails = [self.lanes[self.lanes[b].succ[0]].pts[:13] for b, _, _ in a.yields]
                    a.wait_s = self._narrow_wait(a, np.concatenate([others] + tails))
                    continue
                wait = 0.0
                for sb in np.arange(a.yields[0][1], -0.01, -0.25):
                    sc = sb - 0.9
                    c = a.poly.interp1(sc) if sc >= 0 else a.poly.pts[0] + sc * a.poly.seg_dir[0]
                    if np.min(np.hypot(*(others - np.asarray(c)).T)) >= YIELD_CLEAR:
                        wait = float(sb)
                        break
                a.wait_s = wait

    def _priority_groups(self, J):
        """Unsignalised crossing: approach lane -> 0 on the priority road, 1 on the minor road(s). The priority road
        is a pair of nearly opposite arms (the through road of a T-junction); with a choice, the pair of the higher
        road kind, then the longer one."""
        road_rank = {"alley": 0, "residential": 1, "street": 2, "main": 3}
        arms = J.arms
        best, pair = None, None
        for i in range(len(arms)):
            for k in range(i + 1, len(arms)):
                opp = abs(float(wrap(arms[i]["angle"] - arms[k]["angle"])))
                ri, rk = (road_rank[self.roads[arms[x]["spec"]].kind] for x in (i, k))
                li, lk = (self.roads[arms[x]["spec"]].length for x in (i, k))
                key = (opp > math.radians(120), min(ri, rk), ri + rk, li + lk, opp)
                if best is None or key > best:
                    best, pair = key, (i, k)
        return {arm["in"]: 0 if k in pair else 1 for k, arm in enumerate(arms) if arm["in"] >= 0}

    def _equal_gives_way(self, J, a, b, ranked):
        """True if connector `a` gives way to `b` (equal movements whose paths cross; see _make_yields)."""
        pa, pb = a.pred[0], b.pred[0]
        if (pa in ranked[b.id]) != (pb in ranked[a.id]):
            return pb in ranked[a.id]                 # a already waits for b's approach (or b for a's)
        road_rank = {"alley": 0, "residential": 1, "street": 2, "main": 3}
        ka = road_rank[self.roads[self.lanes[pa].road].kind]
        kb = road_rank[self.roads[self.lanes[pb].road].kind]
        if ka != kb:
            return ka < kb                            # the minor road gives way
        # 左方優先: give way to the car that comes from the left (the right with right-hand traffic); clear cases only
        h = float(self.lanes[pa].heading[-1])
        v = b.poly.pts[0] - J.pos
        side = (math.cos(h) * v[1] - math.sin(h) * v[0]) / max(float(np.hypot(v[0], v[1])), 1e-9)
        if not self.cfg.left_hand_traffic:
            side = -side
        if abs(side) > 0.2:
            return side > 0
        return pa < pb

    @staticmethod
    def _narrow_wait(a, others):
        """Narrow bend: the furthest front-bumper position of connector `a` (down to 12 m before it, on the approach
        lane) where the waiting car's front corners and sides keep NARROW_CLEAR from `others`, the centre lines of the
        turns it gives way to (and the first metres after them), so the inner turn sweeps past it."""
        p0, u0 = a.poly.pts[0], a.poly.seg_dir[0]
        h0 = math.atan2(u0[1], u0[0])
        for sb in np.arange(a.yields[0][1], -12.01, -0.25):
            pts = []
            for back in (0.0, 0.9, 2.2):           # front bumper, front circle, middle of the car
                sp = sb - back
                c = np.asarray(a.poly.interp1(sp)) if sp >= 0 else p0 + sp * u0
                h = a.poly.heading1(sp) if sp >= 0 else h0
                n = np.array([-math.sin(h), math.cos(h)]) * 0.95
                pts += [c + n, c - n]
            if min(float(np.min(np.hypot(*(others - q).T))) for q in pts) >= NARROW_CLEAR:
                return float(sb)
        return -12.0

    def _make_signal_heads(self):
        """Far-side overhead signal (Japanese style) for every signalised approach.

        The head hangs over the outgoing lane of the straight movement, just past
        the junction; the pole stands on that road's curb-side sidewalk. For a
        T-junction approached from the stem, the head stands on the far sidewalk.
        """
        cfg = self.cfg
        side = 1.0 if cfg.left_hand_traffic else -1.0
        for J in self.junctions:
            if not J.signalized:
                continue
            for k, arm in enumerate(J.arms):
                if arm["in"] < 0:
                    continue
                lane = self.lanes[arm["in"]]
                h_in = float(lane.heading[-1])
                straight = [c for c in lane.succ if self.lanes[c].turn == "straight"]
                if straight:
                    out = self.lanes[self.lanes[straight[0]].succ[0]]
                    s_h = min(2.5, out.length / 2)
                    head = out.poly.interp(s_h)
                    h_out = float(out.poly.heading(s_h))
                    n = unit(h_out + math.pi / 2) * side
                    ro = self.roads[out.road].spec
                    base, dirn, d0 = head, n, ro.lane_width / 2 + ro.shoulder + 1.1
                else:
                    u = unit(h_in)
                    p_end = lane.pts[-1]
                    t_exit = _ray_poly_exit(p_end + u * 0.1, u, J.road_poly)
                    head = p_end + u * (t_exit + 1.6)
                    base, dirn, d0 = head, u, 0.35
                # push the pole outwards until it stands clear of every drivable surface
                ring = np.array([[0.6, 0], [-0.6, 0], [0, 0.6], [0, -0.6], [0, 0]])
                d = d0
                while d < d0 + 6.0 and np.any(self.surface(base + dirn * d + ring) == 1):
                    d += 0.25
                pole = base + dirn * d
                if np.any(self.surface(pole + ring) == 1):
                    # dense corner: search the neighbourhood for the closest free spot
                    target = base + dirn * d0
                    g = np.arange(-8.0, 8.01, 0.5)
                    cand = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2) + target
                    cand = cand[np.argsort(np.hypot(*(cand - target).T))]
                    for c in cand:
                        if not np.any(self.surface(c + ring) == 1):
                            pole = c
                            break
                self.signal_heads.append({
                    "junction": J.id, "phase": lane.signal[1], "lane": lane.id,
                    "head": head, "pole": pole, "yaw": h_in + math.pi, "z": 5.4,
                })

    def _make_stop_sign_posts(self):
        """止まれ signs (Town.from_graph): a red inverted triangle on a post at the curb side of every approach with a
        stop line (Lane.stop_sign_s), level with the line, pushed outwards until it stands clear of the drivable
        surface. No random draws: the trees and buildings after it come out as before."""
        side = 1.0 if self.cfg.left_hand_traffic else -1.0
        ring = np.array([[0.35, 0], [-0.35, 0], [0, 0.35], [0, -0.35], [0, 0]])
        self.stop_sign_posts = []
        for lane in self.lanes:
            if lane.stop_sign_s is None:
                continue
            rs = self.roads[lane.road].spec
            p = lane.poly.interp(lane.stop_sign_s)
            h = float(lane.poly.heading(lane.stop_sign_s))
            n = unit(h + math.pi / 2) * side                  # towards the curb
            d = rs.lane_width / 2 + 0.7
            while d < rs.lane_width / 2 + 5.0 and np.any(self.surface(p + n * d + ring) == 1):
                d += 0.25
            self.stop_sign_posts.append({"lane": lane.id, "pole": p + n * d, "yaw": h + math.pi, "z": 2.1})

    def _make_buildings_and_trees(self, buildings=None):
        """Procedural buildings along the roads (or the given boxes {x, y, yaw, l, w, h}, minus those over a road or
        sidewalk) and trees along the sidewalks."""
        cfg, rng = self.cfg, self.rng
        R = Raster(self.bbox, 0.5)
        blocked = np.zeros((R.H, R.W), np.uint8)
        for road in self.roads:
            R.fill(blocked, self.road_polygon(road, road.walk_outer + 0.8), 1)
        for J in self.junctions:
            R.fill(blocked, J.walk_poly, 1)
            # enlarge junction area a bit
            c = J.pos
            R.fill(blocked, (J.walk_poly - c) * 1.15 + c, 1)
        for poly in self.conn_walk_polys:
            R.fill(blocked, poly, 1)
        for sh in self.signal_heads:
            p = sh["pole"]
            R.fill(blocked, box_corners(p[0], p[1], 0.0, 2.5, 2.5), 1)

        palette = np.array([
            [200, 210, 220], [170, 180, 190], [120, 140, 170], [140, 160, 200], [90, 110, 150],
            [185, 200, 215], [150, 150, 150], [110, 120, 130], [200, 190, 170], [160, 120, 100],
            [215, 225, 230], [100, 100, 115],
        ], dtype=float)  # BGR-ish, muted
        # building heights: (low, mid) cumulative probabilities; main roads are lined with taller buildings,
        # residential roads and alleys with houses
        tall = {"main": (0.35, 0.85), "residential": (0.85, 0.98), "alley": (0.9, 0.99)}
        if buildings is not None:
            for b in buildings:
                corners = box_corners(b["x"], b["y"], b["yaw"], b["l"], b["w"])
                test = np.zeros_like(blocked)
                R.fill(test, corners, 1)
                if np.any(test & blocked):
                    continue
                col = palette[rng.integers(len(palette))] * rng.uniform(0.85, 1.1)
                self.buildings.append({**{k: float(b[k]) for k in ("x", "y", "yaw", "l", "w", "h")},
                                       "color": np.clip(col, 0, 255)})
        for road in self.roads if buildings is None else ():
            wo = road.walk_outer
            p_low, p_mid = tall.get(road.kind, (0.68, 0.94))
            for sgn in (1.0, -1.0):
                s = float(rng.uniform(1.0, 6.0))
                while s < road.length - 4.0:
                    w = float(rng.uniform(8.0, 17.0))
                    gap = float(rng.uniform(1.0, 5.0))
                    if rng.random() < cfg.building_prob:
                        d = float(rng.uniform(8.0, 17.0))
                        setback = float(rng.uniform(1.0, 4.5))
                        sc = min(s + w / 2, road.length)
                        p = road.poly.interp(sc)
                        h = float(road.poly.heading(sc))
                        n = unit(h + math.pi / 2) * sgn
                        center = p + n * (wo + setback + d / 2)
                        corners = box_corners(center[0], center[1], h, w, d)
                        test = np.zeros_like(blocked)
                        R.fill(test, corners, 1)
                        if not np.any(test & blocked):
                            r = rng.random()
                            if r < p_low:
                                height = rng.uniform(5.0, 13.0)
                            elif r < p_mid:
                                height = rng.uniform(13.0, 26.0)
                            else:
                                height = rng.uniform(26.0, 42.0)
                            col = palette[rng.integers(len(palette))] * rng.uniform(0.85, 1.1)
                            self.buildings.append({
                                "x": float(center[0]), "y": float(center[1]), "yaw": h,
                                "l": w, "w": d, "h": float(height), "color": np.clip(col, 0, 255),
                            })
                            R.fill(blocked, box_corners(center[0], center[1], h, w + 1.0, d + 1.0), 1)
                    s += w + gap
        # trees along the outer edge of sidewalks
        for road in self.roads:
            if road.spec.sidewalk <= 0:
                continue
            wo = road.walk_outer
            for sgn in (1.0, -1.0):
                s = float(rng.uniform(6.0, 14.0))
                while s < road.length - 6.0:
                    if rng.random() < cfg.tree_prob:
                        p = road.poly.interp(s)
                        h = float(road.poly.heading(s))
                        n = unit(h + math.pi / 2) * sgn
                        pos = p + n * (wo - 0.55)
                        g = rng.uniform(0.75, 1.15)
                        tree = {
                            "x": float(pos[0]), "y": float(pos[1]),
                            "trunk_h": float(rng.uniform(1.9, 2.8)),
                            "r": float(rng.uniform(1.2, 2.0)),
                            "ch": float(rng.uniform(2.2, 3.6)),
                            "color": np.array([40 * g, 125 * g, 60 * g]) * rng.uniform(0.8, 1.2, 3),
                        }
                        if not any(math.hypot(*(pos - sp["pole"])) < 1.5 for sp in self.stop_sign_posts):
                            self.trees.append(tree)
                    s += float(rng.uniform(9.0, 16.0))

    def building_edges(self):
        """The buildings' outlines for line-of-sight tests: segments (4B, 2, 2), building centres (B, 2) and radii (B,)
        (for a quick distance cut; the edges of building i are rows 4i..4i+3). Built on first use."""
        if self._building_edges is None:
            b = self.buildings
            c = box_corners(np.array([x["x"] for x in b]), np.array([x["y"] for x in b]), np.array([x["yaw"] for x in b]),
                            np.array([x["l"] for x in b]), np.array([x["w"] for x in b])).reshape(-1, 4, 2)
            edges = np.stack([c, np.roll(c, -1, axis=1)], axis=2).reshape(-1, 2, 2)
            centre = c.mean(axis=1)
            self._building_edges = (edges, centre, np.linalg.norm(c - centre[:, None], axis=2).max(axis=1)
                                    if len(b) else np.zeros(0))
        return self._building_edges

    def _finalize_static(self):
        """Arrays of static collision boxes: (x, y, yaw, l, w)."""
        boxes = []
        for b in self.buildings:
            boxes.append((b["x"], b["y"], b["yaw"], b["l"], b["w"]))
        for t in self.trees:
            boxes.append((t["x"], t["y"], 0.0, 0.45, 0.45))
        for sh in self.signal_heads:
            boxes.append((sh["pole"][0], sh["pole"][1], 0.0, 0.4, 0.4))
        self.static_boxes = np.array(boxes, float).reshape(-1, 5)
        self.static_corners = box_corners(self.static_boxes[:, 0], self.static_boxes[:, 1], self.static_boxes[:, 2],
                                          self.static_boxes[:, 3], self.static_boxes[:, 4])
        self.road_lanes = [l.id for l in self.lanes if l.kind == "road"]
        # paths for pedestrians: (road id, side) -> polyline, on the sidewalk or in the 路側帯 (in_road; lat: scale
        # of the lateral spread around it). Alleys have none.
        self.walkways = []
        for road in self.roads:
            ww = road.spec.walkway()
            if ww is None:
                continue
            off, lat, in_road = ww
            for sgn in (1.0, -1.0):
                pts = offset_polyline(road.center, sgn * off)
                self.walkways.append({"road": road.id, "side": sgn, "poly": Polyline(pts), "lat": lat,
                                      "in_road": in_road})
        self.in_road_walkways = any(w["in_road"] for w in self.walkways)
        self.narrow_lanes = any(self.lanes[l].width < NARROW_LANE for l in self.road_lanes)
        # road lanes trucks may use: wide lanes from which wide lanes go on (prune dead ends of the wide network)
        self.truck_lanes = None
        if self.narrow_lanes:
            ok = {l for l in self.road_lanes if self.lanes[l].width >= NARROW_LANE}
            changed = True
            while changed:
                changed = False
                for l in sorted(ok):
                    if not any(self.lanes[c].succ[0] in ok for c in self.lanes[l].succ):
                        ok.discard(l)
                        changed = True
            self.truck_lanes = ok or None
        self.speed_varies = len({l.speed_limit for l in self.lanes}) > 1

    # ------------------------------------------------------------------ queries
    def signal_state_for_lane(self, lane_id, t):
        lane = self.lanes[lane_id]
        if lane.signal is None:
            return TL_GREEN
        j, ph = lane.signal
        return self.junctions[j].signal_state(ph, t)

    def total_lane_length(self):
        return sum(l.length for l in self.lanes if l.kind == "road")

    def summary(self):
        nsig = sum(J.signalized for J in self.junctions)
        who = f"seed={self.seed}" if self.name == str(self.seed) else f"{self.name}, seed={self.seed}"
        return (f"Town({who}) junctions={len(self.junctions)} (signalised {nsig}) "
                f"roads={len(self.roads)} lanes={len(self.lanes)} buildings={len(self.buildings)} "
                f"trees={len(self.trees)} size={self.bbox[2]-self.bbox[0]:.0f}x{self.bbox[3]-self.bbox[1]:.0f}m")
