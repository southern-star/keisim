"""Procedural town generator.

A town is a jittered grid of junctions connected by (optionally curved) two-lane
roads. Every junction with three or more arms is signalised with a conflict-free
split-phase controller (one approach green at a time). Lanes form a directed
graph (road lanes -> junction connector lanes -> road lanes) which is used for
routing, traffic and the privileged expert.
"""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from .config import TL_GREEN, TL_RED, TL_YELLOW, TownConfig
from .geometry import (Polyline, arc_bezier, box_corners, cubic_bezier, curvature_of,
                       headings_of, offset_polyline, resample, unit, wrap)

LANE_DS = 0.5


class Lane:
    def __init__(self, lid, pts, kind, speed_limit):
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
        self.stop_s = None                # arc length of the stop line
        self.signal = None                # (junction id, phase index)

    def __repr__(self):
        return f"Lane({self.id},{self.kind},L={self.length:.1f},turn={self.turn})"


class Road:
    def __init__(self, rid, a, b, center, half_width):
        self.id = rid
        self.a, self.b = a, b
        self.center = resample(center, LANE_DS)
        self.poly = Polyline(self.center)
        self.length = self.poly.length
        self.half_width = half_width
        self.lane_ab = -1
        self.lane_ba = -1


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
    """A procedurally generated town. Deterministic given (seed, cfg)."""

    def __init__(self, seed: int, cfg: TownConfig | None = None):
        self.seed = int(seed)
        self.cfg = cfg or TownConfig()
        self.rng = np.random.default_rng(self.seed)
        self.lanes: list[Lane] = []
        self.roads: list[Road] = []
        self.junctions: list[Junction] = []
        self.buildings: list[dict] = []
        self.trees: list[dict] = []
        self.signal_heads: list[dict] = []
        self.markings: list[np.ndarray] = []      # white paint polygons
        self.crosswalks: list[np.ndarray] = []
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self):
        cfg, rng = self.cfg, self.rng
        nx = int(rng.integers(cfg.grid_min, cfg.grid_max + 1))
        ny = int(rng.integers(cfg.grid_min, cfg.grid_max + 1))
        sp = float(rng.uniform(cfg.spacing_min, cfg.spacing_max))
        self.spacing = sp
        nodes = {}
        for i in range(nx):
            for j in range(ny):
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
        specs = []
        for (ka, kb) in kept:
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
            specs.append((a, b, th, phi_a, phi_b))
            self.junctions[a].arms.append({"spec": len(specs) - 1, "end": "a", "angle": wrap(th + phi_a)})
            self.junctions[b].arms.append({"spec": len(specs) - 1, "end": "b", "angle": wrap(th + phi_b + math.pi)})

        # ---- junction radius from the tightest angle between neighbouring arms
        hw = cfg.road_half_width
        wo = cfg.walk_outer
        for J in self.junctions:
            J.arms.sort(key=lambda a: a["angle"])
            angs = np.array([a["angle"] for a in J.arms])
            gaps = np.diff(np.concatenate([angs, [angs[0] + 2 * math.pi]]))
            gmin = float(gaps.min())
            if len(J.arms) >= 3:
                J.radius = max(9.5, wo / math.tan(gmin / 2.0) + 1.5)
                J.signalized = True
            else:
                J.radius = max(3.0, wo / math.tan(min(gmin, 2 * math.pi - gmin) / 2.0) + 1.0)

        # ---- road centre lines
        for sid, (a, b, th, phi_a, phi_b) in enumerate(specs):
            Ja, Jb = self.junctions[a], self.junctions[b]
            p0 = Ja.pos + Ja.radius * unit(th + phi_a)
            p3 = Jb.pos + Jb.radius * unit(th + phi_b + math.pi)
            d = np.linalg.norm(p3 - p0) / 3.0
            pts = cubic_bezier(p0, p0 + d * unit(th + phi_a), p3 - d * unit(th + phi_b), p3, 200)
            road = Road(sid, a, b, pts, hw)
            self.roads.append(road)

        # ---- lanes on roads
        side = 1.0 if cfg.left_hand_traffic else -1.0
        off = side * cfg.lane_width / 2.0
        for road in self.roads:
            ab = Lane(len(self.lanes), offset_polyline(road.center, off), "road", cfg.speed_limit)
            ab.road, ab.start_junction, ab.end_junction = road.id, road.a, road.b
            self.lanes.append(ab)
            ba = Lane(len(self.lanes), offset_polyline(road.center, -off)[::-1].copy(), "road", cfg.speed_limit)
            ba.road, ba.start_junction, ba.end_junction = road.id, road.b, road.a
            self.lanes.append(ba)
            road.lane_ab, road.lane_ba = ab.id, ba.id

        # arms: which lanes come in / go out
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
            J.incoming = [a["in"] for a in J.arms]
            J.outgoing = [a["out"] for a in J.arms]

        # ---- junction connectors + polygons + signals
        for J in self.junctions:
            for ia, arm_in in enumerate(J.arms):
                lin = self.lanes[arm_in["in"]]
                for ib, arm_out in enumerate(J.arms):
                    if ia == ib:
                        continue
                    lout = self.lanes[arm_out["out"]]
                    h0, h1 = lin.heading[-1], lout.heading[0]
                    pts = arc_bezier(lin.pts[-1], h0, lout.pts[0], h1, 64)
                    c = Lane(len(self.lanes), pts, "conn", cfg.speed_limit)
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
                corners_r += [arm["pt"] + n * hw, arm["pt"] - n * hw]
                corners_w += [arm["pt"] + n * wo, arm["pt"] - n * wo]
            J.road_poly = _hull(corners_r)
            J.walk_poly = _hull(corners_w)
            if J.signalized:
                J.phases = len(J.arms)
                J.green = float(rng.uniform(*cfg.signal_green))
                J.yellow = cfg.signal_yellow
                J.allred = cfg.signal_allred
                J.offset = float(rng.uniform(0, J.cycle))
                for k, arm in enumerate(J.arms):
                    lin = self.lanes[arm["in"]]
                    lin.signal = (J.id, k)
                    lin.stop_s = max(1.0, lin.length - cfg.stop_line_setback)

        # swept strips of junction connectors (bends / skewed junctions bulge past the hull)
        self.conn_road_polys, self.conn_walk_polys, self.conn_curb_polys = [], [], []
        hl = cfg.lane_width / 2 + cfg.shoulder
        for lane in self.lanes:
            if lane.kind != "conn":
                continue
            self.conn_road_polys.append(self.strip(lane.pts, hl))
            self.conn_curb_polys.append(self.strip(lane.pts, hl + 0.2))
            self.conn_walk_polys.append(self.strip(lane.pts, hl + cfg.sidewalk_width))
        self._make_bbox()
        self._make_raster()
        self._make_markings()
        self._make_signal_heads()
        self._make_buildings_and_trees()
        self._finalize_static()

    # ------------------------------------------------------------------ helpers
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
        wo, hw = self.cfg.walk_outer, self.cfg.road_half_width
        for road in self.roads:
            R.fill(img, self.road_polygon(road, wo), 2)
        for J in self.junctions:
            R.fill(img, J.walk_poly, 2)
        for poly in self.conn_walk_polys:
            R.fill(img, poly, 2)
        for road in self.roads:
            R.fill(img, self.road_polygon(road, hw), 1)
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
        hw = cfg.road_half_width
        side = 1.0 if cfg.left_hand_traffic else -1.0
        mw = 0.15

        def strip(pts, off, width):
            a = offset_polyline(pts, off + width / 2)
            b = offset_polyline(pts, off - width / 2)
            return np.concatenate([a, b[::-1]])

        for road in self.roads:
            c = road.center
            L = road.length
            s = road.poly.s
            # edge lines (solid)
            for sgn in (1, -1):
                self.markings.append(strip(c, sgn * (hw - 0.35), mw))
            # centre line: solid near junctions, dashed elsewhere
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
            for s0, s1 in seg:
                m = (s >= s0) & (s <= s1)
                if m.sum() >= 2:
                    self.markings.append(strip(c[m], 0.0, mw))
            # stop lines and crosswalks at signalised ends
            for end, J in (("a", self.junctions[road.a]), ("b", self.junctions[road.b])):
                if not J.signalized:
                    continue
                if end == "b":
                    lane = self.lanes[road.lane_ab]
                    s_end = L
                    direction = 1.0
                else:
                    lane = self.lanes[road.lane_ba]
                    s_end = 0.0
                    direction = -1.0
                # stop line across incoming lane (centre-line side to edge)
                sl_s = lane.stop_s
                p = lane.poly.interp(sl_s)
                h = float(lane.poly.heading(sl_s))
                u, n = unit(h), unit(h + math.pi / 2)
                inner = -side * (cfg.lane_width / 2.0)   # towards road centre
                outer = side * (cfg.lane_width / 2.0 + cfg.shoulder - 0.35)
                a0 = p + n * inner
                a1 = p + n * outer
                w = 0.45
                self.markings.append(np.array([a0 - u * w / 2, a1 - u * w / 2, a1 + u * w / 2, a0 + u * w / 2]))
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
                lane = self.lanes[arm["in"]]
                h_in = float(lane.heading[-1])
                straight = [c for c in lane.succ if self.lanes[c].turn == "straight"]
                if straight:
                    out = self.lanes[self.lanes[straight[0]].succ[0]]
                    s_h = min(2.5, out.length / 2)
                    head = out.poly.interp(s_h)
                    h_out = float(out.poly.heading(s_h))
                    n = unit(h_out + math.pi / 2) * side
                    base, dirn, d0 = head, n, cfg.lane_width / 2 + cfg.shoulder + 1.1
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
                    "junction": J.id, "phase": k, "lane": lane.id,
                    "head": head, "pole": pole, "yaw": h_in + math.pi, "z": 5.4,
                })

    def _make_buildings_and_trees(self):
        cfg, rng = self.cfg, self.rng
        R = Raster(self.bbox, 0.5)
        blocked = np.zeros((R.H, R.W), np.uint8)
        wo = cfg.walk_outer
        for road in self.roads:
            R.fill(blocked, self.road_polygon(road, wo + 0.8), 1)
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
        for road in self.roads:
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
                            if r < 0.68:
                                height = rng.uniform(5.0, 13.0)
                            elif r < 0.94:
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
            for sgn in (1.0, -1.0):
                s = float(rng.uniform(6.0, 14.0))
                while s < road.length - 6.0:
                    if rng.random() < cfg.tree_prob:
                        p = road.poly.interp(s)
                        h = float(road.poly.heading(s))
                        n = unit(h + math.pi / 2) * sgn
                        pos = p + n * (wo - 0.55)
                        g = rng.uniform(0.75, 1.15)
                        self.trees.append({
                            "x": float(pos[0]), "y": float(pos[1]),
                            "trunk_h": float(rng.uniform(1.9, 2.8)),
                            "r": float(rng.uniform(1.2, 2.0)),
                            "ch": float(rng.uniform(2.2, 3.6)),
                            "color": np.array([40 * g, 125 * g, 60 * g]) * rng.uniform(0.8, 1.2, 3),
                        })
                    s += float(rng.uniform(9.0, 16.0))

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
        # sidewalk paths for pedestrians: (road id, side) -> polyline
        self.walkways = []
        off = self.cfg.road_half_width + self.cfg.sidewalk_width / 2
        for road in self.roads:
            for sgn in (1.0, -1.0):
                pts = offset_polyline(road.center, sgn * off)
                self.walkways.append({"road": road.id, "side": sgn, "poly": Polyline(pts)})

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
        return (f"Town(seed={self.seed}) junctions={len(self.junctions)} (signalised {nsig}) "
                f"roads={len(self.roads)} lanes={len(self.lanes)} buildings={len(self.buildings)} "
                f"trees={len(self.trees)} size={self.bbox[2]-self.bbox[0]:.0f}x{self.bbox[3]-self.bbox[1]:.0f}m")
