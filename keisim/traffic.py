"""NPC vehicles (lane-following IDM), pedestrians and the shared corridor check."""
from __future__ import annotations

import math

import numpy as np

from .config import TL_GREEN, TL_RED, TL_YELLOW, TrafficConfig
from .geometry import wrap

# ----------------------------------------------------------------------------
# Lane-sequence lookahead
# ----------------------------------------------------------------------------


def lane_lookahead(town, lanes, idx, s, dist, step=2):
    """Polyline following `lanes[idx:]` from arc length `s` for `dist` metres.

    Returns (pts (P,2), cum (P,), curv (P,), stops [(s_path, lane_id)], complete)
    `step` subsamples the 0.5 m lane points (2 -> ~1 m spacing).
    """
    pts_l, curv_l = [], []
    stops = []
    base = 0.0
    first = True
    total = 0.0
    k = idx
    while k < len(lanes) and total < dist:
        lane = town.lanes[lanes[k]]
        n = len(lane.pts)
        ds = lane.length / (n - 1)
        if first:
            j0 = min(n - 1, int(s / ds) + 1)
            p0 = np.array(lane.poly.interp1(float(s)))
            seg = np.concatenate([p0[None], lane.pts[j0::step]], 0)
            cv = np.concatenate([[lane.curv[max(0, j0 - 1)]], lane.curv[j0::step]])
            if lane.stop_s is not None and lane.stop_s >= s - 1e-6:
                stops.append((lane.stop_s - s, lane.id))
            base = lane.length - s
            first = False
        else:
            seg = lane.pts[step::step]
            cv = lane.curv[step::step]
            if lane.stop_s is not None:
                stops.append((base + lane.stop_s, lane.id))
            base += lane.length
        pts_l.append(seg)
        curv_l.append(cv)
        total = base
        k += 1
    pts = np.concatenate(pts_l, 0)
    curv = np.concatenate(curv_l, 0)
    d = np.diff(pts, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
    m = cum <= dist + 1.0
    m[:2] = True
    return pts[m], cum[m], curv[m], stops, total >= dist


NCIRC = 4


def vehicle_circles(xy, yaw, length, width):
    """Cover oriented boxes with NCIRC circles each (vectorised).
    Returns centres (N*NCIRC,2), radii (N*NCIRC,), owner index (N*NCIRC,)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    n = len(xy)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, int)
    length = np.asarray(length, float)
    width = np.asarray(width, float)
    offs = (length / 2 - width / 2)[:, None] * np.linspace(-1.0, 1.0, NCIRC)[None]
    c, s = np.cos(yaw), np.sin(yaw)
    cx = xy[:, 0:1] + c[:, None] * offs
    cy = xy[:, 1:2] + s[:, None] * offs
    return (np.stack([cx, cy], -1).reshape(-1, 2), np.repeat(width / 2 * 1.05, NCIRC),
            np.repeat(np.arange(n), NCIRC))


GRID_MIN_PAIRS = 40000      # corridor_gaps: below this many (path, circle) pairs the dense test is faster
YIELD_GAP = 5.0             # s: a yielding turn goes only if no car with priority reaches the crossing sooner


def near_pairs(origin, radius, cxy):
    """All (k, m) with |cxy[m] - origin[k]| < radius[k], plus some farther ones: the circles in the 3x3 grid cells
    (cell size = the largest radius) around each origin. Unordered."""
    K = len(origin)
    cell = max(float(np.max(radius)), 1.0)
    cg = np.floor(cxy / cell).astype(np.int64)
    og = np.floor(origin / cell).astype(np.int64)
    lo = np.minimum(cg.min(0), og.min(0)) - 2
    cg -= lo
    og -= lo
    W = int(max(cg[:, 1].max(), og[:, 1].max())) + 3
    ckey = cg[:, 0] * W + cg[:, 1]
    order = np.argsort(ckey, kind="stable")
    skey = ckey[order]
    d = np.array([-1, 0, 1])
    qk = np.repeat(np.arange(K), 9)
    qkey = (og[qk, 0] + np.tile(np.repeat(d, 3), K)) * W + og[qk, 1] + np.tile(np.tile(d, 3), K)
    a = np.searchsorted(skey, qkey, "left")
    cnt = np.searchsorted(skey, qkey, "right") - a
    nz = cnt > 0
    qk, a, cnt = qk[nz], a[nz], cnt[nz]
    ks = np.repeat(qk, cnt)
    ms = order[np.repeat(a - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))]
    return ks, ms


def corridor_gaps(paths, cum, valid, origin, heading, half_len, half_w, cxy, cr, cowner, cspeed, cyaw,
                  self_owner, margin):
    """Batched obstacle search along K paths.

    paths (K,P,2), cum (K,P) distance along path, valid (K,P) bool; origin (K,2)
    vehicle centres; heading (K,); half_len/half_w (K,); obstacles are circles:
    cxy (M,2), cr (M,), cowner (M,) owner ids, cspeed (M,), cyaw (M,);
    self_owner (K,) (circles of the same owner are ignored); margin: scalar, (K,),
    (1,M) or (K,M) extra lateral margin.
    Returns gap (K,) bumper-to-obstacle distance along the path (inf if none),
            lead_speed (K,), lead_owner (K,) (-1 if none).
    """
    K, P = paths.shape[:2]
    gap = np.full(K, np.inf)
    lead_v = np.zeros(K)
    lead_o = np.full(K, -1)
    if K == 0 or len(cxy) == 0:
        return gap, lead_v, lead_o
    maxlen = np.where(valid, cum, 0.0).max(1)
    mid = paths[:, P // 2]
    lim = maxlen + 4.0
    if K * len(cxy) < GRID_MIN_PAIRS:
        dm = cxy[None, :, :] - mid[:, None, :]
        dm2 = dm[..., 0] ** 2 + dm[..., 1] ** 2
        ds = cxy[None, :, :] - origin[:, None, :]
        ds2 = ds[..., 0] ** 2 + ds[..., 1] ** 2
        cand = (ds2 < (lim * lim)[:, None]) & (dm2 < ((0.6 * lim + 6.0) ** 2)[:, None])
        cand &= cowner[None, :] != self_owner[:, None]
        ks, ms = np.nonzero(cand)
    else:                       # big towns: only the circles near each path
        ks, ms = near_pairs(origin, lim, cxy)
        dm = cxy[ms] - mid[ks]
        ds = cxy[ms] - origin[ks]
        cand = (ds[:, 0] ** 2 + ds[:, 1] ** 2 < (lim * lim)[ks]) & \
            (dm[:, 0] ** 2 + dm[:, 1] ** 2 < ((0.6 * lim + 6.0) ** 2)[ks]) & (cowner[ms] != self_owner[ks])
        ks, ms = ks[cand], ms[cand]
        o = np.lexsort((ms, ks))            # the dense test's order, so that ties resolve the same way
        ks, ms = ks[o], ms[o]
    if len(ks) == 0:
        return gap, lead_v, lead_o
    u = np.stack([np.cos(heading), np.sin(heading)], -1)
    lon = ((cxy[ms] - origin[ks]) * u[ks]).sum(-1)
    keep = lon > -0.5
    ks, ms, lon = ks[keep], ms[keep], lon[keep]
    if len(ks) == 0:
        return gap, lead_v, lead_o
    diff = paths[ks] - cxy[ms][:, None, :]
    d2 = diff[..., 0] ** 2 + diff[..., 1] ** 2
    d2 = np.where(valid[ks], d2, np.inf)
    p = np.argmin(d2, 1)
    dmin = np.sqrt(d2[np.arange(len(ks)), p])
    if np.ndim(margin) == 2:
        mg = margin[0, ms] if margin.shape[0] == 1 else margin[ks, ms]
    elif np.ndim(margin) == 1:
        mg = margin[ks]
    else:
        mg = margin
    hit = dmin < half_w[ks] + cr[ms] + mg
    if not hit.any():
        return gap, lead_v, lead_o
    ks, ms, p, lon = ks[hit], ms[hit], p[hit], lon[hit]
    s_hit = np.maximum(cum[ks, p], lon)
    g = s_hit - half_len[ks] - cr[ms]
    order = np.lexsort((g, ks))
    ks, ms, g = ks[order], ms[order], g[order]
    first = np.concatenate([[True], ks[1:] != ks[:-1]])
    kf, mf, gf = ks[first], ms[first], g[first]
    gap[kf] = gf
    lead_v[kf] = cspeed[mf] * np.cos(wrap(cyaw[mf] - heading[kf]))
    lead_o[kf] = cowner[mf]
    return gap, lead_v, lead_o


# ----------------------------------------------------------------------------
# NPC vehicles
# ----------------------------------------------------------------------------
CAR_COLORS = np.array([
    [235, 235, 235], [240, 240, 238], [190, 190, 195], [30, 30, 32], [120, 120, 125],
    [140, 60, 25], [40, 40, 170], [30, 110, 190], [70, 70, 50], [180, 140, 90], [60, 130, 60],
], dtype=float)


class Traffic:
    """Lane-following NPC vehicles with IDM longitudinal control."""

    def __init__(self, town, cfg: TrafficConfig, rng, n):
        self.town, self.cfg, self.rng = town, cfg, rng
        self.n = 0
        self.route: list[list[int]] = []
        self.ri = np.zeros(0, int)
        self.s = np.zeros(0)
        self.v = np.zeros(0)
        self.acc = np.zeros(0)
        self.dims = np.zeros((0, 3))
        self.color = np.zeros((0, 3))
        self.kind = np.zeros(0, int)
        self.vf = np.zeros(0)
        self.xy = np.zeros((0, 2))
        self.yaw = np.zeros(0)
        self.stuck = np.zeros(0)
        self.target_n = n
        self.has_yields = any(getattr(l, "yields", None) for l in town.lanes)

    # --------------------------------------------------------------- spawning
    def _sample_dims(self):
        r = self.rng.random()
        if r < 0.62:
            return np.array([self.rng.uniform(4.0, 4.8), self.rng.uniform(1.68, 1.85), self.rng.uniform(1.4, 1.6)]), 0
        if r < 0.78:
            return np.array([self.rng.uniform(3.35, 3.45), 1.48, self.rng.uniform(1.6, 1.8)]), 0  # kei car
        if r < 0.92:
            return np.array([self.rng.uniform(4.6, 5.0), self.rng.uniform(1.8, 1.95), self.rng.uniform(1.75, 2.0)]), 0
        return np.array([self.rng.uniform(6.0, 7.6), self.rng.uniform(2.0, 2.25), self.rng.uniform(2.6, 3.0)]), 1

    def spawn_many(self, n, avoid_xy, avoid_r=12.0, lanes=None, s_range=None):
        town, rng = self.town, self.rng
        cand = [l for l in (lanes if lanes is not None else town.road_lanes)]
        if not cand:
            return
        w = np.array([town.lanes[l].length for l in cand])
        w = w / w.sum()
        tries = 0
        placed = 0
        while placed < n and tries < n * 30:
            tries += 1
            lid = cand[rng.choice(len(cand), p=w)]
            lane = town.lanes[lid]
            if s_range is None:
                s = rng.uniform(4.0, max(4.5, self._spawn_end(lane)))
            else:
                s = rng.uniform(*s_range)
                if s < 0 or s > lane.length:
                    continue
                if self.cfg.box_rule and lane.stop_s is not None and s > lane.stop_s - 20.0:
                    continue
            p = lane.poly.interp(s)
            if avoid_xy is not None and len(avoid_xy) and np.min(np.hypot(*(avoid_xy - p).T)) < avoid_r:
                continue
            if self.n and np.min(np.hypot(*(self.xy - p).T)) < 11.0:
                continue
            self._add(lid, s, p, float(lane.poly.heading(s)))
            placed += 1

    def _spawn_end(self, lane):
        """Last spawn position on a lane. With the box rule, 20 m before a stop line: a car placed right at the line
        at speed could not stop and would roll into the junction regardless of the light and the box."""
        end = lane.length - 4.0
        if self.cfg.box_rule and lane.stop_s is not None:
            end = min(end, lane.stop_s - 20.0)
        return end

    def _add(self, lid, s, p, yaw):
        dims, kind = self._sample_dims()
        col = CAR_COLORS[self.rng.integers(len(CAR_COLORS))] * self.rng.uniform(0.9, 1.08) + self.rng.uniform(-8, 8, 3)
        self.route.append([lid])
        self.ri = np.append(self.ri, 0)
        self.s = np.append(self.s, s)
        vf = self.rng.uniform(0.8, 1.05)
        self.v = np.append(self.v, self.rng.uniform(0.3, 0.8) * self.town.lanes[lid].speed_limit * vf)
        self.acc = np.append(self.acc, 0.0)
        self.dims = np.vstack([self.dims, dims])
        self.color = np.vstack([self.color, np.clip(col, 0, 255)])
        self.kind = np.append(self.kind, kind)
        self.vf = np.append(self.vf, vf)
        self.xy = np.vstack([self.xy, p])
        self.yaw = np.append(self.yaw, yaw)
        self.stuck = np.append(self.stuck, 0.0)
        self.n += 1

    def _extend(self, i, need):
        """Make sure route i has at least `need` metres beyond the current position."""
        town, rng = self.town, self.rng
        r = self.route[i]
        k = self.ri[i]
        rem = town.lanes[r[k]].length - self.s[i]
        for j in range(k + 1, len(r)):
            rem += town.lanes[r[j]].length
        while rem < need:
            last = town.lanes[r[-1]]
            if not last.succ:
                break
            nxt = last.succ[rng.integers(len(last.succ))]
            r.append(nxt)
            rem += town.lanes[nxt].length
        # drop consumed lanes
        if k > 2:
            del r[: k - 1]
            self.ri[i] -= k - 1

    def respawn(self, i, avoid_xy):
        """Teleport NPC i to a random free location."""
        town, rng = self.town, self.rng
        for _ in range(40):
            lid = town.road_lanes[rng.integers(len(town.road_lanes))]
            lane = town.lanes[lid]
            s = rng.uniform(4.0, max(4.5, self._spawn_end(lane)))
            p = lane.poly.interp(s)
            if np.min(np.hypot(*(avoid_xy - p).T)) < 60.0:
                continue
            others = np.delete(self.xy, i, 0)
            if len(others) and np.min(np.hypot(*(others - p).T)) < 12.0:
                continue
            self.route[i] = [lid]
            self.ri[i] = 0
            self.s[i] = s
            self.v[i] = 0.0
            self.xy[i] = p
            self.yaw[i] = float(lane.poly.heading(s))
            self.stuck[i] = 0.0
            return

    # ------------------------------------------------------------------- step
    PC = 100          # cached lookahead points per NPC (~1 m spacing)
    CACHE_LEN = 90.0

    def circles(self):
        return vehicle_circles(self.xy, self.yaw, self.dims[:, 0], self.dims[:, 1])

    def _ensure_cache(self):
        n = self.n
        if getattr(self, "_cn", -1) != n:
            self.c_pts = np.zeros((n, self.PC, 2))
            self.c_cum = np.full((n, self.PC), np.inf)
            self.c_curv = np.zeros((n, self.PC))
            self.c_stop = np.full((n, 4), np.inf)
            self.c_stop_lane = np.full((n, 4), -1, int)
            self.c_odo0 = np.zeros(n)
            self.c_len = np.zeros(n)
            self.odo = np.zeros(n)
            self._cn = n
            for i in range(n):
                self._refresh(i)

    def _refresh(self, i):
        self._extend(i, self.CACHE_LEN + 5.0)
        pts, cum, curv, stops, _ = lane_lookahead(self.town, self.route[i], self.ri[i], self.s[i], self.CACHE_LEN)
        m = min(self.PC, len(pts))
        self.c_pts[i, :m] = pts[:m]
        self.c_pts[i, m:] = pts[m - 1]
        self.c_cum[i, :m] = cum[:m]
        self.c_cum[i, m:] = np.inf
        self.c_curv[i, :m] = curv[:m]
        self.c_curv[i, m:] = 0.0
        self.c_stop[i] = np.inf
        self.c_stop_lane[i] = -1
        for k, (ss, lid) in enumerate(stops[:4]):
            self.c_stop[i, k] = ss
            self.c_stop_lane[i, k] = lid
        self.c_odo0[i] = self.odo[i]
        self.c_len[i] = cum[m - 1]

    def occupancy(self):
        """lane id -> list of (s, v, length) of NPCs currently on that lane."""
        occ = {}
        for i in range(self.n):
            occ.setdefault(self.route[i][self.ri[i]], []).append((float(self.s[i]), float(self.v[i]), float(self.dims[i, 0])))
        return occ

    def box_occupants(self, occ, ego_xy=None):
        """junction id -> incoming lanes of the vehicles inside it: NPCs on its connectors, NPCs past their stop line
        and still moving (committed to enter, e.g. at the end of a yellow), and the ego, as None, while it is within
        the junction radius."""
        lanes = self.town.lanes
        box = {}
        for lid, cars in occ.items():
            L = lanes[lid]
            if L.kind == "conn":
                box.setdefault(L.junction, set()).add(L.pred[0])
            elif L.stop_s is not None and any(s + n / 2 > L.stop_s + 0.5 and v > 1.0 for s, v, n in cars):
                box.setdefault(L.end_junction, set()).add(lid)     # past its stop line and moving: entering
        if ego_xy is not None:
            for J in self.town.junctions:
                if J.signalized and math.hypot(*(np.asarray(ego_xy) - J.pos)) < J.radius:
                    box.setdefault(J.id, set()).add(None)
        return box

    def box_busy(self, conn_id, box):
        """True if a vehicle from an approach of another signal group (or the ego) is inside the junction of
        connector `conn_id`. With split signals every approach is its own group; with two-phase signals the
        opposite approach shares the green and is handled by yielding (Lane.yields)."""
        lanes = self.town.lanes
        c = lanes[conn_id]
        mine = lanes[c.pred[0]].signal
        for p in box.get(c.junction, ()):
            if p is None:
                return True
            if p != c.pred[0] and (mine is None or lanes[p].signal is None or lanes[p].signal[1] != mine[1]):
                return True
        return False

    # ------------------------------------------------------------------- yielding (two-phase signals)
    def yield_tables(self, ego_info=None):
        """Who is where, for must_yield: lane -> [(s, v, length, is_ego, next lane)], front first.
        ego_info = (lane, s on lane, next lane, v, length) adds the ego."""
        on_lane = {}
        for j in range(self.n):
            r, k = self.route[j], self.ri[j]
            on_lane.setdefault(r[k], []).append((float(self.s[j]), float(self.v[j]), float(self.dims[j, 0]), False,
                                                 r[k + 1] if k + 1 < len(r) else None))
        if ego_info is not None:
            lid, s, nxt, v, L = ego_info
            on_lane.setdefault(lid, []).append((s, v, L, True, nxt))
        for lst in on_lane.values():
            lst.sort(key=lambda e: -e[0])
        return on_lane

    def must_yield(self, conn, t, on_lane):
        """True if a car with priority over connector `conn` (Lane) is in its crossing or reaches it within
        YIELD_GAP seconds (see min_arrival)."""
        return self.min_arrival(conn, t, on_lane) < YIELD_GAP

    def min_arrival(self, conn, t, on_lane, moving_only=False):
        """Earliest time [s] a car with priority over connector `conn` can be in its crossing (0: already in it;
        inf: none coming). Cars that will stop at a red or yellow light do not count, nor do cars queued behind
        one that is standing still to go elsewhere (e.g. an oncoming car waiting to turn right itself).
        moving_only: standing cars (and the ones queued behind them) do not count either: for a turn that is already
        past its wait point, a standing car with priority is waiting for it."""
        town = self.town
        lanes = town.lanes
        best = math.inf
        for b, _, s2 in conn.yields:
            vmax = 1.05 * lanes[b].speed_limit
            for s, v, L, ego, _ in on_lane.get(b, ()):
                if s - L / 2 > s2 + 1.5 or (moving_only and v < 0.5):
                    continue                                    # its rear is past the crossing (or it waits)
                d = s2 - (s + L / 2)
                best = min(best, 0.0 if d < 3.0 else self._arrival(d, v, vmax, ego))
            pred = lanes[b].pred[0]
            # a car standing at the entry of one of pred's other connectors blocks the whole approach lane
            if any(e[1] < 0.5 and e[0] < 4.0 for c in lanes[pred].succ if c != b for e in on_lane.get(c, ())):
                continue
            stop_s = lanes[pred].stop_s
            st = town.signal_state_for_lane(pred, t)
            for s, v, L, ego, nxt in on_lane.get(pred, ()):    # front first
                if nxt != b:
                    if v < 0.5:
                        break                                   # standing to go elsewhere: the rest queue behind
                    continue
                if moving_only and v < 0.5:
                    break                                       # waits, and the cars behind queue behind it
                front = s + L / 2
                if not (st != TL_GREEN and stop_s is not None and self._will_stop(st, stop_s - front, v, ego)):
                    best = min(best, self._arrival(lanes[pred].length - front + s2, v, vmax, ego))
                break                                           # the cars behind it arrive later
        return best

    @staticmethod
    def _arrival(d, v, vmax, ego=False):
        """Earliest time a car `d` m away at speed `v` can be there, accelerating up to `vmax` (a car starting
        from its stop line arrives much sooner than d / v suggests): moving NPCs (even creeping) at 2.5 m/s^2,
        standing NPCs after a 1 s reaction at their IDM acceleration (2 m/s^2), the ego at once at 3.5 m/s^2."""
        if d <= 0.0:
            return 0.0
        a, t0 = (3.5, 0.0) if ego else ((2.5, 0.0) if v >= 0.2 else (2.0, 1.0))
        return t0 + Traffic._reach(d, v, vmax, a)

    @staticmethod
    def _reach(d, v, vmax, a):
        t1 = max(0.0, (vmax - v) / a)
        d1 = v * t1 + 0.5 * a * t1 * t1
        if d1 >= d:
            return (-v + math.sqrt(v * v + 2.0 * a * d)) / a
        return t1 + (d - d1) / max(vmax, 0.1)

    @staticmethod
    def _will_stop(st, d, v, ego):
        """Whether a car `d` m before its red / yellow stop line (front bumper) at speed `v` stops there: the NPC
        rule of step() (red: always, if it physically can; yellow: if d > v^2 / 7 + 1), and for the ego only when
        it is already nearly standing before the line (the driving model may take a late yellow)."""
        if ego:
            return v < 2.0 and d > -0.5
        if d < -0.5:
            return False                                        # already past the line
        if st == TL_RED:
            return d + 0.5 >= v * v / 16.0                      # the line counts until 0.5 m past it
        return d > v * v / 7.0 + 1.0

    def _yield_gaps(self, stop_gap, t, ego_info, hl):
        """Turns that give way wait at their connector's wait point while must_yield holds."""
        lanes = self.town.lanes
        tables = None
        for i in range(self.n):
            r, k = self.route[i], self.ri[i]
            cur = lanes[r[k]]
            if cur.kind == "conn":
                if not cur.yields:
                    continue
                c, d = cur, cur.wait_s - (self.s[i] + hl[i])
            elif k + 1 < len(r) and lanes[r[k + 1]].yields:
                c = lanes[r[k + 1]]
                d = cur.length - (self.s[i] + hl[i]) + c.wait_s
            else:
                continue
            if d < -0.5 or d > 40.0:
                continue                                        # past the wait point (committed) or far away
            v = self.v[i]
            if v > 1.0 and d < v * v / 8.0:
                continue                                        # too close to stop gently: go on
            if tables is None:
                tables = self.yield_tables(ego_info)
            if self.must_yield(c, t, tables):
                stop_gap[i] = min(stop_gap[i], max(d, 0.0))

    def exit_blocked(self, conn_id, exit_id, need, occ, ego=None, ignore=None):
        """True if the junction exit `exit_id` (reached via connector `conn_id`) has
        no room for a vehicle needing `need` metres ("don't block the box")."""
        lanes = self.town.lanes
        for s, v, L in occ.get(exit_id, []):
            if s - L / 2 < need and v < 2.0:
                return True
        for c in lanes[exit_id].pred:
            for s, v, L in occ.get(c, []):
                if v < 1.0 and (c == conn_id or s > lanes[c].length - need):
                    return True
        if ego is not None:
            exy, ev = ego
            if ev < 2.0:
                poly = lanes[exit_id].poly
                se, _ = poly.project(exy, 0.0, need + 6.0)
                gap = math.hypot(*(np.asarray(poly.interp1(se)) - exy))   # true distance (no clamping artefacts)
                if 0.1 < se < need + 2.0 and gap < 2.2:
                    return True
        return False

    def step(self, dt, t, ext_xy, ext_r, ext_owner, ext_speed, ext_yaw, ego_xy, ego_v=0.0, ego_yaw=None,
             ego_info=None):
        """ext_*: external obstacles (ego circles with owner -1, pedestrians owner -2-k)."""
        n = self.n
        if n == 0:
            return
        self._ensure_cache()
        cfg, town = self.cfg, self.town
        v = self.v
        look = np.clip(v * 3.5 + 18.0, 20.0, 45.0)
        off = self.odo - self.c_odo0
        for i in np.nonzero(self.c_len - off < look + 2.0)[0]:
            self._refresh(i)
        off = self.odo - self.c_odo0
        rel_all = self.c_cum - off[:, None]
        first_i = np.argmax(rel_all >= -0.6, axis=1)
        PW = 48
        rows = np.arange(n)[:, None]
        widx = np.minimum(first_i[:, None] + np.arange(PW)[None], self.PC - 1)
        paths = self.c_pts[rows, widx]
        rel = rel_all[rows, widx]
        valid = (rel <= look[:, None]) & np.isfinite(rel)
        valid[:, 0] = True
        relc = np.where(valid, rel, 0.0)
        curv = self.c_curv[rows, widx]
        # curve speed envelope
        vc = np.sqrt(cfg.lat_acc / np.maximum(np.abs(curv), 1e-4))
        env = np.where(valid, np.sqrt(vc ** 2 + 4.0 * np.maximum(relc, 0.0)), np.inf)
        vmax = np.minimum(town.cfg.speed_limit * self.vf, env.min(1))
        # traffic lights: first stop line ahead
        hl = self.dims[:, 0] / 2
        d_stop = self.c_stop - off[:, None] - hl[:, None]
        stop_gap = np.full(n, np.inf)
        ahead = d_stop >= -0.5
        has = ahead.any(1)
        first = np.argmax(ahead, 1)
        occ = box = None
        for i in np.nonzero(has)[0]:
            d = d_stop[i, first[i]]
            if d > 45.0:
                continue
            lid = int(self.c_stop_lane[i, first[i]])
            st = town.signal_state_for_lane(lid, t)
            if st == TL_RED or (st == TL_YELLOW and d > v[i] * v[i] / (2 * 3.5) + 1.0):
                stop_gap[i] = d - 0.6
            elif d < 12.0:
                # don't block the box: only enter if the exit has room
                r = self.route[i]
                try:
                    k = r.index(lid, self.ri[i])
                except ValueError:
                    continue
                if k + 2 < len(r):
                    if occ is None:
                        occ = self.occupancy()
                        box = self.box_occupants(occ, ego_xy) if cfg.box_rule else None
                    if self.exit_blocked(r[k + 1], r[k + 2], self.dims[i, 0] + 3.0, occ, (ego_xy, ego_v)) or \
                            (box is not None and self.box_busy(r[k + 1], box)):
                        stop_gap[i] = d - 0.6
        if self.has_yields:          # two-phase signals: turns wait for a gap in the oncoming traffic
            self._yield_gaps(stop_gap, t, ego_info, hl)
        # obstacles: other NPCs + external
        cxy, cr, cow = self.circles()
        csp = self.v[cow]
        cyw = self.yaw[cow]
        if len(ext_xy):
            cxy = np.concatenate([cxy, ext_xy])
            cr = np.concatenate([cr, ext_r])
            cow = np.concatenate([cow, ext_owner])
            csp = np.concatenate([csp, ext_speed])
            cyw = np.concatenate([cyw, ext_yaw])
        margin = np.where(cow < -1, 0.9, 0.35)[None, :]
        gap, lead_v, _ = corridor_gaps(paths, relc, valid, self.xy, self.yaw, hl, self.dims[:, 1] / 2,
                                       cxy, cr, cow, csp, cyw, np.arange(n), margin)
        # IDM
        a_max, b, T, s0 = cfg.idm_a, cfg.idm_b, cfg.idm_T, cfg.idm_s0
        v0 = np.maximum(vmax, 0.5)
        acc = a_max * (1.0 - (v / v0) ** 4)
        for g, vl in ((gap, lead_v), (stop_gap, np.zeros(n))):
            m = np.isfinite(g)
            if m.any():
                gg = np.maximum(g[m], 0.05)
                dv = v[m] - vl[m]
                s_star = s0 + np.maximum(0.0, v[m] * T + v[m] * dv / (2 * math.sqrt(a_max * b)))
                acc_i = a_max * (1.0 - (v[m] / v0[m]) ** 4 - (s_star / gg) ** 2)
                acc[m] = np.minimum(acc[m], acc_i)
        acc = np.clip(acc, -8.0, a_max)
        self.acc = acc
        v_new = np.maximum(0.0, v + acc * dt)
        v_new[(v_new < 0.05) & (acc < 0)] = 0.0
        self.v = v_new
        adv = v_new * dt
        self.odo = self.odo + adv
        self.s = self.s + adv
        lanes = town.lanes
        for i in range(n):
            r = self.route[i]
            k = self.ri[i]
            L = lanes[r[k]].length
            while self.s[i] > L:
                if k + 1 >= len(r):
                    self._extend(i, 10.0)
                    r = self.route[i]
                    k = self.ri[i]
                    if k + 1 >= len(r):
                        self.s[i] = L
                        break
                self.s[i] -= L
                k += 1
                self.ri[i] = k
                L = lanes[r[k]].length
            poly = lanes[r[k]].poly
            si = float(self.s[i])
            self.xy[i, 0], self.xy[i, 1] = poly.interp1(si)
            self.yaw[i] = poly.heading1(si)
        self.stuck = np.where(self.v < 0.1, self.stuck + dt, 0.0)
        self.release_stuck(ego_xy, ego_yaw)

    def release_stuck(self, ego_xy, ego_yaw=None):
        """Teleport vehicles that have been stopped for ages far from the ego and, with cfg.release_hidden, those
        the ego camera cannot see: deadlocks around the ego then clear up without anything vanishing in view."""
        cfg = self.cfg
        first = min(cfg.stuck_far_s, cfg.stuck_hidden_s) if cfg.release_hidden else cfg.stuck_far_s
        for i in np.nonzero(self.stuck > first)[0]:
            d = self.xy[i] - ego_xy
            dist = float(np.hypot(*d))
            go = dist > 60.0 and self.stuck[i] > cfg.stuck_far_s
            if not go and cfg.release_hidden and self.stuck[i] > cfg.stuck_hidden_s and ego_yaw is not None:
                b = math.atan2(d[1], d[0]) - ego_yaw
                off = abs(math.atan2(math.sin(b), math.cos(b)))
                go = dist > cfg.view_dist or (dist > 8.0 and off > math.radians(cfg.view_half_deg))
            if go:
                self.respawn(i, ego_xy[None])
                self._refresh(i)


# ----------------------------------------------------------------------------
# Pedestrians
# ----------------------------------------------------------------------------
SHIRTS = np.array([[200, 60, 40], [40, 40, 200], [40, 160, 40], [230, 230, 230], [30, 30, 30], [0, 140, 230],
                   [150, 50, 150], [60, 200, 220], [120, 120, 120], [200, 150, 80]], float)
PANTS = np.array([[40, 30, 25], [90, 60, 30], [30, 30, 30], [120, 110, 100], [60, 50, 110], [150, 150, 150]], float)
SKIN = np.array([[150, 180, 220], [120, 160, 205], [90, 125, 170], [60, 90, 130]], float)


class Pedestrians:
    WALK, CROSS = 0, 1

    def __init__(self, town, cfg: TrafficConfig, rng, n):
        self.town, self.cfg, self.rng = town, cfg, rng
        ww = town.walkways
        self.n = n
        lens = np.array([w["poly"].length for w in ww])
        p = lens / lens.sum()
        self.w = rng.choice(len(ww), size=n, p=p)
        self.s = np.array([rng.uniform(1.0, ww[k]["poly"].length - 1.0) for k in self.w])
        self.dir = rng.choice([-1.0, 1.0], size=n)
        self.speed = rng.uniform(1.0, 1.6, n)
        self.lat = rng.uniform(-0.6, 0.6, n)
        self.mode = np.zeros(n, int)
        self.c0 = np.zeros((n, 2))
        self.c1 = np.zeros((n, 2))
        self.height = rng.uniform(1.5, 1.85, n)
        self.colors = np.stack([SHIRTS[rng.integers(len(SHIRTS), size=n)] * rng.uniform(0.85, 1.1, (n, 1)),
                                PANTS[rng.integers(len(PANTS), size=n)],
                                SKIN[rng.integers(len(SKIN), size=n)]], 1)
        self.phase = rng.uniform(0, 2 * np.pi, n)
        self.xy = np.zeros((n, 2))
        self.yaw = np.zeros(n)
        self.vel = np.zeros(n)
        self.cooldown = rng.uniform(0, 20, n)
        # partner walkway (other side of the same road)
        self.partner = np.zeros(len(ww), int)
        for i, a in enumerate(ww):
            for j, b in enumerate(ww):
                if i != j and a["road"] == b["road"]:
                    self.partner[i] = j
        self._update_pose()

    def _update_pose(self):
        ww = self.town.walkways
        for i in range(self.n):
            if self.mode[i] == self.WALK:
                poly = ww[self.w[i]]["poly"]
                s = float(self.s[i])
                px, py = poly.interp1(s)
                h = poly.heading1(s)
                lat = self.lat[i]
                self.xy[i, 0] = px - math.sin(h) * lat
                self.xy[i, 1] = py + math.cos(h) * lat
                self.yaw[i] = h if self.dir[i] > 0 else h + math.pi

    def _safe_to_cross(self, i, veh_xy, veh_v, veh_yaw):
        if len(veh_xy) == 0:
            return True
        p = self.xy[i]
        d = p[None] - veh_xy
        dist = np.hypot(d[:, 0], d[:, 1])
        u = np.stack([np.cos(veh_yaw), np.sin(veh_yaw)], -1)
        along = (d * u).sum(1)
        near = (dist < 45.0) & (along > -3.0)
        if not near.any():
            return True
        tta = along[near] / np.maximum(veh_v[near], 0.5)
        return not np.any((along[near] < 18.0) | (tta < 3.5))

    def step(self, dt, veh_xy, veh_v, veh_yaw, trigger=None):
        """trigger: optional boolean mask of pedestrians that should try to cross now."""
        ww = self.town.walkways
        rng = self.rng
        self.cooldown -= dt
        self.phase += self.speed * dt * 4.2 * (self.vel > 0.1)
        for i in range(self.n):
            if self.mode[i] == self.WALK:
                poly = ww[self.w[i]]["poly"]
                self.s[i] += self.dir[i] * self.speed[i] * dt
                if self.s[i] < 1.0 or self.s[i] > poly.length - 1.0:
                    self.dir[i] *= -1
                    self.s[i] = min(max(float(self.s[i]), 1.0), poly.length - 1.0)
                self.vel[i] = self.speed[i]
                want = (rng.random() < self.cfg.cross_rate * dt) or (trigger is not None and trigger[i])
                if want and self.cooldown[i] <= 0 and 10.0 < self.s[i] < poly.length - 10.0:
                    if self._safe_to_cross(i, veh_xy, veh_v, veh_yaw):
                        j = self.partner[self.w[i]]
                        other = ww[j]["poly"]
                        s2, _ = other.project(self.xy[i])
                        p2 = np.array(other.interp1(s2))
                        h2 = other.heading1(s2)
                        lat2 = rng.uniform(-0.5, 0.5)
                        p2 = p2 + np.array([-math.sin(h2), math.cos(h2)]) * lat2
                        self.mode[i] = self.CROSS
                        self.c0[i] = self.xy[i]
                        self.c1[i] = p2
                        self.w[i] = j
                        self.s[i] = s2
                        self.lat[i] = lat2
                        self.speed[i] = rng.uniform(1.1, 1.7)
                    self.cooldown[i] = rng.uniform(15, 40)
            else:
                d = self.c1[i] - self.xy[i]
                L = math.hypot(d[0], d[1])
                if L < 0.15:
                    self.mode[i] = self.WALK
                    continue
                # do not walk into a vehicle standing right in front
                if len(veh_xy):
                    rel = veh_xy - self.xy[i]
                    fwd = (rel @ d) / L
                    side = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]) / L
                    if np.any((fwd > 0.0) & (fwd < 3.2) & (side < 2.6)):
                        self.vel[i] = 0.0
                        continue
                stepl = min(L, self.speed[i] * dt)
                self.xy[i] = self.xy[i] + d / L * stepl
                self.yaw[i] = math.atan2(d[1], d[0])
                self.vel[i] = self.speed[i]
        # refresh walking poses
        self._update_pose()

    def velocity_vectors(self):
        return np.stack([np.cos(self.yaw), np.sin(self.yaw)], -1) * self.vel[:, None]
