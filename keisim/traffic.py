"""NPC vehicles (lane-following IDM), pedestrians and the shared corridor check."""
from __future__ import annotations

import math

import numpy as np

from .config import TL_GREEN, TL_RED, TL_YELLOW, TrafficConfig
from .geometry import box_corners, curvature_of, rays_clear, wrap
from .roadnet import PED_FLASH, PED_GREEN
from .route import (LC_CHECK, LC_PUSH, LC_PUSH_TO, LC_START_MARGIN, blend_weight, change_room, change_window,
                    lane_moves, link_map, pick_move)

# ----------------------------------------------------------------------------
# Lane-sequence lookahead
# ----------------------------------------------------------------------------


def lane_lookahead(town, lanes, idx, s, dist, step=2, change_at=None):
    """Polyline following `lanes[idx:]` from arc length `s` for `dist` metres.

    Returns (pts (P,2), cum (P,), curv (P,), stops [(s_path, lane_id)], complete)
    `step` subsamples the 0.5 m lane points (2 -> ~1 m spacing).
    Two sibling lanes in a row (a lane change, route.py) blend from one into the other: at A-local (a0, lc) from
    change_at[A] (an NPC's planned change), else at the start of A's window.
    """
    if any(lanes[k + 1] in town.lanes[lanes[k]].links for k in range(idx, len(lanes) - 1)):
        return _lookahead_changes(town, lanes, idx, s, dist, step, change_at or {})
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


def _lookahead_changes(town, lanes, idx, s, dist, step, change_at):
    """lane_lookahead with lane changes in `lanes[idx:]`."""
    pts_l, curv_l, stops = [], [], []
    base, lo, k = 0.0, float(s), idx
    while k < len(lanes) and base < dist:
        lane = town.lanes[lanes[k]]
        sib = k + 1 < len(lanes) and lanes[k + 1] in lane.links
        hi = lane.length
        if sib:
            if lane.id in change_at:
                a0, lc = change_at[lane.id]
            else:           # (from where the change before ends: back-to-back changes, e.g. into a right-turn lane)
                win = change_window(lane, lo, town.lanes[lanes[k + 1]])
                a0, lc = (win[0], win[2]) if win is not None else (max(lo, lane.length / 2 - 12.5), 25.0)
            hi = max(lo, a0)
        if hi > lo + 1e-9 or not (pts_l or sib):
            inner = np.nonzero((lane.s > lo + 1e-6) & (lane.s < hi - 1e-6))[0][::step]
            ends = lane.poly.interp(np.array([lo, hi]))
            seg = np.concatenate([ends[:1], lane.pts[inner], ends[1:]])
            cv = np.concatenate([np.interp([lo], lane.s, lane.curv), lane.curv[inner], np.interp([hi], lane.s, lane.curv)])
            if pts_l:
                seg, cv = seg[1:], cv[1:]
            pts_l.append(seg)
            curv_l.append(cv)
        for sl in (lane.stop_s,):
            if sl is not None and lo - 1e-6 <= sl <= hi:
                stops.append((base + sl - lo, lane.id))
        base += hi - lo
        if sib:
            B = town.lanes[lanes[k + 1]]
            u0 = min(1.0, max(0.0, (lo - a0) / lc))
            if u0 < 1.0 - 1e-6:                              # (else the change is already over: no blend)
                m = max(2, int(math.ceil((1.0 - u0) * lc / (0.5 * step))))
                u = np.linspace(u0, 1.0, m + 1)
                sa = a0 + u * lc
                w = blend_weight(u)[:, None]
                seg = (1.0 - w) * lane.poly.interp(sa) + w * B.poly.interp(link_map(lane, B, sa))
                cv = curvature_of(seg)
                d = np.hypot(*np.diff(seg, axis=0).T)
                pts_l.append(seg[1:] if pts_l else seg)      # (from mid-change: the blend is the start)
                curv_l.append(cv[1:] if len(curv_l) else cv)
                base += float(d.sum())
            lo = link_map(lane, B, a0 + lc)
        else:
            lo = 0.0
        k += 1
    pts = np.concatenate(pts_l, 0)
    curv = np.concatenate(curv_l, 0)
    d = np.diff(pts, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
    m = cum <= dist + 1.0
    m[:2] = True
    return pts[m], cum[m], curv[m], stops, base >= dist


def change_pose(A, B, a0, lc, s):
    """Position and heading of a car changing from lane A to its sibling B, at A-local `s` in [a0, a0 + lc]."""
    out = []
    for si in (s, s + 0.3):
        u = min(1.0, max(0.0, (si - a0) / lc))
        w = blend_weight(u)
        pa, pb = A.poly.interp1(min(si, A.length)), B.poly.interp1(min(link_map(A, B, si), B.length))
        out.append(((1.0 - w) * pa[0] + w * pb[0], (1.0 - w) * pa[1] + w * pb[1]))
    (x0, y0), (x1, y1) = out
    return x0, y0, math.atan2(y1 - y0, x1 - x0)


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
LC_GIVE_UP = 8.0            # s: an NPC waiting this long at the end of its lane change window stays in its lane (two
                            # cars side by side that want each other's lane would wait for ever)
PASS_SPEED = 3.0            # m/s: passing someone who walks in the 路側帯 beside the lane (road types)
LOOKBACK_LANE = 60.0        # m: min_arrival's look into the junction before a short approach lane (yield_lookback)
STOP_SIGN_NPC_AT = 1.0      # m: 止まれ: NPCs stop with the front bumper this far before the line (as the expert)
STOP_SIGN_NPC_ZONE = 3.0    # m: ... and the stop counts when they stand within this much of it
NPC_OVER = 3.0              # m: TrafficConfig.light_past_line: a standing NPC this far past its stop line still obeys it
# crosswalks at signals (TrafficConfig.crosswalks, Town.ped_crossings)
CW_REACH = 4.0              # s: a car's stretch of a zebra is taken while someone on it (or setting off) gets there
#                             sooner than this
CW_PAST = 0.5               # m: ... and until they are this far past it
CW_MARGIN = 1.0             # m: a car giving way stops with its front bumper this far before the zebra
CW_LATE = 0.15              # chance that someone waiting still sets off in the flashing green (and hurries)
CW_SPEED_MAX = 1.9          # m/s: the fastest walk over a zebra (to make it before the light changes)
CW_GIVE_UP = 150.0          # s: someone who has waited this long walks on instead
CW_FOCUS = 2.0              # the zebras at the exits of the ego's turns get 1 + this times as many corner people
CW_REACT = (0.3, 3.0)       # s: someone waiting steps off this long into the green (each their own)
CW_CORNER_WALK = (3.0, 45.0)   # m: a corner person walks this far on from the zebra before coming back to it
CW_HELD = (1.5, 1.8, 1.0)   # someone waiting at a zebra does not step off while a car moving faster than [2] m/s comes
#                             within [1] m of their way over it within [0] s (or is there): it could not stop for them.
CW_HELD_REACH = 4.5         # m: ... a car inside the zebra's junction circle or up to this far out of it (on the zebra):
#                             not the cars braking for their red light on the way to it
BIKE_MIN = 0.5              # m: bicycles (TrafficConfig.bicycles): the least room beside one a car passes it with
#                             (with less it follows the bicycle)
BIKE_FREE = 0.95            # m: with this much room (道路交通法 18 条 3 項: about 1 m) it passes at its own speed ...
BIKE_PASS_V = 20.0 / 3.6    # m/s: ... with less, no faster than this
BIKE_PASS_DV = 2.5          # m/s: ... or this much faster than the bicycle
BIKE_KEEP = 0.25            # m: a car moving over to the right for a bicycle keeps this far inside its own lane
BIKE_IN = (22.0, 9.0)       # m: the move over begins / is complete with the bicycle this far ahead of the bumper
BIKE_OUT = (2.0, 8.0)       # m: ... and the move back with it this far behind the rear bumper
BIKE_RATE = 0.8             # m/s sideways (at most 0.15 of the speed, and none while it crawls)
BIKE_ACC = 1.2              # m/s^2 sideways
BIKE_BACK = 0.15            # m sideways per metre driven: back towards the lane's centre
BIKE_END = (3.0, 13.0)      # m before the end of the lane: no moving over from here / the full move up to here
BIKE_LEFT = 25.0            # m before a left turn: 巻き込み確認, the car lets a bicycle beside or just behind it go first
BIKE_LEFT_T = 2.5           # s: "just behind": the bicycle would be beside the car within this time
BIKE_LEFT_WAIT = 6.0        # s: a bicycle that keeps standing beside the car is not waited for longer than this
CW_STALL = 3.0              # s: someone on a zebra held up this long by a car standing in front of them walks back to
#                             their kerb (and waits there again): a car stuck on the zebra and the people it waits for
#                             would otherwise stand for ever


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
        self.ss_done = {}       # car -> lane whose 止まれ line it has stopped at (Town.from_graph)
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
        self.has_lc = any(getattr(l, "links", None) for l in town.lanes)    # multi-lane roads, ramps: lane changes
        self.lcp = {}           # car -> its coming lane change {from, to, a0, lc, a_min, a_max} (A-local, route.py)
        self.bikes = None       # the Bicycles (World, TrafficConfig.bicycles > 0)
        self.shift = np.zeros(0)        # m: how far each car has moved over to the right for a bicycle (_bike_gaps)
        self.shift_v = np.zeros(0)      # m/s: ... and how fast it is moving sideways
        self.shift_tgt = np.zeros(0)    # m: where the shift was heading at the last step (the corridor follows it)
        self._left_wait = {}            # car -> s it has waited before a left turn for a standing bicycle

    # --------------------------------------------------------------- spawning
    def _sample_dims(self):
        """(length, width, height), kind (1: truck). Towns with max_vehicle_dims (keisim.realworld) cap the length
        and width after the draw (the random stream is the same)."""
        r = self.rng.random()
        if r < 0.62:
            return np.array([self.rng.uniform(4.0, 4.8), self.rng.uniform(1.68, 1.85), self.rng.uniform(1.4, 1.6)]), 0
        if r < 0.78:
            return np.array([self.rng.uniform(3.35, 3.45), 1.48, self.rng.uniform(1.6, 1.8)]), 0  # kei car
        if r < 0.92:
            return np.array([self.rng.uniform(4.6, 5.0), self.rng.uniform(1.8, 1.95), self.rng.uniform(1.75, 2.0)]), 0
        dims = np.array([self.rng.uniform(6.0, 7.6), self.rng.uniform(2.0, 2.25), self.rng.uniform(2.6, 3.0)])
        cap = getattr(self.town, "max_vehicle_dims", None)
        if cap is not None:
            dims[:2] = np.minimum(dims[:2], cap)
        return dims, 1

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
        tl = self.town.truck_lanes
        while kind == 1 and tl and lid not in tl:                           # no trucks in alleys
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
            succ = last.succ
            if self.kind[i] == 1 and last.kind == "road" and town.truck_lanes:      # trucks keep out of alleys
                succ = [c for c in succ if town.lanes[c].succ[0] in town.truck_lanes] or succ
            moves = []
            if self.has_lc and last.links and not (len(r) > 1 and r[-2] in last.links):
                moves = lane_moves(town, last, self.s[i] + LC_START_MARGIN if len(r) - 1 == k else 0.0,
                                   town.truck_lanes if self.kind[i] == 1 else None)
            if any(src != last.id or town.lanes[seq[-1]].kind == "road" for _, src, seq in moves):
                # multi-lane road / ramp: the movement first (from any of its lanes), then the lane (as random_route)
                src, seq = pick_move(rng, moves, last.id)
                r.extend(seq)                   # (a lane change: the two lanes count as one road in `rem`)
                rem += sum(town.lanes[x].length for x in seq if x != src)
            else:
                nxt = succ[rng.integers(len(succ))]
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
            if self.kind[i] == 1 and town.truck_lanes and lid not in town.truck_lanes:
                continue
            others = np.delete(self.xy, i, 0)
            if len(others) and np.min(np.hypot(*(others - p).T)) < 12.0:
                continue
            if self.bikes is not None and self.bikes.n and np.min(np.hypot(*(self.bikes.xy - p).T)) < 8.0:
                continue
            self.route[i] = [lid]
            self.ri[i] = 0
            self.s[i] = s
            self.v[i] = 0.0
            self.xy[i] = p
            self.yaw[i] = float(lane.poly.heading(s))
            self.stuck[i] = 0.0
            self.ss_done.pop(i, None)
            self.lcp.pop(i, None)
            if len(self.shift) == self.n:
                self.shift[i] = self.shift_v[i] = 0.0
            if len(self.shift_tgt) == self.n:
                self.shift_tgt[i] = 0.0
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
        change_at = None
        if self.has_lc:
            p = self._lc_plan(i)
            if p is not None:
                change_at = {p["from"]: (p["a0"], p["lc"])}
            elif self.ri[i] + 1 >= len(self.route[i]):              # a change too late was dropped
                self._extend(i, self.CACHE_LEN + 5.0)
        pts, cum, curv, stops, _ = lane_lookahead(self.town, self.route[i], self.ri[i], self.s[i], self.CACHE_LEN,
                                                  change_at=change_at)
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

    # ------------------------------------------------------------------- lane changes (multi-lane roads)
    def _lc_plan(self, i):
        """Car i's coming lane change (route.py) if it is on the lane it changes from; made when first needed, with
        the window from where the car is. A change too late for the window is dropped (the route then goes on from
        the current lane: None, with the route cut after it)."""
        r, k = self.route[i], self.ri[i]
        if k + 1 >= len(r):
            return None
        A = self.town.lanes[r[k]]
        if r[k + 1] not in A.links:
            return None
        p = self.lcp.get(i)
        if p is not None and p["from"] == A.id and p["to"] == r[k + 1]:
            return p
        win = change_window(A, float(self.s[i]) + 2.0, self.town.lanes[r[k + 1]])
        if win is None:
            del r[k + 1:]
            self.lcp.pop(i, None)
            return None
        p = {"from": A.id, "to": r[k + 1], "a0": win[0], "lc": win[2], "a_min": win[0], "a_max": win[1],
             "fresh": True}
        self.lcp[i] = p
        return p

    def cars_by_lane(self):
        """lane id -> indices of the NPCs on it."""
        out = {}
        for j in range(self.n):
            out.setdefault(self.route[j][self.ri[j]], []).append(j)
        return out

    def lane_cars(self, lane_id, by_lane=None, exclude=None, ego=None):
        """(s, v, length) of the cars in lane `lane_id`, s along it: the NPCs on it, on the connectors into it (s < 0:
        just out of the junction behind), changing into it (projected onto it), and the ego (xy, v, yaw, length) if it
        is in it (projected: in it, changing into it or about to enter it)."""
        lanes = self.town.lanes
        L = lanes[lane_id]
        by_lane = self.cars_by_lane() if by_lane is None else by_lane
        out = [(float(self.s[j]), float(self.v[j]), float(self.dims[j, 0])) for j in by_lane.get(lane_id, ())
               if j != exclude]
        for c in L.pred:
            cl = lanes[c].length
            out += [(float(self.s[j]) - cl, float(self.v[j]), float(self.dims[j, 0])) for j in by_lane.get(c, ())
                    if j != exclude]
        out += [(sb, float(self.v[j]), float(self.dims[j, 0])) for j, sb in self.changing_into(lane_id) if j != exclude]
        if ego is not None:
            exy, ev, eyaw, elen = ego
            sb, lat = L.poly.project(np.asarray(exy, float))
            if abs(lat) < L.width / 2 + 0.3 and sb < L.length and math.cos(eyaw - L.poly.heading1(sb)) > 0.5:
                out.append((sb, float(ev), float(elen)))
        return out

    def _lane_change_gaps(self, stop_gap, ego, ego_wait=None, dt=0.05):
        """NPCs about to change lanes need room in the target lane (route.change_room; the ego counts too). Without
        it the change starts later (LC_PUSH, while the window lasts) and the car falls in behind the car in the way
        ahead of it (a virtual lead for IDM: returned bumper gaps and speeds); at the window's end it waits.
        The cars in the target lane leave it a gap there (ファスナー合流): the ones coming up behind a car that waits at
        the end of its window (an NPC, or the ego: ego_wait = (lane, s along it, length)) stop behind it. An NPC that
        has waited there LC_GIVE_UP s, or came to its start too fast to stop without room, stays in its lane (its route
        goes on from there)."""
        n = self.n
        gap, lead_v = np.full(n, np.inf), np.zeros(n)
        lanes = self.town.lanes
        by_lane = None
        waiting = [] if ego_wait is None else [(-1, ego_wait[0], ego_wait[1], ego_wait[2])]
        for i in range(n):
            r, k = self.route[i], self.ri[i]
            if k + 1 >= len(r) or r[k + 1] not in lanes[r[k]].links:
                continue
            p = self._lc_plan(i)
            if p is None or p.pop("fresh", False):
                self._refresh(i)
                if p is None:
                    continue
                r, k = self.route[i], self.ri[i]                 # (the refresh may drop the lanes behind)
            if p.get("go"):
                continue                                         # under way
            d = p["a0"] - self.s[i]
            li, vi = float(self.dims[i, 0]), float(self.v[i])
            dead_end = not lanes[r[k]].succ          # a ramp: it has to merge
            if d <= 0.3 or (p.get("wait", 0.0) > LC_GIVE_UP and not dead_end):
                # at its start without room (or past it: too fast to stop), or waiting there for too long
                if p["a0"] < p["a_max"] and d <= 0.3:
                    p["a0"] = min(p["a_max"], float(self.s[i]) + LC_PUSH_TO)
                elif dead_end:      # past the last start on a ramp: it waits where it is
                    p["a0"] = p["a_max"] = float(self.s[i]) + 0.5
                else:               # it stays in its lane, and its route goes on from there
                    del r[k + 1:]
                    del self.lcp[i]
                self._refresh(i)
                continue
            if d > max(LC_CHECK, vi * vi / 6.0 + 3.0):
                continue                                         # not yet (checked from where it could stop)
            if by_lane is None:
                by_lane = self.cars_by_lane()
            A, B = lanes[p["from"]], lanes[p["to"]]
            s_e = link_map(A, B, float(self.s[i]))
            bad = [(sn - s_e, vn, ln) for sn, vn, ln in self.lane_cars(B.id, by_lane, exclude=i, ego=ego)
                   if not change_room(sn - s_e, vn, ln, vi, li)]
            if not bad:
                if d < 1.5:
                    p["go"] = True                               # room right at its start: it changes
                continue
            if d < LC_PUSH and p["a0"] < p["a_max"]:
                p["a0"] = min(p["a_max"], float(self.s[i]) + LC_PUSH_TO)
                self._refresh(i)
            for dx, vn, ln in bad:
                g = dx - (li + ln) / 2
                if dx > 0.0 and g < gap[i]:
                    gap[i], lead_v[i] = max(g, 0.05), vn
            if p["a0"] >= p["a_max"] - 1e-6:
                stop_gap[i] = min(stop_gap[i], p["a0"] - float(self.s[i]) + 1.5)    # IDM stops 2.5 m short
                if d < 4.0 and vi < 1.0:
                    waiting.append((i, B.id, s_e, li))
                    p["wait"] = p.get("wait", 0.0) + dt
        for j, lane_id, s_w, l_w in waiting:     # a gap for the car waiting to change in
            if by_lane is None:
                by_lane = self.cars_by_lane()
            B = lanes[lane_id]
            followers = [(i, float(self.s[i])) for i in by_lane.get(B.id, ())]
            for c in B.pred:
                followers += [(i, float(self.s[i]) - lanes[c].length) for i in by_lane.get(c, ())]
            for i, s_i in followers:
                g = s_w - s_i - (self.dims[i, 0] + l_w) / 2
                if g > 1.0 and g < gap[i]:                  # fully behind it (the ones beside it go on)
                    gap[i], lead_v[i] = g, 0.0
        # the cars in the target lane behind a car that has begun to change into it follow it from then on (a
        # stopped car could otherwise speed up and come alongside before the changing car is in its corridor)
        for j, s_b in self.changing_into():
            p = self.lcp[j]
            if by_lane is None:
                by_lane = self.cars_by_lane()
            B = lanes[p["to"]]
            followers = [(i, float(self.s[i])) for i in by_lane.get(B.id, ())]
            for c in B.pred:
                followers += [(i, float(self.s[i]) - lanes[c].length) for i in by_lane.get(c, ())]
            for i, s_i in followers:
                if s_i < s_b:
                    g = s_b - s_i - (self.dims[i, 0] + self.dims[j, 0]) / 2
                    if g < gap[i]:
                        gap[i], lead_v[i] = max(g, 0.05), float(self.v[j])
        return gap, lead_v

    def changing_into(self, lane_id=None):
        """[(car, s)]: the cars changing lanes right now (into `lane_id` only, if given), s projected onto the lane
        they change into."""
        lanes = self.town.lanes
        out = []
        for j, p in self.lcp.items():
            if (lane_id is None or p["to"] == lane_id) and p.get("go") and self.route[j][self.ri[j]] == p["from"] and \
                    self.s[j] >= p["a0"]:
                sb, _ = lanes[p["to"]].poly.project(self.xy[j])
                out.append((j, sb))
        return out

    def _lane_end(self, i, k, p):
        """Where car i leaves its route's lane k: the end of a lane change into the next lane, else the lane's end."""
        r = self.route[i]
        if p is not None and p.get("go") and r[k] == p["from"] and k + 1 < len(r) and r[k + 1] == p["to"]:
            return p["a0"] + p["lc"]
        return self.town.lanes[r[k]].length

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
    def yield_tables(self, ego_info=None, visible=None, bikes_visible=None):
        """Who is where, for must_yield: lane -> [(s, v, length, is_ego, next lane, the lane after it, waits)], front
        first. ego_info = (lane, s on lane, next lane, v, length[, the lane after the next[, waits]]) adds the ego;
        waits: it has stood still for TrafficConfig.ego_patience_s (World.step), see min_arrival.
        visible: (n,) bool, only these NPCs are listed (see visible_from).
        With bicycles: ("bike", lane) -> their rows (Bicycles.add_to_tables; bikes_visible: only these of them)."""
        on_lane = {}
        for j in range(self.n):
            if visible is not None and not visible[j]:
                continue
            r, k = self.route[j], self.ri[j]
            on_lane.setdefault(r[k], []).append((float(self.s[j]), float(self.v[j]), float(self.dims[j, 0]), False,
                                                 r[k + 1] if k + 1 < len(r) else None,
                                                 r[k + 2] if k + 2 < len(r) else None, False))
        if ego_info is not None:
            lid, s, nxt, v, L, *more = ego_info
            on_lane.setdefault(lid, []).append((s, v, L, True, nxt, more[0] if more else None,
                                                len(more) > 1 and bool(more[1])))
        if self.bikes is not None:
            self.bikes.add_to_tables(on_lane, bikes_visible)
        for lst in on_lane.values():
            lst.sort(key=lambda e: -e[0])
        return on_lane

    def visible_from(self, cam, yaw, half_fov, max_dist, town=None):
        """NPCs a camera at `cam` (x, y) looking along `yaw` can see -> bool (n,): a corner of the car within
        +-half_fov (rad) of its axis and max_dist of it, and (given `town`) the line to that corner not through a
        building (Town.building_edges, 2D: cars do not hide one another, nor walls and trees)."""
        if self.n == 0:
            return np.zeros(0, bool)
        cam = np.asarray(cam, float)
        cs = box_corners(self.xy[:, 0], self.xy[:, 1], self.yaw, self.dims[:, 0], self.dims[:, 1])    # (n, 4, 2)
        rel = cs - cam
        seen = (np.abs(wrap(np.arctan2(rel[..., 1], rel[..., 0]) - yaw)) < half_fov) & \
            (np.hypot(*(self.xy - cam).T) < max_dist)[:, None]
        if town is not None and seen.any() and len(town.buildings):
            edges, centre, rad = town.building_edges()
            i, k = np.nonzero(seen)
            far = float(np.hypot(*(cs[i, k] - cam).T).max())
            near = np.hypot(*(centre - cam).T) < far + rad + 0.5
            ok = rays_clear(cam, cs[i, k], edges[np.repeat(near, 4)])
            seen[:] = False
            seen[i[ok], k[ok]] = True
        return seen.any(axis=1)

    def must_yield(self, conn, t, on_lane):
        """True if a car with priority over connector `conn` (Lane) is in its crossing or reaches it within
        YIELD_GAP seconds (see min_arrival)."""
        return self.min_arrival(conn, t, on_lane) < YIELD_GAP

    def min_arrival(self, conn, t, on_lane, moving_only=False, yields=None):
        """Earliest time [s] a car with priority over connector `conn` can be in its crossing (0: already in it;
        inf: none coming). Cars that will stop at a red or yellow light do not count, nor do cars queued behind
        one that is standing still to go elsewhere (e.g. an oncoming car waiting to turn right itself).
        moving_only: standing cars (and the ones queued behind them) do not count either: for a turn that is already
        past its wait point, a standing car with priority is waiting for it.
        yields: only these of conn.yields (default: all of them).
        A waiting ego (yield_tables) never counts before the crossing, nor do the cars queued behind it: the cars that
        give way to it wait for it only TrafficConfig.ego_patience_s.
        Towns with yield_lookback (Town.from_graph) at unsignalised junctions: when the approach lane before the
        crossing is shorter than LOOKBACK_LANE and holds nobody bound for it, a car still inside the junction before
        (on one of the approach lane's predecessors) whose route then takes the crossing counts too. Real blocks are
        short (25 m in Sapporo): a car with priority 3 s away was often still in the previous junction, so the minor
        road did not see it until it was too close to stop, and drove into it (one expert collision in 60 routes)."""
        town = self.town
        lanes = town.lanes
        best = math.inf
        lookback = town.yield_lookback and not town.junctions[conn.junction].signalized
        for b, _, s2 in (conn.yields if yields is None else yields):
            vmax = 1.05 * lanes[b].speed_limit
            for s, v, L, ego, *_, waits in on_lane.get(b, ()):
                d = s2 - (s + L / 2)
                if waits and d >= 3.0:
                    break                                       # the ego waits, and the cars behind it queue behind it
                if s - L / 2 > s2 + 1.5 or (moving_only and v < 0.5):
                    continue                                    # its rear is past the crossing (or it waits)
                best = min(best, 0.0 if d < 3.0 else self._arrival(d, v, vmax, ego))
            if self.bikes is not None:
                best = min(best, self._bike_arrival(b, s2, t, on_lane, moving_only))
            pred = lanes[b].pred[0]
            # a car standing at the entry of one of pred's other connectors blocks the whole approach lane
            if any(e[1] < 0.5 and e[0] < 4.0 for c in lanes[pred].succ if c != b for e in on_lane.get(c, ())):
                continue
            stop_s = lanes[pred].stop_s
            st = town.signal_state_for_lane(pred, t)
            seen = False                                        # someone on pred decides for the cars behind
            for s, v, L, ego, nxt, *_, waits in on_lane.get(pred, ()):    # front first
                if nxt != b:
                    if v < 0.5:
                        seen = True
                        break                                   # standing to go elsewhere: the rest queue behind
                    continue
                seen = True
                if (moving_only and v < 0.5) or waits:
                    break                                       # waits, and the cars behind queue behind it
                front = s + L / 2
                if not (st != TL_GREEN and stop_s is not None and self._will_stop(st, stop_s - front, v, ego)):
                    best = min(best, self._arrival(lanes[pred].length - front + s2, v, vmax, ego))
                break                                           # the cars behind it arrive later
            if lookback and not seen and lanes[pred].length < LOOKBACK_LANE:
                for c in lanes[pred].pred:                      # inside the junction before pred
                    for s, v, L, ego, nxt, nxt2, waits in on_lane.get(c, ()):
                        if nxt != pred or nxt2 != b or (moving_only and v < 0.5) or waits:
                            continue
                        front = s + L / 2
                        best = min(best, self._arrival(lanes[c].length - front + lanes[pred].length + s2, v, vmax,
                                                       ego))
                        break                                   # the cars behind it on c arrive later
        return best

    def _bike_arrival(self, b, s2, t, on_lane, moving_only):
        """min_arrival for the bicycles riding beside lane `b` (they have the right of way of the cars on it) and
        for the ones on the lane before it that go on along b: they ride past a queue of cars, so nothing ahead of
        them on the lane holds them up; one that will stop at its red or yellow light does not count."""
        lanes = self.town.lanes
        best = math.inf
        for s, v, L, cruise, _ in on_lane.get(("bike", b), ()):
            if s - L / 2 > s2 + 2.5 or (moving_only and v < 0.5):
                continue                                        # past the crossing (or it waits)
            d = s2 - (s + L / 2)
            best = min(best, 0.0 if d < 3.0 else self._reach(d, v, cruise, 1.0))
        pred = lanes[b].pred[0]
        on = on_lane.get(("bike", pred))
        if on:
            stop_s = lanes[pred].stop_s
            st = self.town.signal_state_for_lane(pred, t)
            for s, v, L, cruise, nxt in on:
                if nxt != b or (moving_only and v < 0.5):
                    continue
                front = s + L / 2
                if st != TL_GREEN and stop_s is not None and stop_s - front > -0.5 and \
                        (st == TL_RED or stop_s - front > v * v / 5.0 + 0.5):
                    continue                                    # it stops at its light (Bicycles.step)
                best = min(best, (0.0 if v >= 0.2 else 0.8) + self._reach(lanes[pred].length - front + s2, v, cruise,
                                                                           1.0))
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

    def _stop_sign_gaps(self, stop_gap, hl):
        """止まれ: a car on a minor road's approach to a priority crossing (Lane.stop_sign_s), or on the lane before
        it, stops once with its front bumper about STOP_SIGN_NPC_AT before the line (the line is an obstacle
        idm_s0 - STOP_SIGN_NPC_AT beyond it), like the expert; standing within STOP_SIGN_NPC_ZONE of the line counts.
        Then _yield_gaps takes it on to its connector's wait point. Without the stop, cars of the minor road reached
        the junction mouth at speed, where a car leaving its left turn into that road swings out (one expert
        collision in 60 Sapporo routes)."""
        lanes = self.town.lanes
        s0 = self.cfg.idm_s0
        for i in range(self.n):
            r, k = self.route[i], self.ri[i]
            front = self.s[i] + hl[i]
            base = 0.0
            for kk in (k, k + 1):
                if kk >= len(r):
                    break
                ln = lanes[r[kk]]
                if ln.stop_sign_s is not None:
                    d = base + ln.stop_sign_s - front
                    if self.ss_done.get(i) == ln.id or d < -0.5 or d > 45.0:
                        break
                    if self.v[i] < 0.05 and d <= STOP_SIGN_NPC_ZONE:
                        self.ss_done[i] = ln.id
                        break
                    stop_gap[i] = min(stop_gap[i], d + s0 - STOP_SIGN_NPC_AT)
                    break
                base += ln.length

    def _crosswalk_gaps(self, stop_gap, hl, peds, t):
        """Crosswalks at signals (TrafficConfig.crosswalks), as the expert: a car whose route crosses a zebra within
        40 m stops with its front bumper CW_MARGIN before it while Pedestrians.crosswalk_blocked holds for its lane's
        stretch; a turn that waits at its connector's wait point holds there instead while the zebra of its exit is
        taken within CW_REACH s plus the drive on to it. Not past the zebra's start, nor when it could not stop at
        6 m/s^2 any more (the corridor check still stops for someone in the way)."""
        if not peds.crosswalk_people(t):
            return                                   # nobody on a zebra or setting off
        lanes, lc = self.town.lanes, self.town.lane_crossings
        s0 = self.cfg.idm_s0
        for i in range(self.n):
            r, k = self.route[i], self.ri[i]
            v = float(self.v[i])
            base = -(self.s[i] + hl[i])                # front bumper -> start of lane r[kk]
            for kk in range(k, min(len(r), k + 4)):
                lid = r[kk]
                for ci, a0, _, u0, u1 in lc.get(lid, ()):
                    d = base + a0
                    if d < -0.5 or d > 40.0:
                        continue
                    gap = x = None
                    conn = lanes[r[kk - 1]] if kk > k else None
                    if conn is not None and conn.kind == "conn" and conn.yields and conn.wait_s is not None:
                        d_wait = base - conn.length + conn.wait_s
                        if d_wait >= -0.5 and peds.crosswalk_blocked(ci, u0, u1, t, CW_REACH + (d - d_wait) / 3.0):
                            gap, x = max(d_wait, 0.0), d_wait       # (as _yield_gaps: the wait point is the obstacle)
                    if gap is None and peds.crosswalk_blocked(ci, u0, u1, t):
                        gap, x = d + s0 - CW_MARGIN, d - CW_MARGIN
                    if gap is not None and not (v > 1.0 and x < v * v / 12.0):
                        stop_gap[i] = min(stop_gap[i], gap)
                base += lanes[lid].length
                if base > 40.0:
                    break

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
             ego_info=None, ped_margin=None, pass_xy=None, ego_len=4.4, ego_wait=None, peds=None):
        """ext_*: external obstacles (ego circles with owner -1, pedestrians owner -2-k). ped_margin: optional lateral
        margin per pedestrian (default 0.9 m); pass_xy: pedestrians walking in a 路側帯, passed at PASS_SPEED.
        peds: the Pedestrians (crosswalks at signals: _crosswalk_gaps)."""
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
        if town.speed_varies:                       # road types: each lane's own limit
            vmax = np.minimum(self._speed_limits() * self.vf, env.min(1))
        else:
            vmax = np.minimum(town.cfg.speed_limit * self.vf, env.min(1))
        # traffic lights: first stop line ahead
        hl = self.dims[:, 0] / 2
        d_stop = self.c_stop - off[:, None] - hl[:, None]
        stop_gap = np.full(n, np.inf)
        ahead = d_stop >= -0.5
        if cfg.light_past_line:     # standing a little over the line: it still binds (TrafficConfig.light_past_line)
            ahead |= (d_stop >= -NPC_OVER) & (v[:, None] < 1.0)
        has = ahead.any(1)
        first = np.argmax(ahead, 1)
        occ = box = None
        for i in np.nonzero(has)[0]:
            d = d_stop[i, first[i]]
            if d > 45.0:
                continue
            lid = int(self.c_stop_lane[i, first[i]])
            st = town.signal_state_for_lane(lid, t)
            if st == TL_RED or (st == TL_YELLOW and (d > v[i] * v[i] / (2 * 3.5) + 1.0 or d < -0.5)):
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
        if town.stop_signs:          # 止まれ (Town.from_graph): stop once at the minor road's stop line
            self._stop_sign_gaps(stop_gap, hl)
        if self.has_yields:          # two-phase signals: turns wait for a gap in the oncoming traffic
            self._yield_gaps(stop_gap, t, ego_info, hl)
        if peds is not None and peds.cw_on:      # crosswalks at signals: give way to the people on them
            self._crosswalk_gaps(stop_gap, hl, peds, t)
        bike_pair = ()
        if self.bikes is not None and self.bikes.n:     # bicycles: pass them with room to spare, or follow them
            g_b, v_b, vmax = self._bike_gaps(stop_gap, paths, relc, valid, hl, vmax, dt)
            bike_pair = ((g_b, v_b),)
        lc_pair = ()
        if self.has_lc:              # multi-lane roads: room for lane changes
            ego = None if ego_yaw is None else (ego_xy, ego_v, ego_yaw, ego_len)
            lc_pair = (self._lane_change_gaps(stop_gap, ego, ego_wait, dt),)
        if self.bikes is not None and len(self.shift) == n and (self.shift > 1e-3).any():
            paths = self._shifted_paths(paths, relc)        # where the cars that moved over for a bicycle will drive
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
        margin = np.where(cow < -1, 0.9, 0.35)
        if ped_margin is not None:
            pm = cow < -1
            margin[pm] = ped_margin[-2 - cow[pm]]
        margin = margin[None, :]
        gap, lead_v, _ = corridor_gaps(paths, relc, valid, self.xy, self.yaw, hl, self.dims[:, 1] / 2,
                                       cxy, cr, cow, csp, cyw, np.arange(n), margin)
        if pass_xy is not None and len(pass_xy):    # slow down to pass people walking beside the lane
            m = len(pass_xy)
            g_pass, _, _ = corridor_gaps(paths, relc, valid, self.xy, self.yaw, hl, self.dims[:, 1] / 2, pass_xy,
                                         np.full(m, 0.35), -2 - np.arange(m), np.zeros(m), np.zeros(m), np.arange(n),
                                         0.9)
            vmax = np.minimum(vmax, np.sqrt(PASS_SPEED ** 2 + 3.0 * np.maximum(g_pass, 0.0)))   # inf: nobody there
        # IDM
        a_max, b, T, s0 = cfg.idm_a, cfg.idm_b, cfg.idm_T, cfg.idm_s0
        v0 = np.maximum(vmax, 0.5)
        acc = a_max * (1.0 - (v / v0) ** 4)
        for g, vl in ((gap, lead_v), (stop_gap, np.zeros(n))) + lc_pair + bike_pair:
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
            p = self.lcp.get(i) if self.has_lc else None
            L = lanes[r[k]].length if p is None else self._lane_end(i, k, p)
            while self.s[i] > L:
                if k + 1 >= len(r):
                    self._extend(i, 10.0)
                    r = self.route[i]
                    k = self.ri[i]
                    if k + 1 >= len(r):
                        self.s[i] = L
                        break
                if p is not None and r[k] == p["from"] and r[k + 1] == p["to"]:   # lane change done
                    self.s[i] = link_map(lanes[r[k]], lanes[r[k + 1]], L) + (self.s[i] - L)
                    del self.lcp[i]
                    p = None
                else:
                    self.s[i] -= L
                k += 1
                self.ri[i] = k
                L = lanes[r[k]].length if p is None else self._lane_end(i, k, p)
            si = float(self.s[i])
            if p is not None and p.get("go") and r[k] == p["from"] and si >= p["a0"]:      # changing lanes
                self.xy[i, 0], self.xy[i, 1], self.yaw[i] = change_pose(lanes[p["from"]], lanes[p["to"]], p["a0"],
                                                                        p["lc"], si)
            else:
                poly = lanes[r[k]].poly
                self.xy[i, 0], self.xy[i, 1] = poly.interp1(si)
                self.yaw[i] = poly.heading1(si)
        if self.bikes is not None and len(self.shift) == n and (self.shift > 1e-4).any():
            # the cars that have moved over to the right for a bicycle
            # (the heading turns about the rear axle, 0.3 of the length behind the centre: the tail does not swing
            # out towards the bicycle)
            dyaw = np.arctan2(self.shift_v, np.maximum(self.v, 2.0))
            side = self.shift + 0.3 * self.dims[:, 0] * np.sin(dyaw)
            self.xy[:, 0] += np.sin(self.yaw) * side
            self.xy[:, 1] -= np.cos(self.yaw) * side
            self.yaw = self.yaw - dyaw
        self.stuck = np.where(self.v < 0.1, self.stuck + dt, 0.0)
        self.release_stuck(ego_xy, ego_yaw)

    def _bike_gaps(self, stop_gap, paths, relc, valid, hl, vmax, dt):
        """Bicycles (TrafficConfig.bicycles) -> (gap, the bicycle's speed) to follow, and vmax with the passing speed.
        A bicycle rides beside the kerb lane, a car's width or less from the lane's centre line. A car coming up
        behind one on its lane moves over to the right inside its lane (it keeps BIKE_KEEP from the lane's edge:
        NPCs do not cross the centre line), by as much as it takes to leave BIKE_FREE beside the bicycle, from
        BIKE_IN ahead of it to BIKE_OUT past it. With BIKE_FREE of room it passes at its own speed; with less, at
        BIKE_PASS_V (or BIKE_PASS_DV faster than the bicycle); with less than BIKE_MIN it stays behind the bicycle
        (narrow lanes, wide cars, a bicycle that has moved out round someone walking, and just before a junction,
        where it does not move over). Bicycles elsewhere (crossing, in a junction) are obstacles like cars, with
        BIKE_MIN beside them, and so is where a moving one will be within 2.4 s (Bicycles.circles).
        Left turns (巻き込み確認): within BIKE_LEFT of its turn a car stops at the end of its lane while a bicycle on
        that lane is beside it or would be within BIKE_LEFT_T s, and lets it go first.
        stop_gap is updated in place; self.shift is what World applies to the cars' positions."""
        bk = self.bikes
        n, nb = self.n, bk.n
        lanes = self.town.lanes
        if len(self.shift) != n:
            self.shift, self.shift_v = np.zeros(n), np.zeros(n)
        hw = self.dims[:, 1] / 2
        bxy, br, bow, bsp, byw = bk.circles(predict=True)
        by_lane = {}
        for j in range(nb):
            by_lane.setdefault(int(bk.lane[j]), []).append(j)
        room = np.zeros((n, nb))            # how far car i can move over where bicycle j is
        target = np.zeros(n)
        half_b = 0.9                        # half a bicycle's length
        hl = np.asarray(hl, float)
        for i in range(n):
            r, k = self.route[i], self.ri[i]
            js = by_lane.get(r[k])
            if not js:
                self._left_wait.pop(i, None)
                continue
            lane = lanes[r[k]]
            if lane.kind != "road":
                continue
            si = float(self.s[i])
            changing = self.has_lc and i in self.lcp
            cap = 0.0 if changing else max(0.0, lane.width / 2 - hw[i] - BIKE_KEEP)
            for j in js:
                sj, vb = float(bk.lane_s[j]), float(bk.v[j])
                ahead = sj - si - hl[i] - half_b            # the car's front bumper to the bicycle's rear wheel
                behind = si - sj - hl[i] - half_b           # its rear bumper to the bicycle's front wheel
                # where the bicycle will be when the car is past it: no passing into the junction
                s_done = sj
                if behind < BIKE_OUT[0]:
                    s_done += vb * (BIKE_OUT[0] - behind) / max(max(self.v[i], vb + BIKE_PASS_DV) - vb, 1.0)
                c = cap * min(max((lane.length - s_done - BIKE_END[0]) / (BIKE_END[1] - BIKE_END[0]), 0.0), 1.0)
                room[i, j] = c
                want = min(max(hw[i] + 0.3 + BIKE_FREE + 0.05 - bk.side[j], 0.0), c)
                if ahead > 0.0:
                    f = (BIKE_IN[0] - ahead) / (BIKE_IN[0] - BIKE_IN[1])
                elif behind > 0.0:
                    f = (BIKE_OUT[1] - behind) / (BIKE_OUT[1] - BIKE_OUT[0])
                else:
                    f = 1.0
                target[i] = max(target[i], want * min(max(f, 0.0), 1.0))
            # 巻き込み確認 before a left turn
            held = False
            if k + 1 < len(r) and lanes[r[k + 1]].kind == "conn" and lanes[r[k + 1]].turn == "left" \
                    and lane.length - si < BIKE_LEFT + hl[i]:
                d = (lane.stop_s if lane.stop_s is not None else lane.length) - si - hl[i]
                if d > -0.5:
                    for j in js:
                        ds, vb = float(bk.lane_s[j]) - si, float(bk.v[j])
                        if not -(hl[i] + half_b + BIKE_LEFT_T * vb + 1.0) < ds < hl[i] + half_b + 1.0:
                            continue
                        if vb < 0.3 and (ds < -(hl[i] + half_b) or self._left_wait.get(i, 0.0) > BIKE_LEFT_WAIT):
                            continue                        # standing behind the car (or it stands there for good)
                        if vb < 0.3 and self.v[i] < 0.3:
                            self._left_wait[i] = self._left_wait.get(i, 0.0) + dt
                        stop_gap[i] = min(stop_gap[i], d - 0.6)
                        held = True
                        break
            if not held and self.v[i] > 1.0:
                self._left_wait.pop(i, None)
        # beside a bicycle (whichever lane it is on by now, e.g. in the junction): the car stays where it is
        if self.shift.any():
            u = np.stack([np.cos(self.yaw), np.sin(self.yaw)], -1)
            rel = bk.xy[None] - self.xy[:, None]
            lon = rel[..., 0] * u[:, None, 0] + rel[..., 1] * u[:, None, 1]
            lat = rel[..., 1] * u[:, None, 0] - rel[..., 0] * u[:, None, 1]
            reach = (hl + half_b)[:, None]
            near = (lon > -(reach + BIKE_OUT[0])) & (lon < reach + 2.0) & (lat > 0.0) & (lat < hw[:, None] + 2.2)
            room = np.where(near, np.maximum(room, self.shift[:, None]), room)
            target = np.where(near.any(1), np.maximum(target, self.shift), target)
        # sideways: towards the target at BIKE_RATE at most, gently (BIKE_ACC); back to the lane's centre from the
        # first metre (a car that stood moved over steers back as it starts: BIKE_BACK per metre driven)
        rate = np.minimum(BIKE_RATE, 0.15 * np.maximum(self.v - 0.5, 0.0))
        back = np.minimum(BIKE_RATE, BIKE_BACK * self.v)
        v_t = np.clip((target - self.shift) * 1.6, -back, rate)
        self.shift_tgt = target
        self.shift_v = self.shift_v + np.clip(v_t - self.shift_v, -BIKE_ACC * dt, BIKE_ACC * dt)
        self.shift_v = np.where((self.v < 0.3) | ((target <= 0.0) & (self.shift < 0.01)), 0.0, self.shift_v)
        self.shift = np.maximum(self.shift + self.shift_v * dt, 0.0)
        me = np.full(n, -1)
        less = room[:, bow]
        hwid = self.dims[:, 1] / 2
        gap, lead_v, _ = corridor_gaps(paths, relc, valid, self.xy, self.yaw, hl, hwid, bxy, br, bow, bsp, byw, me,
                                       BIKE_MIN - less)
        g_free, v_free, _ = corridor_gaps(paths, relc, valid, self.xy, self.yaw, hl, hwid, bxy, br, bow, bsp, byw, me,
                                          BIKE_FREE - less)
        m = np.isfinite(g_free)
        if m.any():
            v_pass = np.maximum(BIKE_PASS_V, v_free[m] + BIKE_PASS_DV)
            vmax = vmax.copy()
            vmax[m] = np.minimum(vmax[m], np.sqrt(v_pass ** 2 + 3.0 * np.maximum(g_free[m], 0.0)))
        return gap, lead_v, vmax

    def _shifted_paths(self, paths, relc):
        """The paths (N,P,2) of step(), moved to the right by where each car's shift (_bike_gaps) will be at each
        point: all of it while the car is moving over or held beside a bicycle; else what is left of it on the way
        back to the lane's centre (a little less than BIKE_BACK per metre, and no faster than the 1.6 /s it steers
        back with). The lane's own line would hide what a car that has moved over is about to drive into, e.g. the
        car waiting to turn right in the middle of the junction."""
        sh = self.shift[:, None]
        tg = self.shift_tgt[:, None] if len(self.shift_tgt) == self.n else np.zeros_like(sh)
        d = np.maximum(relc, 0.0)
        left = np.maximum(sh - (BIKE_BACK - 0.05) * d, sh * np.exp(-1.6 * d / np.maximum(self.v, 1.0)[:, None]))
        o = np.where(tg >= sh - 1e-3, np.maximum(sh, tg), np.maximum(left, tg))
        dp = np.gradient(paths, axis=1)
        ln = np.hypot(dp[..., 0], dp[..., 1])
        ok = ln > 1e-6
        ux = np.where(ok, dp[..., 0] / np.maximum(ln, 1e-6), np.cos(self.yaw)[:, None])
        uy = np.where(ok, dp[..., 1] / np.maximum(ln, 1e-6), np.sin(self.yaw)[:, None])
        return paths + np.stack([uy, -ux], -1) * o[..., None]

    def _speed_limits(self):
        """Speed limit of each NPC's lane, eased down at 1.5 m/s^2 toward lower limits up to 40 m ahead (front
        bumper), so that it enters a slower road at that road's limit."""
        lanes = self.town.lanes
        out = np.empty(self.n)
        for i in range(self.n):
            r, k = self.route[i], self.ri[i]
            v = lanes[r[k]].speed_limit
            d = lanes[r[k]].length - self.s[i] - self.dims[i, 0] / 2
            for j in range(k + 1, len(r)):
                if d > 40.0:
                    break
                if r[j] in lanes[r[j - 1]].siblings:
                    continue                            # a lane change: the same road (and limit)
                vj = lanes[r[j]].speed_limit
                if vj < v:
                    v = min(v, math.sqrt(vj * vj + 3.0 * max(d, 0.0)))
                d += lanes[r[j]].length
            out[i] = v
        return out

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
    WALK, CROSS, WAIT = 0, 1, 2

    def __init__(self, town, cfg: TrafficConfig, rng, n, focus_cw=()):
        """n walkers; with TrafficConfig.crosswalks also the corner people of the zebras at signals (crosswalk_spawn
        each on average, 1 + CW_FOCUS times as many at focus_cw: the zebras at the exits of the ego's turns)."""
        self.town, self.cfg, self.rng = town, cfg, rng
        ww = town.walkways
        if not ww:                      # no sidewalk nor 路側帯 anywhere (keisim/curve_course.py): nobody walks
            n = 0
        pc = town.ped_crossings
        self.cw_on = cfg.crosswalks > 0 and bool(pc) and bool(ww)
        corner_cw = []                  # the zebra of each corner person (random draws only with crosswalks on)
        if self.cw_on:
            focus = set(focus_cw)
            for ci in range(len(pc)):
                corner_cw += [ci] * int(rng.poisson(cfg.crosswalk_spawn * (1.0 + CW_FOCUS * (ci in focus))))
            n += len(corner_cw)
        self.n = n
        lens = np.array([w["poly"].length for w in ww])
        if ww:
            self.w = rng.choice(len(ww), size=n, p=lens / lens.sum())
        else:
            self.w = np.zeros(0, int)
        self.s = np.array([rng.uniform(1.0, ww[k]["poly"].length - 1.0) for k in self.w])
        self.dir = rng.choice([-1.0, 1.0], size=n)
        self.speed = rng.uniform(1.0, 1.6, n)
        self.lat_scale = np.array([w.get("lat", 1.0) for w in ww])   # 路側帯 walkways: a narrower spread
        self.in_road = np.array([bool(w.get("in_road", False)) for w in ww])
        self.ww_len = lens
        self.lat = rng.uniform(-0.6, 0.6, n) * self.lat_scale[self.w]
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
                if i != j and a["road"] == b["road"] and a.get("side") != b.get("side"):
                    self.partner[i] = j
        # crosswalks at signals (TrafficConfig.crosswalks): cw = the zebra someone waits at or crosses on (-1: none),
        # cw_from = the kerb they cross from (0: pts[0]), cw_late = they set off in the flashing green too.
        # The corner people (the last len(corner_cw)) stay round their zebra: over it, they walk CW_CORNER_WALK m on
        # (cw_back: the walkway s where they turn) and come back to cross again; walkers cross on a zebra they come
        # to with the chance TrafficConfig.crosswalks. Random draws only with crosswalks on: without, every episode
        # is as before.
        self.cw = np.full(n, -1)
        self.cw_from = np.zeros(n, int)
        self.cw_wait = np.zeros(n)
        self.cw_late = np.zeros(n, bool)
        self.cw_react = np.zeros(n)             # s into the green before they step off
        self.corner = np.zeros(n, bool)
        self.cw_back = np.full(n, np.nan)
        self.cw_stall = np.zeros(n)             # s held up on the zebra by a standing car (CW_STALL)
        self.cw_retreat = np.zeros(n, bool)     # walking back to the kerb they set off from
        self.cw_skip = np.zeros(n, bool)        # back at it: they let this green go (until their light is red)
        self.cw_held = np.zeros(n, bool)        # a car is about to cross their way (CW_HELD): they wait for it
        self.ww_cw = [[] for _ in ww]          # walkway -> [(zebra, kerb, s of the zebra on it)]
        for ci, c in enumerate(pc):
            for side in (0, 1):
                self.ww_cw[c["ww"][side]].append((ci, side, c["ww_s"][side]))
        self._cw_t = None                      # crosswalk_people's cache (the time it is for)
        for j, ci in enumerate(corner_cw):
            i = n - len(corner_cw) + j
            self.corner[i] = True
            self._start_wait(i, ci, int(rng.integers(2)), there=True)
        self._update_pose()

    def _start_wait(self, i, ci, side, there=False):
        """Person i waits at zebra ci's kerb `side`: spread along the zebra and a little back from the kerb point. They
        walk to that spot (c1; from the walkway in a 路側帯 it is just off the road), or are there already (there)."""
        c, rng = self.town.ped_crossings[ci], self.rng
        d = c["dir"] if side == 0 else -c["dir"]           # towards the road
        along = np.array([-d[1], d[0]])
        self.mode[i], self.cw[i], self.cw_from[i], self.cw_wait[i] = self.WAIT, ci, side, 0.0
        self.cw_skip[i] = False
        self.cw_late[i] = rng.random() < CW_LATE
        self.cw_react[i] = rng.uniform(*CW_REACT)
        self.w[i], self.s[i] = c["ww"][side], c["ww_s"][side]
        self.c1[i] = c["pts"][side] + along * rng.uniform(-1.1, 1.1) - d * rng.uniform(0.0, 0.5)
        if there:
            self.xy[i] = self.c1[i]
            self.yaw[i] = math.atan2(d[1], d[0])
        self.vel[i] = 0.0

    def _start_crossing(self, i, speed):
        """Person i, waiting at a zebra, walks over it to the other kerb, then on along that walkway away from the
        junction."""
        c, rng = self.town.ped_crossings[self.cw[i]], self.rng
        other = 1 - int(self.cw_from[i])
        j = c["ww"][other]
        poly = self.town.walkways[j]["poly"]
        s2 = float(np.clip(c["ww_s"][other] + rng.uniform(-1.1, 1.1), 1.0, poly.length - 1.0))
        p2 = np.array(poly.interp1(s2))
        h2 = poly.heading1(s2)
        lat2 = rng.uniform(-0.5, 0.5) * self.lat_scale[j]
        self.mode[i] = self.CROSS
        self.c0[i] = self.xy[i]
        self.c1[i] = p2 + np.array([-math.sin(h2), math.cos(h2)]) * lat2
        self.w[i], self.s[i], self.lat[i], self.dir[i] = j, s2, lat2, c["away"][other]
        self.speed[i] = speed
        self.cw_stall[i], self.cw_retreat[i] = 0.0, False
        if self.corner[i]:
            back = s2 + c["away"][other] * rng.uniform(*CW_CORNER_WALK)
            self.cw_back[i] = float(np.clip(back, 1.5, poly.length - 1.5))

    def crosswalk_people(self, t):
        """zebra -> (u, sgn, speed) arrays of the people on it (u from its pts[0]; sgn +1 walking towards pts[1]) and
        of those waiting at a kerb while they may set off (green; flashing for the late ones), at u 0 / width.
        Cached for time t."""
        if self._cw_t == t:
            return self._cw_cache
        out = {}
        if self.cw_on:
            pc, J = self.town.ped_crossings, self.town.junctions
            flash = self.cfg.ped_flash
            for i in np.nonzero(self.cw >= 0)[0]:
                ci = int(self.cw[i])
                c = pc[ci]
                sgn = 1.0 if self.cw_from[i] == 0 else -1.0
                if self.mode[i] == self.CROSS:
                    u, v = float((self.xy[i] - c["pts"][0]) @ c["dir"]), float(self.speed[i])
                else:
                    v = self._set_off_speed(i, c, t, flash)
                    if v is None:
                        continue
                    u = 0.0 if sgn > 0 else c["width"]
                out.setdefault(ci, []).append((u, sgn, max(v, 1.0)))
        self._cw_t, self._cw_cache = t, {k: np.array(v) for k, v in out.items()}
        return self._cw_cache

    def crosswalk_blocked(self, ci, u0, u1, t, horizon=CW_REACH):
        """Is the stretch [u0, u1] of zebra ci (a lane's car, Town.lane_crossings) walked on now, or within `horizon` s,
        by someone on the zebra or setting off from its kerb who has not yet passed it (by CW_PAST)?"""
        a = self.crosswalk_people(t).get(ci)
        if a is None:
            return False
        u, sgn, v = a[:, 0], a[:, 1], a[:, 2]
        fwd = (sgn > 0) & (u <= u1 + CW_PAST) & (u0 - u <= horizon * v)
        back = (sgn < 0) & (u >= u0 - CW_PAST) & (u - u1 <= horizon * v)
        return bool(np.any(fwd | back))

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

    def step(self, dt, veh_xy, veh_v, veh_yaw, trigger=None, t=None):
        """trigger: optional boolean mask of pedestrians that should try to cross now. t: the time (the pedestrian
        lights at the zebras, TrafficConfig.crosswalks)."""
        ww = self.town.walkways
        rng = self.rng
        self.cooldown -= dt
        self.phase += self.speed * dt * 4.2 * (self.vel > 0.1)
        if self.cw_on:
            self._update_held(veh_xy, veh_v, veh_yaw)
        for i in range(self.n):
            if self.mode[i] == self.WAIT:
                self._wait_step(i, dt, t)
                continue
            if self.mode[i] == self.WALK:
                poly = ww[self.w[i]]["poly"]
                s_prev = float(self.s[i])
                self.s[i] += self.dir[i] * self.speed[i] * dt
                if self.cw_on:
                    b = self.cw_back[i]
                    if b == b and (s_prev - b) * (self.s[i] - b) <= 0.0:      # a corner person turns back
                        self.dir[i] *= -1
                        self.cw_back[i] = np.nan
                    if self.ww_cw[self.w[i]] and self._reach_zebra(i, s_prev):
                        continue
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
                        lat2 = rng.uniform(-0.5, 0.5) * self.lat_scale[j]
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
                    if self.cw_retreat[i]:          # back at their kerb: wait there again
                        self._back_at_kerb(i)
                        continue
                    self.mode[i] = self.WALK
                    if self.cw[i] >= 0:            # over a zebra: no crossing in the middle of the road for a while
                        self.cw[i] = -1
                        self.cooldown[i] = rng.uniform(15, 40)
                    continue
                # do not walk into a vehicle standing right in front
                if len(veh_xy):
                    rel = veh_xy - self.xy[i]
                    fwd = (rel @ d) / L
                    side = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]) / L
                    if np.any((fwd > 0.0) & (fwd < 3.2) & (side < 2.6)):
                        self.vel[i] = 0.0
                        if self.cw[i] >= 0 and not self.cw_retreat[i]:
                            self.cw_stall[i] += dt
                            if self.cw_stall[i] > CW_STALL:
                                self._retreat(i)
                        continue
                stepl = min(L, self.speed[i] * dt)
                self.xy[i] = self.xy[i] + d / L * stepl
                self.yaw[i] = math.atan2(d[1], d[0])
                self.vel[i] = self.speed[i]
        # refresh walking poses
        self._update_pose()

    def _update_held(self, veh_xy, veh_v, veh_yaw):
        """cw_held of the people waiting at a zebra: a car faster than CW_HELD[2] m/s, in the zebra's junction or on the
        zebra (CW_HELD_REACH), whose straight-line position within CW_HELD[0] s comes within CW_HELD[1] m of their way
        over the zebra (kerb to kerb)."""
        self.cw_held[:] = False
        wait = np.nonzero((self.mode == self.WAIT) & (self.cw >= 0))[0]
        if not len(wait) or not len(veh_xy):
            return
        T, R, V = CW_HELD
        fast = veh_v > V
        if not fast.any():
            return
        fxy = veh_xy[fast]
        u = np.stack([np.cos(veh_yaw[fast]), np.sin(veh_yaw[fast])], -1) * veh_v[fast, None]
        taus = np.linspace(0.0, T, 4)
        pts = (fxy[:, None, :] + u[:, None, :] * taus[None, :, None])           # (cars, taus, 2)
        for i in wait:
            c = self.town.ped_crossings[self.cw[i]]
            J = self.town.junctions[c["junction"]]
            inside = np.hypot(*(fxy - J.pos).T) < J.radius + CW_HELD_REACH
            if not inside.any():
                continue
            p = pts[inside].reshape(-1, 2)
            a, b = c["pts"][0], c["pts"][1]
            ab = b - a
            q = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
            dist = np.hypot(*(p - (a + q[:, None] * ab)).T)
            self.cw_held[i] = bool((dist < R).any())

    def _retreat(self, i):
        """Person i, held up on a zebra by a standing car (CW_STALL), walks back to where they set off from."""
        start = self.c0[i].copy()
        self.c0[i] = self.xy[i]
        self.c1[i] = start
        self.cw_from[i] = 1 - self.cw_from[i]           # now walking towards that kerb (Pedestrians.crosswalk_people)
        self.cw_retreat[i] = True

    def _back_at_kerb(self, i):
        """Person i walked back off a zebra (_retreat): they wait at that kerb again."""
        c = self.town.ped_crossings[self.cw[i]]
        side = 1 - int(self.cw_from[i])
        d = c["dir"] if side == 0 else -c["dir"]
        self.mode[i], self.cw_from[i], self.cw_wait[i] = self.WAIT, side, 0.0
        self.cw_retreat[i], self.cw_stall[i], self.cw_skip[i] = False, 0.0, True
        self.w[i], self.s[i] = c["ww"][side], c["ww_s"][side]
        self.yaw[i] = math.atan2(d[1], d[0])
        self.vel[i] = 0.0

    def _reach_zebra(self, i, s_prev):
        """A walker heading for a junction end of their walkway just passed its zebra: with TrafficConfig.crosswalks
        chance they stop to wait there (True)."""
        s = float(self.s[i])
        here = [(ci, side) for ci, side, s_c in self.ww_cw[self.w[i]]
                if self.dir[i] == -self.town.ped_crossings[ci]["away"][side] and (s_prev - s_c) * (s - s_c) <= 0.0]
        if not here:
            return False
        if self.corner[i] or self.rng.random() < self.cfg.crosswalks:
            ci, side = here[int(self.rng.integers(len(here)))] if len(here) > 1 else here[0]
            self._start_wait(i, ci, side)       # (a scramble's corner: the zebra or the diagonal)
            return True
        return False

    def _set_off_speed(self, i, c, t, flash, react=False):
        """Whether person i, waiting at zebra c, sets off now (react: also past their reaction time; without, about
        to): at the green when there is time to get over before the cars crossing the road get theirs, walking at
        least the speed that takes (up to CW_SPEED_MAX); the late ones also in the flashing green, hurrying. Not
        after stepping back (cw_skip) until their light has turned red, nor while a car is about to cross their way
        (cw_held). Returns that least speed (CW_SPEED_MAX for
        the hurrying late ones), or None."""
        J = self.town.junctions[c["junction"]]
        st, left = J.ped_signal(c["phase"], t, flash)
        if self.cw_skip[i]:
            if st not in (PED_GREEN, PED_FLASH):
                self.cw_skip[i] = False
            return None
        if self.cw_held[i]:
            return None
        if st == PED_GREEN:
            if react and J.green + J.yellow + J.allred - left < self.cw_react[i]:
                return None
            need = c["width"] / max(left - 1.0, 0.1)
            return max(need, 1.0) if need <= CW_SPEED_MAX else None
        if st == PED_FLASH and self.cw_late[i]:
            return CW_SPEED_MAX
        return None

    def _wait_step(self, i, dt, t):
        """Waiting at a zebra: step off once _set_off_speed says so (at their own pace if it is fast enough). First
        they walk to their spot at the kerb (_start_wait)."""
        self.vel[i] = 0.0
        self.cw_wait[i] += dt
        c = self.town.ped_crossings[self.cw[i]]
        to = self.c1[i] - self.xy[i]
        gap = math.hypot(to[0], to[1])
        if gap > 0.02:
            stepl = min(gap, self.speed[i] * dt)
            self.xy[i] = self.xy[i] + to / gap * stepl
            if gap > 0.1:
                self.yaw[i] = math.atan2(to[1], to[0])
            self.vel[i] = self.speed[i]
            if gap - stepl <= 0.02:                      # there: face the road
                d = c["dir"] if self.cw_from[i] == 0 else -c["dir"]
                self.yaw[i] = math.atan2(d[1], d[0])
            return
        if t is not None:
            v = self._set_off_speed(i, c, t, self.cfg.ped_flash, react=True)
            if v is not None:
                hurry = v >= CW_SPEED_MAX                       # (the late ones in the flashing green)
                pace = self.rng.uniform(1.6, CW_SPEED_MAX) if hurry else self.rng.uniform(1.1, 1.6)
                self._start_crossing(i, float(np.clip(max(pace, v), 1.0, CW_SPEED_MAX)))
                return
        if self.cw_wait[i] > CW_GIVE_UP:           # (a light that never lets them over): walk on
            self.mode[i], self.cw[i] = self.WALK, -1
            self.dir[i] = c["away"][int(self.cw_from[i])]

    def velocity_vectors(self):
        return np.stack([np.cos(self.yaw), np.sin(self.yaw)], -1) * self.vel[:, None]

    def waiting_at_kerb(self):
        """People standing at a zebra's kerb for their light (TrafficConfig.crosswalks): off the carriageway, beside the
        cars turning in and out of the road; when they set off, the crosswalk rule takes over (crosswalk_blocked).
        Passed as closely as the people walking in a 路側帯."""
        return self.mode == self.WAIT

    def walking_in_road(self):
        """Pedestrians walking along a 路側帯 (on the road surface, beside the lane; not crossing), away from the ends of
        their walkway: at a junction mouth turning cars sweep across the 路側帯, so there they count like anyone else."""
        return (self.mode == self.WALK) & self.in_road[self.w] & (self.s > 6.0) & (self.s < self.ww_len[self.w] - 6.0)
