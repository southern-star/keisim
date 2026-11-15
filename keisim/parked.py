"""Parked cars (路上駐車) at the kerb of two-way roads, and children darting out from in front of them (飛び出し).

TrafficConfig.parked: parked cars per km of kerb (each side of a road counts), on two-way roads with one lane each
way (streets, residential roads, alleys; not the main roads), away from the ends of the lane (the junctions and
their zebras), from curves, from the other side's parked cars (the road stays passable from one side at a time)
and from the ego's start. 0: none, and no random draw is made for them (every town and evaluation up to here stays
as it was). They stand still with their lights off. Left-hand traffic only.

A car that comes up behind one moves over to the right round it (Traffic._parked_gaps, Expert.parked_scan): inside
its lane when there is room for PK_CLEAR beside the parked car, else over the centre line, which it does only when
the oncoming lane is clear for the whole pass (the side whose lane is blocked gives way); otherwise it waits PK_WAIT
behind the parked car. The expert passes them at PK_PASS_V at most (someone may step out from in front of them).

TrafficConfig.dart_out: the chance that a parked car the ego's route passes hides a child in front of it, who runs
out across the road when the ego comes within DART_TRIGGER of them (World.step). The expert sees them only once the
parked car no longer hides them from its camera (Expert: Pedestrians.HIDE / dart_seen).
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import box_corners, rays_clear

PK_EDGE = (0.2, 0.45)       # m: from the road surface's edge to the parked car's kerb side
PK_FROM = (30.0, 30.0)      # m: no parked car this close to its lane's start (a car waiting behind it leaves room to
                            # come out of the junction) / end (junctions, zebras, stop lines) ...
PK_FROM_JUNCTION = 60.0     # m: ... and this close to a junction of 3+ arms (through bends): the queues at it would
                            # stand beside the parked car (in the oncoming lane before it, behind it after it)
PK_EXIT = 20.0              # m: a car passes parked cars only with nobody in its lane up to PK_OUT + this past them
                            # (it would stand in the oncoming lane, unable to pull back in)
PK_CURVE = 1.0 / 60.0       # 1/m: no parked car where the lane bends more than this, within PK_CURVE_REACH of it
PK_CURVE_REACH = 15.0
PK_SPACING = 45.0           # m: between two parked cars on one kerb (bumper to bumper), but for rows: a car that has
                            # passed one pulls back in (PK_OUT) and has room to wait (PK_WAIT) before the next one
PK_ROW = 0.3                # the chance that a parked car has another one right behind it (PK_ROW_GAP apart)
PK_ROW_GAP = (0.8, 2.5)
PK_GROUP_GAP = 6.0          # m: parked cars of one kerb closer than this (bumper to bumper) are passed in one go
PK_OPPOSITE = 70.0          # m: none on the other side of the road this close (centre to centre, along the road):
                            # the two passing stretches, and the cars waiting before each, stay apart
PK_EGO = 30.0               # m: none this close to the ego's start
PK_FOCUS = 3.0              # the ego route's lanes are 1 + this times as likely to get one (per metre)
PK_CLEAR = 0.5              # m: a car passing a parked car leaves this much room beside it (less where the road is too
                            # narrow for that: at most it drives fully in the oncoming lane, 0.1 m from its far edge)
PK_PASSABLE = 0.3           # m: a parked car stands only where a wide car (half width PK_WIDE_HW) passes it with this
PK_WIDE_HW = 0.95
PK_MARGIN = 0.15            # m: the corridor's margin to parked cars (corridor_gaps): passing them, cars come closer
PK_BIKE_CLEAR = 0.4         # m: a bicycle moves out round a parked car (keisim/bicycles.py) with this much room
PK_KEEP = 0.15              # m: ... and keeps this far inside its own lane when that is enough (no oncoming check)
PK_IN = (16.0, 5.0)         # m before the parked car's rear: the move out begins / is complete
PK_OUT = (1.0, 8.0)         # m past its front bumper: the move back begins / is complete
PK_DECIDE = 22.0            # m (front bumper to the parked car's rear): from here on a car decides to pass or wait
PK_WAIT = 5.0               # m: waiting for the oncoming lane to clear, this far behind the parked car
PK_GAP0 = 2.0               # m: the standstill gap to a parked car the path still runs into (not passing yet)
PK_ONCOMING_T = 2.5         # s: the oncoming lane is clear if nobody gets to the passing stretch before the passing
                            # car is through it, plus this, ...
PK_YIELD_B = 2.5            # m/s^2: ... or everyone coming can stop short of it (they give way to a car out there)
PK_YIELD_D = 4.0            # m: ... with this to spare
PK_ZONE = (18.0, 10.0)      # m: the passing stretch: from this far behind the parked car to this far past it (the
                            # whole move out and back, PK_IN and PK_OUT: the oncoming cars must not be anywhere on it)
PK_PASS_V = 20.0 / 3.6      # m/s: the expert passes a parked car no faster than this, from PK_SLOW_AT before its
PK_SLOW_AT = 8.0            # m ... rear to a metre past its front (a child may run out from in front of it)
PK_SLOW_B = 1.6             # m/s^2: ... slowing down to it
PK_SWING_V = 3.0            # m/s: a car that has not moved out far enough yet comes up to the parked car no faster
PK_RATE = 1.2               # m/s sideways at most (and 0.5 of the speed: a car turns out round a parked car more
PK_ACC = 2.0                # sharply than it moves over for a bicycle), m/s^2 sideways
PK_BACK = 0.25              # m sideways per metre driven: back into the lane past it (0.25 of the speed)
PK_GIVE_ROOM = 6.0          # m: an oncoming car giving way to a car out round parked cars stops this far short of
                            # the passing stretch (room for it to pull back in)
PARKED_OWNER = 2_000_000    # owner ids of their circles (corridor_gaps): this + the parked car's index
DART_TRIGGER = (10.0, 16.0)  # m: a hidden child runs out when the ego's front bumper is this far before their line
DART_SPEED = (2.2, 3.0)     # m/s: running
DART_HEIGHT = (1.0, 1.35)   # m
DART_AHEAD = 0.7            # m in front of the parked car's front bumper, ...
DART_IN = 0.3               # m ... and this far from its kerb side towards its middle
KID_SHIRTS = np.array([[40, 60, 230], [230, 120, 30], [40, 200, 240], [200, 60, 200], [60, 200, 90],
                       [240, 240, 240]], float)    # BGR: red, blue, yellow, pink, green, white


def parked_lanes(town):
    """The lanes parked cars may stand beside: road lanes of two-way roads with one lane each way, not main roads,
    no centre strip, long enough. Empty with right-hand traffic."""
    if not town.cfg.left_hand_traffic:
        return []
    out = []
    for lid in town.road_lanes:
        lane = town.lanes[lid]
        if lane.kind != "road" or lane.road < 0:
            continue
        rs = town.roads[lane.road].spec
        if rs.oneway or rs.lanes != 1 or rs.kind == "main" or rs.median > 0.0:
            continue
        a, b = parked_span(town, lane)
        if b - a < 8.0:
            continue
        out.append(lid)
    return out


def _to_junction(town, lane, ahead, limit=PK_FROM_JUNCTION):
    """Metres from `lane`'s end (ahead) or start (behind) to the next junction of 3+ arms that way, through bends
    (a bend does not stop a queue); `limit` or more: limit."""
    J, lanes = town.junctions, town.lanes
    d, L = 0.0, lane
    for _ in range(8):
        jn = J[L.end_junction if ahead else L.start_junction]
        if len(jn.arms) != 2:
            return d if len(jn.arms) >= 3 else limit
        nxt = L.succ if ahead else L.pred                  # the bend's connector, then the lane beyond it
        if len(nxt) != 1:
            return d
        C = lanes[nxt[0]]
        nn = C.succ if ahead else C.pred
        if len(nn) != 1:
            return d
        L = lanes[nn[0]]
        d += C.length + L.length
        if d >= limit:
            return limit
    return d


def parked_span(town, lane):
    """(from, to): where on `lane` parked cars may stand: PK_FROM from its ends, and PK_FROM_JUNCTION from the
    nearest junction of 3+ arms each way (through bends); behind, from the start of the passing stretch (PK_ZONE):
    the queue in the oncoming lane at that junction would stand on it, the cars waiting to pass would queue back
    into the junction, and the two lock it up."""
    a = max(PK_FROM[0], PK_FROM_JUNCTION + PK_ZONE[0] - _to_junction(town, lane, False, PK_FROM_JUNCTION + PK_ZONE[0]))
    b = max(PK_FROM[1], PK_FROM_JUNCTION - _to_junction(town, lane, True))
    return a, lane.length - b


def opposite_lane(town, lane):
    """The lane of the other direction of a two-way road with one lane each way (-1 if none)."""
    road = town.roads[lane.road]
    if lane.id in road.lanes_ab:
        return road.lanes_ba[0] if road.lanes_ba else -1
    return road.lanes_ab[0] if road.lanes_ab else -1


def _sample_dims(rng, truck_ok):
    """(length, width, height), kind (1: a small truck, only where trucks may drive)."""
    r = rng.random()
    if r < 0.42:
        return np.array([rng.uniform(3.35, 3.45), 1.48, rng.uniform(1.55, 1.8)]), 0           # kei car
    if r < 0.78:
        return np.array([rng.uniform(4.0, 4.8), rng.uniform(1.68, 1.82), rng.uniform(1.4, 1.6)]), 0
    if r < 0.92 or not truck_ok:
        return np.array([rng.uniform(4.6, 5.0), rng.uniform(1.75, 1.9), rng.uniform(1.75, 2.0)]), 0
    return np.array([rng.uniform(4.7, 6.0), rng.uniform(1.85, 2.0), rng.uniform(2.2, 2.6)]), 1   # delivery truck


class ParkedCars:
    """n parked cars: lane (the lane they stand beside), s (their centre's arc length along it), side (their
    centre's offset from the lane's centre line, + to the right: negative, towards the kerb), xy, yaw, dims (L, W,
    H), kind (0 car, 1 truck), color (BGR), opp (the other direction's lane). by_lane: lane -> indices by s."""

    def __init__(self, town, cfg, rng, n, avoid_xy=None, route=None):
        self.town = town
        self.lane = np.zeros(0, int)
        self.s = np.zeros(0)
        self.side = np.zeros(0)
        self.xy = np.zeros((0, 2))
        self.yaw = np.zeros(0)
        self.dims = np.zeros((0, 3))
        self.kind = np.zeros(0, int)
        self.color = np.zeros((0, 3))
        self.opp = np.zeros(0, int)
        self.n = 0
        cand = parked_lanes(town)
        if not cand or n <= 0:
            self._index()
            return
        on_route = set(route.lanes) if route is not None else set()
        w = np.array([town.lanes[l].length * (1.0 + PK_FOCUS * (l in on_route)) for l in cand])
        w = w / w.sum()
        trucks = town.truck_lanes
        tries = 0
        while self.n < n and tries < n * 30:
            tries += 1
            lid = cand[rng.choice(len(cand), p=w)]
            lane = town.lanes[lid]
            dims, kind = _sample_dims(rng, not trucks or lid in trucks)
            hl = dims[0] / 2
            a, b = parked_span(town, lane)
            if b - a < 2 * hl:
                continue
            s = rng.uniform(a + hl, b - hl)
            if not self._try_add(lid, s, dims, kind, rng, avoid_xy):
                continue
            if rng.random() < PK_ROW and self.n < n:             # another one right behind it
                dims2, kind2 = _sample_dims(rng, not trucks or lid in trucks)
                s2 = s - hl - rng.uniform(*PK_ROW_GAP) - dims2[0] / 2
                if s2 - dims2[0] / 2 >= a:
                    self._try_add(lid, s2, dims2, kind2, rng, avoid_xy, row=True)
        self._index()

    def _try_add(self, lid, s, dims, kind, rng, avoid_xy, row=False):
        town = self.town
        lane = town.lanes[lid]
        L, W = float(dims[0]), float(dims[1])
        lo, hi = s - L / 2 - PK_CURVE_REACH, s + L / 2 + PK_CURVE_REACH
        m = (lane.s >= lo) & (lane.s <= hi)
        if m.any() and np.abs(lane.curv[m]).max() > PK_CURVE:
            return False
        same = self.lane == lid
        for j in np.nonzero(same)[0]:                            # this kerb: PK_SPACING apart (a row: touching)
            d = abs(self.s[j] - s) - (L + self.dims[j, 0]) / 2
            if d < (0.5 if row else PK_SPACING):
                return False
        rs = town.roads[lane.road].spec
        off_c = rs.half_width - rng.uniform(*PK_EDGE) - W / 2  # the car's centre from the road's centre line
        lane_off = rs.lane_offsets()[0]
        side = -(off_c - lane_off)                              # from the lane's centre, + to the right
        if side + W / 2 + PK_PASSABLE + 2 * PK_WIDE_HW + 0.1 > lane.width:
            return False                 # a wide car could not get by (narrow alleys: wide cars stand closer in)
        p = np.asarray(lane.poly.interp1(s))
        h = float(lane.poly.heading1(s))
        xy = p + np.array([math.sin(h), -math.cos(h)]) * side
        opp = opposite_lane(town, lane)
        if opp >= 0:                                             # the other kerb: not across the road from it
            O = town.lanes[opp]
            so = O.poly.project(xy)[0]
            for j in np.nonzero(self.lane == opp)[0]:
                if abs(self.s[j] - so) < PK_OPPOSITE:
                    return False
        if avoid_xy is not None and len(avoid_xy) and np.min(np.hypot(*(np.asarray(avoid_xy) - xy).T)) < PK_EGO:
            return False
        from .traffic import CAR_COLORS
        col = CAR_COLORS[rng.integers(len(CAR_COLORS))] * rng.uniform(0.9, 1.08) + rng.uniform(-8, 8, 3)
        self.lane = np.append(self.lane, lid)
        self.s = np.append(self.s, s)
        self.side = np.append(self.side, side)
        self.xy = np.vstack([self.xy, xy])
        self.yaw = np.append(self.yaw, h)
        self.dims = np.vstack([self.dims, dims])
        self.kind = np.append(self.kind, kind)
        self.color = np.vstack([self.color, np.clip(col, 0, 255)])
        self.opp = np.append(self.opp, opp)
        self.n += 1
        return True

    def _index(self):
        """by_lane, and the groups: the parked cars of a kerb less than PK_GROUP_GAP apart (rows) are passed in one
        go: {lane, idx, s_lo (the first one's rear), s_hi (the last one's front), q (the inner edge furthest out),
        opp}; lane_groups: lane -> group ids by s."""
        self.by_lane = {}
        for j in np.argsort(self.s, kind="stable"):
            self.by_lane.setdefault(int(self.lane[j]), []).append(int(j))
        self.groups, self.lane_groups = [], {}
        self.group = np.zeros(self.n, int)
        for lid, js in self.by_lane.items():
            for j in js:
                lo, hi = self.s[j] - self.dims[j, 0] / 2, self.s[j] + self.dims[j, 0] / 2
                gs = self.lane_groups.setdefault(lid, [])
                if gs and lo - self.groups[gs[-1]]["s_hi"] < PK_GROUP_GAP:
                    G = self.groups[gs[-1]]
                    G["idx"].append(j)
                    G["s_hi"] = max(G["s_hi"], float(hi))
                    G["q"] = max(G["q"], self.inner_edge(j))
                else:
                    gs.append(len(self.groups))
                    self.groups.append({"lane": lid, "idx": [j], "s_lo": float(lo), "s_hi": float(hi),
                                        "q": self.inner_edge(j), "opp": int(self.opp[j])})
                self.group[j] = gs[-1]
        # the passing stretch (PK_ZONE round the group) on the other direction's lane: (lo, hi) there
        self.opp_groups = {}            # other direction's lane -> groups whose passing stretch is on it
        for g, G in enumerate(self.groups):
            G["opp_zone"] = None
            if G["opp"] < 0:
                continue
            lane, O = self.town.lanes[G["lane"]], self.town.lanes[G["opp"]]
            z0, z1 = max(G["s_lo"] - PK_ZONE[0], 0.0), min(G["s_hi"] + PK_ZONE[1], lane.length)
            a = O.poly.project(np.asarray(lane.poly.interp1(z1)))[0]
            b = O.poly.project(np.asarray(lane.poly.interp1(z0)))[0]
            G["opp_zone"] = (float(min(a, b)), float(max(a, b)))
            self.opp_groups.setdefault(G["opp"], []).append(g)
        self.corners = box_corners(self.xy[:, 0], self.xy[:, 1], self.yaw, self.dims[:, 0], self.dims[:, 1]) \
            if self.n else np.zeros((0, 4, 2))

    # ---------------------------------------------------------------- queries
    def circles(self):
        """(centres, radii, owner): owner PARKED_OWNER + the parked car's index (vehicle_circles)."""
        from .traffic import vehicle_circles
        c, r, o = vehicle_circles(self.xy, self.yaw, self.dims[:, 0], self.dims[:, 1])
        return c, r, o + PARKED_OWNER

    def inner_edge(self, j):
        """How far right of its lane's centre line parked car j's side towards the traffic is."""
        return float(self.side[j] + self.dims[j, 1] / 2)

    def edges(self, idx=None):
        """Their outlines as segments (M,2,2), for rays_clear (who they hide)."""
        c = self.corners if idx is None else self.corners[idx]
        if not len(c):
            return np.zeros((0, 2, 2))
        return np.stack([c, np.roll(c, -1, axis=1)], 2).reshape(-1, 2, 2)

    def hides(self, cam, pts, near=None):
        """Which of `pts` (N,2) the parked cars hide from `cam` (2,) (seen from above: a parked car is taller than a
        child). near: only the parked cars within this many metres of the camera count."""
        if not self.n or not len(pts):
            return np.zeros(len(pts), bool)
        idx = None
        if near is not None:
            idx = np.nonzero(np.hypot(*(self.xy - np.asarray(cam)).T) < near)[0]
            if not len(idx):
                return np.zeros(len(pts), bool)
        return ~rays_clear(cam, pts, self.edges(idx))

    def scene(self):
        """xy, yaw, dims, color, kind, brake (off) for SceneState's vehicle arrays."""
        return self.xy, self.yaw, self.dims, self.color, self.kind, np.zeros(self.n, bool)

    def collisions(self, corners, near_xy, obb_overlap):
        """Indices of the parked cars an oriented box (corners (4,2), centre near_xy) overlaps."""
        if not self.n:
            return []
        d = np.hypot(*(self.xy - near_xy).T)
        idx = np.nonzero(d < 9.0)[0]
        if not len(idx):
            return []
        return [int(j) for j in idx[obb_overlap(corners, self.corners[idx])]]


def dart_spot(town, pk, j):
    """Where a child hides in front of parked car j (xy), and where they run to: (xy, target xy, walkway index,
    s and lat on it), or None where there is no walkway across the road (alleys)."""
    lane = town.lanes[int(pk.lane[j])]
    road = town.roads[lane.road]
    side = 1.0 if town.cfg.left_hand_traffic else -1.0
    sgn_here = side if lane.id in road.lanes_ab else -side
    ww = town.walkways
    here = [k for k, w in enumerate(ww) if w["road"] == road.id and w["side"] == sgn_here]
    there = [k for k, w in enumerate(ww) if w["road"] == road.id and w["side"] == -sgn_here]
    if not here or not there:
        return None
    s_f = float(pk.s[j] + pk.dims[j, 0] / 2 + DART_AHEAD)
    if s_f > lane.length - 3.0:
        return None
    h = float(lane.poly.heading1(s_f))
    lat = float(pk.side[j] - pk.dims[j, 1] / 2 + DART_IN)       # nearer the kerb than the car's middle
    xy = np.asarray(lane.poly.interp1(s_f)) + np.array([math.sin(h), -math.cos(h)]) * lat
    k = there[0]
    poly = ww[k]["poly"]
    s2, _ = poly.project(xy)
    to = np.asarray(poly.interp1(s2))
    return xy, to, k, float(s2), 0.0
