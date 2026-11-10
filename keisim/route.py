"""Ego routes: a lane sequence flattened into one arc-length parametrised path.

Lane changes (Lane.links: the sibling lanes of multi-lane roads, and the ramps of Town._make_ramps): two linked
lanes next to each other in the sequence, [A, B], mean "drive on A, change to B". The path leaves A at A-local `a0`
and blends into B over `lc` metres (a quintic lateral profile: zero lateral speed and acceleration at both ends), then
follows B. The change sits in A's window [a_min, a_max] (siblings: LC_ENTRY after the junction it came from, done
LC_EXIT before the end of the road: 20 m of solid lane line before a signal + the stop line; ramps: where the two
lanes run side by side); the expert may move it later in the window (Route.retime_change) while it waits for a gap in
B. A ramp in the sequence, [A, R, C], is the left turn at its junction (an entry of Route.junctions).
"""
from __future__ import annotations

import math

import numpy as np

from .config import CMD_LEFT, CMD_RIGHT, CMD_STRAIGHT
from .geometry import Polyline, curvature_of

TURN_TO_CMD = {"left": CMD_LEFT, "straight": CMD_STRAIGHT, "right": CMD_RIGHT}

LC_ENTRY = 10.0          # m: a lane change starts at least this far into the lane (after the junction)
LC_EXIT = 25.0           # m: ... and is complete this far before the lane's end (solid lane line + stop line)
LC_LEN = 40.0            # m: length of a lane change (A-local; ~1.6 m/s^2 lateral at 40 km/h)
LC_MIN = 25.0            # m: ... shorter where the window is short; no lane change below this
LC_START_MARGIN = 8.0    # m: on the route's first lane, a change starts this far after the ego's start
LC_FREE_PROB = 0.25      # random_route: chance of a lane change that the next movement does not need
LC_DS = 0.5              # m: point spacing of the blend
# room for a lane change (the expert and the NPCs alike): checked from LC_CHECK m before its start; the cars in the
# target lane need LC_GAP0 m bumper to bumper, plus the follower's speed times LC_T_REAR behind, plus the closing
# speed times LC_T_CLOSE (ahead: the changing car is faster; behind: the follower is). Without room, a start closer
# than LC_PUSH m moves on to LC_PUSH_TO m ahead (while the window lasts).
LC_CHECK = 12.0
LC_PUSH = 10.0
LC_PUSH_TO = 15.0
LC_GAP0 = 2.0
LC_T_REAR = 0.8
LC_T_CLOSE = 1.5


def blend_weight(u):
    """Lateral share of the target lane, u in [0, 1] along a lane change (quintic: no lateral speed or acceleration
    at either end)."""
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def change_room(dx, v_other, len_other, v, length):
    """True if a car in the target lane at dx (along that lane, + ahead) leaves room for a change at speed v."""
    room = (length + len_other) / 2 + LC_GAP0
    return not (-(room + v_other * LC_T_REAR + max(0.0, v_other - v) * LC_T_CLOSE) < dx <
                room + max(0.0, v - v_other) * LC_T_CLOSE)


def link_map(lane_a, lane_b, s):
    """A-local arc length(s) `s` -> lane_b's, for a change from lane_a into lane_b (Lane.links)."""
    lk = lane_a.links.get(lane_b.id)
    if lk is None or lk[3] is None:                 # siblings: the whole lane, scaled
        return s * lane_b.length / lane_a.length
    return lk[2] + (s - lk[0]) * lk[3]


def change_window(lane_a, s_lo=0.0, lane_b=None):
    """(a_min, a_max, lc): where on `lane_a` a change into lane_b can start, and its length; None if there is no
    room for one (from A-local s_lo on). Siblings (or lane_b None): from LC_ENTRY after the lane's start to LC_EXIT
    before its end; a ramp's lane or a right-turn lane: along the stretch where the two run side by side
    (Lane.links). A change into a right-turn lane is LC_MIN long."""
    lk = None if lane_b is None else lane_a.links.get(lane_b.id)
    if lk is None or lk[3] is None:
        lo = max(LC_ENTRY, s_lo)
        hi = lane_a.length - LC_EXIT
    else:
        lo = max(lk[0], s_lo)
        hi = lk[1]
    if hi - lo < LC_MIN:
        return None
    # into a right-turn lane (Town._make_pockets): a short change (slow cars in its queue), so that its start can wait
    lc = min(LC_MIN if lane_b is not None and lane_b.kind == "pocket" else LC_LEN, hi - lo)
    return lo, hi - lc, lc


class Route:
    # with no junction decision ahead, the target point is the route's end; tp_ahead (m) puts it that far ahead along
    # the route instead (the curve courses' training scenes, keisim/curve_course.py: no junction for a kilometre)
    tp_ahead = None

    def __init__(self, town, lanes, s_start, s_end=None, change_at=None):
        """change_at: {index k of the lane changed to: A-local start of the change} (default: the window start)."""
        self.town = town
        self.lanes = list(lanes)
        self.s_start = float(s_start)
        self._change_at = dict(change_at or {})
        self._build()
        self.s_end = float(min(self.poly.length - 1.0, s_end if s_end is not None else self.poly.length - 5.0))

    def _is_change(self, k):
        if k == 0:
            return False
        a, b = self.town.lanes[self.lanes[k - 1]], self.town.lanes[self.lanes[k]]
        return b.id in getattr(a, "links", ())

    def _build(self):
        town = self.town
        n = len(self.lanes)
        lo = np.zeros(n)                       # A-local start / end of each lane's own part of the path
        hi = np.array([town.lanes[lid].length for lid in self.lanes])
        self.changes = []
        for k in range(1, n):
            if not self._is_change(k):
                continue
            A, B = town.lanes[self.lanes[k - 1]], town.lanes[self.lanes[k]]
            win = change_window(A, lo[k - 1] + (self.s_start + LC_START_MARGIN if k - 1 == 0 else 0.0), B)
            if win is None:                    # too short: change over what is left (rare; random_route avoids it)
                a_min = a_max = max(lo[k - 1], min(A.length - LC_MIN, A.length / 2))
                lc = max(1.0, min(LC_MIN, A.length - a_min - 1.0))
            else:
                a_min, a_max, lc = win
            a0 = float(np.clip(self._change_at.get(k, a_min), a_min, a_max))
            hi[k - 1] = a0
            lo[k] = link_map(A, B, a0 + lc)
            self.changes.append({"k": k, "from": A.id, "to": B.id, "a0": a0, "lc": lc, "a_min": a_min,
                                 "a_max": a_max})
        pts, curv, offs, s0 = [], [], [], []
        base = 0.0                             # route s at the end of the points so far
        ch_i = 0
        for k, lid in enumerate(self.lanes):
            lane = town.lanes[lid]
            change_in = ch_i < len(self.changes) and self.changes[ch_i]["k"] == k
            if change_in:                      # the blend from lanes[k - 1] into this lane
                ch = self.changes[ch_i]
                A = town.lanes[self.lanes[k - 1]]
                m = max(2, int(math.ceil(ch["lc"] / LC_DS)))
                u = np.linspace(0.0, 1.0, m + 1)
                sa = ch["a0"] + u * ch["lc"]
                pa = A.poly.interp(sa)
                pb = lane.poly.interp(link_map(A, lane, sa))
                w = blend_weight(u)[:, None]
                seg = (1.0 - w) * pa + w * pb
                kk = curvature_of(seg)
                if len(kk) > 7:
                    kk = np.convolve(np.pad(kk, 3, mode="edge"), np.ones(7) / 7.0, mode="valid")
                d = np.hypot(*np.diff(seg, axis=0).T)
                cum = base + np.concatenate([[0.0], np.cumsum(d)])
                pts.append(seg[1:-1])
                curv.append(kk[1:-1])
                ch["s_a"] = float(base)
                ch["s_b"] = float(cum[-1])
                s_mid = float(cum[m // 2])     # the route's lane switches from A to B half way
                offs.append(s_mid)
                s0.append(float(lo[k] - (cum[-1] - s_mid)))      # exact on B after the blend
                base = float(cum[-1])
                ch_i += 1
            else:
                offs.append(base)
                s0.append(float(lo[k]))
            full = lo[k] == 0.0 and hi[k] == lane.length
            if full:                           # the whole lane (all routes without lane changes)
                first = k == 0 or change_in
                seg, kk = (lane.pts, lane.curv) if first else (lane.pts[1:], lane.curv[1:])
            else:
                inner = (lane.s > lo[k] + 1e-6) & (lane.s < hi[k] - 1e-6)
                ends = lane.poly.interp(np.array([lo[k], hi[k]]))
                seg = np.concatenate([ends[:1], lane.pts[inner], ends[1:]])
                kk = np.concatenate([np.interp([lo[k]], lane.s, lane.curv), lane.curv[inner],
                                     np.interp([hi[k]], lane.s, lane.curv)])
                if k > 0 and not change_in:
                    seg, kk = seg[1:], kk[1:]
            pts.append(seg)
            curv.append(kk)
            base += lane.length if full else float(hi[k] - lo[k])      # a piece of the lane's own polyline
        self.offsets = np.array(offs)
        self.s0 = np.array(s0)
        self.poly = Polyline(np.concatenate(pts, 0))
        self.junctions = []
        self.stops = []
        self.stop_signs = []            # (s, lane id) of the 止まれ stop lines along the route (Town.from_graph)
        self.crossings = []             # (s0, s1, zebra, u0, u1, lane index): the route's stretches over zebras at
        #                                 signals (Town.lane_crossings; TrafficConfig.crosswalks)
        by_k = {c["k"]: c for c in self.changes}
        for k, lid in enumerate(self.lanes):
            lane = town.lanes[lid]
            if lane.kind == "conn":
                J = town.junctions[lane.junction]
                self.junctions.append({"s_in": offs[k], "s_out": offs[k] + lane.length, "junction": J.id,
                                       "turn": lane.turn, "signalized": J.signalized, "lane": lid,
                                       "decision": len(J.arms) >= 3})
            elif lane.kind == "ramp" and k in by_k and k + 1 in by_k:   # a ramp: the turn it makes at its junction
                self.junctions.append({"s_in": by_k[k]["s_a"], "s_out": by_k[k + 1]["s_b"], "junction": lane.junction,
                                       "turn": lane.turn, "signalized": False, "lane": lid, "decision": True})
            if lane.stop_s is not None and lo[k] <= lane.stop_s <= hi[k]:
                self.stops.append((offs[k] + lane.stop_s - s0[k], lid))
            if lane.stop_sign_s is not None and lo[k] <= lane.stop_sign_s <= hi[k]:
                self.stop_signs.append((offs[k] + lane.stop_sign_s - s0[k], lid))
            for ci, a0, a1, u0, u1 in town.lane_crossings.get(lid, ()):
                if lo[k] - 1e-6 <= a0 and a1 <= hi[k] + 1e-6:
                    self.crossings.append((offs[k] + a0 - s0[k], offs[k] + a1 - s0[k], ci, u0, u1, k))
        # curvature along the flattened path
        curv = np.concatenate(curv)
        self.curv = curv[: len(self.poly.pts)] if len(curv) >= len(self.poly.pts) else np.pad(
            curv, (0, len(self.poly.pts) - len(curv)), mode="edge")

    def zebras(self, town):
        """(zebra, kind, turn) of each of self.crossings: kind "exit" (the zebra at the exit of a junction the route
        turns or goes straight at; turn: its connector's), "entry" (the zebra of the road the route comes in by, before
        the junction) or "other"."""
        out = []
        for _, _, ci, _, _, k in self.crossings:
            jz = town.ped_crossings[ci]["junction"]
            prev = town.lanes[self.lanes[k - 1]] if k > 0 else None
            if prev is not None and prev.kind == "conn" and prev.junction == jz:
                out.append((ci, "exit", prev.turn))
            elif town.lanes[self.lanes[k]].end_junction == jz:
                out.append((ci, "entry", None))
            else:
                out.append((ci, "other", None))
        return out

    def retime_change(self, i, a0):
        """Move lane change i to start at A-local `a0` (clipped to its window) and rebuild the path. The path up to
        the earlier of the two starts is unchanged, and so are the route positions of everything before it."""
        ch = self.changes[i]
        k_end = self.lane_index_at(self.s_end)
        loc_end = self.s_end - self.offsets[k_end] + self.s0[k_end]
        self._change_at[ch["k"]] = float(a0)
        self._build()
        self.s_end = float(min(self.poly.length - 1.0, self.offsets[k_end] + loc_end - self.s0[k_end]))

    @property
    def length(self):
        return self.s_end - self.s_start

    def lane_index_at(self, s):
        return int(np.clip(np.searchsorted(self.offsets, s, side="right") - 1, 0, len(self.lanes) - 1))

    def lane_at(self, s):
        return self.lanes[self.lane_index_at(s)]

    def local_s(self, s, k=None):
        """Arc length on the route's lane at route position `s` (or on lane index k)."""
        if k is None:
            k = self.lane_index_at(s)
        return float(s - self.offsets[k] + self.s0[k])

    def next_decision(self, s):
        """Next junction with a real choice (>=3 arms, signalised or not) that has not been exited."""
        for j in self.junctions:
            if j["decision"] and j["s_out"] > s + 0.5:
                return j
        return None

    def command(self, s):
        j = self.next_decision(s)
        if j is None:
            return CMD_STRAIGHT
        return TURN_TO_CMD[j["turn"]]

    def target_point_world(self, s):
        j = self.next_decision(s)
        if j is None:
            return self.poly.interp(self.s_end if self.tp_ahead is None else min(s + self.tp_ahead, self.s_end))
        return self.poly.interp(min(j["s_out"] + 4.0, self.s_end))

    def curvature_at(self, s):
        return np.interp(s, self.poly.s, self.curv)

    def speed_limit_ahead(self, s_front, look=45.0, b=1.6):
        """Speed limit of the lane at `s_front` (front bumper), eased down at b m/s^2 toward lower limits up to
        `look` m ahead, so that the car enters a slower road at that road's limit."""
        lanes = self.town.lanes
        k = self.lane_index_at(s_front)
        v = lanes[self.lanes[k]].speed_limit
        for j in range(k + 1, len(self.lanes)):
            d = self.offsets[j] - s_front
            if d > look:
                break
            vj = lanes[self.lanes[j]].speed_limit
            if vj < v:
                v = min(v, math.sqrt(vj * vj + 2.0 * b * d))
        return v


def lane_moves(town, cur, s_lo=0.0, truck_lanes=None):
    """The movements from road lane `cur` at the end of its road, for a car that may change lanes from cur-local
    s_lo on: [(turn, src, lanes)], src the lane it is made from (cur, or a sibling it changes to first), lanes the
    route's next lanes: [src, if it changes] + [the junction's connector], or + [a ramp, the lane it merges into]
    (Town._make_ramps), or + [a right-turn lane, its connector] (Town._make_pockets). A change only where its window fits (a ramp after a lane change: from where that ends).
    truck_lanes: only connectors into these lanes (unless none is) and ramps into them."""
    lanes = town.lanes
    srcs = [(cur, s_lo)]
    for b in cur.siblings:
        win = change_window(cur, s_lo, lanes[b])
        if win is not None:
            srcs.append((lanes[b], link_map(cur, lanes[b], win[0] + win[2])))
    moves = []
    for src, lo in srcs:
        pre = [] if src is cur else [src.id]
        sc = src.succ
        if truck_lanes:
            sc = [c for c in sc if lanes[c].succ[0] in truck_lanes] or sc
        moves += [(lanes[c].turn, src.id, pre + [c]) for c in sc]
        for rid in src.links:
            R = lanes[rid]
            if R.kind == "pocket" and change_window(src, lo, R) is not None:      # a right-turn lane (右折専用車線)
                moves += [(lanes[c].turn, src.id, pre + [rid, c]) for c in R.succ
                          if not truck_lanes or lanes[c].succ[0] in truck_lanes]
                continue
            if R.kind != "ramp" or change_window(src, lo, R) is None:
                continue
            (cid,) = R.links
            if not truck_lanes or cid in truck_lanes:
                moves.append((R.turn, src.id, pre + [rid, cid]))
    return moves


def pick_move(rng, moves, cur_id, weights=None):
    """The movement first (weights per turn, or uniform), then the lane: the current one if it has it (else a change;
    LC_FREE_PROB of the time a change it does not need). Returns (src, lanes) of lane_moves."""
    turns = [t for t in ("left", "straight", "right") if any(m[0] == t for m in moves)]
    if weights is None:
        t = turns[rng.integers(len(turns))]
    else:
        w = np.array([weights.get(t, 1.0) for t in turns], float)
        t = turns[rng.choice(len(turns), p=w / w.sum())]
    opts = [m for m in moves if m[0] == t]
    own = [m for m in opts if m[1] == cur_id]
    other = [m for m in opts if m[1] != cur_id]
    pick = own if own and (not other or rng.random() >= LC_FREE_PROB) else other
    _, src, seq = pick[rng.integers(len(pick))]
    return src, seq


def random_route(town, rng, length, start_lane=None, s_start=None, turn_weights=None):
    """Random walk through the lane graph with a mild preference for turns. On multi-lane roads the movement at the
    next junction is drawn first (from all lanes of the road), then the lane: the current one if it has that
    movement (else a lane change to one that has; LC_FREE_PROB of the time a change it does not need)."""
    turn_weights = turn_weights or {"left": 1.0, "straight": 1.0, "right": 1.0}
    if start_lane is None:
        cands = [l for l in town.road_lanes if town.lanes[l].length > 30.0]
        start_lane = cands[rng.integers(len(cands))]
    lane = town.lanes[start_lane]
    if s_start is None:
        s_start = float(rng.uniform(3.0, max(3.5, lane.length - 25.0)))
    lanes = [start_lane]
    total = lane.length - s_start
    s_lo = s_start + LC_START_MARGIN                # A-local s from which the current lane may start a change
    total = _walk(town, rng, lanes, total, s_lo, length, turn_weights)
    return _finish(town, lanes, s_start, total, length)


def ramp_route(town, rng, length, ramp=None, turn_weights=None):
    """A random route that takes one of the town's ramps (Town.ramps, drawn; or index `ramp`). It starts on the road
    before the ramp's approach lane (70 %; else at that road's start), in the approach lane or in a sibling (then it
    changes into the approach lane first), takes the ramp and goes on at random as random_route does."""
    turn_weights = turn_weights or {"left": 1.0, "straight": 1.0, "right": 1.0}
    rp = town.ramps[int(rng.integers(len(town.ramps)))] if ramp is None else town.ramps[ramp]
    A = town.lanes[rp["from"]]
    entries = [A.id] + list(A.siblings)
    X = town.lanes[entries[int(rng.integers(len(entries)))]]
    if X.pred and rng.random() < 0.7:               # from the road before (through the junction before the ramp)
        c = X.pred[int(rng.integers(len(X.pred)))]
        P = town.lanes[town.lanes[c].pred[0]]
        s_start = float(rng.uniform(3.0, max(3.5, P.length - 25.0)))
        lanes = [P.id, c, X.id]
        total = P.length - s_start + town.lanes[c].length + X.length
        s_lo = 0.0
    else:
        s_start = float(rng.uniform(3.0, 8.0))
        lanes = [X.id]
        total = X.length - s_start
        s_lo = s_start + LC_START_MARGIN
    moves = [m for m in lane_moves(town, X, s_lo) if rp["lane"] in m[2]]
    if not moves:                                   # no room to change into the approach lane first: start on it
        s_start = float(rng.uniform(3.0, 8.0))
        lanes, total, s_lo = [A.id], A.length - s_start, s_start + LC_START_MARGIN
        moves = [m for m in lane_moves(town, A, s_lo) if rp["lane"] in m[2]]
    seq = moves[0][2]
    lanes.extend(seq)
    R, C = town.lanes[seq[-2]], town.lanes[seq[-1]]
    total += R.length + C.length
    total = _walk(town, rng, lanes, total, link_map(R, C, R.length), length, turn_weights)
    return _finish(town, lanes, s_start, total, length)


def _walk(town, rng, lanes, total, s_lo, length, turn_weights):
    """random_route's walk: extend `lanes` (in place) until they hold `length` metres (total so far: `total`)."""
    guard = 0
    while total < length + 10.0 and guard < 200:
        guard += 1
        cur = town.lanes[lanes[-1]]
        succ = cur.succ
        if not succ:
            break
        moves = lane_moves(town, cur, s_lo) if cur.links else []
        if any(src != cur.id or town.lanes[seq[-1]].kind == "road" for _, src, seq in moves):
            src, seq = pick_move(rng, moves, cur.id, turn_weights)
        else:                                      # a single lane, no ramp: as before lane changes
            w = np.array([turn_weights.get(town.lanes[c].turn, 1.0) for c in succ], float)
            seq = [succ[rng.choice(len(succ), p=w / w.sum())]]
        lanes.extend(seq)
        if town.lanes[seq[-1]].kind == "conn":
            c = seq[-1]
            total += town.lanes[c].length
            nxt = town.lanes[c].succ[0]
            lanes.append(nxt)
            total += town.lanes[nxt].length
            s_lo = 0.0
        else:                                      # by a ramp: the lane merged into, from the end of the ramp
            R, C = town.lanes[seq[-2]], town.lanes[seq[-1]]
            total += R.length + C.length
            s_lo = link_map(R, C, R.length)
    return total


def _finish(town, lanes, s_start, total, length):
    """The Route of a walk, ending `length` metres on (not inside a junction or on a ramp)."""
    r = Route(town, lanes, s_start)
    r.s_end = float(min(r.s_end, s_start + max(length, 50.0) + 0.0)) if total > length + 40 else r.s_end
    # do not end inside a junction: push the end to the next road lane if needed
    k = r.lane_index_at(r.s_end)
    if town.lanes[r.lanes[k]].kind in ("conn", "ramp"):
        if k + 1 < len(r.lanes):
            r.s_end = float(min(r.poly.length - 1.0, r.offsets[k + 1] + 12.0))
    return r
