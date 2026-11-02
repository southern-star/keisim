"""Ego routes: a lane sequence flattened into one arc-length parametrised path.

Lane changes (multi-lane roads, Town lanes with `siblings`): two lanes of the same road and direction next to each
other in the sequence, [A, B], mean "drive on A, change to B". The path leaves A at A-local `a0` and blends into B
over `lc` metres (a quintic lateral profile: zero lateral speed and acceleration at both ends), then follows B. The
change sits in A's window [a_min, a_max] (LC_ENTRY after the junction it came from, done LC_EXIT before the end of
the road: 20 m of solid lane line before a signal + the stop line); the expert may move it later in the window
(Route.retime_change) while it waits for a gap in B.
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


def change_window(lane_a, s_lo=0.0):
    """(a_min, a_max, lc): where on `lane_a` a change to a sibling lane can start, and its length; None if the lane
    is too short (from A-local s_lo on)."""
    lo = max(LC_ENTRY, s_lo)
    hi = lane_a.length - LC_EXIT
    if hi - lo < LC_MIN:
        return None
    lc = min(LC_LEN, hi - lo)
    return lo, hi - lc, lc


class Route:
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
        return b.id in getattr(a, "siblings", ())

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
            win = change_window(A, lo[k - 1] + (self.s_start + LC_START_MARGIN if k - 1 == 0 else 0.0))
            if win is None:                    # too short: change over what is left (rare; random_route avoids it)
                a_min = a_max = max(lo[k - 1], min(A.length - LC_MIN, A.length / 2))
                lc = max(1.0, min(LC_MIN, A.length - a_min - 1.0))
            else:
                a_min, a_max, lc = win
            a0 = float(np.clip(self._change_at.get(k, a_min), a_min, a_max))
            hi[k - 1] = a0
            lo[k] = (a0 + lc) * B.length / A.length
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
                pb = lane.poly.interp(sa * lane.length / A.length)
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
        for k, lid in enumerate(self.lanes):
            lane = town.lanes[lid]
            if lane.kind == "conn":
                J = town.junctions[lane.junction]
                self.junctions.append({"s_in": offs[k], "s_out": offs[k] + lane.length, "junction": J.id,
                                       "turn": lane.turn, "signalized": J.signalized, "lane": lid,
                                       "decision": len(J.arms) >= 3})
            if lane.stop_s is not None and lo[k] <= lane.stop_s <= hi[k]:
                self.stops.append((offs[k] + lane.stop_s - s0[k], lid))
            if lane.stop_sign_s is not None and lo[k] <= lane.stop_sign_s <= hi[k]:
                self.stop_signs.append((offs[k] + lane.stop_sign_s - s0[k], lid))
        # curvature along the flattened path
        curv = np.concatenate(curv)
        self.curv = curv[: len(self.poly.pts)] if len(curv) >= len(self.poly.pts) else np.pad(
            curv, (0, len(self.poly.pts) - len(curv)), mode="edge")

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
            return self.poly.interp(self.s_end)
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
    guard = 0
    while total < length + 10.0 and guard < 200:
        guard += 1
        cur = town.lanes[lanes[-1]]
        succ = cur.succ
        if not succ:
            break
        sib = [b for b in getattr(cur, "siblings", ()) if change_window(cur, s_lo) is not None]
        if sib:
            moves = [(c, src) for src in [cur.id] + sib for c in town.lanes[src].succ]
            turns = [t for t in ("left", "straight", "right") if any(town.lanes[c].turn == t for c, _ in moves)]
            w = np.array([turn_weights.get(t, 1.0) for t in turns], float)
            t = turns[rng.choice(len(turns), p=w / w.sum())]
            opts = [m for m in moves if town.lanes[m[0]].turn == t]
            own = [m for m in opts if m[1] == cur.id]
            other = [m for m in opts if m[1] != cur.id]
            pick = own if own and (not other or rng.random() >= LC_FREE_PROB) else other
            c, src = pick[rng.integers(len(pick))]
            if src != cur.id:
                lanes.append(src)                  # change to the lane that has the movement
        else:
            w = np.array([turn_weights.get(town.lanes[c].turn, 1.0) for c in succ], float)
            c = succ[rng.choice(len(succ), p=w / w.sum())]
        lanes.append(c)
        total += town.lanes[c].length
        nxt = town.lanes[c].succ[0]
        lanes.append(nxt)
        total += town.lanes[nxt].length
        s_lo = 0.0
    r = Route(town, lanes, s_start)
    r.s_end = float(min(r.s_end, s_start + max(length, 50.0) + 0.0)) if total > length + 40 else r.s_end
    # do not end inside a junction: push the end to the next road lane if needed
    k = r.lane_index_at(r.s_end)
    if town.lanes[r.lanes[k]].kind == "conn":
        if k + 1 < len(r.lanes):
            r.s_end = float(min(r.poly.length - 1.0, r.offsets[k + 1] + 12.0))
    return r
