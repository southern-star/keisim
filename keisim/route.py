"""Ego routes: a lane sequence flattened into one arc-length parametrised path."""
from __future__ import annotations

import numpy as np

from .config import CMD_LEFT, CMD_RIGHT, CMD_STRAIGHT
from .geometry import Polyline

TURN_TO_CMD = {"left": CMD_LEFT, "straight": CMD_STRAIGHT, "right": CMD_RIGHT}


class Route:
    def __init__(self, town, lanes, s_start, s_end=None):
        self.town = town
        self.lanes = list(lanes)
        pts, offs = [], []
        base = 0.0
        for k, lid in enumerate(self.lanes):
            lane = town.lanes[lid]
            offs.append(base)
            pts.append(lane.pts if k == 0 else lane.pts[1:])
            base += lane.length
        self.offsets = np.array(offs)
        self.poly = Polyline(np.concatenate(pts, 0))
        self.s_start = float(s_start)
        self.s_end = float(min(self.poly.length - 1.0, s_end if s_end is not None else self.poly.length - 5.0))
        self.junctions = []
        self.stops = []
        for k, lid in enumerate(self.lanes):
            lane = town.lanes[lid]
            if lane.kind == "conn":
                J = town.junctions[lane.junction]
                self.junctions.append({"s_in": offs[k], "s_out": offs[k] + lane.length, "junction": J.id,
                                       "turn": lane.turn, "signalized": J.signalized, "lane": lid})
            if lane.stop_s is not None:
                self.stops.append((offs[k] + lane.stop_s, lid))
        # curvature along the flattened path
        curv = np.concatenate([town.lanes[l].curv if k == 0 else town.lanes[l].curv[1:]
                               for k, l in enumerate(self.lanes)])
        self.curv = curv[: len(self.poly.pts)] if len(curv) >= len(self.poly.pts) else np.pad(
            curv, (0, len(self.poly.pts) - len(curv)), mode="edge")

    @property
    def length(self):
        return self.s_end - self.s_start

    def lane_index_at(self, s):
        return int(np.clip(np.searchsorted(self.offsets, s, side="right") - 1, 0, len(self.lanes) - 1))

    def lane_at(self, s):
        return self.lanes[self.lane_index_at(s)]

    def next_decision(self, s):
        """Next junction with a real choice (>=3 arms) that has not been exited."""
        for j in self.junctions:
            if j["signalized"] and j["s_out"] > s + 0.5:
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


def random_route(town, rng, length, start_lane=None, s_start=None, turn_weights=None):
    """Random walk through the lane graph with a mild preference for turns."""
    turn_weights = turn_weights or {"left": 1.0, "straight": 1.0, "right": 1.0}
    if start_lane is None:
        cands = [l for l in town.road_lanes if town.lanes[l].length > 30.0]
        start_lane = cands[rng.integers(len(cands))]
    lane = town.lanes[start_lane]
    if s_start is None:
        s_start = float(rng.uniform(3.0, max(3.5, lane.length - 25.0)))
    lanes = [start_lane]
    total = lane.length - s_start
    guard = 0
    while total < length + 10.0 and guard < 200:
        guard += 1
        succ = town.lanes[lanes[-1]].succ
        if not succ:
            break
        w = np.array([turn_weights.get(town.lanes[c].turn, 1.0) for c in succ], float)
        c = succ[rng.choice(len(succ), p=w / w.sum())]
        lanes.append(c)
        total += town.lanes[c].length
        nxt = town.lanes[c].succ[0]
        lanes.append(nxt)
        total += town.lanes[nxt].length
    r = Route(town, lanes, s_start)
    r.s_end = float(min(r.s_end, s_start + max(length, 50.0) + 0.0)) if total > length + 40 else r.s_end
    # do not end inside a junction: push the end to the next road lane if needed
    k = r.lane_index_at(r.s_end)
    if town.lanes[r.lanes[k]].kind == "conn":
        if k + 1 < len(r.lanes):
            r.s_end = float(min(r.poly.length - 1.0, r.offsets[k + 1] + 12.0))
    return r
