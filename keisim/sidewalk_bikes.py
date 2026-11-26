"""Bicycles on the sidewalks (TrafficConfig.sidewalk_bikes, 歩道を走る自転車).

Where they ride. On the sidewalks (Town.walkways that are not in the road: 路側帯 riders are the road bicycles of
keisim/bicycles.py), on the road side of the sidewalk (道路交通法 63 条の 4: 車道寄り), either way along it, at 10-16 km/h,
slowing down for the people walking in front of them and following slower riders.

At a junction. The sidewalks stop at the junction's edge; links (sidewalk_links, built once per town) join the end of
one to the start of another at the same junction:
- "turn": round the corner to the sidewalk of the next road (no road to cross);
- "cross": straight on, over the mouth of the road in front of them, to the sidewalk that goes on on the far side.
  At a two-phase signal (Junction.groups) they cross with the pedestrian light of that road (the light of the people
  on its zebra: Junction.ped_signal), and only if they can get over before it turns red; at other signals they do
  not cross there. At a junction without signals they look first: careful riders (SW_CAREFUL) slow down to
  SW_LOOK_V before the kerb and wait while a car would reach their way within SW_GAP_CAREFUL s; the others keep
  their speed and stop only for a car closer than SW_GAP_CARELESS s (出会い頭: they come out from behind the corner).
- or they turn back.

What the others see. Riders in the carriageway (on a "cross" link past its kerb) are obstacles for the cars (World:
as people, through the corridor check) and for the expert, with where they will be in 0.8, 1.6 and 2.4 s along their
way (Expert.plan). The people on foot keep out of the way of the moving ones. The renderers draw them as bicycles
(World.scene_state: after the road bicycles).
"""
from __future__ import annotations

import math

import numpy as np

from .bicycles import BIKE_LEN, BIKE_W, FRAME_COLORS, bike_circles
from .geometry import Polyline, wrap
from .roadnet import PED_GREEN

SW_CRUISE = (2.8, 4.4)      # m/s on the sidewalk
SW_TURN_V = 2.2             # m/s round a corner
SW_PED_V = 1.2              # m/s passing someone walking in front of them
SW_PASS_V = 2.0             # m/s passing an oncoming rider
SW_DODGE = 0.5              # m: how far they move over for them
SW_SIDE_RATE = 0.8          # m/s sideways
SW_CAREFUL = 0.6            # share of the riders who slow down and look before riding across a road without signals
SW_LOOK_V = 1.4             # m/s: their speed at the kerb
SW_GAP_CAREFUL = 3.5        # s: a careful rider waits while a car would reach their way sooner than this
SW_GAP_CARELESS = 2.0       # s: the others stop only for a car this close (if they still can)
SW_RED_RUN = 0.15           # share of the riders who cross against the pedestrian red when they see no car coming
#                             (as at a junction without signals, by their own look: 信号無視)
SW_ACC, SW_BRAKE = 0.9, 3.0         # m/s^2: setting off, braking
SW_HARD = 4.5               # m/s^2: the hardest braking (a careless rider who saw a car late)
SW_KERB = 0.6               # m before the kerb (the edge of the carriageway along their way) where they wait
SW_ROAD_SIDE = 0.25         # of the sidewalk's width: their line, from its middle towards the road
P_CROSS, P_TURN = 0.55, 0.35        # what they do at a junction (the rest: turn back)
RIDE, LINK, WAIT = 0, 1, 2
BACK_OFF_S, BACK_OFF_V = 3.0, 0.5  # s, m/s: a rider stuck in front of a standing car backs off (SidewalkRiders.step)
CAR_LOOK, CAR_MARGIN = 6.0, 0.3     # m: riders stop short of a car's body this far ahead of their front (+ margin)
ROAD_OUT = 0.8              # m: in_road: past the far edge of the carriageway still in it
ENTER_V, ENTER_D = 1.0, 2.0         # m/s, m: in_road: riding into the carriageway
CORNER_BOW = 0.15           # of the chord: how far a corner link bows out (its middle: half of that)
TURN_MAX = 18.0             # m: a corner link at most this long
CROSS_MAX = 45.0            # m: a crossing link at most this long
CROSS_ANGLE = 0.6           # rad: straight on within this of the way they ride
COMING_V, COMING_D = 2.0, 6.0      # m/s, m: Expert: riders coming into a road (SidewalkRiders.coming)
ZEBRA_NEAR = 10.0           # m: on_zebras counts a rider whom the light lets go from this far before their sidewalk's end


def _end(w, end):
    """Point and the heading out of walkway `w` at its end `end` (0: start, 1: end)."""
    poly = w["poly"]
    if end == 1:
        return poly.pts[-1].copy(), float(poly.heading1(poly.length))
    return poly.pts[0].copy(), float(poly.heading1(0.0)) + math.pi


def _road_side(town, w):
    """Unit lateral (in the walkway's left-normal units) of the rider's line from the walkway's middle: towards the
    road, SW_ROAD_SIDE of the sidewalk's width."""
    return -w["side"] * SW_ROAD_SIDE * town.roads[w["road"]].spec.sidewalk


def _line_pt(town, w, s):
    poly = w["poly"]
    x, y = poly.interp1(s)
    h = poly.heading1(s)
    off = _road_side(town, w)
    return np.array([x - math.sin(h) * off, y + math.cos(h) * off]), h


def _seg_x(a, b, c, d):
    """Whether segments ab and cd cross -> (t along ab) or None."""
    r, s = b - a, d - c
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return None
    t = ((c - a)[0] * s[1] - (c - a)[1] * s[0]) / den
    u = ((c - a)[0] * r[1] - (c - a)[1] * r[0]) / den
    return t if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0 else None


def _drivable(town, J, road, pts):
    """Which of `pts` (k, 2) the cars can drive on at junction J: the carriageway of `road` (within its half width of
    its centre line), the junction's road surface (Junction.road_poly), or a connector's swept strip (a turn cuts the
    corner inside the hull's edge)."""
    import cv2
    out = np.array([abs(road.poly.project(x)[1]) < road.half_width for x in pts])
    if J.road_poly is not None and len(J.road_poly) >= 3:
        poly = np.asarray(J.road_poly, np.float32).reshape(-1, 1, 2)
        out |= np.array([cv2.pointPolygonTest(poly, (float(x[0]), float(x[1])), False) >= 0 for x in pts])
    for cid in J.connectors:
        ln = town.lanes[cid]
        hw = ln.width / 2 + 0.3
        far = np.hypot(*(pts - ln.pts.mean(axis=0)).T) > ln.length / 2 + hw + 1.0
        for k in np.nonzero(~far & ~out)[0]:
            s_k, lat = ln.poly.project(pts[k])
            if 0.0 < s_k < ln.length and abs(lat) < hw:
                out[k] = True
    return out


def sidewalk_links(town):
    """The links between the sidewalks' ends at each junction (cached on the town): (walkway, end) -> [link], a link
    a dict: kind ("turn" | "cross"), pts (its line, (k, 2)), poly, to (walkway, end it comes onto), and for "cross":
    road, arm, junction, u_in / u_out (the stretch of the line over the carriageway), phase (the pedestrian light's
    phase at a two-phase signal, else None), signal (the junction has signals)."""
    out = town.__dict__.get("_sw_links")
    if out is not None:
        return out
    out = {}
    ww = town.walkways
    ends = {}                                   # junction -> [(walkway, end)]
    for i, w in enumerate(ww):
        if w["in_road"]:
            continue
        r = town.roads[w["road"]]
        ends.setdefault(r.a, []).append((i, 0))
        ends.setdefault(r.b, []).append((i, 1))
    for jid, lst in ends.items():
        J = town.junctions[jid]
        arms = [(k, town.roads[a["road"]], a["end"]) for k, a in enumerate(J.arms) if "road" in a]
        for i, e in lst:
            p, h = _end(ww[i], e)
            p = p + np.array([-math.sin(h), math.cos(h)]) * (_road_side(town, ww[i]) * (1 if e == 1 else -1))
            links = []
            for i2, e2 in lst:
                if ww[i2]["road"] == ww[i]["road"]:
                    continue
                q, h2o = _end(ww[i2], e2)
                h2 = h2o + math.pi                              # the way they ride on along it
                q = q + np.array([-math.sin(h2o), math.cos(h2o)]) * (_road_side(town, ww[i2]) * (1 if e2 == 1 else -1))
                dist = float(np.hypot(*(q - p)))
                crossed = []
                for k, road, rend in arms:
                    c = road.center                         # (from the junction's centre: the road starts at
                    seg = np.vstack([J.pos, c[:min(len(c), 40)]]) if rend == "a" else \
                        np.vstack([c[-min(len(c), 40):], J.pos])   # its edge, where the riders cross)
                    for m in range(len(seg) - 1):
                        t = _seg_x(p, q, seg[m], seg[m + 1])
                        if t is not None:
                            crossed.append((k, road, t))
                            break
                straight = abs(float(wrap(h2 - h))) < CROSS_ANGLE
                if crossed:                                 # a road's mouth: only between the two corners beside it
                    k, road, _ = crossed[0]
                    end_pt = road.center[0] if J.arms[k]["end"] == "a" else road.center[-1]
                    ax = (end_pt - J.pos) / max(float(np.hypot(*(end_pt - J.pos))), 1e-6)
                    if min(float((p - J.pos) @ ax), float((q - J.pos) @ ax)) < 1.0:
                        continue                            # (not both on that road's side of the junction)
                if not crossed and (dist < TURN_MAX or (straight and dist < CROSS_MAX)):
                    # round the corner: the chord between the two ends, bowed out away from the junction (where the
                    # two sidewalk lines meet is inside the junction's road surface: its hull cuts the corners)
                    mid = (p + q) / 2
                    out_dir = mid - J.pos
                    out_dir = out_dir / max(float(np.hypot(*out_dir)), 1e-6)
                    ctrl = mid + out_dir * min(CORNER_BOW * dist, 1.5)
                    f = np.linspace(0.0, 1.0, 13)[:, None]
                    pts = (1 - f) ** 2 * p + 2 * (1 - f) * f * ctrl + f ** 2 * q
                    if _drivable(town, J, town.roads[ww[i]["road"]], pts).any():
                        continue                                # (it would take them over the junction's road)
                    links.append({"kind": "turn", "pts": pts, "poly": Polyline(pts), "to": (i2, e2)})
                elif len(crossed) == 1 and dist < CROSS_MAX and straight:
                    k, road, _ = crossed[0]
                    phase, zebra, zsgn, via = None, -1, 0.0, [p, q]
                    if J.signalized:
                        if J.groups is None or J.phases != (3 if J.scramble else 2):
                            continue                            # other signals: they do not cross there
                        phase = 2 if J.scramble else 1 - J.groups[k]
                        zebra = next((ci for ci, c in enumerate(town.ped_crossings)
                                      if c["junction"] == jid and c["arm"] == k), -1)
                        if zebra < 0:
                            continue
                        k0, k1 = town.ped_crossings[zebra]["pts"]
                        zsgn = 1.0 if np.hypot(*(k0 - p)) <= np.hypot(*(k1 - p)) else -1.0
                        if zsgn < 0:
                            k0, k1 = k1, k0
                        via = [p, k0, k1, q]                    # on the zebra, kerb to kerb
                    pts = [via[0]]
                    for a, b in zip(via[:-1], via[1:]):
                        n = max(2, int(math.ceil(float(np.hypot(*(b - a))) / 0.5)) + 1)
                        pts += list(a + (b - a) * np.linspace(0.0, 1.0, n)[1:, None])
                    pts = np.array(pts)
                    poly = Polyline(pts)
                    inside = np.nonzero(_drivable(town, J, road, poly.pts))[0]
                    if not len(inside):
                        continue
                    links.append({"kind": "cross", "pts": poly.pts, "poly": poly, "to": (i2, e2), "road": road.id,
                                  "arm": k, "junction": jid, "u_in": float(poly.s[inside[0]]),
                                  "u_out": float(poly.s[inside[-1]]), "phase": phase, "signal": J.signalized,
                                  "zebra": zebra, "zsgn": zsgn})
            out[(i, e)] = links
    town.__dict__["_sw_links"] = out
    return out


def sidewalk_walkways(town):
    return [i for i, w in enumerate(town.walkways) if not w["in_road"] and w["poly"].length > 12.0]


class SidewalkRiders:
    def __init__(self, town, cfg, rng, n, avoid_xy=None, avoid_r=15.0):
        self.town, self.cfg, self.rng = town, cfg, rng
        self.links = sidewalk_links(town)
        ws = sidewalk_walkways(town)
        self.n = 0
        cols = ("w", "s", "dir", "v", "cruise", "careful", "red_run", "mode", "u", "side", "held", "crank", "wheel", "roll", "steer", "foot",
                "look", "effort", "height", "yaw", "wait", "kind")
        for c in cols:
            setattr(self, c, np.zeros(0))
        self.xy = np.zeros((0, 2))
        self.colors = np.zeros((0, 4, 3))
        self.link = []                          # the link each rider is on, or has chosen (None)
        if not ws or n <= 0:
            return
        lens = np.array([town.walkways[i]["poly"].length for i in ws])
        tries = 0
        from .traffic import PANTS, SHIRTS, SKIN
        while self.n < n and tries < 30 * n:
            tries += 1
            i = ws[int(rng.choice(len(ws), p=lens / lens.sum()))]
            L = town.walkways[i]["poly"].length
            s = float(rng.uniform(3.0, L - 3.0))
            p, h = _line_pt(town, town.walkways[i], s)
            if avoid_xy is not None and len(avoid_xy) and np.min(np.hypot(*(avoid_xy - p).T)) < avoid_r:
                continue
            if self.n and np.min(np.hypot(*(self.xy - p).T)) < 8.0:
                continue
            d = float(rng.choice([-1.0, 1.0]))
            kind = 2 if rng.random() < 0.2 else 0           # city bikes, some slow riders
            ap = lambda a, x: np.append(a, x)               # noqa: E731
            self.w, self.s, self.dir = ap(self.w, i), ap(self.s, s), ap(self.dir, d)
            cr = float(rng.uniform(*SW_CRUISE)) * (0.8 if kind == 2 else 1.0)
            self.cruise, self.v = ap(self.cruise, cr), ap(self.v, cr * rng.uniform(0.7, 1.0))
            self.careful = ap(self.careful, float(rng.random() < SW_CAREFUL))
            self.red_run = ap(self.red_run, float(rng.random() < SW_RED_RUN))
            self.mode, self.u = ap(self.mode, RIDE), ap(self.u, 0.0)
            self.crank, self.wheel = ap(self.crank, rng.uniform(0, 2 * math.pi)), ap(self.wheel, rng.uniform(0, 6.3))
            for c in ("roll", "steer", "foot", "look", "wait", "side", "held"):
                setattr(self, c, ap(getattr(self, c), 0.0))
            self.effort = ap(self.effort, 0.3)
            self.height = ap(self.height, rng.uniform(1.5, 1.85))
            self.yaw = ap(self.yaw, h if d > 0 else h + math.pi)
            self.kind = ap(self.kind, kind)
            self.xy = np.vstack([self.xy, p])
            col = np.stack([FRAME_COLORS[rng.integers(len(FRAME_COLORS))] * rng.uniform(0.9, 1.05),
                            SHIRTS[rng.integers(len(SHIRTS))] * rng.uniform(0.85, 1.1),
                            PANTS[rng.integers(len(PANTS))], SKIN[rng.integers(len(SKIN))]])
            self.colors = np.concatenate([self.colors, col[None]])
            self.link.append(None)
            self.n += 1
        self.kind = self.kind.astype(int)
        self.mode = self.mode.astype(int)
        self.w = self.w.astype(int)

    # ------------------------------------------------------------------ what the others see
    def in_road(self):
        """The riders on a crossing link in its carriageway (or within ROAD_OUT m past it), or riding into it (faster
        than ENTER_V within ENTER_D m of its kerb). Riders standing at the kerb for their light or a gap are not."""
        out = np.zeros(self.n, bool)
        for i in range(self.n):
            lk = self.link[i]
            if self.mode[i] != RIDE and isinstance(lk, dict) and lk["kind"] == "cross":
                u = self.u[i]
                out[i] = (lk["u_in"] - 0.3 <= u <= lk["u_out"] + ROAD_OUT) or \
                    (self.v[i] > ENTER_V and lk["u_in"] - ENTER_D <= u < lk["u_in"])
        return out

    def ahead(self, i, tau):
        """Where rider i will be in `tau` s along their way (at their speed now)."""
        if self.mode[i] == RIDE:
            w = self.town.walkways[self.w[i]]
            L = w["poly"].length
            s = self.s[i] + self.dir[i] * self.v[i] * tau
            lk = self.link[i]
            if isinstance(lk, dict) and not 0.0 <= s <= L:     # on into the link
                x, y = lk["poly"].interp1(min(abs(s - (L if s > L else 0.0)), lk["poly"].length))
                return np.array([x, y])
            return _line_pt(self.town, w, min(max(s, 0.0), L))[0]
        lk = self.link[i]
        x, y = lk["poly"].interp1(min(self.u[i] + self.v[i] * tau, lk["poly"].length))
        return np.array([x, y])

    def on_zebras(self, t):
        """For Pedestrians.crosswalk_people: zebra -> [(u from its pts[0], sgn, speed)] of the riders on a zebra, and of
        those coming up to one (on their sidewalk's last ZEBRA_NEAR m, or on the link before its kerb) whom the light
        lets go, at its kerb (u 0 / width) at their cruising speed."""
        out = {}
        pc = self.town.ped_crossings
        for i in range(self.n):
            lk = self.link[i]
            if not isinstance(lk, dict) or lk["kind"] != "cross" or lk.get("zebra", -1) < 0:
                continue
            c = pc[lk["zebra"]]
            sgn = lk["zsgn"]
            if self.mode[i] == LINK and (self.u[i] >= lk["u_in"] - 0.3 or
                                         (self.v[i] > ENTER_V and self.u[i] >= lk["u_in"] - ENTER_D)):
                u = float((self.xy[i] - c["pts"][0]) @ c["dir"])     # on it, or riding onto it
                out.setdefault(lk["zebra"], []).append((u, sgn, max(float(self.v[i]), 1.0)))
                continue
            to_kerb = lk["u_in"] - self.u[i]
            if self.v[i] < 0.15 and to_kerb < SW_KERB + 0.5 and self.held[i]:
                continue                                 # standing at the kerb, a car in their way: it goes first
            if self.mode[i] == RIDE:
                w = self.town.walkways[self.w[i]]
                left = (w["poly"].length - self.s[i]) if self.dir[i] > 0 else self.s[i]
                if left > ZEBRA_NEAR:
                    continue
                to_kerb = lk["u_in"] + left
            J = self.town.junctions[lk["junction"]]
            st, rem = J.ped_signal(lk["phase"], t, self.cfg.ped_flash)
            if st == PED_GREEN and rem > (lk["u_out"] - lk["u_in"] + 2.0) / max(self.cruise[i], 2.5) + 1.0:
                # (as far before the zebra's end as they are from its kerb: crosswalk_blocked times their arrival)
                u = -to_kerb if sgn > 0 else c["width"] + to_kerb
                out.setdefault(lk["zebra"], []).append((u, sgn, max(float(self.v[i]), 1.0)))
        return out

    def coming(self):
        """Where the riders on their way over a road will be in 0.8, 1.6 and 2.4 s (at their speed now): those in it,
        and those riding on at more than COMING_V within COMING_D of its kerb (the ones slowing down to look, or
        stopping for their light, are not on their way yet) -> [(rider, point)]."""
        out = []
        road = self.in_road()
        for i in range(self.n):
            lk = self.link[i]
            if not isinstance(lk, dict) or lk["kind"] != "cross":
                continue
            if self.mode[i] == LINK:
                to_kerb = lk["u_in"] - self.u[i]
            else:
                w = self.town.walkways[self.w[i]]
                to_kerb = lk["u_in"] + ((w["poly"].length - self.s[i]) if self.dir[i] > 0 else self.s[i])
            if not (road[i] or (self.v[i] > COMING_V and to_kerb < COMING_D)):
                continue
            for tau in (0.8, 1.6, 2.4):
                out.append((i, self.ahead(i, tau)))
        return out

    def crossing_ahead(self):
        """The riders on their way over a road without a zebra (Expert.rider_crossings) -> [(rider, link, m from the
        rider to the link's kerb (negative past it))]: those riding on (not standing for a gap or a car) whose link is
        chosen, on its sidewalk's last ZEBRA_NEAR + 6 m or on the link, not yet over the far edge."""
        out = []
        for i in range(self.n):
            lk = self.link[i]
            if not isinstance(lk, dict) or lk["kind"] != "cross" or lk.get("zebra", -1) >= 0:
                continue
            if self.mode[i] == LINK:
                d = lk["u_in"] - self.u[i]
                if self.u[i] > lk["u_out"]:
                    continue
            else:
                w = self.town.walkways[self.w[i]]
                left = (w["poly"].length - self.s[i]) if self.dir[i] > 0 else self.s[i]
                if left > ZEBRA_NEAR + 6.0:
                    continue
                d = lk["u_in"] + left
            if self.v[i] < 0.15 and d > -0.3:
                continue                                 # standing before the road (for a gap, or a car in the way)
            out.append((i, lk, d))
        return out

    def circles(self):
        """The riders near or in the carriageway as circles for the cars: centres, owner (rider index)."""
        idx = np.nonzero(self.in_road())[0]
        if not len(idx):
            return np.zeros((0, 2)), np.zeros(0, int)
        c, o = bike_circles(self.xy[idx], self.yaw[idx])
        return c, idx[o]

    def visible_from(self, cam, yaw, half_fov, max_dist, town=None):
        """As Bicycles.visible_from (by their centres)."""
        if self.n == 0:
            return np.zeros(0, bool)
        rel = self.xy - np.asarray(cam, float)
        seen = (np.abs(wrap(np.arctan2(rel[:, 1], rel[:, 0]) - yaw)) < half_fov) & (np.hypot(*rel.T) < max_dist)
        if town is not None and seen.any() and len(town.buildings):
            from .geometry import rays_clear
            edges, _, _ = town.building_edges()
            i = np.nonzero(seen)[0]
            seen[i] = rays_clear(cam, self.xy[i], edges)
        return seen

    def scene(self):
        """Bicycles.scene's arrays for these riders."""
        pose = np.stack([self.roll + 0.07 * self.foot, self.steer, self.crank, self.wheel, self.foot, self.look,
                         self.effort], -1) if self.n else np.zeros((0, 7))
        return self.xy, self.yaw, pose, self.kind, self.height, self.colors

    # ------------------------------------------------------------------ moving
    def _choose(self, i):
        """Pick what rider i does at the end of their sidewalk: a link, or "back" (turn round)."""
        e = 1 if self.dir[i] > 0 else 0
        links = self.links.get((int(self.w[i]), e), [])
        opts = [(lk, P_CROSS if lk["kind"] == "cross" else P_TURN) for lk in links]
        kinds = {lk["kind"] for lk in links}
        opts = [(lk, p / sum(1 for x in links if x["kind"] == lk["kind"])) for lk, p in opts]
        opts.append(("back", 1.0 - sum(P_CROSS if k == "cross" else P_TURN for k in kinds)))
        w = np.array([p for _, p in opts])
        return opts[int(self.rng.choice(len(opts), p=w / w.sum()))][0]

    def _gap(self, lk, cars_xy, cars_v, cars_yaw, cars_r):
        """The earliest time a car reaches the carriageway stretch of crossing link `lk` (straight on along its heading
        at its speed; a car standing in it: 0)."""
        a = lk["poly"].interp1(lk["u_in"])
        b = lk["poly"].interp1(lk["u_out"])
        a, b = np.array(a), np.array(b)
        mid = (a + b) / 2
        if not len(cars_xy):
            return math.inf
        d = np.hypot(*(cars_xy - mid).T)
        near = np.nonzero(d < 60.0)[0]
        best = math.inf
        ab = b - a
        L = float(np.hypot(*ab)) or 1.0
        t = ab / L
        nrm = np.array([-t[1], t[0]])
        for j in near:
            p = cars_xy[j]
            rel = p - a
            lon, lat = float(rel @ t), float(rel @ nrm)
            if -1.5 - cars_r[j] < lon < L + 1.5 + cars_r[j] and abs(lat) < cars_r[j] + 1.2:
                return 0.0                               # in their way now
            u = np.array([math.cos(cars_yaw[j]), math.sin(cars_yaw[j])])
            den = float(u @ nrm)
            if abs(den) < 0.2 or cars_v[j] < 0.3:
                continue
            k = -lat / den                               # metres along its heading to the line of their way
            if k < -cars_r[j]:
                continue
            x = lon + k * float(u @ t)
            if -2.0 - cars_r[j] < x < L + 2.0 + cars_r[j]:
                best = min(best, max(k - cars_r[j] - 1.0, 0.0) / cars_v[j])
        return best

    def _path_pt(self, i, d):
        """The point `d` m ahead of rider i along their way (on into their link, if they have one)."""
        if self.mode[i] == RIDE:
            w = self.town.walkways[self.w[i]]
            L = w["poly"].length
            s = self.s[i] + self.dir[i] * d
            lk = self.link[i]
            if isinstance(lk, dict) and not 0.0 <= s <= L:
                x, y = lk["poly"].interp1(min(abs(s - (L if s > L else 0.0)), lk["poly"].length))
                return np.array([x, y])
            return _line_pt(self.town, w, min(max(s, 0.0), L))[0]
        lk = self.link[i]
        x, y = lk["poly"].interp1(min(self.u[i] + d, lk["poly"].length))
        return np.array([x, y])

    def _car_ahead(self, i, cars_xy, cars_yaw, cars_dims):
        """m along their way to the first point within CAR_MARGIN of a car's body (inf: none within CAR_LOOK m)."""
        if not len(cars_xy):
            return math.inf
        near = np.nonzero(np.hypot(*(cars_xy - self.xy[i]).T) < CAR_LOOK + 5.0)[0]
        if not len(near):
            return math.inf
        c, s_ = np.cos(cars_yaw[near]), np.sin(cars_yaw[near])
        hl, hw = cars_dims[near, 0] / 2 + CAR_MARGIN, cars_dims[near, 1] / 2 + CAR_MARGIN
        for d in np.arange(0.5, CAR_LOOK + 0.01, 0.5):
            p = self._path_pt(i, float(d) + BIKE_LEN / 2)
            rel = p - cars_xy[near]
            lon = rel[:, 0] * c + rel[:, 1] * s_
            lat = -rel[:, 0] * s_ + rel[:, 1] * c
            if np.any((np.abs(lon) < hl) & (np.abs(lat) < hw)):
                return float(d)
        return math.inf

    def step(self, dt, t, cars_xy, cars_v, cars_yaw, cars_dims, peds_xy):
        """cars_*: every car (NPCs and the ego): centres, speeds, headings, (length, width)."""
        town = self.town
        cars_r = cars_dims[:, 0] / 2 if len(cars_dims) else np.zeros(0)
        for i in range(self.n):
            v_want = float(self.cruise[i])
            look = 0.0
            if self.mode[i] == RIDE:
                w = town.walkways[self.w[i]]
                L = w["poly"].length
                left = (L - self.s[i]) if self.dir[i] > 0 else self.s[i]
                if self.link[i] is None and left < 8.0:
                    self.link[i] = self._choose(i)
                lk = self.link[i]
                if lk == "back":
                    v_want = min(v_want, math.sqrt(2 * SW_BRAKE * max(left - 0.5, 0.0)) + 0.6)
                elif isinstance(lk, dict):
                    v_want = min(v_want, self._link_speed(i, lk, -max(left, 0.0), cars_xy, cars_v, cars_yaw, cars_r,
                                                          t))
            else:
                lk = self.link[i]
                v_want = min(v_want, self._link_speed(i, lk, self.u[i], cars_xy, cars_v, cars_yaw, cars_r, t))
                if lk["kind"] == "cross" and (not lk["signal"] or self.red_run[i]) and self.u[i] < lk["u_in"] and \
                        self.v[i] < 2.0:
                    look = math.sin(t * 2.5 + i)         # looking both ways at the kerb
            # people walking and oncoming riders in front of them: past them slowly, moved over to one side (keep
            # left of an oncoming rider; away from someone walking); a rider going their way: followed
            hd = np.array([math.cos(self.yaw[i]), math.sin(self.yaw[i])])
            dodge = 0.0
            if len(peds_xy):
                rel = peds_xy - self.xy[i]
                lon = rel @ hd
                lat = hd[0] * rel[:, 1] - hd[1] * rel[:, 0]             # + = on their left
                m = (lon > -0.5) & (lon < 6.0) & (np.abs(lat - self.side[i]) < 1.0)
                if m.any():
                    j = int(np.argmin(np.where(m, lon, np.inf)))
                    dodge = -SW_DODGE if lat[j] > self.side[i] else SW_DODGE
                    v_want = min(v_want, SW_PED_V)
            if self.n > 1:
                rel = self.xy - self.xy[i]
                lon = rel @ hd
                lat = hd[0] * rel[:, 1] - hd[1] * rel[:, 0]
                same = np.cos(self.yaw - self.yaw[i]) > 0.0
                near = (lon > 0.3) & (np.abs(lat) < 1.2)
                near[i] = False
                m = near & same & (lon < 6.0)
                if m.any():
                    v_want = min(v_want, max(float(lon[m].min()) - BIKE_LEN - 0.8, 0.0) * 1.2)
                m = near & ~same & (lon < 9.0)
                if m.any():
                    dodge = SW_DODGE
                    v_want = min(v_want, SW_PASS_V)
            if self.v[i] > 0.05 or v_want > 0.05:       # a car's body in their way (one turning or standing in
                d_car = self._car_ahead(i, cars_xy, cars_yaw, cars_dims)     # the road they cross): stop short of it
                if d_car < math.inf:
                    v_want = min(v_want, math.sqrt(2 * SW_HARD * max(d_car - 0.4, 0.0)))
            if self.mode[i] == LINK and self.v[i] < 0.05 and self.wait[i] > BACK_OFF_S and \
                    self._car_ahead(i, cars_xy, cars_yaw, cars_dims) < math.inf and \
                    self.u[i] > self.link[i]["u_in"] - SW_KERB:
                # a car standing in their way while they stand in its way (it waits for them): they push the bicycle
                # back towards the kerb they came from, out of its way
                self.u[i] = max(self.u[i] - BACK_OFF_V * dt, self.link[i]["u_in"] - SW_KERB)
            if self.mode[i] != RIDE:
                dodge = 0.0                              # (on a link: along its line)
            ds = dodge - self.side[i]
            self.side[i] += float(np.clip(ds, -SW_SIDE_RATE * dt, SW_SIDE_RATE * dt))
            dv = v_want - self.v[i]
            a = min(dv / 0.6, SW_ACC) if dv > 0 else max(dv / dt, -(SW_HARD if v_want < 0.1 else SW_BRAKE))
            self.v[i] = max(0.0, self.v[i] + a * dt)
            self.effort[i] = 0.6 if a > 0.2 else (0.3 if self.v[i] > 0.3 else 0.0)
            self.look[i] = look
            self._advance(i, dt)
            self.foot[i] = 1.0 if self.v[i] < 0.15 else 0.0
            self.wait[i] = self.wait[i] + dt if self.v[i] < 0.15 else 0.0
            if self.wait[i] > 60.0 and self.mode[i] == LINK and self.u[i] < 1.0:   # (a light that never lets them go)
                self._back(i)

    def _link_speed(self, i, lk, u, cars_xy, cars_v, cars_yaw, cars_r, t):
        """The speed rider i wants at `u` on link `lk` (u < 0: still on their sidewalk, -u m before its end): round a
        corner slowly; before a road to cross: its light, or the look (careful riders slow down to SW_LOOK_V and wait
        for a gap)."""
        if lk["kind"] == "turn":
            return math.sqrt(SW_TURN_V ** 2 + 2 * SW_BRAKE * max(-u, 0.0))
        to_kerb = lk["u_in"] - SW_KERB - u
        if u > lk["u_in"] - 0.3:                         # in the road: get over
            return max(self.cruise[i], 2.5)
        stop = math.sqrt(2 * SW_BRAKE * max(to_kerb, 0.0))
        if lk["signal"]:
            if self.v[i] ** 2 / (2 * SW_HARD) > to_kerb + SW_KERB - 0.2:
                return self.cruise[i]                    # too close to stop before the road: on over it
            J = self.town.junctions[lk["junction"]]
            st, left = J.ped_signal(lk["phase"], t, self.cfg.ped_flash)
            need = (lk["u_out"] - u + 1.0) / max(self.cruise[i], 2.5)
            self.held[i] = to_kerb <= 3.0 and self._gap(lk, cars_xy, cars_v, cars_yaw, cars_r) <= 0.0
            if st == PED_GREEN and left > need + 1.0 and not self.held[i]:
                return self.cruise[i]
            if not self.red_run[i]:
                return stop
        gap = self._gap(lk, cars_xy, cars_v, cars_yaw, cars_r)
        if self.careful[i]:
            v = min(self.cruise[i], math.sqrt(SW_LOOK_V ** 2 + 2 * SW_BRAKE * max(to_kerb, 0.0)))
            if to_kerb < 3.0 and gap < SW_GAP_CAREFUL:
                return stop
            return v
        if to_kerb < 4.0 and gap < SW_GAP_CARELESS:
            hard = math.sqrt(2 * SW_HARD * max(to_kerb + SW_KERB - 0.2, 0.0))
            return hard if hard < self.v[i] + 0.5 else stop
        return self.cruise[i]

    def _advance(self, i, dt):
        town = self.town
        step = float(self.v[i]) * dt
        if self.mode[i] == RIDE:
            w = town.walkways[self.w[i]]
            L = w["poly"].length
            s = self.s[i] + self.dir[i] * step
            if s < 0.0 or s > L:
                lk = self.link[i]
                if isinstance(lk, dict):
                    self.mode[i], self.u[i] = LINK, abs(s - (L if s > L else 0.0))
                else:
                    self._back(i)
                    s = min(max(s, 0.0), L)
                    self.s[i] = s
                    p, h = _line_pt(town, w, s)
                    self._pose(i, p, h + (0.0 if self.dir[i] > 0 else math.pi), dt)
                    return
            else:
                self.s[i] = s
                p, h = _line_pt(town, w, s)
                self._pose(i, p, h + (0.0 if self.dir[i] > 0 else math.pi), dt)
                return
        lk = self.link[i]
        self.u[i] += step
        if self.u[i] >= lk["poly"].length:
            w2, e2 = lk["to"]
            extra = self.u[i] - lk["poly"].length
            L2 = town.walkways[w2]["poly"].length
            self.w[i], self.dir[i] = w2, (1.0 if e2 == 0 else -1.0)
            self.s[i] = extra if e2 == 0 else L2 - extra
            self.mode[i], self.u[i], self.link[i], self.held[i] = RIDE, 0.0, None, 0.0
            p, h = _line_pt(town, town.walkways[w2], self.s[i])
            self._pose(i, p, h + (0.0 if self.dir[i] > 0 else math.pi), dt)
            return
        x, y = lk["poly"].interp1(self.u[i])
        h = lk["poly"].heading1(self.u[i])
        self._pose(i, np.array([x, y]), h, dt)

    def _back(self, i):
        """Turn round (at the end of a sidewalk with nowhere to go, or after waiting too long at a kerb)."""
        self.mode[i], self.u[i], self.link[i], self.held[i] = RIDE, 0.0, None, 0.0
        self.dir[i] *= -1.0
        self.wait[i] = 0.0

    def _pose(self, i, p, h, dt):
        if self.side[i]:                                 # moved over (to their left: +)
            p = p + np.array([-math.sin(h), math.cos(h)]) * self.side[i]
        dyaw = float(wrap(h - self.yaw[i]))
        rate = dyaw / max(dt, 1e-3)
        self.xy[i] = p
        self.yaw[i] = h
        v = float(self.v[i])
        self.roll[i] = float(np.clip(math.atan(v * rate / 9.81), -0.4, 0.4))
        self.steer[i] = float(np.clip(math.atan(1.08 * rate / max(v, 0.5)), -0.5, 0.5))
        self.wheel[i] = (self.wheel[i] + v * dt / 0.33) % (2 * math.pi)
        if self.effort[i] > 0.0:
            self.crank[i] = (self.crank[i] + v * dt / 4.8 * 2 * math.pi) % (2 * math.pi)
