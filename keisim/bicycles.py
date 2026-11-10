"""Bicycles (TrafficConfig.bicycles): riders along the left edge of the road.

Where they ride. A bicycle is a vehicle (軽車両): it keeps to the left edge of the road surface, BIKE_EDGE from it (in
the 路側帯 where the road has one), on its own track beside the kerb lane (bike_track), goes straight on or turns left
at junctions (a right turn would be a 二段階右折, which they do not do yet: it is taken only where a road offers
nothing else), obeys the cars' lights and 止まれ lines and gives way where a car on its lane would.

How they move. Each rider has a type (city bike, sport bike, slow rider) with a cruising speed, a mass, a rolling
and an air resistance, a pedalling power and gears. The speed comes from the forces: pedalling (at most the power
allows, and the push of the legs from a standstill), coasting (only the resistances slow the bike down, so it rolls
a long way) and braking. A rider stops pedalling well before a red light and rolls up to it, brakes late, puts the
left foot down, and sets off after a moment of reaction in a low gear. Bends are taken at the speed the lean allows.
They filter slowly past a queue of standing cars where there is room on the left, follow what they cannot pass, and
move out to the right round someone walking in the 路側帯 or a slower rider after a look over the shoulder, when no
car is coming up from behind.

What the renderers get (scene): the lean (from speed and yaw rate, with the foot-down lean at a stop), the steering
angle, the crank angle (it turns only while the rider pedals, at the cadence the gear gives), the wheel angle, the
foot on the ground, the turn of the head, and the effort.

Everything here runs only with TrafficConfig.bicycles > 0.
"""
from __future__ import annotations

import math

import numpy as np

from .config import TL_RED, TL_YELLOW, TrafficConfig
from .geometry import cumlen, curvature_of, headings_of, wrap

BIKE_LEN, BIKE_W = 1.8, 0.6
BIKE_R = 0.3                # the circles that cover a bike (NCB of them along it)
NCB = 3
BIKE_EDGE = 0.5             # m: the bike's line from the edge of the road surface
BIKE_OWNER = 1_000_000      # owner ids of bike circles in the bikes' own corridor check
G = 9.81
RHO = 1.2
WHEELBASE = 1.08

# rider types: share, cruising speed (m/s), push from a standstill (m/s^2), peak power (W), mass with the bike (kg),
# comfortable braking (m/s^2), drag area CdA (m^2), rolling resistance, metres per crank turn of each gear, the
# cadence band (turns/s) the rider keeps to by shifting, wheel radius, lateral acceleration in bends
TYPES = (
    dict(name="city", p=0.62, cruise=(3.3, 4.7), a0=(0.9, 1.3), pmax=(160.0, 240.0), mass=88.0, b=1.4, cda=0.60,
         crr=0.008, gears=(3.6, 4.8, 6.2), cad=(0.75, 1.15), wheel=0.33, alat=(1.2, 1.7)),
    dict(name="sport", p=0.22, cruise=(5.0, 6.6), a0=(1.2, 1.7), pmax=(300.0, 420.0), mass=80.0, b=1.8, cda=0.42,
         crr=0.005, gears=(2.9, 3.7, 4.6, 5.7, 7.0), cad=(1.15, 1.6), wheel=0.34, alat=(1.7, 2.4)),
    dict(name="slow", p=0.16, cruise=(2.5, 3.3), a0=(0.6, 0.9), pmax=(90.0, 140.0), mass=85.0, b=1.1, cda=0.62,
         crr=0.009, gears=(3.6, 4.8), cad=(0.6, 1.0), wheel=0.33, alat=(0.9, 1.3)),
)
B_MAX = 4.5                 # m/s^2: the hardest braking
TAU_V = 1.1                 # s: how fast the rider closes in on the speed they want
OUT = 0.85                  # m: how far a rider moves out to the right round someone in their way
OUT_RATE = 0.7              # m/s sideways (and at most 0.22 of the speed)
OUT_BLEND = 6.0             # m along the road for the planned sideways move
MARGIN_CAR = 0.2            # m beside a car (filtering past a queue needs this much room)
MARGIN_BIKE = 0.25
MARGIN_WALK = 0.25          # someone walking along the road
MARGIN_CROSS = 0.55         # someone crossing it
MARGIN_TURN = 0.15          # where a turning car will be
FILTER_SIDE = 0.9           # m beside a car: closer than this the rider passes it at most FILTER_DV faster
FILTER_DV = 2.5             # m/s
S0_BIKE, S0_CAR = 0.8, 1.4  # m: standing distance behind people and bikes / behind cars
T_HEAD = 0.9                # s: time gap when following
LOOK_BACK = 0.6             # s: the look over the shoulder before moving out
STUCK_S = 45.0              # s: a rider standing this long out of the ego's sight is put elsewhere

SIG, SIGN, YIELD, ZEBRA = range(4)
PEDAL, COAST, BRAKE = range(3)

# colours are B, G, R like the cars' and the pedestrians'
FRAME_COLORS = np.array([[200, 200, 205], [35, 35, 38], [235, 235, 235], [40, 40, 190], [110, 60, 30], [190, 200, 120],
                         [40, 190, 220], [60, 110, 50], [150, 90, 170], [90, 140, 230]], float)
SPORT_COLORS = np.array([[35, 35, 38], [40, 40, 200], [235, 235, 235], [200, 120, 30], [60, 200, 230],
                         [70, 170, 60]], float)


# ----------------------------------------------------------------------------
# the bike's line along a lane
# ----------------------------------------------------------------------------
class BikeTrack:
    """A lane's bicycle line: pts, cum (arc length along it), lane_s (the lane's arc length at each point), heading
    (unwrapped), curv, length."""
    __slots__ = ("pts", "cum", "lane_s", "heading", "curv", "length", "off")

    def at(self, lane_s):
        """Arc length along the track at the lane's arc length `lane_s`."""
        return float(np.interp(lane_s, self.lane_s, self.cum))

    def lane_at(self, u):
        return float(np.interp(u, self.cum, self.lane_s))


def edge_offset(town, lane):
    """Road lanes: from the lane's centre line to the bike's line on its left (m)."""
    if lane.kind == "road" and lane.road >= 0:
        return lane.width / 2 + town.roads[lane.road].spec.shoulder - BIKE_EDGE
    return max(lane.width / 2 - 0.3, 0.0)


def bike_track(town, lid) -> BikeTrack:
    """The bicycle line of lane `lid` (cached on the town): the lane's centre line moved left to BIKE_EDGE from the
    road's edge; on a connector from the entry road's offset to the exit road's (smoothly), and never farther inside
    a bend than 0.6 of its radius."""
    cache = town.__dict__.setdefault("_bike_tracks", {})
    tr = cache.get(lid)
    if tr is not None:
        return tr
    lane = town.lanes[lid]
    n = len(lane.pts)
    if lane.kind == "road":
        off = np.full(n, edge_offset(town, lane))
    else:
        d0 = edge_offset(town, town.lanes[lane.pred[0]]) if lane.pred else edge_offset(town, lane)
        d1 = edge_offset(town, town.lanes[lane.succ[0]]) if lane.succ else d0
        x = lane.s / max(lane.length, 1e-6)
        off = d0 + (d1 - d0) * (x * x * (3.0 - 2.0 * x))
        lim = 0.6 / np.maximum(lane.curv, 1e-6)                 # left bends: stay well outside the centre of the arc
        if (lim < off).any():
            off = np.minimum(off, lim)
            m = min(9, n)
            ker = np.ones(m) / m
            sm = np.convolve(np.pad(off, m // 2, mode="edge"), ker, mode="valid")[:n]
            keep = np.minimum(1.0, np.minimum(lane.s, lane.length - lane.s) / 2.0)   # the ends stay on the roads' lines
            off = np.minimum(off, sm * keep + off * (1.0 - keep))
    nl = np.stack([-np.sin(lane.heading), np.cos(lane.heading)], -1)
    pts = lane.pts + nl * off[:, None]
    tr = BikeTrack()
    tr.pts = pts
    tr.off = off
    tr.cum = cumlen(pts)
    tr.lane_s = lane.s
    tr.heading = np.unwrap(headings_of(pts))
    k = curvature_of(pts)
    if len(k) > 7:
        k = np.convolve(np.pad(k, 3, mode="edge"), np.ones(7) / 7.0, mode="valid")
    tr.curv = k
    tr.length = float(tr.cum[-1])
    cache[lid] = tr
    return tr


def bike_lanes(town):
    """The lanes bicycles ride beside: the kerb lanes of the roads (cached)."""
    out = town.__dict__.get("_bike_lanes")
    if out is None:
        out = [l.id for l in town.lanes if l.kind == "road" and l.lane_index == 0 and l.road >= 0 and l.length > 12.0]
        town.__dict__["_bike_lanes"] = out
    return out


def bike_circles(xy, yaw):
    """NCB circles of radius BIKE_R along each bike -> centres (n * NCB, 2), owner (n * NCB,)."""
    n = len(xy)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0, int)
    offs = np.linspace(-(BIKE_LEN / 2 - BIKE_R), BIKE_LEN / 2 - BIKE_R, NCB)
    c, s = np.cos(yaw), np.sin(yaw)
    cx = xy[:, 0:1] + c[:, None] * offs[None]
    cy = xy[:, 1:2] + s[:, None] * offs[None]
    return np.stack([cx, cy], -1).reshape(-1, 2), np.repeat(np.arange(n), NCB)


class _Cat:
    """The tracks of a rider's next lanes in one piece: pts, cum (0 at the start of the current lane), head, curv,
    stops [(u, lane, kind, data)], first (the length of the current lane's track)."""
    __slots__ = ("key", "pts", "cum", "head", "curv", "stops", "first")


class Bicycles:
    P = 48          # points of a rider's lookahead: 24 every 0.5 m, then 24 every 1 m

    def __init__(self, town, cfg: TrafficConfig, rng, n, avoid_xy=None, avoid_r=10.0, car_xy=None, car_r=None,
                 ahead=None):
        """n riders on random kerb lanes, clear of avoid_xy (the ego) and of the car circles car_xy / car_r.
        ahead: (lane, s0, s1) puts one of them on that stretch (in front of the ego)."""
        self.town, self.cfg, self.rng = town, cfg, rng
        self.n = 0
        self.route: list[list[int]] = []
        self._cat: list = []
        z = lambda *shape, dt=float: np.zeros(shape, dt)      # noqa: E731
        self.ri = z(0, dt=int)
        for name in ("u", "v", "acc", "lat", "lat_t", "dwell", "crank", "wheel", "roll", "steer", "foot", "look",
                     "effort", "wob", "wob_f", "cruise", "a0", "pmax", "alat", "react", "wait", "stuck", "yaw",
                     "height", "lane_s", "shift_t", "look_t", "yawrate", "out", "stood", "gap", "v_want"):
            setattr(self, name, z(0))
        self.type = z(0, dt=int)
        self.lane = z(0, dt=int)            # the lane each rider is beside (route[ri])
        self.side = z(0)                    # m: how far left of that lane's centre line the rider is
        self.mode = z(0, dt=int)
        self.gear = z(0, dt=int)
        self.ss_done = z(0, dt=int)
        self.xy = z(0, 2)
        self.colors = z(0, 4, 3)            # frame, shirt, pants, skin (B, G, R)
        lanes = bike_lanes(town)
        if not lanes or n <= 0:
            return
        w = np.array([town.lanes[l].length for l in lanes])
        w = w / w.sum()
        if ahead is not None:
            lid, s0, s1 = ahead
            if s1 > s0:
                self._try_add(lid, float(rng.uniform(s0, s1)), avoid_xy, avoid_r, car_xy, car_r)
        tries = 0
        while self.n < n and tries < n * 30:
            tries += 1
            lid = lanes[rng.choice(len(lanes), p=w)]
            L = town.lanes[lid].length
            self._try_add(lid, float(rng.uniform(3.0, L - 3.0)), avoid_xy, avoid_r, car_xy, car_r)

    # --------------------------------------------------------------- spawning
    def _free(self, p, avoid_xy, avoid_r, car_xy, car_r, skip=None):
        if avoid_xy is not None and len(avoid_xy) and np.min(np.hypot(*(avoid_xy - p).T)) < avoid_r:
            return False
        if car_xy is not None and len(car_xy) and np.any(np.hypot(*(car_xy - p).T) < car_r + BIKE_R + 0.45):
            return False
        if self.n:
            d = np.hypot(*(self.xy - p).T)
            if skip is not None:
                d[skip] = np.inf
            if np.min(d) < 9.0:
                return False
        return True

    def _try_add(self, lid, s, avoid_xy, avoid_r, car_xy, car_r):
        town, rng = self.town, self.rng
        tr = bike_track(town, lid)
        u = tr.at(s)
        j = int(np.clip(np.searchsorted(tr.cum, u), 1, len(tr.cum) - 1))
        p = tr.pts[j]
        if not self._free(p, avoid_xy, avoid_r, car_xy, car_r):
            return False
        ty = int(rng.choice(len(TYPES), p=[t["p"] for t in TYPES]))
        T = TYPES[ty]
        cruise = float(rng.uniform(*T["cruise"]))
        frame = (SPORT_COLORS if ty == 1 else FRAME_COLORS)
        from .traffic import PANTS, SHIRTS, SKIN
        col = np.stack([frame[rng.integers(len(frame))] * rng.uniform(0.9, 1.05),
                        SHIRTS[rng.integers(len(SHIRTS))] * rng.uniform(0.85, 1.1),
                        PANTS[rng.integers(len(PANTS))], SKIN[rng.integers(len(SKIN))]])
        ap = lambda a, x: np.append(a, x)       # noqa: E731
        self.route.append([lid])
        self._cat.append(None)
        self.ri = ap(self.ri, 0)
        self.u = ap(self.u, u)
        self.v = ap(self.v, cruise * rng.uniform(0.6, 1.0))
        self.acc = ap(self.acc, 0.0)
        self.lat = ap(self.lat, 0.0)
        self.lat_t = ap(self.lat_t, 0.0)
        self.out = ap(self.out, OUT * rng.uniform(0.9, 1.15))
        self.dwell = ap(self.dwell, 1.0)
        self.crank = ap(self.crank, rng.uniform(0, 2 * math.pi))
        self.wheel = ap(self.wheel, rng.uniform(0, 2 * math.pi))
        self.roll = ap(self.roll, 0.0)
        self.steer = ap(self.steer, 0.0)
        self.foot = ap(self.foot, 0.0)
        self.look = ap(self.look, 0.0)
        self.look_t = ap(self.look_t, 0.0)
        self.effort = ap(self.effort, 0.3)
        self.wob = ap(self.wob, rng.uniform(0, 2 * math.pi))
        self.wob_f = ap(self.wob_f, rng.uniform(0.55, 0.9))
        self.cruise = ap(self.cruise, cruise)
        self.a0 = ap(self.a0, rng.uniform(*T["a0"]))
        self.pmax = ap(self.pmax, rng.uniform(*T["pmax"]))
        self.alat = ap(self.alat, rng.uniform(*T["alat"]))
        self.react = ap(self.react, rng.uniform(0.3, 1.1))
        self.wait = ap(self.wait, 0.0)
        self.stuck = ap(self.stuck, 0.0)
        self.stood = ap(self.stood, 0.0)
        self.gap = ap(self.gap, np.inf)             # to whoever is in the way (inf: nobody), for tests and plots
        self.v_want = ap(self.v_want, 0.0)          # the speed the rider wants
        self.yaw = ap(self.yaw, float(tr.heading[j]))
        self.yawrate = ap(self.yawrate, 0.0)
        self.height = ap(self.height, rng.uniform(1.5, 1.82))
        self.lane_s = ap(self.lane_s, s)
        self.lane = ap(self.lane, lid)
        self.side = ap(self.side, float(np.interp(s, tr.lane_s, tr.off)))
        self.shift_t = ap(self.shift_t, 0.0)
        self.type = ap(self.type, ty)
        self.mode = ap(self.mode, PEDAL)
        self.gear = ap(self.gear, len(T["gears"]) // 2)
        self.ss_done = ap(self.ss_done, -1)
        self.xy = np.vstack([self.xy, p])
        self.colors = np.concatenate([self.colors, np.clip(col, 0, 255)[None]])
        self.n += 1
        return True

    def _extend(self, i):
        """At least three lanes beyond the current one on rider i's route: straight on or left at each junction, on
        to a kerb lane (a right turn only where there is nothing else)."""
        town, rng = self.town, self.rng
        lanes = town.lanes
        r, k = self.route[i], self.ri[i]
        while len(r) - k < 4:
            last = lanes[r[-1]]
            if not last.succ:
                break
            if last.kind != "road":
                r.append(last.succ[0])
                continue
            ok = [c for c in last.succ if lanes[c].kind == "conn" and lanes[c].succ
                  and lanes[lanes[c].succ[0]].kind == "road" and lanes[lanes[c].succ[0]].lane_index == 0]
            if not ok:
                break
            pick = [c for c in ok if lanes[c].turn != "right"] or ok
            wts = np.array([1.5 if lanes[c].turn == "straight" else 1.0 for c in pick])
            r.append(pick[rng.choice(len(pick), p=wts / wts.sum())])
        if k > 2:
            del r[: k - 1]
            self.ri[i] -= k - 1

    def respawn(self, i, avoid_xy, car_xy, car_r):
        town, rng = self.town, self.rng
        lanes = bike_lanes(town)
        for _ in range(40):
            lid = lanes[rng.integers(len(lanes))]
            tr = bike_track(town, lid)
            u = float(rng.uniform(3.0, tr.length - 3.0))
            j = int(np.clip(np.searchsorted(tr.cum, u), 1, len(tr.cum) - 1))
            p = tr.pts[j]
            if not self._free(p, avoid_xy, 60.0, car_xy, car_r, skip=i):
                continue
            self.route[i] = [lid]
            self._cat[i] = None
            self.ri[i] = 0
            self.u[i] = u
            self.v[i] = 0.0
            self.lat[i] = self.lat_t[i] = 0.0
            self.xy[i] = p
            self.yaw[i] = float(tr.heading[j])
            self.lane_s[i] = tr.lane_at(u)
            self.lane[i] = lid
            self.side[i] = float(np.interp(self.lane_s[i], tr.lane_s, tr.off))
            self.stuck[i] = 0.0
            self.ss_done[i] = -1
            return True
        return False

    # ------------------------------------------------------------------ paths
    def _cat_of(self, i):
        r, k = self.route[i], self.ri[i]
        if len(r) - k < 4:
            self._extend(i)
            r, k = self.route[i], self.ri[i]
        key = tuple(r[k:k + 4])
        cat = self._cat[i]
        if cat is not None and cat.key == key:
            return cat
        town = self.town
        lanes = town.lanes
        cat = _Cat()
        cat.key = key
        pts, cum, head, curv, stops = [], [], [], [], []
        base = 0.0
        h_off = 0.0
        for j, lid in enumerate(key):
            tr = bike_track(town, lid)
            lane = lanes[lid]
            sl = slice(1, None) if j else slice(None)
            if j:                                   # keep the heading continuous over the joins
                h_off += 2 * math.pi * round((head[-1][-1] - (tr.heading[0] + h_off)) / (2 * math.pi))
            else:
                cat.first = tr.length
            pts.append(tr.pts[sl])
            cum.append(base + tr.cum[sl])
            head.append(tr.heading[sl] + h_off)
            curv.append(tr.curv[sl])
            if lane.stop_s is not None:
                stops.append((base + tr.at(lane.stop_s), lid, SIG, None))
            if lane.stop_sign_s is not None:
                stops.append((base + tr.at(lane.stop_sign_s), lid, SIGN, None))
            if lane.kind == "conn" and getattr(lane, "yields", None) and getattr(lane, "wait_s", None) is not None:
                stops.append((base + tr.at(lane.wait_s), lid, YIELD, None))
            for ci, a0, _, u0, u1 in town.lane_crossings.get(lid, ()):
                stops.append((base + tr.at(a0), lid, ZEBRA, (ci, u0, u1)))
            base += tr.length
        cat.pts = np.concatenate(pts)
        cat.cum = np.concatenate(cum)
        cat.head = np.concatenate(head)
        cat.curv = np.concatenate(curv)
        cat.stops = sorted(stops, key=lambda e: e[0])
        self._cat[i] = cat
        return cat

    @staticmethod
    def _at(cat, u):
        """Point, heading and curvature of the track at arc length u."""
        cum = cat.cum
        j = int(min(max(np.searchsorted(cum, u, "right") - 1, 0), len(cum) - 2))
        f = min(max((u - cum[j]) / max(cum[j + 1] - cum[j], 1e-9), 0.0), 1.0)
        p = cat.pts[j] + f * (cat.pts[j + 1] - cat.pts[j])
        h = cat.head[j] + f * (cat.head[j + 1] - cat.head[j])
        k = cat.curv[j] + f * (cat.curv[j + 1] - cat.curv[j])
        return p, float(h), float(k)

    def circles(self, predict=False):
        """The bikes as circles for the cars' corridor checks -> centres, radii, owner (bike index), speed, yaw.
        predict: also where each moving bike will be in 0.8, 1.6 and 2.4 s along its track (one circle each), so
        that a car turning across a bike's line sees it coming."""
        cxy, own = bike_circles(self.xy, self.yaw)
        if predict and self.n:
            ex, eo = [], []
            for i in np.nonzero(self.v > 0.5)[0]:
                cat = self._cat[i]
                if cat is None:
                    continue
                for tau in (0.8, 1.6, 2.4):
                    p, h, _ = self._at(cat, self.u[i] + self.v[i] * tau)
                    ex.append(p + np.array([math.sin(h), -math.cos(h)]) * self.lat[i])
                    eo.append(i)
            if ex:
                cxy = np.concatenate([cxy, np.array(ex)])
                own = np.concatenate([own, np.array(eo, int)])
        return cxy, np.full(len(cxy), BIKE_R), own, self.v[own], self.yaw[own]

    def visible_from(self, cam, yaw, half_fov, max_dist, town=None):
        """The riders a camera at `cam` looking along `yaw` can see (as Traffic.visible_from, by their centres)."""
        if self.n == 0:
            return np.zeros(0, bool)
        cam = np.asarray(cam, float)
        rel = self.xy - cam
        seen = (np.abs(wrap(np.arctan2(rel[:, 1], rel[:, 0]) - yaw)) < half_fov) & (np.hypot(*rel.T) < max_dist)
        if town is not None and seen.any() and len(town.buildings):
            from .geometry import rays_clear
            edges, _, _ = town.building_edges()
            i = np.nonzero(seen)[0]
            seen[i] = rays_clear(cam, self.xy[i], edges)
        return seen

    def add_to_tables(self, on_lane, visible=None):
        """Traffic.yield_tables: the riders under ("bike", lane) -> [(s on the lane, v, length, cruising speed,
        next lane)], for Traffic.min_arrival: whoever gives way to a car on that lane gives way to them too."""
        for i in range(self.n):
            if visible is not None and not visible[i]:
                continue
            r, k = self.route[i], self.ri[i]
            on_lane.setdefault(("bike", r[k]), []).append(
                (float(self.lane_s[i]), float(self.v[i]), BIKE_LEN, float(self.cruise[i]),
                 r[k + 1] if k + 1 < len(r) else None))

    def scene(self):
        """For the renderers: xy, yaw, pose (n, 7: roll (+ = leaning left), steer (+ = left), crank angle (0 = right
        pedal on top, turning forwards), wheel angle, foot (1 = left foot on the ground), look (-1 = head turned
        back over the right shoulder, +1 = left), effort 0..1), kind (0 city, 1 sport, 2 slow), height (the rider's),
        colors (n, 4, 3: frame, shirt, pants, skin; B, G, R)."""
        roll = self.roll + np.where(self.type == 1, 0.12, 0.07) * self.foot     # standing: tipped onto that foot
        pose = np.stack([roll, self.steer, self.crank, self.wheel, self.foot, self.look, self.effort], -1)
        return self.xy, self.yaw, pose, self.type, self.height, self.colors

    # ------------------------------------------------------------------- step
    def step(self, dt, t, traffic, peds, ego_xy, ego_v, ego_yaw, ego_circ, ego_info=None, ego_pred=None):
        """ego_circ: (centres, radii) of the ego's circles; ego_pred: (k, 2) where the ego will be shortly, when it
        is about to turn (World.step)."""
        n = self.n
        if n == 0:
            return
        town, cfg = self.town, self.cfg
        lanes = town.lanes
        P = self.P
        # ---- lookahead paths: along the home line and moved out
        cats = [self._cat_of(i) for i in range(n)]
        idx = np.concatenate([np.arange(24), 24 + 2 * np.arange(P - 24)])
        pH = np.zeros((n, P, 2))
        pO = np.zeros((n, P, 2))
        rel = np.zeros((n, P))
        valid = np.zeros((n, P), bool)
        curv = np.zeros((n, P))
        look = 12.0 + 3.0 * self.v
        for i in range(n):
            cat = cats[i]
            j0 = int(np.searchsorted(cat.cum, self.u[i]))
            jj = np.minimum(j0 + idx, len(cat.cum) - 1)
            d = cat.cum[jj] - self.u[i]
            h = cat.head[jj]
            nr = np.stack([np.sin(h), -np.cos(h)], -1)
            base = cat.pts[jj]
            x = np.clip(d / OUT_BLEND, 0.0, 1.0)
            lat = self.lat[i]
            pH[i] = base + nr * (lat * (1.0 - x))[:, None]
            pO[i] = base + nr * (lat + (self.out[i] - lat) * x)[:, None]
            rel[i] = d
            valid[i] = (d <= look[i]) & (j0 + idx < len(cat.cum))
            curv[i] = cat.curv[jj]
        valid[:, 0] = True
        relc = np.where(valid, np.maximum(rel, 0.0), 0.0)
        # ---- obstacles
        from .traffic import corridor_gaps
        tr = traffic
        cxy, cr, cow = tr.circles()
        csp, cyw = tr.v[cow], tr.yaw[cow]
        ncar = len(cxy)
        parts = [(cxy, cr, cow, csp, cyw, np.full(ncar, MARGIN_CAR))]
        exy, er = ego_circ
        parts.append((exy, er, np.full(len(exy), -1), np.full(len(exy), ego_v), np.full(len(exy), ego_yaw),
                      np.full(len(exy), MARGIN_CAR)))
        # where turning cars will be (the cars on a turn's connector or about to enter one)
        t_xy, t_r, t_o, t_v, t_yaw, t_car = self._turning(tr)
        if ego_pred is not None and len(ego_pred):
            m = len(ego_pred)
            t_xy = np.concatenate([t_xy, ego_pred])
            t_r = np.concatenate([t_r, np.full(m, 0.95)])
            t_o = np.concatenate([t_o, np.full(m, -1)])
            t_v = np.concatenate([t_v, np.full(m, ego_v)])
            t_yaw = np.concatenate([t_yaw, np.full(m, ego_yaw)])
            t_car = np.concatenate([t_car, np.tile(ego_xy, (m, 1))])
        n_turn = len(t_xy)
        parts.append((t_xy, t_r, t_o, t_v, t_yaw, np.full(n_turn, MARGIN_TURN)))
        bxy, bown = bike_circles(self.xy, self.yaw)
        parts.append((bxy, np.full(len(bxy), BIKE_R), BIKE_OWNER + bown, self.v[bown], self.yaw[bown],
                      np.full(len(bxy), MARGIN_BIKE)))
        if peds is not None and peds.n:
            cross = peds.mode == peds.CROSS
            pm = np.where(cross, MARGIN_CROSS, MARGIN_WALK)
            parts.append((peds.xy, np.full(peds.n, 0.35), -2 - np.arange(peds.n), peds.vel, peds.yaw, pm))
            if cross.any():
                ci = np.nonzero(cross)[0]
                vv = peds.velocity_vectors()[ci]
                for tau in (0.8, 1.6):
                    parts.append((peds.xy[ci] + vv * tau, np.full(len(ci), 0.35), -2 - ci, peds.vel[ci],
                                  peds.yaw[ci], np.full(len(ci), MARGIN_CROSS)))
        oxy, orr, oow, osp, oyw, omg = (np.concatenate([p[k] for p in parts]) for k in range(6))
        margin = np.broadcast_to(omg[None], (n, len(oxy))).copy()
        if n_turn:
            # a turning car's future does not count for a rider it comes up behind (the rider would brake for a car
            # that is still behind them): only for the riders the car is beside or ahead of
            a = ncar + len(exy)
            u = np.stack([np.cos(self.yaw), np.sin(self.yaw)], -1)
            lon = ((t_car[None] - self.xy[:, None]) * u[:, None]).sum(-1)
            margin[:, a:a + n_turn] = np.where(lon > -1.5, MARGIN_TURN, -np.inf)
        hl = np.full(n, BIKE_LEN / 2)
        hw = np.full(n, BIKE_W / 2)
        me = BIKE_OWNER + np.arange(n)
        # the home line with every rider on it (for moving back in), then the two paths for the rider's speed. There,
        # of two riders going the same way, one does not count for the other (two riders standing side by side,
        # one of them half moved out, would otherwise wait for each other for ever): a rider a length behind; of two
        # riders side by side the one further out (it gives way: the rider on the line goes first), and with both
        # on the same line the one behind
        gHf, _, _ = corridor_gaps(pH, relc, valid, self.xy, self.yaw, hl, hw, oxy, orr, oow, osp, oyw, me, margin)
        ub = np.stack([np.cos(self.yaw), np.sin(self.yaw)], -1)
        rel_ = self.xy[None] - self.xy[:, None]
        du = (rel_ * ub[:, None]).sum(-1)                                   # [i, j]: j ahead of i by
        rgt = (rel_ * np.stack([ub[:, 1], -ub[:, 0]], -1)[:, None]).sum(-1)       # [i, j]: j to the right of i by
        par = np.cos(self.yaw[None] - self.yaw[:, None]) > 0.5
        dlat = self.lat[None] - self.lat[:, None]                           # [i, j]: j further out than i by
        ar = np.arange(n)
        lower = np.where(np.abs(dlat) > 0.25, dlat > 0.0,
                         np.where(np.abs(du) > 0.15, du < 0.0, ar[None] > ar[:, None]))
        ignore = par & ((du <= -BIKE_LEN) | ((np.abs(du) < BIKE_LEN) & lower))
        a = ncar + len(exy) + n_turn
        margin[:, a:a + len(bxy)] = np.where(ignore[:, bown], -np.inf, MARGIN_BIKE)
        # riders between this one and its home line, level with it or not yet a length behind: no moving back in
        # on top of the rider just passed (the home path only reaches the line some metres ahead)
        dv = np.maximum(self.v[None] - self.v[:, None], 0.0)
        left = par & (rgt < 0.3) & (rgt > -(self.lat[:, None] + 0.7)) & (ar[None] != ar[:, None])
        beside = (left & (du > -(BIKE_LEN + 0.5 + dv)) & (du < BIKE_LEN + 0.4)).any(1)
        gH, vH, oH = corridor_gaps(pH, relc, valid, self.xy, self.yaw, hl, hw, oxy, orr, oow, osp, oyw, me, margin)
        gO, vO, oO = corridor_gaps(pO, relc, valid, self.xy, self.yaw, hl, hw, oxy, orr, oow, osp, oyw, me, margin)
        # the cars close beside the line (filtering): only the cars and the ego
        nv = ncar + len(exy)
        gF, vF, _ = corridor_gaps(pH, relc, valid, self.xy, self.yaw, hl, hw, oxy[:nv], orr[:nv], oow[:nv], osp[:nv],
                                  oyw[:nv], me, FILTER_SIDE)
        # vehicles for the look behind (cars and the ego)
        vxy = np.vstack([tr.xy, np.asarray(ego_xy, float)[None]]) if tr.n else np.asarray(ego_xy, float)[None]
        vv_ = np.append(tr.v, ego_v)
        tables = None
        # ---- each rider
        for i in range(n):
            T = TYPES[self.type[i]]
            cat = cats[i]
            v = float(self.v[i])
            u = float(self.u[i])
            r, k = self.route[i], self.ri[i]
            cur = lanes[r[k]]
            front = u + BIKE_LEN / 2
            # -- moving out / back in
            home = self.lat_t[i] < 0.01
            lead_slow = math.isfinite(gH[i]) and vH[i] < self.cruise[i] - 0.8 and (oH[i] <= -2 or oH[i] >= BIKE_OWNER)
            if home:
                want = lead_slow and gH[i] < 5.0 + 2.5 * v and gO[i] > gH[i] + 8.0 and cur.kind == "road" \
                    and cat.first - u > 14.0 and u > 3.0
                if want and self._rear_clear(i, vxy, vv_):
                    self.look_t[i] += dt
                    if self.look_t[i] >= LOOK_BACK:
                        self.lat_t[i] = self.out[i]
                        self.look_t[i] = 0.0
                else:
                    self.look_t[i] = max(0.0, self.look_t[i] - 2 * dt)
            else:
                self.look_t[i] = 0.0
                if (gHf[i] > max(9.0, 2.5 * v) or (gHf[i] >= gO[i] and gO[i] < 6.0)) and not beside[i]:
                    self.lat_t[i] = 0.0
            out = self.lat_t[i] > 0.01
            gap, lead_v, lead_o = (gO[i], vO[i], oO[i]) if out else (gH[i], vH[i], oH[i])
            # -- the speed the rider wants
            v_t = min(self.cruise[i], cur.speed_limit)
            b = T["b"]
            why_stop = False
            a_req = 0.0             # the braking it takes to stop where the rider has to (<= 0)
            # bends: the speed the lean allows, braking gently into them
            vc = np.sqrt(self.alat[i] / np.maximum(np.abs(curv[i]), 1e-3))
            env = np.where(valid[i], np.sqrt(vc * vc + 2.0 * 0.9 * relc[i]), np.inf)
            v_t = min(v_t, float(env.min()))
            # lights, 止まれ, giving way, zebras
            stop_d = math.inf
            for su, lid, kind, data in cat.stops:
                d = su - front
                if d > 40.0:
                    break
                if d < -0.5:
                    continue
                if kind == SIG:
                    st = town.signal_state_for_lane(lid, t)
                    if st == TL_RED or (st == TL_YELLOW and d > v * v / (2 * 2.5) + 0.5):
                        stop_d = min(stop_d, d - 0.3)
                elif kind == SIGN:
                    if self.ss_done[i] == lid:
                        continue
                    if v < 0.05 and d <= 2.0:
                        self.ss_done[i] = lid
                        continue
                    stop_d = min(stop_d, d - 0.4)
                elif kind == YIELD:
                    if v > 1.0 and d < v * v / (2 * 3.0):
                        continue
                    if tables is None:
                        tables = tr.yield_tables(ego_info)
                    if tr.must_yield(lanes[lid], t, tables):
                        stop_d = min(stop_d, max(d - 0.2, 0.0))
                elif peds is not None and peds.cw_on:
                    if not (v > 1.0 and d < v * v / (2 * B_MAX)) and peds.crosswalk_blocked(data[0], data[1], data[2], t):
                        stop_d = min(stop_d, d - 0.8)
            if math.isfinite(stop_d):
                v_t = min(v_t, math.sqrt(2.0 * b * max(stop_d, 0.0)))
                why_stop = stop_d < v * v / (2 * 0.35) + 3.0
                a_req = -v * v / (2.0 * max(stop_d, 0.05))
            # whoever is in the way
            if math.isfinite(gap):
                s0 = S0_CAR if 0 <= lead_o < BIKE_OWNER or lead_o == -1 else S0_BIKE
                room = max(gap - s0, 0.0)
                vl = max(float(lead_v), 0.0)
                v_obs = min(math.sqrt(vl * vl + 2.0 * b * room), vl + room / T_HEAD)
                if lead_v < 0.0:                                # coming towards the rider
                    v_obs = min(v_obs, math.sqrt(2.0 * b * max(room + 1.5 * float(lead_v), 0.0)))
                v_t = min(v_t, v_obs)
                why_stop = why_stop or (vl < 0.5 and gap < v * v / (2 * 0.35) + 3.0)
                if v > vl:
                    a_req = min(a_req, -(v * v - vl * vl) / (2.0 * max(room, 0.05)))
            # moved out with a junction close: no riding into it side by side (the two may part there, one across
            # the other) - the rider falls in behind whoever on its left it has not passed
            if self.lat[i] > 0.25 and (cur.kind != "road" or cat.first - u < 14.0):
                for j in np.nonzero(left[i] & (du[i] > -BIKE_LEN) & (du[i] < 8.0))[0]:
                    v_t = min(v_t, max(0.0, float(self.v[j]) + (du[i, j] - BIKE_LEN - S0_BIKE) / T_HEAD))
            # filtering past cars close beside the line
            if math.isfinite(gF[i]):
                v_t = min(v_t, math.sqrt((max(float(vF[i]), 0.0) + FILTER_DV) ** 2 + 2.0 * 1.0 * max(gF[i], 0.0)))
            # -- reaction before setting off
            if v < 0.05:
                self.stood[i] += dt
                if v_t > 0.4:
                    self.wait[i] += dt
                    if self.wait[i] < (self.react[i] if self.stood[i] > 1.0 else 0.15):
                        v_t = 0.0
                else:
                    self.wait[i] = 0.0
            else:
                self.wait[i] = 0.0
                if v > 0.5:
                    self.stood[i] = 0.0
            self.gap[i], self.v_want[i] = gap, v_t
            # -- forces
            mode_was = self.mode[i]
            m = T["mass"]
            a_coast = -(T["crr"] * G + 0.5 * RHO * T["cda"] * v * v / m) if v > 0.02 else 0.0
            a_cmd = (v_t - v) / TAU_V
            # riders brake late: once stopping in time takes most of their comfortable braking, they brake as hard
            # as it takes (the speed profile above alone, followed with a lag, would stop them too late)
            if a_req < -0.7 * b or (mode_was == BRAKE and a_req < -0.4 * b):
                a_cmd = min(a_cmd, a_req)
            a_cmd = max(a_cmd, -B_MAX)
            a_ped = min(self.a0[i], self.pmax[i] / (m * max(v, 0.6))) + a_coast     # the most pedalling gives
            mode = self.mode[i]
            self.dwell[i] += dt
            halt = v_t <= 0.0 and v < 1.2       # the last metre of a stop: on the brakes (no rolling on at a crawl)
            if mode == PEDAL:
                if a_cmd < a_coast - 0.03 or (why_stop and v > 1.6) or v_t <= 0.0:
                    mode = COAST if (a_cmd > a_coast - 0.45 and not halt) else BRAKE
            elif mode == COAST:
                if a_cmd <= a_coast - 0.45 or halt:
                    mode = BRAKE
                elif a_cmd > a_coast + 0.2 and self.dwell[i] > 0.5 and not (why_stop and v > 1.6) and v_t > 0.0:
                    mode = PEDAL
            else:
                if a_cmd > a_coast - 0.2 and self.dwell[i] > 0.3 and not halt:
                    mode = COAST if (a_cmd < a_coast + 0.2 or (why_stop and v > 1.6) or v_t <= 0.0) else PEDAL
            if mode != self.mode[i]:
                self.mode[i] = mode
                self.dwell[i] = 0.0
            if mode == PEDAL:
                a = min(a_cmd, a_ped)
                eff = max(0.0, (a - a_coast) * m * max(v, 0.6)) / self.pmax[i]
            elif mode == COAST:
                a, eff = a_coast, 0.0
            else:
                a, eff = min(a_cmd, a_coast, -0.9 if halt else 0.0), 0.0
            self.effort[i] += (min(eff, 1.0) - self.effort[i]) * min(1.0, dt / 0.3)
            v_new = max(0.0, v + a * dt)
            if v_new < 0.06 and a <= 0.0:
                v_new = 0.0
            self.acc[i] = a
            self.v[i] = v_new
            # -- along the track
            u += v_new * dt
            while u > cat.first:
                if k + 1 >= len(r):
                    u = cat.first
                    self.v[i] = v_new = 0.0
                    break
                u -= cat.first
                k += 1
                self.ri[i] = k
                cat = self._cat_of(i)
                r, k = self.route[i], self.ri[i]
            self.u[i] = u
            trk = bike_track(town, r[k])
            self.lane_s[i] = trk.lane_at(u)
            self.lane[i] = r[k]
            # -- sideways: the move out and back, and the wobble (larger when slow, none when standing)
            rate = min(OUT_RATE, 0.22 * v_new)
            dl = float(np.clip(self.lat_t[i] - self.lat[i], -rate * dt, rate * dt))
            self.lat[i] += dl
            self.wob[i] += 2 * math.pi * self.wob_f[i] * dt * min(1.0, v_new / 1.0)
            amp = (0.012 + 0.06 * math.exp(-v_new / 1.1)) * min(1.0, v_new / 0.5)
            wob = amp * math.sin(self.wob[i])
            dwob = amp * math.cos(self.wob[i]) * 2 * math.pi * self.wob_f[i] * min(1.0, v_new / 1.0)
            p, h, kap = self._at(cat, u)
            nr = np.array([math.sin(h), -math.cos(h)])
            self.xy[i] = p + nr * (self.lat[i] + wob)
            self.side[i] = float(np.interp(self.lane_s[i], trk.lane_s, trk.off)) - self.lat[i] - wob
            yaw = h - math.atan2(dl / dt + dwob, max(v_new, 1.5))
            # -- what the renderers show
            w = wrap(yaw - self.yaw[i]) / dt if v_new > 0.05 else 0.0
            self.yaw[i] = yaw
            self.yawrate[i] += (w - self.yawrate[i]) * min(1.0, dt / 0.25)
            lean = math.atan(v_new * self.yawrate[i] / G)
            self.roll[i] += (float(np.clip(lean, -0.55, 0.55)) - self.roll[i]) * min(1.0, dt / 0.2)
            steer = math.atan(WHEELBASE * w / max(v_new, 1.2)) if v_new > 0.05 else self.steer[i] * 0.9
            self.steer[i] += (float(np.clip(steer, -0.6, 0.6)) - self.steer[i]) * min(1.0, dt / 0.12)
            self.wheel[i] = (self.wheel[i] + v_new * dt / T["wheel"]) % (2 * math.pi)
            gears = T["gears"]
            g = min(int(self.gear[i]), len(gears) - 1)
            cad = v_new / gears[g]
            self.shift_t[i] += dt
            if self.shift_t[i] > 1.5:
                if cad > T["cad"][1] and g + 1 < len(gears):
                    g, self.shift_t[i] = g + 1, 0.0
                elif cad < T["cad"][0] and g > 0:
                    g, self.shift_t[i] = g - 1, 0.0
            self.gear[i] = g
            ph = self.crank[i]
            stopping = v_new < 0.9 and (a <= 0.0 or v_t <= 0.0)
            if mode == PEDAL and v_new > 0.05:
                ph += 2 * math.pi * v_new / gears[g] * dt
            else:               # cranks to level while coasting; the right pedal up and forward for the start at a stop
                goal = math.pi / 3 if (stopping or v_new < 0.05) else (math.pi / 2 if math.cos(ph - math.pi / 2) > 0
                                                                       else 3 * math.pi / 2)
                ph += float(np.clip(wrap(goal - ph), -3.0 * dt, 3.0 * dt))
            self.crank[i] = ph % (2 * math.pi)
            foot_t = 1.0 if (v_new < 0.05 or stopping) else 0.0
            self.foot[i] += float(np.clip(foot_t - self.foot[i], -dt / 0.45, dt / 0.3))
            look_t = -1.0 if self.look_t[i] > 0.0 else 0.0
            self.look[i] += float(np.clip(look_t - self.look[i], -dt / 0.2, dt / 0.25))
            self.stuck[i] = self.stuck[i] + dt if v_new < 0.1 else 0.0
        self._release(ego_xy, ego_yaw, cxy, cr)

    def _rear_clear(self, i, vxy, vv):
        """No car beside rider i or coming up from behind within the time it takes to move out and pass."""
        if len(vxy) == 0:
            return True
        yaw = self.yaw[i]
        d = vxy - self.xy[i]
        lon = d[:, 0] * math.cos(yaw) + d[:, 1] * math.sin(yaw)
        right = d[:, 0] * math.sin(yaw) - d[:, 1] * math.cos(yaw)
        reach = 6.0 + 3.5 * np.maximum(vv - self.v[i], 0.0) + 0.5 * vv
        if np.any((lon < 3.0) & (lon > -reach) & (right > -1.0) & (right < 3.6)):
            return False
        # the other riders: one beside on the right, or a faster one coming up within a few metres
        d = self.xy - self.xy[i]
        lon = d[:, 0] * math.cos(yaw) + d[:, 1] * math.sin(yaw)
        right = d[:, 0] * math.sin(yaw) - d[:, 1] * math.cos(yaw)
        reach = 2.0 + 2.5 * np.maximum(self.v - self.v[i], 0.0)
        near = (lon < 1.9) & (lon > -reach) & (right > -0.3) & (right < 2.0)
        near[i] = False
        return not near.any()

    def _turning(self, tr):
        """Where the cars that turn at a junction will be in 0.6, 1.2 and 1.8 s (one circle each): the cars moving on
        a turn's connector or within 12 m of one on their route -> xy, r, owner, v, yaw, the car's position now."""
        lanes = self.town.lanes
        xs, rs, os_, vs, ys, cs = [], [], [], [], [], []
        if tr.n and getattr(tr, "_cn", -1) == tr.n:
            for j in np.nonzero(tr.v > 0.5)[0]:
                r, k = tr.route[j], tr.ri[j]
                cur = lanes[r[k]]
                if cur.kind == "conn":
                    turn = cur.turn != "straight"
                else:
                    turn = k + 1 < len(r) and lanes[r[k + 1]].kind == "conn" and lanes[r[k + 1]].turn != "straight" \
                        and cur.length - tr.s[j] < 12.0
                if not turn:
                    continue
                off = tr.odo[j] - tr.c_odo0[j]
                for tau in (0.6, 1.2, 1.8):
                    q = int(np.searchsorted(tr.c_cum[j], off + tr.v[j] * tau))
                    if q >= tr.PC or not np.isfinite(tr.c_cum[j, q]):
                        break
                    xs.append(tr.c_pts[j, q])
                    rs.append(tr.dims[j, 1] / 2)
                    os_.append(j)
                    vs.append(tr.v[j])
                    ys.append(tr.yaw[j])
                    cs.append(tr.xy[j])
        if not xs:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros((0, 2))
        return np.array(xs), np.array(rs), np.array(os_, int), np.array(vs), np.array(ys), np.array(cs)

    def _release(self, ego_xy, ego_yaw, car_xy, car_r):
        """Riders standing for STUCK_S out of the ego's sight (or at the end of a road with no way on) go elsewhere."""
        cfg = self.cfg
        for i in np.nonzero(self.stuck > STUCK_S)[0]:
            d = self.xy[i] - ego_xy
            dist = float(np.hypot(*d))
            go = dist > 60.0
            if not go and ego_yaw is not None:
                bearing = abs(wrap(math.atan2(d[1], d[0]) - ego_yaw))
                go = dist > cfg.view_dist or (dist > 8.0 and bearing > math.radians(cfg.view_half_deg))
            if go:
                self.respawn(i, np.asarray(ego_xy, float)[None], car_xy, car_r)
