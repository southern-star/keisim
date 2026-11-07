"""Curve test courses: one open two-way road of left and right curves without junctions, to see how a driving model
keeps its lane where the road gives few cues (scripts/curve_test.py). The road runs east from a dead end: a lead
straight, then for each radius a 90 deg left curve, a straight, a 90 deg right curve and a straight, to a dead end;
driven back the other way every curve turns the other way. Variants differ only in the paint and the guardrails:

    lines          white centre line (dashed) and edge lines (the procedural towns' streets)
    edge           edge lines only, no centre line (KeiSim's residential roads)
    nolines        no paint at all (KeiSim's alleys have none either, but are narrower)
    nolines_guard  no paint, a guardrail (ガードレール) along both edges
    lines_guard    white lines and guardrails

Each variant also comes without buildings: <variant>_nobld (e.g. nolines_guard_nobld) has no KeiSim buildings and no
KeiView lots (houses, fields, parking), only the road, its guardrails and KeiView's utility poles in open grass.

Every variant has 3.0 m lanes, a 0.5 m shoulder and no sidewalk (the road surface, then grass or a building lot),
30 km/h. KeiView fills its extra lots (houses, fields) only within PARCEL_REACH of the road and draws no far-town
ring around the course: beyond, grass. The town is a Town.from_graph (buildings along the road as KeiSim places them, no signals), named
curve_<variant>_<COURSE_VERSION> (KeiView caches it under that name: bump the version when the geometry changes).
"""
from __future__ import annotations

import math

import numpy as np

from .config import town_config
from .geometry import Polyline
from .roadnet import RoadSpec, Town
from .route import Route

COURSE_VERSION = "v3"     # v3: KeiView lots kept off the road and behind the guardrails (KeiSim as v2)
RADII = (15.0, 25.0, 40.0, 60.0, 100.0, 150.0)     # m, centre line
LEAD = 70.0                          # m of straight from the start before the first curve (and after the last)
GAP = 50.0                           # m of straight between curves
DS = 0.5
PARCEL_REACH = 60.0                  # m: KeiView fills lots only this far from the road (grass beyond)

VARIANTS = {
    "lines": {"center": "dashed", "edge": "lane"},
    "edge": {"center": "none", "edge": "lane"},
    "nolines": {"center": "none", "edge": "none"},
    "nolines_guard": {"center": "none", "edge": "none", "guardrail": "both"},
    "lines_guard": {"center": "dashed", "edge": "lane", "guardrail": "both"},
}
NOBLD = "_nobld"                     # variant suffix: no buildings (the guardrails stay)
VARIANT_NAMES = list(VARIANTS) + [v + NOBLD for v in VARIANTS]


def split_variant(variant):
    """(the VARIANTS key, whether the course has buildings) of a variant name (VARIANT_NAMES)."""
    base = variant.removesuffix(NOBLD)
    if base not in VARIANTS:
        raise ValueError(f"unknown curve course variant {variant!r} (known: {VARIANT_NAMES})")
    return base, base == variant


def course_line(radii=RADII, lead=LEAD, gap=GAP):
    """Centre line of the course from its west end, and its curves: (pts (N, 2), [{s0, s1, radius, turn}]) with
    s along the centre line, turn "left" / "right" when driven from the start."""
    pts = [np.zeros(2)]
    h = 0.0
    curves = []
    s = 0.0

    def straight(length):
        nonlocal s
        n = max(1, int(round(length / DS)))
        d = np.array([math.cos(h), math.sin(h)])
        for k in range(1, n + 1):
            pts.append(pts[-1] + d * (length / n))
        s += length

    def arc(r, turn):
        nonlocal h, s
        sign = 1.0 if turn == "left" else -1.0
        c = pts[-1] + r * np.array([-math.sin(h), math.cos(h)]) * sign
        n = max(4, int(round(r * math.pi / 2 / DS)))
        a0 = h - sign * math.pi / 2                 # angle of the start point seen from the centre
        for k in range(1, n + 1):
            a = a0 + sign * (math.pi / 2) * k / n
            pts.append(c + r * np.array([math.cos(a), math.sin(a)]))
        curves.append({"s0": s, "s1": s + r * math.pi / 2, "radius": r, "turn": turn})
        h += sign * math.pi / 2
        s += r * math.pi / 2

    straight(lead)
    for i, r in enumerate(radii):
        arc(r, "left")
        straight(gap)
        arc(r, "right")
        straight(gap if i + 1 < len(radii) else lead)
    return np.array(pts), curves


def curve_town(variant="nolines", radii=RADII, seed=0, cfg=None):
    """The course of `variant` (VARIANT_NAMES) as a Town; town.course = {variant, curves (centre-line spans)}."""
    base, with_buildings = split_variant(variant)
    spec = RoadSpec("residential", 3.0, 0.5, 0.0, 30 / 3.6, **VARIANTS[base])
    pts, curves = course_line(radii)
    cfg = cfg or town_config("roadtypes")
    town = Town.from_graph(seed, cfg, [tuple(pts[0]), tuple(pts[-1])], [(0, 1, pts, spec)],
                           buildings=None if with_buildings else [], name=f"curve_{variant}_{COURSE_VERSION}")
    town.course = {"variant": variant, "centre": pts, "curves": curves}
    town.parcel_reach = PARCEL_REACH           # KeiView's extra lots (scripts/export_town.py): only near the road
    town.extra_lots = with_buildings           # _nobld: none at all
    return town


def course_route(town, reverse=False, margin=8.0, start=None):
    """The route along the whole course: from its west end (reverse: from its east end, every curve turned the other
    way), margin m in from the dead end (or from route s `start`), to margin m before the other one. Also the curves
    along it: [{s0, s1 (route s), radius (of the lane driven), turn}]."""
    road = town.roads[0]
    lid = road.lanes_ba[0] if reverse else road.lanes_ab[0]
    lane = town.lanes[lid]
    r = Route(town, [lid], max(margin, start or 0.0), s_end=lane.length - margin)
    centre = Polyline(town.course["centre"])
    out = []
    for c in town.course["curves"]:
        turn = c["turn"] if not reverse else ("right" if c["turn"] == "left" else "left")
        # the curve's own ends projected onto the route (driven the other way, the route meets them in reverse order)
        a, b = (r.poly.project(centre.interp1(x))[0] for x in (c["s0"], c["s1"]))
        # the lane runs half a lane inside the centre line on a left curve (left-hand traffic), outside on a right one
        side = 1.0 if town.cfg.left_hand_traffic else -1.0
        rad = c["radius"] - side * (1 if turn == "left" else -1) * lane.width / 2
        out.append({"s0": float(min(a, b)), "s1": float(max(a, b)), "radius": float(rad), "turn": turn})
    return r, sorted(out, key=lambda c: c["s0"])


# ---------------------------------------------------------------------------------------------- training scenes
# The curve course found that, on a road without white lines, KeiPilot is pulled towards a guardrail in a tight curve
# (towards the car's left rail: inwards on a left curve, outwards onto it on a right one; README, curve course). The
# training scenes are random winding roads of that kind, for scripts/collect.py --curve_scenes: mostly tight curves,
# mostly no paint, a guardrail on one side, the other or both (of varying height), with or without buildings. The
# expert's labels hold the lane centre; the collector's virtual cameras and noise put the car off it, towards the rails.
SCENE_RADII = ((0.6, 10.0, 30.0), (0.3, 30.0, 80.0), (0.1, 80.0, 200.0))     # (probability, R lo, R hi) of a curve
SCENE_PAINT = ((0.6, "none", "none"), (0.2, "none", "lane"), (0.2, "dashed", "lane"))   # (probability, centre, edge)
SCENE_GUARD = ((0.3, "left"), (0.3, "right"), (0.3, "both"), (0.1, None))
SCENE_BUILDINGS = 0.5                # probability of KeiSim buildings along the road
SCENE_TP_AHEAD = 50.0                # m: the target point ahead along the route (curve_test.py --tp ahead)
SCENE_MAX_HEADING = math.radians(110)   # the road keeps within this of its first heading (it never loops back)
SCENE_CLEAR = 30.0                   # m: parts of the road more than 80 m apart along it stay this far apart


def _pick(rng, table):
    k = rng.choice(len(table), p=[t[0] for t in table])
    return table[k][1:] if len(table[k]) > 2 else table[k][1]


def _scene_attempt(rng, lead):
    """One random winding centre line (scene_line): (pts (N, 2), curves)."""
    pts, h, s, curves = [np.zeros(2)], 0.0, 0.0, []

    def straight(length):
        nonlocal s
        n = max(1, int(round(length / DS)))
        d = np.array([math.cos(h), math.sin(h)])
        for _ in range(n):
            pts.append(pts[-1] + d * (length / n))
        s += length

    def arc(r, ang, turn):
        nonlocal h, s
        sign = 1.0 if turn == "left" else -1.0
        c = pts[-1] + r * np.array([-math.sin(h), math.cos(h)]) * sign
        n = max(4, int(round(r * ang / DS)))
        a0 = h - sign * math.pi / 2
        for k in range(1, n + 1):
            a = a0 + sign * ang * k / n
            pts.append(c + r * np.array([math.cos(a), math.sin(a)]))
        curves.append({"s0": s, "s1": s + r * ang, "radius": r, "turn": turn})
        h += sign * ang
        s += r * ang

    straight(lead)
    n_curves = int(rng.integers(5, 9))
    for i in range(n_curves):
        lo, hi = _pick(rng, SCENE_RADII)
        r = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        ang = math.radians(rng.uniform(40.0, 180.0))
        turn = "left" if rng.random() < 0.5 else "right"
        if abs(h + (ang if turn == "left" else -ang)) > SCENE_MAX_HEADING:
            turn = "right" if turn == "left" else "left"
        ang = min(ang, SCENE_MAX_HEADING - (h if turn == "left" else -h))
        if ang < math.radians(30):
            continue
        arc(r, ang, turn)
        straight(rng.uniform(10.0, 60.0) if i + 1 < n_curves else lead)
    return np.array(pts), curves


def scene_line(rng, lead=40.0):
    """A random winding centre line: 5-8 curves of random radius (SCENE_RADII), angle (40-180 deg) and direction,
    straights of 10-60 m between them. (pts (N, 2), [{s0, s1, radius, turn}]) as course_line()."""
    for _ in range(100):
        P, curves = _scene_attempt(rng, lead)
        sub = P[::10]                                     # every 5 m
        ss = np.arange(len(sub)) * 5.0
        far = np.abs(ss[:, None] - ss[None, :]) > 80.0
        d = np.hypot(*(sub[:, None] - sub[None, :]).transpose(2, 0, 1))
        if len(curves) >= 4 and not np.any(far & (d < SCENE_CLEAR)):
            return P, curves
    raise RuntimeError("no scene line without a loop in 100 tries")


def scene_town(seed, cfg=None):
    """A random training scene (a Town, named scene_<seed>): one two-way road of scene_line() between two dead ends.
    town.course = {variant, centre, curves, spec}: what was drawn."""
    rng = np.random.default_rng([int(seed), 4242])
    pts, curves = scene_line(rng)
    center, edge = _pick(rng, SCENE_PAINT)
    guard = _pick(rng, SCENE_GUARD)
    lo = float(rng.uniform(0.35, 0.55))
    beam = (lo, lo + float(rng.uniform(0.3, 0.4))) if guard else None
    lane, shoulder = float(rng.uniform(2.75, 3.25)), float(rng.uniform(0.25, 0.75))
    with_buildings = bool(rng.random() < SCENE_BUILDINGS)
    spec = RoadSpec("residential", lane, shoulder, 0.0, 30 / 3.6, center, edge, guardrail=guard, guardrail_beam=beam)
    cfg = cfg or town_config("roadtypes")
    town = Town.from_graph(int(seed), cfg, [tuple(pts[0]), tuple(pts[-1])], [(0, 1, pts, spec)],
                           buildings=None if with_buildings else [], name=f"scene_{int(seed)}")
    town.course = {"variant": "scene", "centre": pts, "curves": curves,
                   "spec": {"center": center, "edge": edge, "guardrail": guard, "beam": beam, "lane_width": lane,
                            "shoulder": shoulder, "buildings": with_buildings}}
    town.parcel_reach = PARCEL_REACH
    town.extra_lots = with_buildings
    return town


def scene_route(town, rng, length, margin=8.0):
    """A route of up to `length` m along a scene's road, either way, from a random start; its target point stays
    SCENE_TP_AHEAD m ahead (Route.tp_ahead)."""
    road = town.roads[0]
    lid = road.lanes_ba[0] if rng.random() < 0.5 else road.lanes_ab[0]
    L = town.lanes[lid].length
    length = min(float(length), L - 2 * margin)
    s0 = float(rng.uniform(margin, L - margin - length))
    r = Route(town, [lid], s0, s_end=s0 + length)
    r.tp_ahead = SCENE_TP_AHEAD
    return r
