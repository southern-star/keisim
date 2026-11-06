"""Curve test courses: one open two-way road of left and right curves without junctions, to see how a driving model
keeps its lane where the road gives few cues (scripts/curve_test.py). The road runs east from a dead end: a lead
straight, then for each radius a 90 deg left curve, a straight, a 90 deg right curve and a straight, to a dead end;
driven back the other way every curve turns the other way. Variants differ only in the paint and the guardrails:

    lines          white centre line (dashed) and edge lines (the procedural towns' streets)
    edge           edge lines only, no centre line (KeiSim's residential roads)
    nolines        no paint at all (KeiSim's alleys have none either, but are narrower)
    nolines_guard  no paint, a guardrail (ガードレール) along both edges
    lines_guard    white lines and guardrails

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

COURSE_VERSION = "v2"
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
    """The course of `variant` (VARIANTS) as a Town; town.course = {variant, curves (centre-line spans)}."""
    spec = RoadSpec("residential", 3.0, 0.5, 0.0, 30 / 3.6, **VARIANTS[variant])
    pts, curves = course_line(radii)
    cfg = cfg or town_config("roadtypes")
    town = Town.from_graph(seed, cfg, [tuple(pts[0]), tuple(pts[-1])], [(0, 1, pts, spec)],
                           name=f"curve_{variant}_{COURSE_VERSION}")
    town.course = {"variant": variant, "centre": pts, "curves": curves}
    town.parcel_reach = PARCEL_REACH           # KeiView's extra lots (scripts/export_town.py): only near the road
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
