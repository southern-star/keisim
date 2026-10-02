"""Export KeiSim towns to JSON for the KeiView web viewer (web/).

    uv run scripts/export_town.py --town 1000 1001 1002 --out web/towns

Everything stays in KeiSim's frame (metres, +x east, +y north); the viewer maps
(x, y) -> three.js (x - ox, -(y - oy)) with (ox, oy) = "origin". Plain polygons
are rings [[x, y], ...]. Surface regions are shapes {"outer": ring, "holes":
[ring, ...]} obtained by rasterising KeiSim's own road / sidewalk polygons (the
same layering as keisim/render/maptex.py) and tracing the result, so the
viewer gets clean unions (curb lines) without doing polygon booleans in JS.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from keisim.geometry import box_corners  # noqa: E402
from keisim.roadnet import Raster, Town  # noqa: E402

REGION_RES = 0.1        # m per pixel for the road / sidewalk rasters
REGION_EPS = 0.03       # m, simplification tolerance of traced outlines
LINE_EPS = 0.02         # m, simplification tolerance of centre lines / paint


def r2(a):
    return np.round(np.asarray(a, float), 2).tolist()


def simplify(pts, eps, closed):
    pts = np.asarray(pts, np.float32)
    if len(pts) <= 3:
        return pts
    out = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps, closed).reshape(-1, 2)
    return out if len(out) >= (3 if closed else 2) else pts


def smooth_ring(ring, k=3):
    """Circular moving average (removes raster staircase, rounds corners by ~k px)."""
    if len(ring) < 2 * k + 3:
        return ring
    pad = np.concatenate([ring[-k:], ring, ring[:k]])
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.stack([np.convolve(pad[:, 0], ker, "valid"), np.convolve(pad[:, 1], ker, "valid")], 1)


def trace_shapes(mask, R: Raster, min_area=1.0):
    """Binary mask -> list of {"outer", "holes"} in world coordinates."""
    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]

    def to_world(c):
        c = c.reshape(-1, 2).astype(np.float64)
        ring = np.stack([R.x0 + (c[:, 0] + 0.5) * R.res, R.y1 - (c[:, 1] + 0.5) * R.res], 1)
        ring = smooth_ring(ring)
        return simplify(ring, REGION_EPS, True)

    shapes = []
    for i, c in enumerate(contours):
        if hier[i][3] != -1:            # a hole; attached to its parent below
            continue
        if cv2.contourArea(c) * R.res * R.res < min_area:
            continue
        holes = []
        j = hier[i][2]
        while j != -1:
            if cv2.contourArea(contours[j]) * R.res * R.res >= min_area:
                holes.append(r2(to_world(contours[j])))
            j = hier[j][0]
        shapes.append({"outer": r2(to_world(c)), "holes": holes})
    return shapes


def surface_masks(town: Town):
    """Road and sidewalk-only masks with the same polygons (and order) as maptex.py."""
    R = Raster(town.bbox, REGION_RES)
    walk = np.zeros((R.H, R.W), np.uint8)
    road = np.zeros((R.H, R.W), np.uint8)
    wo, hw = town.cfg.walk_outer, town.cfg.road_half_width
    for rd in town.roads:
        R.fill(walk, town.road_polygon(rd, wo), 1)
        R.fill(road, town.road_polygon(rd, hw), 1)
    for J in town.junctions:
        R.fill(walk, J.walk_poly, 1)
        R.fill(road, J.road_poly, 1)
    for poly in town.conn_walk_polys:
        R.fill(walk, poly, 1)
    for poly in town.conn_road_polys:
        R.fill(road, poly, 1)
    return R, road, walk & (1 - road)


def road_samples(town: Town):
    """(road, centre-line points, headings, arc lengths) for building_frontage()."""
    return [(rd, rd.center, rd.poly.heading(rd.poly.s), rd.poly.s) for rd in town.roads]


def building_frontage(town: Town, b, samples):
    """Which road a building faces, on which side, and how far back it sits.

    KeiSim places a building at p + n * (walk_outer + setback + depth/2) where
    p = road.interp(sc), yaw = road heading at sc and n the road normal on side
    sgn. Recover (road, sc, sgn, setback) by finding the road sample whose heading
    equals yaw and whose normal line passes through the building centre (plain
    nearest-point projection fails for buildings that overhang a road's end).
    """
    c = np.array([b["x"], b["y"]])
    u = np.array([math.cos(b["yaw"]), math.sin(b["yaw"])])
    v = np.array([-u[1], u[0]])
    wo, half_d = town.cfg.walk_outer, b["w"] / 2
    best = None
    for rd, pts, hs, ss in samples:
        rel = c - pts
        along = rel @ u                                 # 0 where the normal line hits the building centre
        lat = rel @ v
        mis = np.abs(np.sin(hs - b["yaw"]))
        setback = np.abs(lat) - half_d - wo
        score = np.abs(along) + 50.0 * mis + 5.0 * np.clip(0.5 - setback, 0, None) + 5.0 * np.clip(setback - 5.0, 0, None)
        i = int(np.argmin(score))
        if best is None or score[i] < best[0]:
            best = (float(score[i]), rd.id, float(ss[i] + along[i]), float(lat[i]), float(setback[i]))
    _, rid, s, lat, setback = best
    return {"road": rid, "s": round(s, 2), "side": 1 if lat > 0 else -1,   # +1 = left of the road direction a->b
            "setback": round(setback, 2)}


PARCEL_RES = 0.5        # m per pixel for the free-space raster used to place parcels


class FreeSpace:
    """Occupancy raster for placing KeiView's extra parcels (off-road only; KeiSim is untouched)."""

    def __init__(self, town: Town):
        R = Raster(town.bbox, PARCEL_RES)
        self.R, self.town = R, town
        self.blocked = np.zeros((R.H, R.W), np.uint8)
        cfg = town.cfg
        for rd in town.roads:
            R.fill(self.blocked, town.road_polygon(rd, cfg.walk_outer + 0.35), 1)
        for J in town.junctions:
            c = J.pos
            R.fill(self.blocked, (J.walk_poly - c) * 1.12 + c, 1)
        for poly in town.conn_walk_polys:
            R.fill(self.blocked, poly, 1)
        for sh in town.signal_heads:
            p = sh["pole"]
            R.fill(self.blocked, box_corners(p[0], p[1], 0.0, 2.0, 2.0), 1)

    def rect(self, cx, cy, yaw, l, w):
        return box_corners(cx, cy, yaw, l, w)

    def _roi(self, corners, pad=1):
        px = self.R.to_px(corners)
        c0 = max(int(np.floor(px[:, 0].min())) - pad, 0)
        r0 = max(int(np.floor(px[:, 1].min())) - pad, 0)
        c1 = min(int(np.ceil(px[:, 0].max())) + pad, self.R.W)
        r1 = min(int(np.ceil(px[:, 1].max())) + pad, self.R.H)
        return c0, r0, c1, r1

    def free(self, corners, outside=None):
        """True if the polygon covers no blocked pixel (nor any pixel of the optional `outside` mask)."""
        c0, r0, c1, r1 = self._roi(corners)
        if c1 - c0 < 2 or r1 - r0 < 2:
            return False
        px = self.R.to_px(corners)
        if px[:, 0].min() < 0 or px[:, 1].min() < 0 or px[:, 0].max() > self.R.W or px[:, 1].max() > self.R.H:
            return False
        m = np.zeros((r1 - r0, c1 - c0), np.uint8)
        q = np.round((px - [c0, r0]) * 8.0).astype(np.int32)
        cv2.fillPoly(m, [q.reshape(-1, 1, 2)], 1, lineType=cv2.LINE_8, shift=3)
        occ = self.blocked[r0:r1, c0:c1]
        if outside is not None:
            occ = occ | outside[r0:r1, c0:c1]
        return not np.any(m & occ)

    def mark(self, corners):
        self.R.fill(self.blocked, corners, 1)


def make_parcels(town: Town, buildings):
    """Extra parcels around KeiSim's buildings so the town reads as a dense Japanese suburb.

    row 1: street-frontage gaps between KeiSim buildings; row 2: lots behind the frontage row;
    row 3: whatever block interior is left (fields, vacant lots, small parks). Each parcel is a
    lot {fx, fy (frontage centre), face (heading toward its road), w (width), d (depth), kind}.
    """
    cfg = town.cfg
    wo = cfg.walk_outer
    rng = np.random.default_rng(town.seed * 7919 + 17)
    fs = FreeSpace(town)
    for b in buildings:                      # KeiSim lots: from the sidewalk edge to the back of the box
        nx, ny = -math.sin(b["yaw"]) * b["side"], math.cos(b["yaw"]) * b["side"]
        cx, cy = b["x"] - nx * b["setback"] / 2, b["y"] - ny * b["setback"] / 2
        fs.mark(fs.rect(cx, cy, b["yaw"], b["l"] + 0.8, b["w"] + b["setback"] + 0.8))
    parcels = []

    def kind_for(row):
        r = rng.random()
        if row == 1:
            return "house" if r < 0.62 else "parking" if r < 0.76 else "field" if r < 0.9 else "vacant"
        if row == 2:
            return "house" if r < 0.5 else "field" if r < 0.8 else "parking" if r < 0.88 else "vacant"
        return "field" if r < 0.55 else "vacant" if r < 0.75 else "park" if r < 0.87 else "house"

    def add(center, yaw, l, d, n, row, outside=None):
        # test a slightly shrunk rectangle: boundary pixels of neighbours (0.5 m raster) may touch
        if not fs.free(fs.rect(center[0], center[1], yaw, l - 1.0, d - 1.0), outside):
            return False
        fs.mark(fs.rect(center[0], center[1], yaw, l + 0.6, d + 0.6))
        front = center - n * (d / 2)
        face = math.atan2(-n[1], -n[0])
        parcels.append({"fx": round(float(front[0]), 2), "fy": round(float(front[1]), 2), "face": round(face, 4),
                        "w": round(float(l), 2), "d": round(float(d), 2), "row": row, "kind": kind_for(row)})
        return True

    # rows 1 and 2 along every road side (same walk as KeiSim's own building placement)
    for row, (off0, off1) in ((1, (0.4, 0.4)), (2, (14.5, 17.5))):
        for rd in town.roads:
            for sgn in (1.0, -1.0):
                s = float(rng.uniform(0.5, 3.0))
                while s < rd.length - 3.0:
                    w = float(rng.uniform(8.5, 12.5))
                    d = float(rng.uniform(11.0, 14.0))
                    sc = min(s + w / 2, rd.length)
                    p = rd.poly.interp(sc)
                    h = float(rd.poly.heading(sc))
                    n = np.array([-math.sin(h), math.cos(h)]) * sgn
                    off = float(rng.uniform(off0, off1))
                    if add(p + n * (wo + off + d / 2), h, w, d, n, row):
                        s += w + float(rng.uniform(0.3, 1.0))
                    else:
                        s += 1.5
    # row 3: fill the remaining interior of every block with a grid aligned to the block
    free = (1 - fs.blocked).astype(np.uint8)
    n_lab, labels, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    for k in range(1, n_lab):
        if st[k, cv2.CC_STAT_AREA] * PARCEL_RES ** 2 < 150:
            continue
        outside = (labels != k).astype(np.uint8)
        ys, xs = np.nonzero(labels == k)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
        th = -math.radians(ang)                      # pixel rows grow southwards -> flip the angle
        u = np.array([math.cos(th), math.sin(th)])
        v = np.array([-u[1], u[0]])
        cw = np.array([fs.R.x0 + (cx + 0.5) * PARCEL_RES, fs.R.y1 - (cy + 0.5) * PARCEL_RES])
        half = max(rw, rh) * PARCEL_RES / 2 + 10
        dstep = float(rng.uniform(13.0, 17.0))
        for a in np.arange(-half, half, dstep):
            b0 = -half
            while b0 < half:
                w = float(rng.uniform(10.0, 18.0))
                c = cw + u * (b0 + w / 2) + v * (a + dstep / 2)
                if add(c, math.atan2(u[1], u[0]), w, dstep - 0.8, v, 3, outside):
                    b0 += w + 0.6
                else:
                    b0 += 2.0
    return parcels


def export(seed: int) -> dict:
    town = Town(seed)
    cfg = town.cfg
    x0, y0, x1, y1 = town.bbox
    origin = [round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1)]
    R, road_mask, walk_mask = surface_masks(town)

    roads = [{"id": rd.id, "a": rd.a, "b": rd.b, "length": round(rd.length, 2),
              "center": r2(simplify(rd.center, LINE_EPS, False))} for rd in town.roads]
    junctions = [{"id": J.id, "pos": r2(J.pos), "radius": round(J.radius, 2), "arms": len(J.arms),
                  "signalized": bool(J.signalized), "phases": J.phases, "green": round(J.green, 9),
                  "yellow": J.yellow, "allred": J.allred, "offset": round(J.offset, 9)}
                 for J in town.junctions]
    signals = [{"junction": sh["junction"], "phase": sh["phase"], "head": r2(sh["head"]),
                "pole": r2(sh["pole"]), "yaw": round(float(sh["yaw"]), 4), "z": sh["z"]}
               for sh in town.signal_heads]
    buildings, samples = [], road_samples(town)
    for b in town.buildings:
        d = {"x": round(b["x"], 2), "y": round(b["y"], 2), "yaw": round(float(b["yaw"]), 4),
             "l": round(b["l"], 2), "w": round(b["w"], 2), "h": round(b["h"], 2)}
        d.update(building_frontage(town, b, samples))
        buildings.append(d)
    trees = [{"x": round(t["x"], 2), "y": round(t["y"], 2), "trunk_h": round(t["trunk_h"], 2),
              "r": round(t["r"], 2), "ch": round(t["ch"], 2)} for t in town.trees]
    return {
        "format": "keisim-town", "version": 1, "seed": seed, "summary": town.summary(),
        "origin": origin, "bbox": r2(town.bbox),
        "cfg": {"lane_width": cfg.lane_width, "shoulder": cfg.shoulder, "sidewalk_width": cfg.sidewalk_width,
                "road_half_width": cfg.road_half_width, "walk_outer": cfg.walk_outer,
                "left_hand_traffic": cfg.left_hand_traffic, "speed_limit": cfg.speed_limit},
        "surfaces": {"road": trace_shapes(road_mask, R), "sidewalk": trace_shapes(walk_mask, R)},
        "markings": [r2(simplify(p, LINE_EPS, True)) for p in town.markings],
        "crosswalks": [r2(simplify(p, LINE_EPS, True)) for p in town.crosswalks],
        "roads": roads, "junctions": junctions, "signals": signals,
        "buildings": buildings, "trees": trees,
        "parcels": make_parcels(town, buildings),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--town", type=int, nargs="+", default=[1000])
    ap.add_argument("--out", default="web/towns")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for seed in args.town:
        data = export(seed)
        path = os.path.join(args.out, f"town_{seed}.json")
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"{path}  {os.path.getsize(path) / 1024:.0f} KB  {data['summary']}")


if __name__ == "__main__":
    main()
