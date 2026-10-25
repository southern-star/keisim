"""Sight lines at the 止まれ lines of a real town: can the ego camera see a car with priority D m up its lane?

For every minor approach with a 止まれ line (Lane.stop_sign_s), every connector after it that gives way, and every
lane it gives way to (crossing at s2 on that lane): the point D m upstream of the crossing (back along the lane, its
approach lane, then the straightest predecessor). The camera (CameraConfig: 0.6 m ahead of the car's centre) sits on
the approach looking along it, at
  stop: the front bumper STOP_SIGN_MARGIN before the line (where the expert stands), or
  wait: the front bumper at the connector's wait point (where it gives way from).
Prints, for cars coming from the side (the lane's heading there more than 45 deg from the camera's opposite; the
oncoming lanes of right turns are counted apart), the share whose point is inside the horizontal field of view and
not behind a building (Town.building_edges, 2D boxes: KeiView's walls and hedges are not in them, so it is a bound).
KeiPilot's camera is 100 deg wide: from the stop position it sees a side car 30 m up the priority road 1 % of the
time in Sapporo (dev log 10 of the README).

usage: uv run python tools/sightline.py [--town osm_sapporo]
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from keisim.config import CameraConfig  # noqa: E402
from keisim.expert import STOP_SIGN_MARGIN  # noqa: E402
from keisim.geometry import rays_clear, wrap  # noqa: E402
from keisim.realworld import osm_town  # noqa: E402
from keisim.vehicle import EgoVehicle  # noqa: E402

BBOX = {"osm_sapporo": (43.088, 141.360, 43.097, 141.372)}
DS = [10, 20, 30, 42]                   # m up the priority lane (42 m: 5 s at 30 km/h)
FOVS = [100, 120, 150, 180]


def straightest_pred(lanes, lid):
    h0 = float(lanes[lid].poly.heading(0.0))
    return min(lanes[lid].pred, key=lambda p: abs(wrap(float(lanes[p].poly.heading(lanes[p].length)) - h0)))


def back_along(lanes, lid, s, back):
    """(point, heading) `back` m before arc length s of lane lid, into the straightest predecessors; None at a dead end."""
    while back > s:
        back -= s
        if not lanes[lid].pred:
            return None
        lid = straightest_pred(lanes, lid)
        s = lanes[lid].length
    return np.asarray(lanes[lid].poly.interp(s - back), float), float(lanes[lid].poly.heading(s - back))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--town", default="osm_sapporo", choices=sorted(BBOX))
    args = ap.parse_args()
    town = osm_town(BBOX[args.town], cache=f"data/osm/{args.town}.json", name=args.town)
    lanes = town.lanes
    cam_x, hl = CameraConfig().x, EgoVehicle.LENGTH / 2
    edges = town.building_edges()[0]
    rows = {"stop": [], "wait": []}     # (D index, bearing deg, clear, from the side)
    n_lines = 0
    for lin in lanes:
        if lin.stop_sign_s is None:
            continue
        n_lines += 1
        for cid in lin.succ:
            conn = lanes[cid]
            if not conn.yields:
                continue
            cams = {"stop": back_along(lanes, lin.id, lin.stop_sign_s, STOP_SIGN_MARGIN + hl - cam_x)}
            w = conn.wait_s - hl + cam_x
            cams["wait"] = (np.asarray(conn.poly.interp(w), float), float(conn.poly.heading(w))) if w >= 0 else \
                back_along(lanes, lin.id, lin.length, -w)
            for b, _s1, s2 in conn.yields:
                for k, d in enumerate(DS):
                    p = back_along(lanes, b, s2, d)
                    if p is None:
                        continue
                    for where, cam in cams.items():
                        if cam is None:
                            continue
                        (c, h), (q, hq) = cam, p
                        brg = abs(math.degrees(wrap(math.atan2(q[1] - c[1], q[0] - c[0]) - h)))
                        side = abs(math.degrees(wrap(hq - h))) <= 135.0
                        clear = bool(rays_clear(c, q[None], edges)[0])
                        rows[where].append((k, brg, clear, side))
    print(f"{args.town}: {n_lines} 止まれ lines")
    for where, rr in rows.items():
        r = np.array(rr, float)
        side = r[:, 3] > 0
        print(f"\ncamera at the {where} position: cars from the side within the view and not behind a building "
              f"(oncoming: {int((~side & (r[:, 0] == 0)).sum())} lane pairs, left out)")
        print("   D m  " + "".join(f"{f:>7d}deg" for f in FOVS) + "   pairs")
        for k, d in enumerate(DS):
            m = side & (r[:, 0] == k)
            seen = [np.mean((r[m, 1] < f / 2) & (r[m, 2] > 0)) for f in FOVS]
            print(f"  {d:4d}  " + "".join(f"{x:>10.0%}" for x in seen) + f"   {int(m.sum())}")


if __name__ == "__main__":
    main()
