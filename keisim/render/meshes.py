"""Low-poly meshes for the painter's-algorithm camera renderer.

Every drawable is a list of planar convex faces with up to 8 vertices.
Faces carry: base colour (BGR), semantic class, a painter's sub-order within
their object (parts first, decals after their part) and flags.
Decals (windows, lights, lamps) are coplanar faces nudged 2 cm outwards, so a
plain back-face test gives exactly the visibility of the parent face.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import SEM

KMAX = 8


class FaceBuffer:
    """Accumulates faces; converts to packed numpy arrays."""

    def __init__(self):
        self.verts, self.nv, self.color, self.sem, self.order = [], [], [], [], []
        self.emissive, self.lamp, self.lod = [], [], []

    def add(self, pts, color, sem, order, emissive=False, lamp=-1, lod=False):
        pts = np.asarray(pts, np.float64)
        n = len(pts)
        pad = np.concatenate([pts, np.repeat(pts[-1:], KMAX - n, 0)], 0) if n < KMAX else pts[:KMAX]
        self.verts.append(pad)
        self.nv.append(min(n, KMAX))
        self.color.append(np.asarray(color, np.float64))
        self.sem.append(sem)
        self.order.append(order)
        self.emissive.append(emissive)
        self.lamp.append(lamp)
        self.lod.append(lod)

    def __len__(self):
        return len(self.verts)

    def pack(self):
        if not self.verts:
            return None
        return {
            "verts": np.array(self.verts, np.float32),
            "nv": np.array(self.nv, np.int32),
            "color": np.array(self.color, np.float32),
            "sem": np.array(self.sem, np.uint8),
            "order": np.array(self.order, np.float32),
            "emissive": np.array(self.emissive, bool),
            "lamp": np.array(self.lamp, np.int32),
            "lod": np.array(self.lod, bool),
        }


def _orient(pts, outward):
    """Make the vertex winding give a normal along `outward`."""
    n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    if np.dot(n, outward) < 0:
        return pts[::-1].copy()
    return pts


def box_faces(x0, x1, y0, y1, z0, z1, skip=()):
    """Return dict name -> (4,3) face polygon with outward winding."""
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    F = {
        "front": np.array([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]),
        "back": np.array([[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]]),
        "left": np.array([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]]),
        "right": np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]),
        "top": np.array([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]),
        "bottom": np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]),
    }
    out = {}
    for k, v in F.items():
        if k in skip:
            continue
        out[k] = _orient(v, v.mean(0) - c)
    return out


def frustum_faces(xb0, xb1, xt0, xt1, yb, yt, z0, z1):
    """Prism with rectangular bottom [xb0,xb1]x[-yb,yb] at z0 and top
    [xt0,xt1]x[-yt,yt] at z1 (a car cabin)."""
    b = np.array([[xb0, -yb, z0], [xb1, -yb, z0], [xb1, yb, z0], [xb0, yb, z0]])
    t = np.array([[xt0, -yt, z1], [xt1, -yt, z1], [xt1, yt, z1], [xt0, yt, z1]])
    c = (b.mean(0) + t.mean(0)) / 2
    faces = {
        "front": np.array([b[1], b[2], t[2], t[1]]),
        "back": np.array([b[0], b[3], t[3], t[0]]),
        "left": np.array([b[3], b[2], t[2], t[3]]),
        "right": np.array([b[0], b[1], t[1], t[0]]),
        "top": t.copy(),
    }
    return {k: _orient(v, v.mean(0) - c) for k, v in faces.items()}


def ngon(center, u, v, r, n=8):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False) + math.pi / n
    return center[None] + r * (np.cos(a)[:, None] * u[None] + np.sin(a)[:, None] * v[None])


def transform(pts, x, y, yaw, z=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.empty_like(pts, dtype=np.float64)
    out[..., 0] = x + c * pts[..., 0] - s * pts[..., 1]
    out[..., 1] = y + s * pts[..., 0] + c * pts[..., 1]
    out[..., 2] = pts[..., 2] + z
    return out


# ----------------------------------------------------------------------------
# Static scene (buildings, trees, signal poles) -- built once per town
# ----------------------------------------------------------------------------

LAMP_COLORS_ON = {0: np.array([40, 40, 255.0]), 1: np.array([0, 190, 255.0]), 2: np.array([170, 255, 40.0])}
LAMP_OFF = np.array([38, 40, 44.0])


class StaticScene:
    def __init__(self, town):
        fb = FaceBuffer()
        objs = []  # (cx, cy, radius, f0, f1)
        rng = np.random.default_rng(town.seed + 99)
        side = 1.0 if town.cfg.left_hand_traffic else -1.0

        for b in town.buildings:
            f0 = len(fb)
            l, w, h = b["l"], b["w"], b["h"]
            col = np.asarray(b["color"], float)
            faces = box_faces(-l / 2, l / 2, -w / 2, w / 2, 0.0, h, skip=("bottom",))
            for k, f in faces.items():
                shade = 0.8 if k == "top" else 1.0
                fb.add(transform(f, b["x"], b["y"], b["yaw"]), col * shade, SEM["building"], 0.0)
            # window bands (level-of-detail decals)
            glass = np.array([120, 95, 70.0]) * rng.uniform(0.7, 1.2)
            floor_h = 3.2
            nfl = int((h - 1.0) // floor_h)
            for fl in range(nfl):
                z0 = fl * floor_h + 1.1
                z1 = z0 + 1.3
                if z1 > h - 0.4:
                    break
                for name, (a0, a1, fixed, axis) in {
                    "front": (-w / 2 + 0.8, w / 2 - 0.8, l / 2 + 0.02, "x"),
                    "back": (-w / 2 + 0.8, w / 2 - 0.8, -l / 2 - 0.02, "x"),
                    "left": (-l / 2 + 0.8, l / 2 - 0.8, w / 2 + 0.02, "y"),
                    "right": (-l / 2 + 0.8, l / 2 - 0.8, -w / 2 - 0.02, "y"),
                }.items():
                    if a1 - a0 < 1.0:
                        continue
                    if axis == "x":
                        q = np.array([[fixed, a0, z0], [fixed, a1, z0], [fixed, a1, z1], [fixed, a0, z1]])
                        out = np.array([np.sign(fixed), 0, 0])
                    else:
                        q = np.array([[a0, fixed, z0], [a1, fixed, z0], [a1, fixed, z1], [a0, fixed, z1]])
                        out = np.array([0, np.sign(fixed), 0])
                    q = _orient(q, out)
                    fb.add(transform(q, b["x"], b["y"], b["yaw"]), glass, SEM["building"], 0.5, lod=True)
            objs.append((b["x"], b["y"], 0.5 * math.hypot(l, w) + 0.1 * h, f0, len(fb)))

        for t in town.trees:
            f0 = len(fb)
            th = t["trunk_h"]
            trunk = box_faces(-0.15, 0.15, -0.15, 0.15, 0.0, th + 0.3, skip=("bottom", "top"))
            for f in trunk.values():
                fb.add(transform(f, t["x"], t["y"], 0.0), np.array([40, 60, 85.0]), SEM["vegetation"], 0.0)
            r, ch = t["r"], t["ch"]
            n = 6
            ang = np.linspace(0, 2 * math.pi, n, endpoint=False) + rng.uniform(0, 1)
            ring_lo = np.stack([r * 0.75 * np.cos(ang), r * 0.75 * np.sin(ang), np.full(n, th)], -1)
            ring_mid = np.stack([r * np.cos(ang), r * np.sin(ang), np.full(n, th + ch * 0.45)], -1)
            top = np.array([0.0, 0.0, th + ch])
            col = np.asarray(t["color"], float)
            center = np.array([0, 0, th + ch * 0.45])
            for i in range(n):
                j = (i + 1) % n
                quad = np.array([ring_lo[i], ring_lo[j], ring_mid[j], ring_mid[i]])
                quad = _orient(quad, quad.mean(0) - center)
                fb.add(transform(quad, t["x"], t["y"], 0.0), col * rng.uniform(0.9, 1.1), SEM["vegetation"], 1.0)
                tri = np.array([ring_mid[i], ring_mid[j], top])
                tri = _orient(tri, tri.mean(0) - center)
                fb.add(transform(tri, t["x"], t["y"], 0.0), col * 1.08, SEM["vegetation"], 1.0)
            bottom = _orient(ring_lo[::-1].copy(), np.array([0, 0, -1.0]))
            fb.add(transform(bottom, t["x"], t["y"], 0.0), col * 0.7, SEM["vegetation"], 1.0)
            objs.append((t["x"], t["y"], r + 0.5, f0, len(fb)))

        # signal heads: pole + arm + housing + three lamp decals (JP order: green, yellow, red from left)
        grey = np.array([150, 150, 150.0])
        dark = np.array([55, 55, 60.0])
        for hi, sh in enumerate(town.signal_heads):
            f0 = len(fb)
            px, py = sh["pole"]
            hx, hy = sh["head"]
            zc = sh["z"]
            pole = box_faces(-0.13, 0.13, -0.13, 0.13, 0.0, zc + 0.5, skip=("bottom",))
            for f in pole.values():
                fb.add(transform(f, px, py, 0.0), grey, SEM["pole"], 0.0)
            # arm along the pole->head direction
            ax, ay = hx - px, hy - py
            L = math.hypot(ax, ay)
            ayaw = math.atan2(ay, ax)
            arm = box_faces(0.0, L, -0.08, 0.08, zc + 0.30, zc + 0.46, skip=())
            for f in arm.values():
                fb.add(transform(f, px, py, ayaw), grey, SEM["pole"], 1.0)
            yaw = sh["yaw"]  # housing faces incoming traffic (its +x points at the driver)
            hw_, hh_, hd_ = 1.05, 0.38, 0.18
            hous = box_faces(-hd_, hd_, -hw_, hw_, zc - hh_, zc + hh_)
            for f in hous.values():
                fb.add(transform(f, hx, hy, yaw), dark, SEM["pole"], 2.0)
            # lamps on +x face. driver's left = housing's local -y (housing faces the driver)
            u = np.array([0.0, 1.0, 0.0])
            v = np.array([0.0, 0.0, 1.0])
            for slot, yl in enumerate((-0.68 * side, 0.0, 0.68 * side)):
                # slot 0=green (driver's left in JP), 1=yellow, 2=red
                c = np.array([hd_ + 0.02, yl, zc])
                poly = ngon(c, u, v, 0.30, 8)
                poly = _orient(poly, np.array([1.0, 0, 0]))
                fb.add(transform(poly, hx, hy, yaw), LAMP_OFF, SEM["pole"], 2.5, emissive=True, lamp=hi * 3 + slot)
            cx, cy = (px + hx) / 2, (py + hy) / 2
            objs.append((cx, cy, L / 2 + 1.5, f0, len(fb)))

        self.faces = fb.pack()
        self.objs = np.array(objs, np.float64).reshape(-1, 5)


# ----------------------------------------------------------------------------
# Dynamic templates
# ----------------------------------------------------------------------------

def _template(parts):
    """parts: list of (faces dict or list, role, order) -> arrays in unit space."""
    fb = FaceBuffer()
    roles = []
    for faces, role, order in parts:
        items = faces.values() if isinstance(faces, dict) else faces
        for f in items:
            fb.add(f, np.zeros(3), 0, order)
            roles.append(role)
    p = fb.pack()
    p["role"] = np.array(roles, np.int32)
    return p


# colour roles
R_BODY, R_GLASS, R_DARK, R_HEAD, R_TAIL, R_SKIN, R_SHIRT, R_PANTS, R_HAIR, R_ROOF = range(10)


def car_template():
    parts = []
    parts.append((box_faces(-0.43, 0.43, -0.46, 0.46, 0.03, 0.26, skip=("top",)), R_DARK, 0.0))
    parts.append((box_faces(-0.5, 0.5, -0.5, 0.5, 0.22, 0.62, skip=("bottom",)), R_BODY, 1.0))
    cab = frustum_faces(-0.36, 0.20, -0.30, 0.06, 0.46, 0.40, 0.62, 1.0)
    roof = {"top": cab.pop("top")}
    parts.append((cab, R_GLASS, 2.0))
    parts.append((roof, R_ROOF, 2.0))
    # lights as decals
    head = []
    tail = []
    for yl in (0.28, -0.28):
        q = np.array([[0.502, yl - 0.12, 0.44], [0.502, yl + 0.12, 0.44], [0.502, yl + 0.12, 0.55], [0.502, yl - 0.12, 0.55]])
        head.append(_orient(q, np.array([1.0, 0, 0])))
        q2 = q.copy()
        q2[:, 0] = -0.502
        tail.append(_orient(q2, np.array([-1.0, 0, 0])))
    parts.append((head, R_HEAD, 1.5))
    parts.append((tail, R_TAIL, 1.5))
    return _template(parts)


def truck_template():
    parts = []
    parts.append((box_faces(-0.45, 0.45, -0.46, 0.46, 0.02, 0.2, skip=("top",)), R_DARK, 0.0))
    parts.append((box_faces(0.2, 0.5, -0.5, 0.5, 0.16, 0.62, skip=("bottom",)), R_BODY, 1.0))
    parts.append((box_faces(0.28, 0.46, -0.47, 0.47, 0.62, 0.8, skip=("bottom",)), R_GLASS, 2.0))
    parts.append((box_faces(-0.5, 0.17, -0.5, 0.5, 0.16, 1.0, skip=("bottom",)), R_ROOF, 1.0))
    head, tail = [], []
    for yl in (0.3, -0.3):
        q = np.array([[0.502, yl - 0.1, 0.28], [0.502, yl + 0.1, 0.28], [0.502, yl + 0.1, 0.36], [0.502, yl - 0.1, 0.36]])
        head.append(_orient(q, np.array([1.0, 0, 0])))
        q2 = q.copy()
        q2[:, 0] = -0.502
        tail.append(_orient(q2, np.array([-1.0, 0, 0])))
    parts.append((head, R_HEAD, 1.5))
    parts.append((tail, R_TAIL, 1.5))
    return _template(parts)


def ped_template():
    """Unit pedestrian (height 1). Legs are separate so they can swing."""
    parts = []
    legs_l = box_faces(-0.05, 0.05, 0.015, 0.1, 0.0, 0.47, skip=("bottom",))
    legs_r = box_faces(-0.05, 0.05, -0.1, -0.015, 0.0, 0.47, skip=("bottom",))
    parts.append((legs_l, R_PANTS, 0.0))
    parts.append((legs_r, R_PANTS, 0.1))
    parts.append((box_faces(-0.075, 0.075, -0.13, 0.13, 0.46, 0.83, skip=("bottom",)), R_SHIRT, 1.0))
    parts.append((box_faces(-0.065, 0.065, -0.06, 0.06, 0.84, 0.99, skip=("bottom",)), R_SKIN, 2.0))
    t = _template(parts)
    # remember which faces belong to which leg for animation
    t["leg"] = np.zeros(len(t["role"]), np.int32)
    n_l = len(legs_l)
    n_r = len(legs_r)
    t["leg"][:n_l] = 1
    t["leg"][n_l:n_l + n_r] = -1
    return t


CAR_T = car_template()
TRUCK_T = truck_template()
PED_T = ped_template()
