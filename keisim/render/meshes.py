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
GUARDRAIL_BEAM = (0.45, 0.80)    # m above the ground: the beam panel of a guardrail (KeiView web/src/world/guardrails.js)
GUARDRAIL_POST = 0.75            # m: its posts


class FaceBuffer:
    """Accumulates faces; converts to packed numpy arrays."""

    def __init__(self):
        self.verts, self.nv, self.color, self.sem, self.order = [], [], [], [], []
        self.emissive, self.lamp, self.lod, self.guard = [], [], [], []

    def add(self, pts, color, sem, order, emissive=False, lamp=-1, lod=False, guard=False):
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
        self.guard.append(guard)          # a guardrail face (CameraConfig.guardrail_light)

    def __len__(self):
        return len(self.verts)

    def pack(self):
        # no faces (a town with nothing standing: the curve course's lines_nobld / nolines_nobld): empty arrays
        return {
            "verts": np.array(self.verts, np.float32).reshape(-1, KMAX, 3),
            "nv": np.array(self.nv, np.int32),
            "color": np.array(self.color, np.float32).reshape(-1, 3),
            "sem": np.array(self.sem, np.uint8),
            "order": np.array(self.order, np.float32),
            "emissive": np.array(self.emissive, bool),
            "lamp": np.array(self.lamp, np.int32),
            "lod": np.array(self.lod, bool),
            "guard": np.array(self.guard, bool),
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

        # 止まれ signs (Town.from_graph): post + red inverted triangle with a white rim, facing the driver (+x)
        red, white, back = np.array([45, 40, 200.0]), np.array([235, 235, 232.0]), np.array([120, 120, 125.0])
        for sp in town.stop_sign_posts:
            f0 = len(fb)
            px, py = sp["pole"]
            zc = sp["z"]
            post = box_faces(-0.035, 0.035, -0.035, 0.035, 0.0, zc + 0.2, skip=("bottom",))
            for f in post.values():
                fb.add(transform(f, px, py, 0.0), grey, SEM["pole"], 0.0)
            for col, a, x, out in ((back, 0.9, 0.035, -1.0), (white, 0.9, 0.04, 1.0), (red, 0.75, 0.045, 1.0)):
                h = a * math.sqrt(3) / 2
                tri = np.array([[x, -a / 2, zc + h / 3], [x, a / 2, zc + h / 3], [x, 0.0, zc - 2 * h / 3]])
                tri = _orient(tri, np.array([out, 0.0, 0.0]))
                fb.add(transform(tri, px, py, sp["yaw"]), col, SEM["pole"], 1.0 + (x - 0.035) * 10)
            objs.append((px, py, 1.0, f0, len(fb)))

        # ガードレール (Town.guardrails): a white beam panel (0.45-0.80 m) on every GUARDRAIL_DS piece of the line, on
        # the road's side of a post (every second piece, 0-0.75 m); objects of ~10 m for the culling
        rail, post = np.array([228, 230, 232.0]), np.array([205, 205, 210.0])
        for gr in getattr(town, "guardrails", ()):
            p = gr["pts"]
            beam_lo, beam_hi = gr.get("beam", GUARDRAIL_BEAM)
            post_h = GUARDRAIL_POST if "beam" not in gr else beam_hi - 0.05
            for k0 in range(0, len(p) - 1, 5):
                f0 = len(fb)
                for k in range(k0, min(k0 + 5, len(p) - 1)):
                    a, b = p[k], p[k + 1]
                    d = b - a
                    L = float(math.hypot(*d))
                    yaw = math.atan2(d[1], d[0])
                    # the road is on the left of a -> b for side -1 (right edge), on the right for side +1
                    inward = -gr["side"]
                    beam = box_faces(0.0, L, 0.06 * inward - 0.03, 0.06 * inward + 0.03, beam_lo, beam_hi, skip=("bottom",))
                    for kk, f in beam.items():
                        fb.add(transform(f, a[0], a[1], yaw), rail * (0.85 if kk == "top" else 1.0), SEM["pole"], 0.5,
                               guard=True)
                    if k % 2 == 0:
                        pf = box_faces(-0.05, 0.05, -0.05, 0.05, 0.0, post_h, skip=("bottom",))
                        for f in pf.values():
                            fb.add(transform(f, a[0], a[1], yaw), post, SEM["pole"], 0.0, guard=True)
                seg = p[k0:k0 + 6]
                c = seg.mean(0)
                objs.append((c[0], c[1], float(np.hypot(*(seg - c).T).max()) + 1.0, f0, len(fb)))

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


# ----------------------------------------------------------------------------
# Bicycles: built every frame from the riders' poses (keisim/bicycles.py, Bicycles.scene)
# ----------------------------------------------------------------------------
B_FRAME, B_TYRE, B_DARK, B_METAL, B_SHIRT, B_PANTS, B_SKIN, B_HAIR, B_RED = range(9)
BIKE_STEER_X = 0.42         # m ahead of the middle of the bike: the steering axis (taken as upright)
BIKE_BB = (-0.06, 0.29)     # x, z of the crank axle
BIKE_CRANK = 0.17


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _wind(faces, outward):
    """faces (..., 4, 3): reverse the ones whose normal points against `outward` (..., 3)."""
    n = np.cross(faces[..., 1, :] - faces[..., 0, :], faces[..., 2, :] - faces[..., 0, :])
    flip = (n * outward).sum(-1) < 0
    faces[flip] = faces[flip][:, ::-1]
    return faces


def _beam(a, b, side, ws, wn, caps=False):
    """A bar from a to b ((n, 3) each), 2 ws wide along `side` and 2 wn thick: (n, 4 or 6, 4, 3). The faces come
    in the order: against the third axis (d x side), along side, along the third axis, against side, the two ends."""
    d = _unit(b - a)
    s = _unit(side - (side * d).sum(-1, keepdims=True) * d)
    m = np.cross(d, s)
    ws = np.asarray(ws, float).reshape(-1, 1)
    wn = np.asarray(wn, float).reshape(-1, 1)
    c = [-ws * s - wn * m, ws * s - wn * m, ws * s + wn * m, -ws * s + wn * m]
    faces = [np.stack([a + c[i], b + c[i], b + c[j], a + c[j]], 1) for i, j in ((0, 1), (1, 2), (2, 3), (3, 0))]
    out = [-m, s, m, -s]
    if caps:
        faces += [np.stack([a + ci for ci in c], 1), np.stack([b + ci for ci in c], 1)]
        out += [-d, d]
    return _wind(np.stack(faces, 1), np.stack(out, 1))


def _wheel(c, u, v, r, t=0.06, w=0.022, k=8):
    """A wheel as a ring (you see through it): centre c, in-plane unit vectors u, v ((n, 3) each), radius r (n,).
    Both sides of the ring and the tread: (n, 3 k, 4, 3)."""
    ang = np.arange(k + 1) * (2 * math.pi / k)
    nrm = np.cross(u, v)
    ring = np.cos(ang)[None, :, None] * u[:, None] + np.sin(ang)[None, :, None] * v[:, None]
    r = np.asarray(r, float).reshape(-1, 1, 1)
    o = c[:, None] + r * ring
    i = c[:, None] + (r - t) * ring
    off = w * nrm[:, None]
    side = np.stack([o[:, :-1], o[:, 1:], i[:, 1:], i[:, :-1]], 2)
    nk = np.broadcast_to(nrm[:, None], o[:, :-1].shape)
    left = _wind(side + off[:, None], nk)
    right = _wind(side - off[:, None], -nk)
    tread = _wind(np.stack([o[:, :-1] - off, o[:, 1:] - off, o[:, 1:] + off, o[:, :-1] + off], 2),
                  ring[:, :-1] + ring[:, 1:])
    return np.concatenate([left, right, tread], 1)


def _knee(hip, tgt, l1, l2, fwd):
    """Two-bone leg: the knee (bending towards `fwd`) and the ankle the leg can reach."""
    d = tgt - hip
    dist = np.linalg.norm(d, axis=-1, keepdims=True)
    e = d / np.maximum(dist, 1e-6)
    l1, l2 = l1[:, None], l2[:, None]
    dc = np.clip(dist, 0.05, 0.995 * (l1 + l2))
    a = (l1 ** 2 - l2 ** 2 + dc ** 2) / (2 * dc)
    hk = np.sqrt(np.maximum(l1 ** 2 - a ** 2, 0.0))
    nb = _unit(fwd - (fwd * e).sum(-1, keepdims=True) * e)
    return hip + a * e + hk * nb, hip + e * dc


def bike_faces(pose, kind, height):
    """Bikes with their riders. pose (n, 7), kind (n,), height (n,) as Bicycles.scene gives them. Returns the faces
    (n, F, 4, 3) in each bike's own frame (x forward, y left, z up from the ground under its middle; the lean about
    the line of the tyres on the ground is applied), their colour roles (n, F) and semantic classes (F,).

    The same build as KeiView's (web/src/world/bikes.js), with far fewer parts: wheels as rings, the frame's tubes
    (a step-through frame with a basket and a carrier for the city bike and the slow rider, a diamond frame with
    low bars for the sport bike), the saddle, and a rider whose legs follow the pedals (the left foot goes to the
    ground at a stop), whose hands are on the bars and whose head turns for the look behind."""
    n = len(kind)
    roll, steer, crank, foot, look, effort = (pose[:, k] for k in (0, 1, 2, 4, 5, 6))
    sport = kind == 1
    city = ~sport
    h = np.asarray(height, float)
    cr, sr = np.cos(roll), np.sin(roll)
    cs, ss = np.cos(steer), np.sin(steer)
    one = np.ones(n)

    def P(x, y, z):                 # a point (or a direction) of the upright bike -> leaned
        x, y, z = x * one, y * one, z * one
        return np.stack([x, y * cr + z * sr, -y * sr + z * cr], -1)

    def S(x, y, z):                 # the same for the steered front end
        x, y = x * one, y * one
        dx = x - BIKE_STEER_X
        return P(BIKE_STEER_X + dx * cs - y * ss, dx * ss + y * cs, z)

    xv, yv, zv = P(1, 0, 0), P(0, 1, 0), P(0, 0, 1)
    yf = P(-ss, cs, 0)
    bx, bz = BIKE_BB
    rw = np.where(sport, 0.34, 0.33)
    rear, front, bb = P(-0.54, 0, rw), S(0.54, 0, rw), P(bx, 0, bz)
    # the rider's size sets the saddle: the leg is almost straight with the pedal at the bottom
    l1, l2 = 0.245 * h, 0.245 * h
    ext = np.where(sport, 0.93, np.where(kind == 0, 0.87, 0.85))
    hip_d = ext * (l1 + l2) - 0.08
    hx, hz = bx - 0.309 * hip_d, bz + 0.951 * hip_d
    sz = hz - 0.10
    seat = P(hx, 0, sz)
    head_t, head_b = P(0.36, 0, np.where(sport, 0.88, 1.0)), P(0.44, 0, 0.64)
    hcx, hcz, bw = np.where(sport, 0.565, 0.24), np.where(sport, 0.915, 1.07), np.where(sport, 0.24, 0.31)
    parts = []                      # (faces, role, of the rider)

    def add(f, role, rider=False, only=None):
        if only is not None:        # a part this kind of bike does not have: faces of no area (culled)
            f = f * only[:, None, None, None]
        parts.append((f, role, rider))

    add(_wheel(rear, xv, zv, rw), B_TYRE)
    add(_wheel(front, P(cs, ss, 0), zv, rw), B_TYRE)
    tube = 0.018
    add(_beam(head_b, front, yf, tube, tube), B_FRAME)                               # fork
    add(_beam(head_b, head_t, yv, 0.02, 0.02), B_FRAME)                              # head tube
    add(_beam(head_t, S(hcx, 0, hcz), yf, tube, tube), B_DARK)                       # stem
    add(_beam(S(hcx, -bw, hcz), S(hcx, bw, hcz), xv, tube, tube, caps=True), B_DARK)  # bars
    add(_beam(head_b, bb, yv, 0.022, 0.022), B_FRAME)                                # down tube
    add(_beam(P(0.38, 0, 0.84), P(hx + 0.05, 0, sz - 0.13), yv, tube, tube), B_FRAME, only=sport)    # top tube
    add(_beam(bb, seat, yv, tube, tube), B_FRAME)                                    # seat tube
    add(_beam(P(bx + 0.75 * (hx - bx), 0, bz + 0.75 * (sz - bz)), rear, yv, tube, tube), B_FRAME)    # seat stays
    add(_beam(bb, rear, yv, tube, tube), B_FRAME)                                    # chain stays
    add(_beam(P(hx - 0.15, 0, sz + 0.015), P(hx + 0.12, 0, sz + 0.015), yv, 0.075, 0.025, caps=True), B_DARK)
    add(_beam(P(-0.86, 0, 0.71), P(-0.38, 0, 0.71), yv, 0.075, 0.012), B_METAL, only=city)           # carrier
    add(_beam(S(0.50, 0, 0.86), S(0.78, 0, 0.86), yf, 0.17, 0.11, caps=True), B_METAL, only=city)    # basket
    rx, rz = np.where(sport, hx - 0.16, -0.875), np.where(sport, sz - 0.07, 0.69)
    refl = np.stack([P(rx, -0.035, rz - 0.03), P(rx, 0.035, rz - 0.03), P(rx, 0.035, rz + 0.03),
                     P(rx, -0.035, rz + 0.03)], 1)[:, None]
    add(_wind(refl, -xv[:, None]), B_RED)

    # ---- the rider
    lean = np.where(sport, 0.62, np.where(kind == 0, 0.24, 0.12)) + 0.10 * effort
    lt = 0.30 * h
    nx, nz = hx + lt * np.sin(lean), hz + lt * np.cos(lean)
    neck = P(nx, 0, nz)
    add(_beam(P(hx, 0, hz - 0.05), neck, yv, 0.105 * h, 0.055 * h, caps=True), B_SHIRT, True)
    psi = look * 1.1
    crown = P(nx + 0.15 * h * np.sin(0.3 * lean), 0, nz + 0.15 * h * np.cos(0.3 * lean))
    hd = _beam(P(nx, 0, nz + 0.01), crown, P(-np.sin(psi), np.cos(psi), 0), 0.045 * h, 0.055 * h, caps=True)
    add(hd[:, :1], B_SKIN, True)                    # the face
    add(hd[:, 1:], B_HAIR, True)                    # hair, or the sport rider's helmet
    arm = 0.026 * h
    for sgn in (1.0, -1.0):
        add(_beam(P(nx - 0.02, sgn * 0.115 * h, nz - 0.04), S(hcx, sgn * (bw - 0.03), hcz + 0.03), xv, arm, arm),
            B_SHIRT, True)
    ground = np.stack([hx + 0.10, 0.27 * one, 0.09 * one], -1)        # the left ankle with that foot on the ground
    for sgn, ph in ((-1.0, crank), (1.0, crank + math.pi)):
        ankle = P(bx + BIKE_CRANK * np.sin(ph) - 0.04, sgn * 0.11, bz + BIKE_CRANK * np.cos(ph) + 0.09)
        if sgn > 0:
            ankle = ankle + foot[:, None] * (ground - ankle)
        hip = P(hx, sgn * 0.055 * h, hz)
        knee, ankle = _knee(hip, ankle, l1, l2, P(1.0, sgn * 0.10, 0.2))
        add(_beam(hip, knee, yv, 0.036 * h, 0.036 * h), B_PANTS, True)
        add(_beam(knee, ankle, yv, 0.028 * h, 0.028 * h), B_PANTS, True)
        flat = foot[:, None] if sgn > 0 else 0.0     # the shoe lies along the bike; flat on the ground at a stop
        toe = xv * 0.17 - zv * 0.085 * (1 - flat) - np.array([0, 0, 0.065]) * flat
        heel = -xv * 0.07 - zv * 0.06 * (1 - flat) - np.array([0, 0, 0.065]) * flat
        add(_beam(ankle + heel, ankle + toe, yv, 0.045, 0.03, caps=True), B_DARK, True)

    verts = np.concatenate([p[0] for p in parts], 1)
    role = np.concatenate([np.full(p[0].shape[1], p[1], np.int32) for p in parts])
    sem = np.concatenate([np.full(p[0].shape[1], SEM["pedestrian"] if p[2] else SEM["vehicle"], np.uint8)
                          for p in parts])
    return verts, role, sem


def bike_palette(kind, colors):
    """(n, 9, 3) colours by role (B, G, R) from Bicycles.scene's colours (frame, shirt, pants, skin)."""
    n = len(kind)
    pal = np.zeros((n, 9, 3))
    pal[:, B_FRAME] = colors[:, 0]
    pal[:, B_TYRE] = (28, 28, 30)
    pal[:, B_DARK] = (40, 40, 44)
    pal[:, B_METAL] = (165, 165, 168)
    pal[:, B_SHIRT], pal[:, B_PANTS], pal[:, B_SKIN] = colors[:, 1], colors[:, 2], colors[:, 3]
    pal[:, B_HAIR] = (30, 34, 42)
    pal[kind == 1, B_HAIR] = (225, 225, 228)            # the sport rider's helmet
    pal[kind == 2, B_HAIR] = (150, 190, 205)            # the slow rider's hat
    pal[:, B_RED] = (40, 40, 200)
    return pal
