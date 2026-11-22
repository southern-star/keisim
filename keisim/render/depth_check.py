"""Checks of KeiView's depth images against what must hold in a KeiSim town (ReconView's G0, docs/recon_view.md):
the flat ground (a) and the same surfaces seen from another place or by another camera (b). Used by
tests/test_recon.py and tools/recon_g0.py."""
from __future__ import annotations

import math

import numpy as np

from ..config import SEM
from . import rays as RY

# heights above KeiSim's flat ground of what KeiView draws (web/src/world/roads.js): the asphalt, the paint 8 mm above
# it, the sidewalk slab (web/src/town.js SIDEWALK_Y)
GROUND_Z = {"road": 0.0, "marking": 0.008, "sidewalk": 0.15}


def plane_errors(cfg, dep, seg, pose, step=1):
    """{class: relative errors} of a camera's depth image against the flat ground of its labels (check a)."""
    C, R = RY.level_frame(cfg, *pose)
    D = RY.camera_rays(cfg, step) @ R
    r = RY.decode_depth(dep[::step, ::step])
    seg = seg[::step, ::step]
    out = {}
    for name, z in GROUND_Z.items():
        m = (seg == SEM[name]) & (r > 0) & (D[..., 2] < -1e-6)
        want = (z - C[2]) / np.where(m, D[..., 2], -1.0)
        out[name] = np.abs(r[m] - want[m]) / want[m]
    return out


def heights(cfg, dep, seg, pose, name, max_r=40.0):
    """Heights [m] above the ground of the pixels of a class within max_r, from their depth."""
    C, R = RY.level_frame(cfg, *pose)
    r = RY.decode_depth(dep)
    return RY.unproject(r, RY.camera_rays(cfg), C, R)[..., 2][(seg == SEM[name]) & (r > 0) & (r < max_r)]


def pixel_angle(cfg):
    """The angle between neighbouring pixels at the image centre [rad]."""
    return math.radians(cfg.fov_deg) / cfg.width if cfg.projection == "cylindrical" \
        else 2.0 * math.tan(math.radians(cfg.fov_deg) / 2.0) / cfg.width


def reproject(cfg_a, dep_a, pose_a, cfg_b, dep_b, pose_b, tol=0.02):
    """Check (b): camera A's pixels as world points in camera B's image. Returns (the share of them that B looks
    through: B's nearest surface within the footprint of A's pixel is more than tol behind the point, or B shows none;
    how many landed in B; the median relative difference to B's depth where B's four pixels around the point are one
    smooth surface).
    A pixel of A says that there is a surface at its distance somewhere inside the pixel, so B is asked for its nearest
    surface inside that footprint (at least the 2x2 pixels around the point: against B's one nearest pixel, the flat
    ground alone fails, a pixel row of the wide camera being 19 % of the distance 30 m ahead)."""
    Ca, Ra = RY.level_frame(cfg_a, *pose_a)
    ra = RY.decode_depth(dep_a)
    m = ra > 0
    P = RY.unproject(ra[m], RY.camera_rays(cfg_a)[m], Ca, Ra)
    Cb, Rb = RY.level_frame(cfg_b, *pose_b)
    uv, r, ok = RY.project(cfg_b, P, Cb, Rb)
    uv, r = uv[ok], r[ok]
    rb = RY.decode_depth(dep_b)
    far = np.where(rb > 0, rb, np.inf)
    half = max(0.5, 0.5 * pixel_angle(cfg_a) / pixel_angle(cfg_b))            # A's pixel in B's pixels, from its centre
    near = np.full(len(r), np.inf)
    k = int(math.ceil(half))
    for dv in range(-k, k + 2):
        for du in range(-k, k + 2):
            u = np.floor(uv[:, 0]).astype(int) + du
            v = np.floor(uv[:, 1]).astype(int) + dv
            inside = (np.abs(u - uv[:, 0]) <= half + 0.5) & (np.abs(v - uv[:, 1]) <= half + 0.5)
            inside &= (u >= 0) & (u < cfg_b.width) & (v >= 0) & (v < cfg_b.height)
            near = np.where(inside, np.minimum(near, far[np.clip(v, 0, cfg_b.height - 1), np.clip(u, 0, cfg_b.width - 1)]), near)
    through = float(np.mean(near > r * (1 + tol)))
    one = far[np.clip(np.rint(uv[:, 1]).astype(int), 0, cfg_b.height - 1), np.clip(np.rint(uv[:, 0]).astype(int), 0, cfg_b.width - 1)]
    reproject.nearest_pixel = float(np.mean(one > r * (1 + tol)))            # against B's one nearest pixel (the record)
    u0, v0 = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    q = (u0 >= 0) & (v0 >= 0) & (u0 < cfg_b.width - 1) & (v0 < cfg_b.height - 1)
    u0, v0, fu, fv = u0[q], v0[q], uv[q, 0] - u0[q], uv[q, 1] - v0[q]
    n4 = np.stack([rb[v0, u0], rb[v0, u0 + 1], rb[v0 + 1, u0], rb[v0 + 1, u0 + 1]])
    smooth = (n4.min(0) > 0) & (n4.max(0) < 1.03 * n4.min(0))
    interp = n4[0] * (1 - fu) * (1 - fv) + n4[1] * fu * (1 - fv) + n4[2] * (1 - fu) * fv + n4[3] * fu * fv
    return through, int(ok.sum()), float(np.median(np.abs(interp[smooth] - r[q][smooth]) / r[q][smooth]))
