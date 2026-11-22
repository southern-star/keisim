"""Depth images of KeiView's cameras: the encoding, every pixel's line of sight, and back to the world (ReconView,
docs/recon_view.md). web/src/unproject.js is the same in JavaScript (web/tools/unproject_check.mjs compares them).

Depth is the straight distance r [m] from the camera centre to the surface, not z along the optical axis: a
cylindrical camera is stitched from pinhole faces (rig.py), z would jump at their seams.
* 16 bit: q = round(65535 ln(r / 0.25) / ln(400 / 0.25)); q = 0: no surface (sky, or beyond 400 m); nearer than
  0.25 m: q = 1. One step is 0.011 % of r.
* A line of sight is a unit vector in the camera's level frame (x right, y down, z forward: the rows of level_frame,
  the camera's heading without its pitch), and P = C + r (d.x right + d.y down + d.z forward).
* Pixel centres: KeiView draws pinhole images with pixel centres at half-integer coordinates, so pixel (v, u) of a
  pinhole camera looks through (u + 0.5, v + 0.5). The cylinder is stitched with maps shifted by half a pixel
  (keiview.py), which puts its pixel (v, u) at the integer coordinates (u, v) of rig.py's formulas.
* Half resolution (keipilot/data.py ShardWriter._seg, cv2.resize INTER_NEAREST): pixel (i, j) of the small image is
  pixel (2i, 2j) of the full one, not the centre of a 2x2 block: camera_rays(cfg, step=2).
"""
from __future__ import annotations

import math
from dataclasses import replace

import cv2
import numpy as np

from .camera import camera_pose
from .rig import cylinder_faces

DEPTH_NEAR, DEPTH_FAR = 0.25, 400.0
_LOG_RANGE = math.log(DEPTH_FAR / DEPTH_NEAR)
# stitch_depth "planar": second differences of 1/z (relative) under which a face pixel's neighbourhood counts as one
# plane; the 16 bit steps alone give up to 2.2e-4
PLANAR_TOL = 1e-3


# ---------------------------------------------------------------------- encoding
def encode_depth(r):
    """uint16 codes of distances r [m]; 0 where there is no surface (r <= 0, not finite, or beyond DEPTH_FAR)."""
    r = np.asarray(r, np.float64)
    ok = np.isfinite(r) & (r > 0) & (r <= DEPTH_FAR * (1 + 1e-9))
    q = np.rint(65535.0 * np.log(np.maximum(np.where(ok, r, DEPTH_NEAR), 1e-9) / DEPTH_NEAR) / _LOG_RANGE)
    return np.where(ok, np.clip(q, 1, 65535), 0).astype(np.uint16)


def decode_depth(q):
    """Distances [m] (float64) of uint16 codes; 0 where there is no surface."""
    q = np.asarray(q)
    return np.where(q > 0, DEPTH_NEAR * np.exp(q.astype(np.float64) * (_LOG_RANGE / 65535.0)), 0.0)


# ---------------------------------------------------------------------- lines of sight
def _pinhole(cfg):
    f = (cfg.width / 2.0) / math.tan(math.radians(cfg.fov_deg) / 2.0)
    return f, math.radians(cfg.pitch_deg)


def _cylinder(cfg):
    f = cfg.width / math.radians(cfg.fov_deg)
    return f, cfg.height / 2.0 - f * math.tan(math.radians(cfg.pitch_deg))


def camera_rays(cfg, step=1):
    """Unit lines of sight (H // step, W // step, 3) of a KeiView camera's pixels (step * i, step * j), in the camera's
    level frame (x right, y down, z forward)."""
    u, v = np.meshgrid(np.arange(0, cfg.width, step, dtype=np.float64), np.arange(0, cfg.height, step, dtype=np.float64))
    if cfg.projection == "cylindrical":
        f, cy = _cylinder(cfg)
        phi = (u - cfg.width / 2.0) / f
        d = np.stack([np.sin(phi), -(cy - v) / f, np.cos(phi)], -1)
    elif cfg.projection == "pinhole":
        f, p = _pinhole(cfg)
        a, b = (u + 0.5 - cfg.width / 2.0) / f, (v + 0.5 - cfg.height / 2.0) / f
        d = np.stack([a, b * math.cos(p) + math.sin(p), math.cos(p) - b * math.sin(p)], -1)
    else:
        raise ValueError(f"unknown projection {cfg.projection!r}")
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def level_frame(cfg, x, y, yaw, offset=None):
    """(C, R): the camera centre and its level frame (rows: right, down, forward) for the car at (x, y, yaw)."""
    return camera_pose(replace(cfg, pitch_deg=0.0), x, y, yaw, offset)


def unproject(r, rays, C, R):
    """World points (..., 3) of distances r [m] along rays (camera_rays) from the camera at level_frame (C, R).
    Pixels without a surface (r = 0) come back at C."""
    return C + (np.asarray(r, np.float64)[..., None] * rays) @ R


def project(cfg, pts, C, R):
    """Where world points (N, 3) are in a KeiView camera's image: (uv (N, 2) in pixel indices, as camera_rays counts
    them; distance r from the camera centre; in front of the camera and inside the image)."""
    pc = (np.asarray(pts, np.float64) - C) @ R.T
    r = np.linalg.norm(pc, axis=1)
    if cfg.projection == "cylindrical":
        f, cy = _cylinder(cfg)
        h = np.hypot(pc[:, 0], pc[:, 2])
        phi = np.arctan2(pc[:, 0], pc[:, 2])
        uv = np.stack([cfg.width / 2.0 + f * phi, cy + f * pc[:, 1] / np.maximum(h, 1e-9)], 1)
        ok = h > 1e-6
    else:
        f, p = _pinhole(cfg)
        cp, sp = math.cos(p), math.sin(p)
        yc, zc = pc[:, 1] * cp - pc[:, 2] * sp, pc[:, 1] * sp + pc[:, 2] * cp      # pitched down by p
        ok = zc > 1e-6
        zs = np.maximum(zc, 1e-9)
        uv = np.stack([cfg.width / 2.0 + f * pc[:, 0] / zs - 0.5, cfg.height / 2.0 + f * yc / zs - 0.5], 1)
    ok = ok & (uv[:, 0] > -0.5) & (uv[:, 0] < cfg.width - 0.5) & (uv[:, 1] > -0.5) & (uv[:, 1] < cfg.height - 0.5)
    return uv, r, ok


# ---------------------------------------------------------------------- the cylinder's depth from its faces
class DepthStitcher:
    """Depth of a cylindrical camera from the depth images of its pinhole faces (cylinder_faces), drawn by KeiView.

    "nearest": every pixel takes the face pixel cv2.remap(INTER_NEAREST) picks, the one its label comes from. That
    face pixel looks up to half a face pixel (0.22 deg) past the cylinder pixel's own line of sight, which is up to
    3 % of the distance on the ground 12 m ahead.
    "planar" (the default): the same face pixel, and where the surface around it is one plane (1/z of its four
    neighbours in the face on one line each way), the distance where the cylinder pixel's own line of sight meets
    that plane; 1/z of a plane is linear in a pinhole image, so this is exact for planes. At edges and on curved
    surfaces it is "nearest".
    """

    def __init__(self, cfg, n=3, oversample=1.25, mode="planar", tol=PLANAR_TOL):
        if mode not in ("nearest", "planar"):
            raise ValueError(f"mode must be nearest or planar (got {mode!r})")
        self.cfg, self.mode, self.tol = cfg, mode, tol
        self.faces, map_x, map_y, _ = cylinder_faces(cfg, n, oversample)
        self.maps = (map_x - 0.5, map_y - 0.5)                  # KeiView's pixel centres (keiview.py)
        Wf, Hf = self.faces[0].width, self.faces[0].height
        self.Wf, self.Hf, self.n = Wf, Hf, n
        # the face pixel of every cylinder pixel, as cv2.remap picks it for the labels (-1: outside the atlas)
        idx = np.arange(n * Wf * Hf, dtype=np.float32).reshape(Hf, n * Wf)
        self.idx = cv2.remap(idx, *self.maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=-1).astype(np.int64)
        self.ok = self.idx >= 0
        i = np.where(self.ok, self.idx, 0)
        self.row, self.col = i // (n * Wf), i % (n * Wf)
        ff = (Wf / 2.0) / math.tan(math.radians(self.faces[0].fov_deg) / 2.0)
        # |(a, b, 1)| of every face pixel centre (r = z * s), and of every cylinder pixel's place in its face
        a = ((np.arange(n * Wf) % Wf) + 0.5 - Wf / 2.0) / ff
        b = (np.arange(Hf) + 0.5 - Hf / 2.0) / ff
        self.s_atlas = np.sqrt(a[None] ** 2 + b[:, None] ** 2 + 1.0)
        mx, my = self.maps[0].astype(np.float64), self.maps[1].astype(np.float64)
        self.ax, self.ay = mx - self.col, my - self.row          # from the picked pixel to the place, in face pixels
        face = self.col // Wf
        self.s_px = np.sqrt(((mx - face * Wf + 0.5 - Wf / 2.0) / ff) ** 2 + ((my + 0.5 - Hf / 2.0) / ff) ** 2 + 1.0)
        # the picked pixel's neighbours in its own face
        self.in_x = (self.col % Wf > 0) & (self.col % Wf < Wf - 1)
        self.in_y = (self.row > 0) & (self.row < Hf - 1)

    def __call__(self, faces_q):
        """uint16 depth (H, W) of the cylinder from the uint16 depth images of the faces (top row first)."""
        atlas = np.concatenate(faces_q, 1)
        q0 = np.where(self.ok, atlas[self.row, self.col], 0).astype(np.uint16)
        if self.mode == "nearest":
            return q0
        r = decode_depth(atlas)
        w = np.where(r > 0, self.s_atlas / np.maximum(r, 1e-9), 0.0)          # 1/z; 0: no surface
        row, col = self.row, self.col
        w0 = w[row, col]
        xm, xp = w[row, np.maximum(col - 1, 0)], w[row, np.minimum(col + 1, w.shape[1] - 1)]
        ym, yp = w[np.maximum(row - 1, 0), col], w[np.minimum(row + 1, w.shape[0] - 1), col]
        tol = self.tol * w0
        plane = (self.ok & self.in_x & self.in_y & (w0 > 0) & (xm > 0) & (xp > 0) & (ym > 0) & (yp > 0)
                 & (np.abs(xm + xp - 2 * w0) <= tol) & (np.abs(ym + yp - 2 * w0) <= tol))
        wx = np.where(self.ax >= 0, xp - w0, w0 - xm) * self.ax
        wy = np.where(self.ay >= 0, yp - w0, w0 - ym) * self.ay
        wp = w0 + wx + wy
        plane &= wp > 0
        out = encode_depth(self.s_px / np.where(plane, wp, 1.0))
        return np.where(plane, out, q0).astype(np.uint16)
