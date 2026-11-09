"""Cylindrical cameras for KeiSim's CPU renderer: a wide view stitched from pinhole faces.

CameraRenderer draws pinhole images only (an exact ground homography, straight polygon edges). A cylindrical camera
(CameraConfig.projection) is drawn as FACES pinhole images side by side around its vertical axis and resampled with
one cv2.remap:
* column u looks (u - W/2) / f to the right of the camera's heading, with f = W / fov (radians);
* row v looks at the elevation e with tan e = (cy - v) / f, and cy = H/2 - f tan(pitch).
The horizon is the straight row cy, and every row is one elevation. The faces colour their sky by each pixel's
elevation (CameraRenderer's sky_tan_top), so the sky has no seams. KeiView draws the same faces with its own renderer
and stitches them with the same maps (cylinder_faces, keisim/render/keiview.py).
"""
from __future__ import annotations

import math
from dataclasses import replace

import cv2
import numpy as np

from ..config import SEM, CameraConfig
from .camera import CameraRenderer, camera_pose


class CylinderRenderer:
    NEAR = 0.12
    FACES = 3                 # pinhole faces over the field of view (60 deg each for 180 deg)
    OVERSAMPLE = 1.25         # face pixels per cylinder pixel at the centre of a face

    def __init__(self, town, tex, static, cfg: CameraConfig):
        self.town, self.tex, self.static, self.cfg = town, tex, static, cfg
        self.W, self.H = cfg.width, cfg.height
        self.fov = math.radians(cfg.fov_deg)
        self.f = self.W / self.fov
        self.cy = self.H / 2.0 - self.f * math.tan(math.radians(cfg.pitch_deg))
        faces, self.map_x, self.map_y, tan_top = cylinder_faces(cfg, self.FACES, self.OVERSAMPLE)
        self.faces = [CameraRenderer(town, tex, static, fc, sky_tan_top=tan_top) for fc in faces]
        self.face_size = (faces[0].width, faces[0].height)

    # ------------------------------------------------------------------ pose
    def camera_pose(self, x, y, yaw, offset=None):
        """Centre and level frame of the rig (rows: right, down, forward); the pitch is in cy."""
        return camera_pose(replace(self.cfg, pitch_deg=0.0), x, y, yaw, offset)

    def project_ground(self, pts_world, C, R, scale=1, z=0.0):
        """Project world points (N,2) at height z into the image. Returns (N,2), valid mask."""
        P = np.concatenate([pts_world, np.full((len(pts_world), 1), z)], 1)
        pc = (P - C) @ R.T
        h = np.hypot(pc[:, 0], pc[:, 2])
        phi = np.arctan2(pc[:, 0], pc[:, 2])
        ok = (h > self.NEAR) & (np.abs(phi) < self.fov / 2)
        uv = np.stack([self.W / 2.0 + self.f * phi, self.cy + self.f * pc[:, 1] / np.maximum(h, 1e-6)], 1)
        return uv * scale, ok

    def render(self, pose, scene, weather, want_rgb=True, want_seg=True, offset=None):
        views = [fc.render(pose, scene, weather, want_rgb=want_rgb, want_seg=want_seg, offset=offset)
                 for fc in self.faces]
        rgb = seg = None
        if want_rgb:
            rgb = cv2.remap(np.concatenate([v[0] for v in views], 1), self.map_x, self.map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        if want_seg:
            seg = cv2.remap(np.concatenate([v[1] for v in views], 1), self.map_x, self.map_y, cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=int(SEM["sky"]))
        return rgb, seg


def cylinder_faces(cfg: CameraConfig, n=3, oversample=1.25):
    """The n pinhole faces of a cylindrical camera, side by side over its field of view, and where each pixel of the
    cylinder is in the atlas of the faces side by side (pixel centres at integer coordinates, as CameraRenderer draws):
    (face configs, map_x, map_y, tan of the elevation of the top row). KeiView draws the same faces
    (keisim/render/keiview.py)."""
    W, H = cfg.width, cfg.height
    fov = math.radians(cfg.fov_deg)
    f = W / fov
    cy = H / 2.0 - f * math.tan(math.radians(cfg.pitch_deg))
    span = fov / n
    # direction of every pixel in the level frame of the rig: x right, y down, z forward, with a horizontal length of 1
    phi = (np.arange(W) - W / 2.0) / f
    tan_e = (cy - np.arange(H)) / f
    k = np.clip(np.floor((phi + fov / 2) / span), 0, n - 1).astype(int)
    rel = phi - (k - (n - 1) / 2.0) * span                 # azimuth from the centre of the pixel's face
    # faces look down at the middle of the elevations they show
    e_top, e_bot = math.atan(tan_e[0]), math.atan(tan_e[-1])
    p = -(e_top + e_bot) / 2
    cp, sp = math.cos(p), math.sin(p)
    dx = np.broadcast_to(np.sin(rel)[None], (H, W))
    dz = np.broadcast_to(np.cos(rel)[None], (H, W))
    dy = np.broadcast_to(-tan_e[:, None], (H, W))
    yc, zc = dy * cp - dz * sp, dy * sp + dz * cp            # pitched down by p (as camera_pose)
    ff = oversample * f
    Wf = int(math.ceil(2 * ff * math.tan(span / 2) + 2))
    Hf = int(math.ceil(2 * ff * np.abs(yc / zc).max() + 2))
    faces = []
    for i in range(n):
        centre = (i - (n - 1) / 2.0) * span                 # + to the right
        faces.append(replace(cfg, projection="pinhole", width=Wf, height=Hf,
                             fov_deg=math.degrees(2 * math.atan(Wf / 2.0 / ff)), pitch_deg=math.degrees(p),
                             yaw_deg=cfg.yaw_deg - math.degrees(centre), name=f"{cfg.name}_face{i}"))
    map_x = (k * Wf + Wf / 2.0 + ff * dx / zc).astype(np.float32)
    map_y = (Hf / 2.0 + ff * yc / zc).astype(np.float32)
    return faces, map_x, map_y, float(tan_e[0])


def make_renderer(town, tex, static, cfg: CameraConfig):
    """KeiSim's renderer of one camera of a rig."""
    if cfg.projection == "pinhole":
        return CameraRenderer(town, tex, static, cfg)
    if cfg.projection == "cylindrical":
        return CylinderRenderer(town, tex, static, cfg)
    raise ValueError(f"unknown projection {cfg.projection!r}")
