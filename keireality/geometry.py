"""Calibration and same-time surface visibility. All input depths are radial uint16 codes."""
from dataclasses import replace
import math

import numpy as np

from keipilot.filldata import CHASE
from keisim.render import rays
from keisim.render.camera import camera_pose


UNSEEN, VISIBLE, UNKNOWN = 0, 1, 255
CHASE_CAMERA = replace(CHASE, width=256, height=128)
ORBIT_ANGLES = (0, -36, 36, -18, 18)


def orbit_camera(angle):
    """A calibrated orbit looking at the same point 12 m ahead; 4–6 m over the training arc."""
    if angle == 0:
        return CHASE_CAMERA
    a = math.radians(angle)
    x, y = -8 * math.cos(a), 8 * math.sin(a)
    z = 4 + abs(angle) / 18
    return replace(CHASE_CAMERA, name=f"orbit_{angle:+g}", x=x, y=y, z=z,
                   yaw_deg=math.degrees(math.atan2(-y, 12 - x)),
                   pitch_deg=math.degrees(math.atan2(z, math.hypot(12 - x, y))))


def view_pose(cam):
    """In the ego frame (+forward, +left, +up): position, forward, up, vertical FOV in degrees, aspect."""
    pos, rotation = camera_pose(cam, 0., 0., 0.)
    aspect = cam.width / cam.height
    vfov = math.degrees(2 * math.atan(math.tan(math.radians(cam.fov_deg) / 2) / aspect))
    return np.asarray([*pos, *rotation[2], *-rotation[1], vfov, aspect], dtype=np.float32)


def relative_pose(previous, current):
    """SE(2) displacement in the previous ego frame; yaw in radians."""
    if previous is None:
        return np.zeros(3, np.float32)
    dx, dy, angle = np.subtract(current, previous)
    c, s = math.cos(previous[2]), math.sin(previous[2])
    return np.asarray([c * dx + s * dy, -s * dx + c * dy, math.atan2(math.sin(angle), math.cos(angle))],
                      dtype=np.float32)


def visibility_now(target, target_codes, cameras, input_codes):
    """Test each target surface in both driving cameras. A discontinuity never creates an intermediate surface.

    A nearest-pixel depth match within 3% gives support. Foreground occlusion and outside-FOV give unseen.
    Missing depth, a surface in front of the measured input, and ambiguous edges give unknown. Sky has no surface.
    """
    depth = rays.decode_depth(target_codes)
    points = rays.unproject(depth, rays.camera_rays(target), *rays.level_frame(target, 0., 0., 0.)).reshape(-1, 3)
    surface = depth.ravel() > 0
    mask = np.full(surface.size, UNSEEN, np.uint8)
    mask[~surface] = UNKNOWN
    for cam, codes in zip(cameras, input_codes, strict=True):
        uv, distance, inside = rays.project(cam, points, *rays.level_frame(cam, 0., 0., 0.))
        idx = np.flatnonzero(surface & inside & (mask != VISIBLE))
        if not idx.size:
            continue
        u, v = uv[idx].T
        d = rays.decode_depth(codes)
        nearest = d[np.floor(v + .5).astype(int), np.floor(u + .5).astype(int)]
        expected = distance[idx]
        match = (nearest > 0) & (np.abs(nearest - expected) <= .03 * expected)
        # Where projected pixels straddle a depth edge, a neighbour may see the target or even sky.
        u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
        ambiguous = (nearest == 0) | (nearest > 1.03 * expected)
        for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
            ud, vd = u0 + du, v0 + dv
            contributes = (np.abs(u - ud) < 1 - 1e-7) & (np.abs(v - vd) < 1 - 1e-7)
            neighbour = d[vd.clip(0, cam.height - 1), ud.clip(0, cam.width - 1)]
            ambiguous |= contributes & ((neighbour == 0) | (neighbour >= .97 * expected))
        mask[idx[ambiguous]] = UNKNOWN
        mask[idx[match]] = VISIBLE
    return mask.reshape(depth.shape)
