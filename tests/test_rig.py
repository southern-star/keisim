"""Camera rigs (keisim/config.py CAMERA_RIGS, keisim/render/rig.py; run: python -m pytest tests -q)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keisim.config import SEM, CameraConfig, EnvConfig, set_camera_rig  # noqa: E402
from keisim.env import KeiEnv  # noqa: E402

GROUND = [SEM[k] for k in ("road", "marking", "sidewalk", "terrain")]


def rig_env(rig="wide_narrow"):
    env = KeiEnv(set_camera_rig(EnvConfig(), rig))
    obs = env.reset(town_seed=5, episode_seed=1, weather="noon")
    return env, obs


def test_default_rig_is_the_front_camera():
    cfg = EnvConfig()
    assert cfg.cameras == (CameraConfig(),) == set_camera_rig(EnvConfig(), "front").cameras
    obs = KeiEnv(cfg).reset(town_seed=5, episode_seed=1)
    assert "rgbs" not in obs and obs["rgb"].shape == (160, 320, 3)


def test_wide_narrow_rig_renders_every_camera():
    env, obs = rig_env()
    assert [c.name for c in env.cfg.cameras] == ["wide", "narrow"]
    assert len(obs["rgbs"]) == len(obs["segs"]) == 2 and obs["rgbs"][0] is obs["rgb"]
    for rgb, seg in zip(obs["rgbs"], obs["segs"]):
        assert rgb.shape == (160, 320, 3) and rgb.dtype == np.uint8 and seg.shape == (160, 320)
    views = env.render_cameras(offset=(0.8, 0.1, 0.0), want_rgb=False)
    assert [v[0] for v in views] == [None, None] and all(v[1].shape == (160, 320) for v in views)


def test_cylinder_ground_matches_the_map():
    """The wide camera's segmentation (pinhole faces resampled into the cylinder) shows the map's ground class where
    the cylindrical projection puts each ground point, 3-40 m away in every direction of its 180 deg."""
    env, obs = rig_env()
    cam, tex = env.assets.cameras[0], env.assets.tex
    C, R = cam.camera_pose(*env.world.ego.pose)
    r, a = np.meshgrid(np.linspace(3.0, 40.0, 60), np.radians(np.linspace(-88.0, 88.0, 120)))
    x, y, yaw = env.world.ego.pose
    pts = np.stack([C[0] + r * np.cos(yaw - a), C[1] + r * np.sin(yaw - a)], -1).reshape(-1, 2)
    uv, ok = cam.project_ground(pts, C, R)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok &= (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
    # the map's class at the point, where it is the same 0.3 m around (no class edges)
    sem0, res, ras = tex.sem[0], tex.res, tex.raster
    col = np.round((pts[:, 0] - ras.x0) / res - 0.5).astype(int)
    row = np.round((ras.y1 - pts[:, 1]) / res - 0.5).astype(int)
    ok &= (col >= 3) & (col < sem0.shape[1] - 3) & (row >= 3) & (row < sem0.shape[0] - 3)
    want = np.full(len(pts), 255)
    for i in np.nonzero(ok)[0]:
        patch = sem0[row[i] - 3:row[i] + 4, col[i] - 3:col[i] + 4]
        if (patch == patch[3, 3]).all():
            want[i] = patch[3, 3]
    got = obs["segs"][0][np.clip(v, 0, cam.H - 1), np.clip(u, 0, cam.W - 1)]
    m = ok & (want != 255) & np.isin(got, GROUND)         # not hidden by an object
    assert m.sum() > 1500
    assert np.mean(got[m] == want[m]) > 0.97


def test_cylinder_has_no_seams():
    """Sky and ground of the faces (without objects) resampled into the cylinder: every sky row is one colour (the
    faces colour it by elevation; by their rows, a row varied by 20 levels), and across a seam the ground changes no
    more than next to it."""
    import cv2
    env, _ = rig_env()
    cam = env.assets.cameras[0]
    faces = []
    for fc in cam.faces:
        out = np.empty((fc.H, fc.W, 3), np.uint8)
        C, R = fc.camera_pose(*env.world.ego.pose)
        fc._ground(out, C, R, 1, False, env.weather)
        faces.append(out)
    img = cv2.remap(np.concatenate(faces, 1), cam.map_x, cam.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    d = np.abs(np.diff(img.astype(int), axis=1)).max(2)            # d[:, u]: between columns u and u + 1
    face = np.floor((np.arange(cam.W) - cam.W / 2.0) / cam.f / (cam.fov / cam.FACES) + cam.FACES / 2.0)
    seams = np.nonzero(np.diff(face) != 0)[0]
    assert len(seams) == cam.FACES - 1
    # (the top rows, where the sky gradient is steepest, vary up to 5 levels from the resampling)
    sky, ground = slice(8, int(cam.cy) - 3), slice(int(cam.cy) + 3, cam.H)
    rows = img[sky].astype(int)
    assert (rows.max(1) - rows.min(1)).max() <= 2
    for u in seams:
        near = np.concatenate([d[ground, u - 4:u - 1], d[ground, u + 2:u + 5]], 1).mean()
        assert d[ground, u].mean() < 1.5 * near + 1
    assert math.isclose(cam.cy, 80.0 - cam.f * math.tan(math.radians(8.0)))
