"""ReconView's geometry (docs/recon_view.md): depth codes, lines of sight, the cylinder's depth from its faces, Python
against JavaScript, and KeiView's depth pass (run: python -m pytest tests/test_recon.py -q). The KeiView tests are
skipped without `npm install` in web/ or a Chrome."""
import base64
import json
import math
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from keisim.config import CAMERA_RIGS, SEM, EnvConfig, set_camera_rig  # noqa: E402
from keisim.render import rays as RY  # noqa: E402
from keisim.render.depth_check import heights, plane_errors, reproject  # noqa: E402
from keisim.render.camera import CameraRenderer  # noqa: E402
from keisim.render.rig import CylinderRenderer  # noqa: E402

WIDE, NARROW = CAMERA_RIGS["wide_narrow"]
POSE = (12.5, -7.25, 0.6)


def rel_err(got, want, m):
    return np.abs(got[m] - want[m]) / want[m]


# ---------------------------------------------------------------------- codes
def test_depth_codes():
    r = np.array([0.0, -1.0, np.nan, np.inf, 0.1, 0.25, 1.0, 12.0, 399.99, 400.0, 400.5])
    q = RY.encode_depth(r)
    assert q.dtype == np.uint16
    assert q[:4].tolist() == [0, 0, 0, 0] and q[4] == 1 and q[5] == 1 and q[9] == 65535 and q[10] == 0
    back = RY.decode_depth(q)
    assert back[0] == 0 and back[10] == 0
    assert np.allclose(back[6:10], r[6:10], rtol=0.6e-4)                    # half a step: 0.0056 %
    step = RY.decode_depth(np.array([30001])) / RY.decode_depth(np.array([30000])) - 1
    assert abs(step[0] - math.log(1600) / 65535) < 1e-8                      # 0.011 % a step
    codes = np.arange(1, 65536)
    assert np.array_equal(RY.encode_depth(RY.decode_depth(codes)), codes)


# ---------------------------------------------------------------------- lines of sight
def test_half_resolution_is_every_other_pixel():
    """cv2.resize INTER_NEAREST to half size (keipilot/data.py _seg) keeps pixel (2i, 2j), so the lines of sight of a
    half-resolution depth image are those of the full image's even pixels, not of the centres of 2x2 blocks."""
    a = np.arange(160 * 320, dtype=np.uint16).reshape(160, 320)
    half = cv2.resize(a, (160, 80), interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(half, a[::2, ::2])
    for cfg in (WIDE, NARROW):
        full = RY.camera_rays(cfg)
        assert full.shape == (160, 320, 3) and np.allclose(np.linalg.norm(full, axis=-1), 1.0)
        assert np.array_equal(RY.camera_rays(cfg, 2), full[::2, ::2])


def test_rays_match_keisim_projection():
    """A point along a pixel's line of sight projects back to that pixel: with rays.project, and with KeiSim's own
    renderers (whose pinhole pixel centres are half a pixel from KeiView's)."""
    rng = np.random.default_rng(0)
    for cfg in (WIDE, NARROW):
        C, R = RY.level_frame(cfg, *POSE)
        d = RY.camera_rays(cfg)
        r = rng.uniform(2.0, 60.0, d.shape[:2])
        P = RY.unproject(r, d, C, R).reshape(-1, 3)
        v, u = np.mgrid[0:cfg.height, 0:cfg.width]
        uv, rr, ok = RY.project(cfg, P, C, R)
        assert ok.all() and np.allclose(rr, r.ravel())
        assert np.abs(uv - np.stack([u.ravel(), v.ravel()], 1)).max() < 1e-6
        # KeiSim: points on the ground
        g = d[..., 1] > 0.05
        rg = cfg.z / d[..., 1][g]
        Pg = RY.unproject(rg, d[g], C, R)
        assert np.abs(Pg[:, 2]).max() < 1e-9
        if cfg.projection == "cylindrical":
            cam = CylinderRenderer.__new__(CylinderRenderer)
            cam.cfg, cam.W, cam.H = cfg, cfg.width, cfg.height
            cam.fov = math.radians(cfg.fov_deg)
            cam.f = cam.W / cam.fov
            cam.cy = cam.H / 2.0 - cam.f * math.tan(math.radians(cfg.pitch_deg))
            Ck, Rk = cam.camera_pose(*POSE)
            uvk, okk = cam.project_ground(Pg[:, :2], Ck, Rk)
            shift = 0.0
        else:
            cam = CameraRenderer.__new__(CameraRenderer)
            cam.cfg, cam.W, cam.H = cfg, cfg.width, cfg.height
            cam.f = (cam.W / 2.0) / math.tan(math.radians(cfg.fov_deg) / 2.0)
            Ck, Rk = cam.camera_pose(*POSE)
            uvk, okk = cam.project_ground(Pg[:, :2], Ck, Rk)
            shift = 0.5                                                       # KeiView's pixel centres
        assert okk.mean() > 0.99                                              # (the cylinder's first column is its edge)
        assert np.abs(uvk - shift - np.stack([u[g], v[g]], 1))[okk].max() < 1e-6


# ---------------------------------------------------------------------- the cylinder's depth from its faces
def _hit(C, D, wall=None):
    """Distance along world rays D (.., 3) from C to the ground z = 0 and, with wall = (x0, half width, height), to
    the wall x = x0, |y| <= half width, z <= height. 0: nothing hit."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(D[..., 2] < -1e-9, -C[2] / D[..., 2], np.inf)
        if wall is not None:
            x0, hw, top = wall
            tw = np.where(D[..., 0] > 1e-9, (x0 - C[0]) / D[..., 0], np.inf)
            Pw = C + tw[..., None] * D
            tw = np.where(np.isfinite(tw) & (np.abs(Pw[..., 1]) <= hw) & (Pw[..., 2] <= top) & (Pw[..., 2] >= 0), tw, np.inf)
            t = np.minimum(t, tw)
    return np.where(np.isfinite(t), t, 0.0)


def _scene(wall=None, pose=(0.0, 0.0, 0.0)):
    """(face depth codes, exact distance along every cylinder pixel's line of sight) of the wide camera."""
    st = RY.DepthStitcher(WIDE)
    faces = []
    for fc in st.faces:
        C, R = RY.level_frame(fc, *pose)
        faces.append(RY.encode_depth(_hit(C, RY.camera_rays(fc) @ R, wall)))
    C, R = RY.level_frame(WIDE, *pose)
    return faces, _hit(C, RY.camera_rays(WIDE) @ R, wall)


def test_cylinder_depth_of_a_plane():
    """Check (a) without a renderer: the ground through the faces of the wide camera. Stitched "planar", the depth is
    the distance along the cylinder pixel's own line of sight (median < 0.5 %, 95th percentile < 2 %, at full and at
    half resolution); stitched by the nearest face pixel alone it is not (the reason for "planar")."""
    faces, want = _scene()
    planar = RY.decode_depth(RY.DepthStitcher(WIDE, mode="planar")(faces))
    nearest = RY.decode_depth(RY.DepthStitcher(WIDE, mode="nearest")(faces))
    m = (want > 0) & (want < RY.DEPTH_FAR)
    assert m.sum() > 25000 and ((planar > 0) == (nearest > 0)).all() and (planar[m] > 0).all()
    for sl in (np.s_[:, :], np.s_[::2, ::2]):
        e = rel_err(planar[sl], want[sl], m[sl])
        assert np.median(e) < 0.005 and np.percentile(e, 95) < 0.02
        assert np.percentile(e, 95) < 0.001                                  # in fact exact to the 16 bit steps
        assert np.percentile(rel_err(nearest[sl], want[sl], m[sl]), 95) > 0.02


def test_cylinder_depth_keeps_edges():
    """At the edges of a wall on the ground, "planar" does not blend the two surfaces: every pixel is the nearest face
    pixel's depth or on one of the two planes its neighbourhood shows."""
    for wall in ((10.0, 3.0, 2.5), (25.0, 1.0, 6.0)):
        faces, want = _scene(wall, pose=(0.0, 0.4, 0.05))
        planar = RY.DepthStitcher(WIDE, mode="planar")(faces)
        nearest = RY.DepthStitcher(WIDE, mode="nearest")(faces)
        assert ((planar > 0) == (nearest > 0)).all()
        p, n = RY.decode_depth(planar), RY.decode_depth(nearest)
        m = (want > 0) & (n > 0)
        exact = np.abs(p - want) <= 2e-3 * want
        assert (exact | (planar == nearest))[m].all()
        assert exact[m].mean() > 0.97 and (np.abs(n - want) <= 2e-3 * want)[m].mean() < 0.5


# ---------------------------------------------------------------------- Python against JavaScript
def _js_cam(cfg):
    return {"width": cfg.width, "height": cfg.height, "fov": cfg.fov_deg, "x": cfg.x, "y": cfg.y, "z": cfg.z,
            "pitch": cfg.pitch_deg, "yaw": cfg.yaw_deg, "projection": cfg.projection, "name": cfg.name}


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a, "<u2").tobytes()).decode()


def test_python_and_javascript_agree(tmp_path):
    """Check (d): web/src/unproject.js gives the world points of keisim/render/rays.py within 1 mm (and the same codes,
    projections and stitched cylinder depth), for src/rig.js's own copy of the rig."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("no node")
    rng = np.random.default_rng(1)
    r = np.concatenate([[0.0, 0.1, 0.25, 400.0, 401.0], np.exp(rng.uniform(math.log(0.2), math.log(420.0), 500))])
    q = RY.encode_depth(r)
    fx = {"rig": "wide_narrow", "pose": list(POSE),
          "encode": {"r": r.tolist(), "q": q.tolist(), "back": RY.decode_depth(q).tolist()}, "cams": []}
    for index, cfg in enumerate(CAMERA_RIGS["wide_narrow"]):
        C, R = RY.level_frame(cfg, *POSE)
        for step in (1, 2):
            d = RY.camera_rays(cfg, step)
            qd = RY.encode_depth(np.exp(rng.uniform(math.log(0.3), math.log(390.0), d.shape[:2])))
            qd[rng.random(qd.shape) < 0.1] = 0
            world = RY.unproject(RY.decode_depth(qd), d, C, R)
            pts = C + rng.normal(0, 20.0, (300, 3))
            uv, rr, ok = RY.project(cfg, pts, C, R)
            fx["cams"].append({"index": index, "step": step, "q": _b64(qd), "world": world.ravel().tolist(),
                               "proj": {"pts": pts.tolist(), "uv": uv.tolist(), "r": rr.tolist(), "ok": ok.tolist()}})
    faces, _ = _scene((10.0, 3.0, 2.5), pose=(0.0, 0.4, 0.05))
    fx["stitch"] = {"index": 0, "atlas": _b64(np.concatenate(faces, 1)),
                    "nearest": _b64(RY.DepthStitcher(WIDE, mode="nearest")(faces)),
                    "planar": _b64(RY.DepthStitcher(WIDE, mode="planar")(faces))}
    assert _js_cam(WIDE)["fov"] == 180.0
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fx))
    res = subprocess.run([node, "tools/unproject_check.mjs", str(path)], cwd=os.path.join(ROOT, "web"),
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert res.stdout.count("ok  ") == 1 + 2 * 4 + 2


# ---------------------------------------------------------------------- KeiView's depth pass
def keiview_rig_env():
    if not os.path.isdir(os.path.join(ROOT, "web", "node_modules", "puppeteer-core")):
        pytest.skip("KeiView needs `npm install` in web/")
    from keisim.env import KeiEnv
    env = KeiEnv(set_camera_rig(EnvConfig(renderer="keiview"), "wide_narrow"))
    try:
        env.reset(town_seed=5, episode_seed=1, weather="noon")
    except RuntimeError as e:
        env.close()
        pytest.skip(f"KeiView did not start: {e}")
    return env


def test_keiview_depth():
    """KeiView's depth pass on the wide_narrow rig (G0): asking for depth changes no image or label (and without it
    the request is the old one); depth and labels show the same pixels (c); the ground's depth is the distance to the
    flat ground along each pixel's line of sight, at full and half resolution (a); and the points of one frame are
    surfaces in a frame taken 3 m further on, and in the other camera (b)."""
    env = keiview_rig_env()
    try:
        kv, pose, state, t = env.keiview(), env.world.ego.pose, env.world.scene_state(), env.world.t
        with pytest.raises(ValueError):
            from keisim.env import KeiEnv
            KeiEnv(set_camera_rig(EnvConfig(), "wide_narrow")).render_cameras_depth()
        before = kv.render_rig(pose, state, t)
        views, deps = env.render_cameras_depth()
        after = kv.render_rig(pose, state, t)
        for (rgb0, seg0), (rgb1, seg1), (rgb2, seg2) in zip(before, views, after):
            assert np.array_equal(rgb0, rgb1) and np.array_equal(seg0, seg1)
            assert np.array_equal(rgb0, rgb2) and np.array_equal(seg0, seg2)
        cams = env.cfg.cameras
        for cfg, (_, seg), dep in zip(cams, views, deps):
            assert dep.shape == (cfg.height, cfg.width) and dep.dtype == np.uint16
            assert np.mean((dep > 0) == (seg != SEM["sky"])) >= 0.995                           # (c)
            for step in (1, 2):                                                                  # (a)
                errs = plane_errors(cfg, dep, seg, pose, step)
                e = np.concatenate([errs["road"], errs["marking"]])
                assert len(e) > 2000 / step ** 2
                assert np.median(e) < 0.005 and np.percentile(e, 95) < 0.02, (cfg.name, step)
                # the sidewalk class is the slab 0.15 m up and the kerb's face below it: the slab is the median
                assert np.median(errs["sidewalk"]) < 0.005, (cfg.name, step)
            z = heights(cfg, dep, seg, pose, "sidewalk")
            assert np.mean((z > -0.04) & (z < 0.19)) > 0.98                                      # kerb foot to kerb top
        # (b) the same scene from 3 m further on and 0.5 m to the left, and from the other camera
        x, y, yaw = pose
        pose_b = (x + 3.0 * math.cos(yaw) - 0.5 * math.sin(yaw), y + 3.0 * math.sin(yaw) + 0.5 * math.cos(yaw), yaw + 0.03)
        _, deps_b = kv.render_rig_depth(pose_b, state, t)
        for ia, ib, dep_b, pb in ((0, 0, deps_b[0], pose_b), (1, 1, deps_b[1], pose_b), (0, 1, deps_b[1], pose_b),
                                  (0, 1, deps[1], pose), (1, 0, deps[0], pose)):
            through, n, diff = reproject(cams[ia], deps[ia], pose, cams[ib], dep_b, pb)
            assert n > 2000 and through < 0.03 and diff < 0.005, (cams[ia].name, cams[ib].name, through, diff)
    finally:
        env.close()
