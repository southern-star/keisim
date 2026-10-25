"""Camera rigs (keisim/config.py CAMERA_RIGS, keisim/render/rig.py; run: python -m pytest tests -q)."""
import math
import os
import sys

import numpy as np
import pytest

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


def test_keiview_draws_the_rig():
    """KeiView draws every camera of the rig in one request (the cylinder from the pinhole faces of KeiSim's
    CylinderRenderer, stitched with the same maps), with the ground and the sky where KeiSim has them (the houses and
    trees differ between the renderers). Skipped without KeiView (npm install in web/, a Chrome)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(os.path.join(root, "web", "node_modules", "puppeteer-core")):
        pytest.skip("KeiView needs `npm install` in web/")
    env = KeiEnv(set_camera_rig(EnvConfig(renderer="keiview"), "wide_narrow"))
    try:
        try:
            obs = env.reset(town_seed=5, episode_seed=1, weather="noon")
        except RuntimeError as e:
            pytest.skip(f"KeiView did not start: {e}")
        state = env.world.scene_state()
        keisim = [cam.render(env.world.ego.pose, state, env.weather) for cam in env.assets.cameras]
        for rgb, seg, (_, ks) in zip(obs["rgbs"], obs["segs"], keisim):
            assert rgb.shape == (160, 320, 3) and seg.shape == (160, 320)
            assert (np.isin(seg, GROUND) == np.isin(ks, GROUND)).mean() > 0.93
            assert ((seg == SEM["sky"]) == (ks == SEM["sky"])).mean() > 0.85
        # the first camera alone is the same image as in the whole rig
        assert np.array_equal(env.render_camera()[1], obs["segs"][0])
    finally:
        env.close()


def _labels(i):
    from keipilot.data import LABEL_KEYS
    return {k: (np.zeros(2, np.float32) if k == "tp" else np.zeros((10, 2), np.float32) if k == "path" else i)
            for k in LABEL_KEYS}


def test_rig_shards_dataset_and_model(tmp_path):
    """Shards of a rig hold every camera (jpg_c1, seg_c1); the dataset gives (K,3,H,W) images and (K,h,w)
    segmentations; shards of different rigs do not mix; a two-camera KeiPilot starts from a one-camera checkpoint
    (every camera with its position embeddings) and trains on the batch."""
    import torch

    from keipilot.data import DrivingDataset, ShardWriter, load_shards
    from keipilot.model import KeiPilot, compute_loss

    w = ShardWriter(str(tmp_path / "rig"), "expert", frames_per_shard=3)
    for i in range(5):
        cams = [np.full((160, 320, 3), 20 * i + 100 * c, np.uint8) for c in range(2)]
        segs = [np.full((160, 320), c + 1, np.uint8) for c in range(2)]
        w.add(cams[0], segs[0], extra=[(cams[1], segs[1])], **_labels(i))
    w.flush()
    data, files = load_shards([str(tmp_path / "rig")])
    assert data["n_cams"] == 2 and len(files) == 2 and len(data["cmd"]) == 5
    ds = DrivingDataset(data, np.arange(5), train=False)
    s = ds[3]
    assert s["img"].shape == (2, 3, 160, 320) and s["seg"].shape == (2, 80, 160)
    assert abs(float(s["img"][1].float().mean()) - 160) <= 2 and s["seg"][1].unique().tolist() == [2]
    assert DrivingDataset(data, np.arange(5), train=True)[0]["img"].shape == (2, 3, 160, 320)
    one_cam = ShardWriter(str(tmp_path / "one"), "expert")
    one_cam.add(cams[0], segs[0], **_labels(0))
    one_cam.flush()
    with pytest.raises(ValueError):
        load_shards([str(tmp_path / "rig"), str(tmp_path / "one")])

    torch.manual_seed(0)
    one = KeiPilot(pretrained=False, speed_input=True, mem_input=True)
    two = KeiPilot(pretrained=False, speed_input=True, mem_input=True, n_cams=2)
    assert two.load_compatible(one.state_dict()) == []
    assert torch.equal(two.pos3[1], one.pos3[0]) and torch.equal(two.pos4[0], one.pos4[0])
    batch = torch.utils.data.default_collate([ds[i] for i in range(2)])
    out = two(batch["img"], batch["cmd"], batch["tp"], speed=batch["v"], stand_m=batch["stand_m"])
    assert out["path"].shape == (2, 10, 2) and out["seg"].shape[0] == 4
    loss, logs = compute_loss(two, out, batch)
    loss.backward()
    assert np.isfinite(logs["seg"]) and float(two.pos3.grad[1].abs().sum()) > 0


def test_agent_drives_a_rig_model(tmp_path):
    """KeiPilotAgent feeds a rig model the images of every camera (obs["rgbs"]); one image is refused."""
    import torch

    from keipilot.agent import KeiPilotAgent, checkpoint_rig
    from keipilot.model import KeiPilot

    ck = tmp_path / "rig.pt"
    cfg = {"speed_input": True, "mem_input": True, "n_cams": 2}
    torch.save({"model": KeiPilot(pretrained=False, **cfg).state_dict(), "model_cfg": cfg, "rig": "wide_narrow"}, ck)
    assert checkpoint_rig(str(ck)) == "wide_narrow"
    agent = KeiPilotAgent(str(ck), device="cpu", brake_hold=True, red_hold=True)
    env, obs = rig_env()
    action, p = agent.act(obs["rgbs"], obs["command"], obs["target_point"], 0.0, 0.1, with_seg=True)
    assert np.isfinite(action).all() and p["seg"].ndim == 2
    with pytest.raises(ValueError):
        agent.act(obs["rgb"], obs["command"], obs["target_point"], 0.0, 0.1)
