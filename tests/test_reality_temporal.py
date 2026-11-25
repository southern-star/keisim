from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from keireality.geometry import CHASE_CAMERA, ORBIT_ANGLES, orbit_camera, relative_pose, view_pose
from keireality.history import HistoryWindow, compact_mem
from keireality.spatial import SpatialRenderer, camera_grid, ray_points
from keireality.stream import TemporalDisplay
from keireality.temporal_metrics import correspondence
from keisim.config import CAMERA_RIGS, SEM
from keisim.render import rays
from scripts.label_reality_history import historical_labels


@pytest.mark.parametrize("angle", ORBIT_ANGLES)
def test_torch_sampling_matches_native_geometry(angle):
    cam = orbit_camera(angle)
    pts = ray_points(torch.tensor(view_pose(cam))[None], 128, 256, (23.,))[0, 1, ::8, ::8]
    ref = rays.unproject(23., rays.camera_rays(cam, 8), *rays.level_frame(cam, 0, 0, 0))
    np.testing.assert_allclose(pts, ref, atol=5e-6)
    for source in CAMERA_RIGS["wide_narrow"]:
        uv, _, inside = rays.project(source, ref.reshape(-1, 3), *rays.level_frame(source, 0, 0, 0))
        grid, valid = camera_grid(pts.reshape(-1, 3), source, 8)
        np.testing.assert_array_equal(valid, inside)
        actual = ((grid[valid].numpy()+1)*[source.width, source.height] - 8)/2
        np.testing.assert_allclose(actual, uv[inside], atol=1e-4)


def test_history_se2_capacity_discontinuities_and_episode_identity():
    window = HistoryWindow()
    poses = [(i*.7, .1*i*i, .095*i) for i in range(11)]
    for i, pose in enumerate(poses):
        refs, motion = window.push(i, "first", .1*i, relative_pose(poses[i-1] if i else None, pose))
        assert refs == list(range(i-1, max(-1, i-8), -1))
        for k, j in enumerate(refs):
            x, y, a = relative_pose(pose, poses[j])
            np.testing.assert_allclose(motion[k], [x, y, math.sin(a), math.cos(a), (j-i)*.1], atol=1e-5)
    for name, t in (("first", 1.2), ("first", .6), ("second", .7)):
        refs, _ = window.push(99, name, t, np.ones(3))
        assert refs == []


def test_history_teacher_and_vacated_actors():
    cam = replace(CHASE_CAMERA, width=16, height=8, yaw_deg=0, pitch_deg=0)
    center, rotation = rays.level_frame(cam, 0, 0, 0)
    directions = rays.camera_rays(cam)
    tab = center[None], (directions @ rotation)[None]
    d = rays.encode_depth(np.full((1, 8, 16), 20.))
    now = np.zeros_like(d, dtype=np.uint8)
    now[:, 0] = 255
    now[:, 1] = 1
    seg = np.full_like(now, SEM["road"])
    seg[:, 2] = SEM["vehicle"]
    old_seg = seg.copy()
    old_seg[:, 3] = SEM["vehicle"]
    old_d = d.copy()
    old_d[:, 3] = rays.encode_depth(8.)
    old_d[:, 4] = 0
    pose = (0, 0, 0)
    labels = historical_labels(now, seg, d, pose, tab, [(pose, old_d, old_seg)], [cam])
    assert np.all(labels[0, 0] == 255) and np.all(labels[0, 1] == 1)
    assert np.all(labels[0, 2:4] == 0) and np.all(labels[0, 4] == 255) and np.all(labels[0, 5:] == 2)
    assert np.array_equal(historical_labels(now, seg, d, pose, tab, [], [cam]), now)
    (_, _), (vacated, pix) = correspondence(cam, directions, (seg[0], d[0], pose), (old_seg[0], old_d[0], pose))
    np.testing.assert_array_equal(vacated, np.arange(48, 64))
    np.testing.assert_array_equal(vacated, pix)


@torch.inference_mode()
def test_stream_repeated_view_and_missing_history():
    torch.set_num_threads(2)
    torch.manual_seed(23)
    model = SpatialRenderer(support=True, temporal=True).eval()
    stream = TemporalDisplay(model)
    features = dict(mem=torch.randn(1, 500, 256).half(), c2=torch.randn(1, 2, 128, 20, 40).half(),
                    c3=torch.randn(1, 2, 256, 10, 20).half())
    view = torch.tensor(view_pose(CHASE_CAMERA))[None]
    with pytest.raises(ValueError, match="observe"):
        stream.render(view)
    for i in range(3):
        stream.observe(features, "a", float(i)/10, np.array([1., 0., .1]))
        assert stream.valid.sum() == i
    np.testing.assert_array_equal(stream.past[0, 0], compact_mem(features["mem"][0].numpy()))
    before = stream.render(view)
    stream.render(torch.tensor(view_pose(orbit_camera(-36)))[None])
    stream.past[~stream.valid] = 99
    after = stream.render(view)
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))
    stream.observe(features, "b", .3, np.ones(3))
    assert not stream.valid.any()
    first = stream.render(view)
    stream.past[:] = -99
    assert all(torch.equal(a, b) for a, b in zip(first, stream.render(view), strict=True))
