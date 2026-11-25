"""KeiReality's calibration, acquisition, isolation and cache contracts."""
import base64
from dataclasses import replace
import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from keipilot.model import KeiPilot
from keireality.dataset import checked_shards, file_hash, json_hash, validate_plan, write_json, write_shard
from keireality.encoder import extract_mem, state_hash
from keireality.geometry import CHASE_CAMERA, UNKNOWN, UNSEEN, VISIBLE, relative_pose, view_pose, visibility_now
from keireality.learning import RealityFrames
from keireality.metrics import ImageScores, regions
from keireality.renderer import SingleFrameRenderer
from keireality.snapshot import capture_env, render_snapshot
from keisim.config import CAMERA_RIGS
from keisim.render import rays
from keisim.render.camera import camera_pose
from keisim.render.keiview import KeiViewRenderer


@pytest.mark.parametrize("yaw", [0, math.pi / 2, -math.pi / 2, math.pi])
def test_chase_alignment_and_round_trip(yaw):
    cam = CHASE_CAMERA
    x, y = 12., 23.
    c, rot = camera_pose(cam, x, y, yaw)
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.])
    ego = np.array([x, y, 0.])
    np.testing.assert_allclose(c, ego - 8 * forward + [0, 0, 4])
    aim = ego + 12 * forward - c
    np.testing.assert_allclose(rot[2], aim / np.linalg.norm(aim), atol=1e-12)
    assert view_pose(cam)[-2:] == pytest.approx([50., 2.])
    for camera in (*CAMERA_RIGS["wide_narrow"], cam):
        direction = rays.camera_rays(camera, step=8)
        d = 8 + np.arange(direction.shape[1])[None, :] * .75
        frame = rays.level_frame(camera, x, y, yaw)
        pts = rays.unproject(d, direction, *frame)
        uv, distance, ok = rays.project(camera, pts.reshape(-1, 3), *frame)
        u, v = np.meshgrid(np.arange(0, camera.width, 8), np.arange(0, camera.height, 8))
        assert ok.all()
        np.testing.assert_allclose(uv, np.stack([u.ravel(), v.ravel()], 1), atol=1e-8)
        np.testing.assert_allclose(distance.reshape(u.shape), np.broadcast_to(d, u.shape))
    np.testing.assert_allclose(relative_pose((x, y, yaw), (x + forward[0], y + forward[1], yaw + .1)),
                               [1., 0., .1], atol=1e-6)


def test_visibility_occlusion_unknown_and_boundary():
    cam = replace(CHASE_CAMERA, width=16, height=8, pitch_deg=0)
    target = rays.encode_depth(np.full((8, 16), 20.))
    d = target.copy()
    d[:, :4] = rays.encode_depth(5.)       # nearer occluder
    d[:, 6:8] = 0                         # missing input surface
    d[:, 12:] = rays.encode_depth(30.)     # target in front of input depth
    target[0] = 0                         # sky/no target surface
    mask = visibility_now(cam, target, [cam], [d])
    assert np.all(mask[0] == UNKNOWN)
    assert np.all(mask[1:, :4] == UNSEEN)
    assert np.all(mask[1:, 4:6] == VISIBLE)
    assert np.all(mask[1:, 6:8] == UNKNOWN)
    assert np.all(mask[1:, 8:12] == VISIBLE)
    assert np.all(mask[1:, 12:] == UNKNOWN)
    assert np.all(visibility_now(cam, target, [cam, cam], [d, target])[1:] == VISIBLE)
    behind = replace(cam, yaw_deg=180)
    assert np.all(visibility_now(cam, target, [behind], [d])[1:] == UNSEEN)
    # Halfway across a foreground/background edge the target is at neither measured distance.
    cam = replace(cam, x=0., y=0., fov_deg=90.)
    edge = np.full_like(target, rays.encode_depth(5.))
    edge[:, 8:] = rays.encode_depth(35.)
    shifted = replace(cam, y=1.)
    result = visibility_now(cam, target, [shifted], [edge])
    assert result[4, 7] == UNKNOWN


def test_snapshot_color_order_and_single_scene_request():
    env = capture_env("soft")
    w, h = 2, 2
    top = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [10, 20, 30]]], np.uint8)
    wire = {"w": w, "h": h, "rgb": base64.b64encode(top[::-1].tobytes()).decode(),
            "sem": base64.b64encode(bytes([3, 4, 1, 2])).decode()}
    bgr, seg = KeiViewRenderer._decode(wire)
    scene = SimpleNamespace(veh_xy=np.array([[10., 0.]]))
    env.world = SimpleNamespace(ego=SimpleNamespace(pose=(1., 2., 0.), v=2.), t=11.3,
                                scene_state=lambda include_ego: scene)
    env.assets = SimpleNamespace(town=SimpleNamespace(lanes=[SimpleNamespace(kind="conn")]))
    env.route = SimpleNamespace(lane_at=lambda s: 0)
    env.s_ego, env.plan = 0, {"reason": "cruise"}
    calls = []

    def render(pose, frozen, t):
        calls.append((pose, t))
        assert not np.shares_memory(frozen.veh_xy, scene.veh_xy)
        images = [(np.tile(bgr, (c.height // 2, c.width // 2, 1)),
                   np.tile(seg, (c.height // 2, c.width // 2))) for c in env.cfg.cameras]
        return images, [np.zeros(s.shape, np.uint16) for _, s in images]

    env._keiview = SimpleNamespace(W=320, H=256, render_rig_depth=render)
    frame, _ = render_snapshot(env, 3)
    assert calls == [((1., 2., 0.), 11.3)]
    np.testing.assert_array_equal(frame["rgb_target"][0, :2, :2], top)
    np.testing.assert_array_equal(frame["rgb_in"][1, :2, :2], top)
    np.testing.assert_array_equal(frame["seg_target"][0, :2, :2], [[1, 2], [3, 4]])
    assert frame["depth_in"].shape == (2, 160, 320)
    assert frame["visibility_now"].shape == (1, 128, 256)
    assert frame["junction"]


def test_dataset_split_continuity_resume_and_damage(tmp_path):
    ep = dict(split="train", town_seed=4, episode_seed=0, frames=4)
    validate_plan([ep, dict(ep, split="test", town_seed=5)])
    for collision in (dict(ep), dict(ep, split="val", episode_seed=1)):
        with pytest.raises(ValueError, match="overlap"):
            validate_plan([ep, collision])
    frames = [dict(step=np.int32(i), t_sim=np.float64(7.5 + i / 10)) for i in range(4)]
    with pytest.raises(ValueError, match="consecutive"):
        write_shard(tmp_path, ep, frames[::2], "key")
    a = write_shard(tmp_path, ep, frames[:2], "key")
    b = write_shard(tmp_path, ep, frames[2:], "key")
    assert checked_shards(tmp_path, ep, "key") == [a, b]
    with pytest.raises(ValueError, match="incompatible"):
        checked_shards(tmp_path, ep, "other-key")
    path = tmp_path / a["path"] / "step.npy"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="corrupt"):
        checked_shards(tmp_path, ep, "key")


def test_partial_episode_replays_render_history(tmp_path, monkeypatch):
    from scripts import collect_reality
    ep = dict(town_seed=0, episode_seed=4, split="train", start_step=1, frames=4)
    args = SimpleNamespace(gl="hw", shard_size=2, audit=False)
    env = SimpleNamespace(done=False, status="running", world=SimpleNamespace(ego=SimpleNamespace(pose=(0., 0., 0.))))
    env.keiview = lambda: SimpleNamespace(gl="hw", gpu="test")
    env.expert_action = lambda: None
    def reset(**kwargs):
        env.world.ego.pose = (0., 0., 0.)
        env.render_count = 0
    def advance(action):
        env.world.ego.pose = (env.world.ego.pose[0] + 1, 0., 0.)
    env.reset, env.step = reset, advance
    def render(env, step, previous, audit=False):
        if env.interrupt and step == 3:
            raise RuntimeError("interrupted")
        env.render_count += 1
        return dict(step=np.int32(step), t_sim=np.float64(step / 10),
                    state_sha256=np.asarray(str(env.world.ego.pose).encode(), dtype="S64"),
                    render_count=np.int32(env.render_count)), {}
    monkeypatch.setattr(collect_reality, "render_snapshot", render)
    env.interrupt = True
    with pytest.raises(RuntimeError, match="interrupted"):
        collect_reality.collect_episode(env, tmp_path, ep, "key", args)
    first, = checked_shards(tmp_path, ep, "key")
    env.interrupt = False
    rows, done = collect_reality.collect_episode(env, tmp_path, ep, "key", args)
    assert rows[0] == first and done["count"] == 4 and done["captured_this_run"] == 2
    np.testing.assert_array_equal(np.load(tmp_path / rows[1]["path"] / "render_count.npy"), [3, 4])


def test_shared_decoder_tokens_parity_and_inference_inputs():
    torch.manual_seed(9)
    torch.set_num_threads(2)
    pilot = KeiPilot(pretrained=False, n_cams=2).eval().requires_grad_(False)
    state = state_hash(pilot)
    image = torch.randint(256, (1, 2, 3, 160, 320), dtype=torch.uint8)
    seen = []
    handle = pilot.decoder.register_forward_pre_hook(lambda mod, args: seen.append(args[1]))
    try:
        mem = extract_mem(pilot, image, parity=True)
    finally:
        handle.remove()
    assert torch.equal(mem, seen[0]) and torch.equal(mem, seen[1]) and not mem.requires_grad
    assert not torch.equal(mem, extract_mem(pilot, image.flip(1)))
    assert state_hash(pilot) == state
    head = SingleFrameRenderer().eval()
    pose = torch.tensor(view_pose(CHASE_CAMERA))[None]
    assert tuple(inspect.signature(head.forward).parameters) == ("mem", "view_pose")
    sample = dict(mem=mem, view_pose=pose, rgb_target=image, town_seed=0, ego_pose_world=[1., 2., 3.])
    with torch.inference_mode():
        a = head(sample["mem"], sample["view_pose"])
        sample.update(rgb_target=torch.zeros_like(image), town_seed=99, ego_pose_world=[0., 0., 0.])
        b = head(sample["mem"], sample["view_pose"])
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert a[0].shape == (1, 3, 128, 256) and a[1].shape == (1, 13, 128, 256)
    with pytest.raises(TypeError):
        head(**sample)


def test_normal_dataset_path_with_independent_teachers(tmp_path):
    ep = dict(split="train", town_seed=5, episode_seed=1, frames=2)
    plan = {"episodes": [ep]}
    contract = json_hash(plan)
    frames = []
    for i in range(2):
        frames.append(dict(step=np.int32(i), t_sim=np.float64(i / 10), ego_pose_world=np.full(3, i*99.),
                           view_pose=view_pose(CHASE_CAMERA)[None], rgb_in=np.zeros((2, 2, 2, 3), np.uint8),
                           rgb_target=np.full((1, 128, 256, 3), i*255, np.uint8),
                           seg_target=np.full((1, 128, 256), i*11, np.uint8),
                           visibility_now=np.full((1, 128, 256), i, np.uint8)))
    row = write_shard(tmp_path, ep, frames, contract)
    write_json(tmp_path / "plan.json", plan)
    write_json(tmp_path / "manifest.json", dict(complete=True, shards=[row], contract_sha256=contract,
                                                episodes={row["episode_id"]: {"count": 2}}))
    cache = tmp_path / "cache"
    dest = cache / row["path"]
    dest.mkdir(parents=True)
    np.save(dest / "mem.npy", np.zeros((2, 500, 256), np.float16))
    write_json(cache / "manifest.json", dict(collection_sha256=contract, shards={row["path"]: dict(
        count=2, sha256=file_hash(dest / "mem.npy"), rgb_sha256=row["fields"]["rgb_in"]["sha256"])}))
    batch = next(iter(torch.utils.data.DataLoader(RealityFrames(tmp_path, cache, "train"), batch_size=2)))
    assert not torch.equal(batch["rgb"][0], batch["rgb"][1])
    assert not torch.equal(batch["semantic"][0], batch["semantic"][1])
    assert not torch.equal(batch["visibility"][0], batch["visibility"][1])
    with torch.inference_mode():
        rgb, sem = SingleFrameRenderer().eval()(batch["mem"], batch["view_pose"])
    assert torch.equal(rgb[0], rgb[1]) and torch.equal(sem[0], sem[1])


def test_recon_fidelity_and_coverage_are_independent():
    truth = np.ones((12, 20, 3), np.float32)
    support = np.zeros((12, 20), bool)
    support[:, :10] = True
    pred = truth * support[..., None]
    scores = ImageScores()
    scores.add(pred, truth, regions(np.full(support.shape, VISIBLE, np.uint8)), support)
    visible = scores.report()["visible"]
    assert visible["l1"] == 0 and visible["ssim"] == pytest.approx(1)
    assert visible["coverage"] == .5 and 0 < visible["ssim_pixels"] < visible["valid"]
