"""One synchronous KeiView request for the driving cameras and calibrated teacher views."""
import copy
from dataclasses import asdict, replace
from hashlib import sha256

import numpy as np

from keisim.config import EnvConfig, set_camera_rig
from keisim.env import KeiEnv
from .geometry import CHASE_CAMERA, ORBIT_ANGLES, orbit_camera, relative_pose, view_pose, visibility_now


def capture_env(gl="hw", orbit=False):
    cfg = set_camera_rig(EnvConfig(renderer="keiview", weather="noon", keiview_gl=gl), "wide_narrow")
    # Extra shots share the same scene and shadows as the driving cameras in this offline environment.
    targets = [orbit_camera(a) for a in ORBIT_ANGLES] if orbit else [replace(CHASE_CAMERA)]
    cfg.extra_cameras = (*cfg.extra_cameras, *targets)
    return KeiEnv(cfg)


def calibration(env):
    info = dict(camera_in=[asdict(c) for c in env.cfg.cameras[:2]], target=asdict(CHASE_CAMERA),
                renderer="keiview", quality=env.cfg.keiview_quality, roads=env.cfg.keiview_roads,
                peds=env.cfg.keiview_peds, weather=env.cfg.weather, dt=env.cfg.dt,
                action_repeat=env.cfg.action_repeat, color="RGB", row_order="top-first",
                depth="KeiSim radial uint16, 0=no surface", ego_body=False,
                view_pose="ego x-forward y-left z-up: position[3], forward[3], up[3], vertical_fov_deg, aspect",
                delta_pose="dx,dy in previous ego frame [m], wrapped dyaw [rad]; zero at episode start",
                visibility_now={"unseen": 0, "visible": 1, "unknown_or_no_surface": 255})
    if len(env.cfg.cameras) > 3:
        info["targets"] = [asdict(c) for c in env.cfg.cameras[2:]]
    return info


def state_digest(pose, t, scene):
    digest = sha256(np.asarray([*pose, t], dtype="<f8").tobytes())
    for name, value in sorted(vars(scene).items()):
        if value is not None:
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def render_snapshot(env, step, previous_pose=None, audit=False):
    world, kv = env.world, env.keiview()
    pose, t = tuple(world.ego.pose), float(world.t)
    scene = copy.deepcopy(world.scene_state(include_ego=False))
    digest = state_digest(pose, t, scene)
    cams = env.cfg.cameras
    if tuple(c.name for c in cams[:3]) != ("wide", "narrow", "chase") or cams[2] != CHASE_CAMERA:
        raise ValueError("capture requires the wide, narrow, chase rig")
    if any(c.width > kv.W or c.height > kv.H for c in cams[2:]):
        raise ValueError("teacher does not fit the KeiView canvas")
    images, depths = kv.render_rig_depth(pose, scene, t)
    checks = {}
    if audit:
        # Bracket a teacher render by normal two-camera renders to measure native repeatability.
        before, _ = kv.render_rig_depth(pose, scene, t, which=[0, 1])
        images, depths = kv.render_rig_depth(pose, scene, t)
        after, _ = kv.render_rig_depth(pose, scene, t, which=[0, 1])
        for i, cam in enumerate(cams[:2]):
            diff = np.abs(images[i][0].astype(np.int16) - before[i][0])
            repeat = np.abs(after[i][0].astype(np.int16) - before[i][0])
            checks[cam.name] = dict(rgb_max=int(diff.max()), rgb_mae=float(diff.mean()),
                                    repeat_max=int(repeat.max()), repeat_mae=float(repeat.mean()),
                                    seg_changed=int(np.count_nonzero(images[i][1] != before[i][1])))
    if digest != state_digest(world.ego.pose, world.t, world.scene_state(include_ego=False)):
        raise RuntimeError("rendering advanced or mutated the simulation")
    rgb = [np.ascontiguousarray(bgr[..., ::-1]) for bgr, _ in images]
    lane = env.assets.town.lanes[env.route.lane_at(env.s_ego)]
    frame = dict(step=np.int32(step), t_sim=np.float64(t), ego_pose_world=np.array(pose, np.float64),
                 ego_speed=np.float32(world.ego.v), delta_pose=relative_pose(previous_pose, pose),
                 view_pose=np.stack([view_pose(c) for c in cams[2:]]), rgb_in=np.stack(rgb[:2]), rgb_target=np.stack(rgb[2:]),
                 seg_in=np.stack([s for _, s in images[:2]]), seg_target=np.stack([s for _, s in images[2:]]),
                 depth_in=np.stack(depths[:2]), depth_target=np.stack(depths[2:]),
                 visibility_now=np.stack([visibility_now(c, dep, cams[:2], depths[:2])
                                          for c, dep in zip(cams[2:], depths[2:], strict=True)]),
                 state_sha256=np.asarray(digest, dtype="S64"), reason=np.asarray(env.plan["reason"], dtype="U24"),
                 junction=np.bool_(lane.kind == "conn"))
    return frame, checks
