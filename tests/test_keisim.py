"""Basic invariants of the simulator (run: python -m pytest tests -q)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keisim.config import EnvConfig  # noqa: E402
from keisim.env import KeiEnv  # noqa: E402
from keisim.geometry import normals_of  # noqa: E402
from keisim.roadnet import Town  # noqa: E402


def test_lanes_on_road_and_poles_off_road():
    for seed in range(5):
        town = Town(seed)
        for lane in town.lanes:
            n = normals_of(lane.pts)
            pts = np.concatenate([lane.pts, lane.pts + n * 0.95, lane.pts - n * 0.95])
            assert (town.surface(pts) == 1).all(), f"lane {lane.id} leaves the road in town {seed}"
            if lane.kind == "road":
                assert lane.succ, "every road lane must continue into a junction"
        for sh in town.signal_heads:
            assert town.surface(sh["pole"][None])[0] != 1


def test_signal_phases_are_exclusive():
    town = Town(3)
    for J in town.junctions:
        if not J.signalized:
            continue
        for t in np.linspace(0, 200, 400):
            greens = sum(J.signal_state(p, t) != 0 for p in range(J.phases))  # not red
            assert greens <= 1


def test_deterministic_replay():
    env = KeiEnv(EnvConfig())
    poses = []
    for _ in range(2):
        env.reset(town_seed=7, episode_seed=11, render=False)
        xs = []
        for _ in range(150):
            _, _, done, _ = env.step(env.expert_action())
            xs.append(env.world.ego.pose)
            if done:
                break
        poses.append(np.array(xs))
    assert np.allclose(poses[0], poses[1])


def test_expert_completes_short_route():
    env = KeiEnv(EnvConfig())
    env.reset(town_seed=1003, episode_seed=2, route_length=250, render=False)
    done = False
    while not done:
        _, _, done, info = env.step(env.expert_action())
    assert info["status"] == "success", info
    assert info["DS"] > 0.99, info


def test_camera_render_shapes():
    env = KeiEnv(EnvConfig())
    obs = env.reset(town_seed=5, episode_seed=1)
    assert obs["rgb"].shape == (160, 320, 3) and obs["rgb"].dtype == np.uint8
    assert obs["seg"].shape == (160, 320)
    assert obs["expert"]["path"].shape == (10, 2)
