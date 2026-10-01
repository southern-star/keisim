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


def test_counterfactual_speed_labels_match_expert():
    """target_for_speed(v, plan) must reproduce the expert's own target at its actual speed, including the
    red / yellow / overshoot / don't-block-the-box cases, so relabelling with other speeds is exact."""
    from keisim.expert import target_for_speed

    env = KeiEnv(EnvConfig())
    seen = set()
    for town, ep in ((1003, 2), (1001, 5), (7, 11)):
        env.reset(town_seed=town, episode_seed=ep, route_length=500, render=False)
        for _ in range(3000):
            plan = env.plan
            assert abs(target_for_speed(env.world.ego.v, plan) - plan["target_speed"]) < 1e-9, plan
            seen.add(plan["reason"])
            for v in (0.0, 2.5, 6.0, 11.0):           # every counterfactual speed gives a valid label
                assert 0.0 <= target_for_speed(v, plan) <= env.world.town.cfg.speed_limit + 1e-6
            _, _, done, _ = env.step(env.expert_action())
            if done:
                break
    assert "red_light" in seen


def test_light_rules():
    """Red and yellow: stop at the line unless even a hard stop (6 m/s^2) is impossible, whatever the remaining
    yellow time; green: go (unless the junction exit is full)."""
    from keisim.config import TL_GREEN, TL_RED, TL_YELLOW
    from keisim.expert import light_stop

    assert light_stop(5.0, 10.0, TL_YELLOW, 2.4, False) == "red_light"      # 3.9 m needed: stop, time left or not
    assert light_stop(10.0, 8.0, TL_YELLOW, 0.5, False) is None             # 8.6 m needed: too close, go on
    assert light_stop(10.0, 12.0, TL_YELLOW, 0.5, False) == "red_light"     # still possible -> stop (v3 went on)
    assert light_stop(10.0, 9.0, TL_RED, 0.0, False) == "red_light"         # red: 8.6 m needed at 6 m/s^2
    assert light_stop(11.0, 8.0, TL_RED, 0.0, False) is None                # 10.4 m needed: cannot stop
    assert light_stop(8.0, 6.0, TL_GREEN, 0.0, False) is None
    assert light_stop(3.0, 8.0, TL_GREEN, 0.0, True) == "junction_blocked"  # don't block the box


def test_camera_render_shapes():
    env = KeiEnv(EnvConfig())
    obs = env.reset(town_seed=5, episode_seed=1)
    assert obs["rgb"].shape == (160, 320, 3) and obs["rgb"].dtype == np.uint8
    assert obs["seg"].shape == (160, 320)
    assert obs["expert"]["path"].shape == (10, 2)


def test_speed_input_starts_as_camera_only_model():
    """A camera-only checkpoint loaded into a speed-input KeiPilot must behave exactly as before."""
    import torch

    from keipilot.model import KeiPilot

    torch.manual_seed(0)
    base = KeiPilot(pretrained=False).eval()
    spd = KeiPilot(pretrained=False, speed_input=True).eval()
    missing, unexpected = spd.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("speed_") for k in missing)
    img = torch.randint(0, 256, (2, 3, 160, 320), dtype=torch.uint8)
    cmd, tp = torch.tensor([0, 1]), torch.tensor([[20.0, 3.0], [15.0, -2.0]])
    with torch.no_grad():
        a = base(img, cmd, tp, with_seg=False)
        b = spd(img, cmd, tp, with_seg=False, speed=torch.tensor([0.0, 8.0]))
        c = spd(img, cmd, tp, with_seg=False)
    for k in ("path", "speed_logits", "tl_logits"):
        assert torch.allclose(a[k], b[k], atol=1e-5) and torch.allclose(a[k], c[k], atol=1e-5)


def test_brake_hold_blocks_throttle_burst():
    """A model that loses a pedestrian for a few frames mid-stop must keep braking; braking harder is never
    limited, and the hold ends on its own."""
    from keipilot.agent import BrakeHold

    h = BrakeHold()
    assert h(3.75, 7.5, 0.1) == 3.75                       # firm braking starts a hold
    burst = [h(9.98, 7.0, 0.1) for _ in range(5)]          # pedestrian lost: the model asks for 10 m/s
    assert max(burst) < 4.1                                # target rises only slowly -> still braking
    assert h(0.0, 6.0, 0.1) == 0.0                         # braking harder is never limited
    out = [h(8.0, 0.0, 0.1) for _ in range(20)]            # pedestrian gone, standing: hold runs out
    assert out[-1] == 8.0 and out[0] < 1.0


def test_history_model_starts_as_single_frame_model():
    """A single-frame checkpoint loaded into a history model must behave exactly as before, with or without a
    previous frame."""
    import torch

    from keipilot.model import KeiPilot

    for mode in ("frame", "diff"):
        _check_history_start(mode)


def _check_history_start(mode):
    import torch

    from keipilot.model import KeiPilot

    torch.manual_seed(0)
    base = KeiPilot(pretrained=False, speed_input=True).eval()
    hist = KeiPilot(pretrained=False, speed_input=True, history=True, history_mode=mode).eval()
    assert hist.load_compatible(base.state_dict()) == []
    img = torch.randint(0, 256, (2, 3, 160, 320), dtype=torch.uint8)
    prev = torch.randint(0, 256, (2, 3, 160, 320), dtype=torch.uint8)
    cmd, tp, v = torch.tensor([0, 1]), torch.tensor([[20.0, 3.0], [15.0, -2.0]]), torch.tensor([0.0, 8.0])
    with torch.no_grad():
        a = base(img, cmd, tp, with_seg=False, speed=v)
        b = hist(img, cmd, tp, with_seg=False, speed=v, img_prev=prev, has_prev=torch.tensor([True, False]))
        c = hist(img, cmd, tp, with_seg=False, speed=v)
    for k in ("path", "speed_logits", "tl_logits"):
        assert torch.allclose(a[k], b[k], atol=1e-5) and torch.allclose(a[k], c[k], atol=1e-5)


def test_agent_drives_with_history_and_brake_hold(tmp_path):
    """End to end through KeiPilotAgent.act (CPU): plain numbers out, frame history filled after history_dt."""
    import torch

    from keipilot.agent import KeiPilotAgent
    from keipilot.model import KeiPilot

    ck = tmp_path / "m.pt"
    model = KeiPilot(pretrained=False, speed_input=True, history=True, history_dt=0.2)
    torch.save({"model": model.state_dict(), "model_cfg": {"speed_input": True, "history": True, "history_dt": 0.2}}, ck)
    agent = KeiPilotAgent(str(ck), device="cpu", brake_hold=True)
    agent.reset()
    rgb = np.zeros((160, 320, 3), np.uint8)
    for _ in range(4):
        action, p = agent.act(rgb, 1, (20.0, 0.0), 3.0, 0.1)
    assert len(agent.frames) == 3 and np.isfinite(action).all() and isinstance(p["target_speed"], float)


def test_red_hold_blocks_creeping():
    """Standing with the light head sure of red/yellow: no creeping; moving, or not sure: untouched."""
    from keipilot.agent import RedHold

    h = RedHold()
    red = np.array([0.9, 0.05, 0.03, 0.02])
    assert h(0.6, 0.3, red) == 0.0                                # creeping at the line on red: held
    assert h(0.6, 0.3, np.array([0.3, 0.1, 0.5, 0.1])) == 0.6     # not sure it is red: untouched
    assert h(5.0, 6.0, red) == 5.0                                # moving: untouched
    assert h(8.0, 0.0, red) == 8.0                                # standing far from the light: may pull up
