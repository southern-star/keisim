"""Closed-loop agent wrapper: camera frame -> KeiPilot -> plan -> controls."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from keisim.control import PlanFollower

from .model import KeiPilot


def load_model(ckpt, device="cuda"):
    sd = torch.load(ckpt, map_location="cpu")
    cfg = sd.get("model_cfg", {})
    model = KeiPilot(pretrained=False, **cfg)
    model.load_state_dict(sd["model"])
    model.eval().to(device)
    return model


class BrakeHold:
    """Safety layer between the model and the speed controller. After a firm braking command (target at least
    `dv` below the current speed), the target may rise again only slowly (`rise` m/s per second) for `hold` s.
    A pedestrian lost for a frame or two then no longer turns into a burst of throttle in the middle of a stop;
    braking harder is never limited."""

    def __init__(self, dv=1.5, hold=1.5, rise=0.5):
        self.dv, self.hold, self.rise = dv, hold, rise
        self.reset()

    def reset(self):
        self.left, self.cap = 0.0, None

    def __call__(self, target, v, dt):
        if target < v - self.dv:                         # firm braking: (re)start the hold at this target
            self.left = self.hold
            self.cap = target if self.cap is None else min(self.cap, target)
        elif self.left > 0.0:
            self.left -= dt
            self.cap += self.rise * dt
        if self.left <= 0.0:
            self.cap = None
            return target
        return min(target, self.cap)


class RedHold:
    """Safety layer: while the model's own light head is confident the light ahead is red or yellow (`p` or more)
    and the car is (nearly) standing, the target speed is held at zero, so it cannot creep over the line."""

    def __init__(self, p=0.8, v=1.0):
        self.p, self.v = p, v

    def __call__(self, target, speed, tl_probs):
        if speed < self.v and float(tl_probs[0] + tl_probs[1]) >= self.p:
            return 0.0
        return target


class KeiPilotAgent:
    def __init__(self, ckpt, device=None, brake_hold=False, red_hold=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(ckpt, self.device).to(memory_format=torch.channels_last)
        self.follower = PlanFollower()
        self.amp = self.device.startswith("cuda")
        self.hold = BrakeHold() if brake_hold else None
        self.red_hold = RedHold() if red_hold else None
        self.frames = None                 # history models: the last frames seen, one per control step

    def reset(self):
        self.follower.reset()
        if self.hold:
            self.hold.reset()
        self.frames = None

    def _tensor(self, bgr):
        img = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1)))[None].to(self.device)
        return img.contiguous(memory_format=torch.channels_last)

    @torch.no_grad()
    def plan(self, rgb_bgr, command, target_point, with_seg=False, speed=None, prev_bgr=None):
        """prev_bgr: for history models, the frame `history_dt` s earlier (None: not available yet)."""
        img = self._tensor(rgb_bgr)
        cmd = torch.tensor([int(command)], device=self.device)
        tp = torch.tensor(np.asarray(target_point, np.float32)[None], device=self.device)
        spd = None if speed is None else torch.tensor([float(speed)], device=self.device)
        prev = has_prev = None
        if self.model.history:
            prev = self._tensor(prev_bgr) if prev_bgr is not None else torch.zeros_like(img)
            has_prev = torch.tensor([prev_bgr is not None], device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            out = self.model(img, cmd, tp, with_seg=with_seg, speed=spd, img_prev=prev, has_prev=has_prev)
        v, p = self.model.decode_speed(out["speed_logits"])
        res = {
            "path": out["path"][0].float().cpu().numpy(),
            "target_speed": float(v[0]),
            "speed_probs": p[0].cpu().numpy(),
            "tl": int(out["tl_logits"][0].argmax()),
            "tl_probs": out["tl_logits"][0].float().softmax(-1).cpu().numpy(),
        }
        if with_seg:
            res["seg"] = out["seg"][0].float().argmax(0).cpu().numpy().astype(np.uint8)
        return res

    def act(self, rgb_bgr, command, target_point, speed, dt, with_seg=False):
        prev = None
        if self.model.history:
            if self.frames is None:
                self.frames = deque(maxlen=int(round(self.model.history_dt / dt)) + 1)
            self.frames.append(np.array(rgb_bgr, copy=True))
            if len(self.frames) == self.frames.maxlen:
                prev = self.frames[0]
        p = self.plan(rgb_bgr, command, target_point, with_seg=with_seg, speed=speed, prev_bgr=prev)
        if self.hold or self.red_hold:
            p["target_speed_raw"] = p["target_speed"]
        if self.hold:
            p["target_speed"] = self.hold(p["target_speed"], speed, dt)
        if self.red_hold:
            p["target_speed"] = self.red_hold(p["target_speed"], speed, p["tl_probs"])
        action = self.follower(p["path"], p["target_speed"], speed, dt)
        return action, p
