"""Closed-loop agent wrapper: camera frame -> KeiPilot -> plan -> controls."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from keisim.control import PlanFollower

from .data import STAND_CAP, stand_update
from .model import KeiPilot


def load_model(ckpt, device="cuda"):
    sd = torch.load(ckpt, map_location="cpu")
    cfg = sd.get("model_cfg", {})
    model = KeiPilot(pretrained=False, **cfg)
    model.load_state_dict(sd["model"])
    model.eval().to(device)
    model.rig = sd.get("rig", "front")          # the camera rig it was trained on (keisim CAMERA_RIGS)
    return model


def checkpoint_rig(ckpt):
    """Camera rig of a checkpoint (scripts set the environment's cameras from it)."""
    return torch.load(ckpt, map_location="cpu").get("rig", "front")


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
    """Safety layer against creeping over the stop line: while the model's own light head is confident the light
    ahead is red or yellow (`p` or more), the car is (nearly) standing and the model itself only asks for a creep
    (target below `creep` m/s, i.e. it believes it is at the stopping point), the target is held at zero. Far from
    the line the model asks for more than a creep, so pulling up to the line is never blocked."""

    def __init__(self, p=0.8, v=1.0, creep=1.0):
        self.p, self.v, self.creep = p, v, creep

    def __call__(self, target, speed, tl_probs):
        if speed < self.v and target < self.creep and float(tl_probs[0] + tl_probs[1]) >= self.p:
            return 0.0
        return target


class YieldHold:
    """Safety layer for giving way, on the model's own yield_tta head (the earliest arrival of a car with priority;
    trained on the frames before the junctions where the ego gives way). While the car is slow (below `v` m/s:
    pulling up to a crossing or creeping into it) and the head says a car arrives within `on` s, the target is held
    at zero until that arrival plus `margin` s, even if the head loses the car in between (it often does as the car
    comes close at the side of the image) or the speed head says go. A hold ends at the earliest arrival read
    during it: a car really coming keeps its arrival time, while a phantom reading that stays the same (a wall read
    as a car always 3.7 s away) would otherwise push the end back for ever. The next short reading starts a new
    hold, but only while the car moves or within `stand` s of coming to a stand: a phantom seen while standing
    would chain holds for ever, and the model's own target already keeps the car waiting for what it sees.
    A hold starts only while the model itself slows down or stands (its target `brake` m/s or more below the speed,
    or below `brake`): the layer keeps a yield the model began until the car it saw has passed, but starts no stop
    on the head alone. Off its training frames (crossings where the ego has priority, scenery) the head often reads
    3-4 s with no car near, most of all while standing, and holds on those readings made 131 of 300 Sapporo routes
    stand longer (11-01). brake None or negative: no such condition (the layer as first evaluated)."""

    def __init__(self, on=4.0, margin=1.0, v=3.5, stand=4.0, brake=0.5):
        self.on, self.margin, self.v, self.stand = on, margin, v, stand
        self.brake = None if brake is None or brake < 0 else brake
        self.reset()

    def reset(self):
        self.t, self.until, self.stood = 0.0, -1.0, 0.0

    def __call__(self, target, speed, tta, dt):
        self.t += dt
        self.stood = self.stood + dt if speed < 0.1 else 0.0
        if speed < self.v and tta < self.on:
            end = self.t + max(tta, 0.0) + self.margin
            if self.t < self.until:
                self.until = min(self.until, end)
            elif self.stood <= self.stand and (self.brake is None or target < max(speed - self.brake, self.brake)):
                self.until = end
        return 0.0 if self.t < self.until else target


class KeiPilotAgent:
    def __init__(self, ckpt, device=None, brake_hold=False, red_hold=False, yield_hold=None):
        """yield_hold: None (off), True (YieldHold's defaults) or its (on, margin, v, stand[, brake])."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(ckpt, self.device).to(memory_format=torch.channels_last)
        self.follower = PlanFollower()
        self.amp = self.device.startswith("cuda")
        self.hold = BrakeHold() if brake_hold else None
        self.red_hold = RedHold() if red_hold else None
        self.yield_hold = None if not yield_hold else YieldHold() if yield_hold is True else YieldHold(*yield_hold)
        if self.yield_hold and not self.model.aux_yield:
            raise ValueError("yield_hold needs a model with the yield_tta head (train.py --aux_weight)")
        self.frames = None                 # history models: the last frames seen, one per control step
        self.stand_m = STAND_CAP           # memory-input models: metres since the ego last stood

    def reset(self):
        self.follower.reset()
        if self.hold:
            self.hold.reset()
        if self.yield_hold:
            self.yield_hold.reset()
        self.frames = None
        self.stand_m = STAND_CAP

    def _tensor(self, bgr):
        if isinstance(bgr, (list, tuple)):          # a camera rig: (1, K, 3, H, W)
            return torch.stack([self._tensor(b)[0] for b in bgr])[None]
        img = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1)))[None].to(self.device)
        return img.contiguous(memory_format=torch.channels_last)

    @torch.no_grad()
    def plan(self, rgb_bgr, command, target_point, with_seg=False, speed=None, prev_bgr=None, stand_m=None):
        """rgb_bgr: the camera image, or the list of a rig's images (obs["rgbs"]) for multi-camera models.
        prev_bgr: for history models, the frame `history_dt` s earlier (None: not available yet).
        stand_m: for memory-input models, metres since the ego last stood (None: unknown)."""
        if self.model.n_cams != (len(rgb_bgr) if isinstance(rgb_bgr, (list, tuple)) else 1):
            raise ValueError(f"the model takes {self.model.n_cams} camera images")
        img = self._tensor(rgb_bgr)
        cmd = torch.tensor([int(command)], device=self.device)
        tp = torch.tensor(np.asarray(target_point, np.float32)[None], device=self.device)
        spd = None if speed is None else torch.tensor([float(speed)], device=self.device)
        stand = None if stand_m is None else torch.tensor([float(stand_m)], device=self.device)
        prev = has_prev = None
        if self.model.history:
            prev = self._tensor(prev_bgr) if prev_bgr is not None else torch.zeros_like(img)
            has_prev = torch.tensor([prev_bgr is not None], device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            out = self.model(img, cmd, tp, with_seg=with_seg, speed=spd, img_prev=prev, has_prev=has_prev,
                             stand_m=stand)
        v, p = self.model.decode_speed(out["speed_logits"])
        res = {
            "path": out["path"][0].float().cpu().numpy(),
            "target_speed": float(v[0]),
            "speed_probs": p[0].cpu().numpy(),
            "tl": int(out["tl_logits"][0].argmax()),
            "tl_probs": out["tl_logits"][0].float().softmax(-1).cpu().numpy(),
        }
        if with_seg:        # (rigs: the first camera's)
            res["seg"] = out["seg"][0].float().argmax(0).cpu().numpy().astype(np.uint8)
        if "yield_tta" in out:
            res["yield_tta"] = float(out["yield_tta"][0])
        return res

    def act(self, rgb_bgr, command, target_point, speed, dt, with_seg=False):
        prev = None
        if self.model.history:
            if self.frames is None:
                self.frames = deque(maxlen=int(round(self.model.history_dt / dt)) + 1)
            self.frames.append(np.array(rgb_bgr, copy=True))
            if len(self.frames) == self.frames.maxlen:
                prev = self.frames[0]
        self.stand_m = stand_update(self.stand_m, speed, dt)
        p = self.plan(rgb_bgr, command, target_point, with_seg=with_seg, speed=speed, prev_bgr=prev,
                      stand_m=self.stand_m)
        if self.hold or self.red_hold or self.yield_hold:
            p["target_speed_raw"] = p["target_speed"]
        if self.hold:
            p["target_speed"] = self.hold(p["target_speed"], speed, dt)
        if self.red_hold:     # judged on the model's own target, before the brake hold capped it
            p["target_speed"] = min(p["target_speed"], self.red_hold(p["target_speed_raw"], speed, p["tl_probs"]))
        if self.yield_hold:
            p["target_speed"] = min(p["target_speed"], self.yield_hold(p["target_speed_raw"], speed, p["yield_tta"], dt))
        action = self.follower(p["path"], p["target_speed"], speed, dt)
        return action, p
