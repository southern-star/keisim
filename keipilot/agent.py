"""Closed-loop agent wrapper: camera frame -> KeiPilot -> plan -> controls."""
from __future__ import annotations

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


class KeiPilotAgent:
    def __init__(self, ckpt, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(ckpt, self.device).to(memory_format=torch.channels_last)
        self.follower = PlanFollower()
        self.amp = self.device.startswith("cuda")

    def reset(self):
        self.follower.reset()

    @torch.no_grad()
    def plan(self, rgb_bgr, command, target_point, with_seg=False, speed=None):
        img = torch.from_numpy(np.ascontiguousarray(rgb_bgr[:, :, ::-1].transpose(2, 0, 1)))[None].to(self.device)
        img = img.contiguous(memory_format=torch.channels_last)
        cmd = torch.tensor([int(command)], device=self.device)
        tp = torch.tensor(np.asarray(target_point, np.float32)[None], device=self.device)
        spd = None if speed is None else torch.tensor([float(speed)], device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            out = self.model(img, cmd, tp, with_seg=with_seg, speed=spd)
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
        p = self.plan(rgb_bgr, command, target_point, with_seg=with_seg, speed=speed)
        action = self.follower(p["path"], p["target_speed"], speed, dt)
        return action, p
