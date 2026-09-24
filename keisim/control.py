"""Low-level controllers shared by the privileged expert and the learned agent.

The expert and the model output the same kind of plan (a path in the ego frame
and a target speed); using the exact same controllers for both guarantees that
a perfect prediction reproduces the expert's driving.
"""
from __future__ import annotations

import math

import numpy as np

from .vehicle import EgoVehicle


class PurePursuit:
    def __init__(self, wheelbase=EgoVehicle.WHEELBASE, lr=EgoVehicle.LR, max_steer=EgoVehicle.MAX_STEER,
                 k=0.45, ld_min=3.5, ld_max=11.0):
        self.L, self.lr, self.max_steer = wheelbase, lr, max_steer
        self.k, self.ld_min, self.ld_max = k, ld_min, ld_max

    def lookahead(self, v):
        return float(np.clip(self.ld_min + self.k * v, self.ld_min, self.ld_max))

    def __call__(self, path, v):
        """path: (N,2) points in the ego frame (x fwd, y left), ordered by distance."""
        path = np.asarray(path, float)
        # reference = rear axle
        pts = path + np.array([self.lr, 0.0])
        pts = np.concatenate([[[0.0, 0.0]], pts], 0)
        d = np.hypot(pts[:, 0], pts[:, 1])
        ld = self.lookahead(v)
        idx = np.nonzero(d >= ld)[0]
        if len(idx) == 0:
            tgt = pts[-1]
        else:
            i = idx[0]
            a, b = pts[i - 1], pts[i]
            da, db = d[i - 1], d[i]
            t = (ld - da) / max(db - da, 1e-6)
            tgt = a + (b - a) * np.clip(t, 0, 1)
        l2 = max(float(tgt @ tgt), 1e-3)
        curv = 2.0 * tgt[1] / l2
        delta = math.atan(curv * self.L)
        return float(np.clip(delta / self.max_steer, -1.0, 1.0))


class SpeedController:
    """Inverse-dynamics feed-forward + PI feedback on speed (asymmetric gains)."""

    def __init__(self, kp_acc=1.5, kp_dec=3.0, ki=0.3, a_max=2.5, b_max=6.5):
        self.kp_acc, self.kp_dec, self.ki = kp_acc, kp_dec, ki
        self.a_max, self.b_max = a_max, b_max
        self.i = 0.0

    def reset(self):
        self.i = 0.0

    def __call__(self, target, v, dt):
        if target < 0.05:
            self.i = 0.0
            if v < 0.3:
                return 0.0, 1.0
            a_des = -min(self.b_max, max(4.0, 2.0 * v))
        else:
            e = target - v
            self.i = float(np.clip(self.i + e * dt, -2.0, 2.0))
            kp = self.kp_acc if e > 0 else self.kp_dec
            a_des = float(np.clip(kp * e + self.ki * self.i, -self.b_max, self.a_max))
        res = EgoVehicle.resistance(v)
        need = a_des + res
        if need >= 0:
            return float(np.clip(need / EgoVehicle.max_accel(v), 0.0, 1.0)), 0.0
        return 0.0, float(np.clip(-need / EgoVehicle.B_MAX, 0.0, 1.0))


class PlanFollower:
    """(path, target speed) -> (steer, throttle, brake)."""

    def __init__(self):
        self.pp = PurePursuit()
        self.sc = SpeedController()

    def reset(self):
        self.sc.reset()

    def __call__(self, path, target_speed, v, dt):
        steer = self.pp(path, v)
        thr, brk = self.sc(target_speed, v, dt)
        return np.array([steer, thr, brk])
