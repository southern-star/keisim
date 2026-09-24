"""Ego vehicle: kinematic bicycle with steering lag and simple powertrain."""
from __future__ import annotations

import math

import numpy as np

from .geometry import box_corners


class EgoVehicle:
    LENGTH = 4.5
    WIDTH = 1.85
    HEIGHT = 1.5
    WHEELBASE = 2.7
    LF = 1.35
    LR = 1.35
    MAX_STEER = 0.6          # rad at the wheels
    STEER_TAU = 0.12         # s, first-order steering lag
    A_MAX = 3.5              # m/s^2 at standstill
    B_MAX = 8.0              # m/s^2 full brake

    def __init__(self, x, y, yaw, v=0.0):
        self.x, self.y, self.yaw, self.v = float(x), float(y), float(yaw), float(v)
        self.steer = 0.0
        self.acc = 0.0
        self.control = np.zeros(3)

    @classmethod
    def max_accel(cls, v):
        return cls.A_MAX * max(0.25, 1.0 - v / 25.0)

    @staticmethod
    def resistance(v):
        return (0.12 if v > 0.05 else 0.0) + 0.02 * v + 0.0015 * v * v

    def step(self, steer, throttle, brake, dt):
        steer = float(np.clip(steer, -1, 1))
        throttle = float(np.clip(throttle, 0, 1))
        brake = float(np.clip(brake, 0, 1))
        self.control = np.array([steer, throttle, brake])
        target = steer * self.MAX_STEER
        self.steer += (target - self.steer) * min(1.0, dt / self.STEER_TAU)
        a = throttle * self.max_accel(self.v) - brake * self.B_MAX - self.resistance(self.v)
        v_new = max(0.0, self.v + a * dt)
        self.acc = (v_new - self.v) / dt
        v_mid = 0.5 * (self.v + v_new)
        beta = math.atan(self.LR / (self.LF + self.LR) * math.tan(self.steer))
        self.x += v_mid * math.cos(self.yaw + beta) * dt
        self.y += v_mid * math.sin(self.yaw + beta) * dt
        self.yaw += v_mid / self.LR * math.sin(beta) * dt
        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.v = v_new

    @property
    def xy(self):
        return np.array([self.x, self.y])

    @property
    def pose(self):
        return (self.x, self.y, self.yaw)

    def corners(self):
        return box_corners(self.x, self.y, self.yaw, self.LENGTH, self.WIDTH)
