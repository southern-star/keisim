"""A causal 10 Hz window shared by recorded training sequences and streaming inference."""
from collections import deque
import math

import numpy as np

from .geometry import relative_pose


def compact_mem(mem):
    """100 tokens per time. Both ResNet levels are aligned at input pixels 32*i, 32*j."""
    lead = mem.shape[:-2]
    fine = mem[..., :400, :].reshape(*lead, 2, 10, 20, 256)[..., ::2, ::2, :]
    coarse = mem[..., 400:, :].reshape(*lead, 2, 5, 10, 256)
    merged = .5 * fine.astype(np.float32) + .5 * coarse.astype(np.float32)
    return np.moveaxis(merged, -1, -3).astype(np.float16)


class HistoryWindow:
    """push() returns only past items, newest first. A discontinuity starts a new coordinate frame."""
    def __init__(self):
        self.items = deque(maxlen=7)
        self.episode = self.time = None
        self.pose = np.zeros(3, np.float64)

    def push(self, item, episode, time, delta):
        if not math.isfinite(time) or not np.isfinite(delta).all():
            raise ValueError("time and odometry must be finite")
        if self.episode != episode or self.time is None or not math.isclose(time - self.time, .1, abs_tol=1e-6):
            self.items.clear()
            self.pose[:] = 0
        else:
            dx, dy, yaw = delta
            c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
            self.pose += (c*dx - s*dy, s*dx + c*dy, yaw)
        relative = np.zeros((7, 5), np.float32)
        relative[:, 3] = 1
        ids = []
        for k, (old, old_time, old_pose) in enumerate(reversed(self.items)):
            x, y, a = relative_pose(self.pose, old_pose)
            relative[k] = [x, y, math.sin(a), math.cos(a), old_time - time]
            ids.append(old)
        self.items.append((item, time, self.pose.copy()))
        self.episode, self.time = episode, time
        return ids, relative
