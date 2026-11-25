"""Pixel-weighted RGB, semantics and support diagnostics. Geometry here is offline scoring only."""
from collections import defaultdict

import cv2
import numpy as np

from keisim.config import SEM
from keisim.render import rays
from .metrics import local_ssim


def correspondence(camera, directions, current, old, previous_camera=None):
    """Static target pixels projected into the previous target view, including newly vacated actor pixels."""
    seg, depth, pose = current
    old_seg, old_depth, old_pose = old
    idx = np.flatnonzero((seg >= 1) & (seg <= 7) & (depth > 0))
    center, rotation = rays.level_frame(camera, *pose)
    points = center + (directions.reshape(-1, 3)[idx] @ rotation) * rays.decode_depth(depth.ravel()[idx])[:, None]
    old_camera = camera if previous_camera is None else previous_camera
    uv, dist, ok = rays.project(old_camera, points, *rays.level_frame(old_camera, *old_pose))
    u, v = np.floor(uv[ok] + .5).astype(int).T
    prev = v * old_camera.width + u
    idx, dist = idx[ok], dist[ok]
    old_r = rays.decode_depth(old_depth.ravel()[prev])
    old_c = old_seg.ravel()[prev]
    overlap = (old_c == seg.ravel()[idx]) & (old_r > 0) & (np.abs(old_r-dist) <= .03*dist)
    vacated = (old_c >= SEM["vehicle"]) & (old_r > 0) & (old_r < .97*dist)
    return (idx[overlap], prev[overlap]), (idx[vacated], prev[vacated])


def pixel_scores(pred, labels, gt, seg, support_truth, vis, support_prob=None, warp=None, previous=None):
    visible = vis == 1
    static = (seg >= 1) & (seg <= 7)
    error = np.abs(pred - gt).mean(-1)
    stats = {}
    def measure(key, arr):
        stats[key] = (float(arr.sum(dtype=np.float64)), arr.size)
    def ratio(key, numerator, denominator):
        stats[key] = (int(numerator.sum()), int(denominator.sum()))

    measure("all_l1", error)
    measure("visible_l1", error[visible])
    measure("visible_ssim", local_ssim(pred, gt)[visible])
    for k in ("vehicle", "road", "pedestrian", "tl_red", "tl_green"):
        truth, found = seg == SEM[k], labels == SEM[k]
        ratio("visible_" + k + "_iou", visible & truth & found, visible & (truth | found))
        measure("visible_" + k + "_l1", error[visible & truth])
    ratio("phantom_fpr", static & (labels >= SEM["vehicle"]), static)
    if support_truth is not None:
        past = support_truth == 2
        measure("past_static_l1", error[past])
        measure("no_support_l1", error[support_truth == 0])
        if support_prob is not None:
            predicted = support_prob.argmax(-1)
            prob = support_prob[..., 1] + support_prob[..., 2]
            known = support_truth != 255
            supported = known & (support_truth > 0)
            ratio("unknown_iou", (predicted == 3) & ~known, (predicted == 3) | ~known)
            measure("support_brier", np.square(prob[known] - supported[known]))
            for k, mask in (("past", past), ("supported", supported)):
                found = (predicted == 2) if k == "past" else (prob >= .5)
                ratio("support_" + k + "_precision", known & found & mask, known & found)
                ratio("support_" + k + "_recall", found & mask, mask)
            boundary = cv2.dilate(support_truth, np.ones((3, 3), np.uint8)) != cv2.erode(support_truth, np.ones((3, 3), np.uint8))
            for k, mask in (("boundary", boundary), ("interior", ~boundary)):
                measure("support_" + k + "_error", (predicted != support_truth)[known & mask])
    if warp is not None:
        (i, j), (vacated, _) = warp
        old_pred, old_gt = previous
        diff = (pred - gt).reshape(-1, 3)[i] - (old_pred - old_gt).reshape(-1, 3)[j]
        measure("warped_residual_l1", np.abs(diff).mean(-1))
        measure("vacated_rgb_l1", error.ravel()[vacated])
        measure("vacated_actor_rate", (labels.ravel()[vacated] >= SEM["vehicle"]))
    return stats


class Totals:
    def __init__(self):
        self.sums = defaultdict(lambda: np.zeros(2, np.float64))
        self.frames = 0

    def add(self, stats):
        self.frames += 1
        for key, pair in stats.items():
            self.sums[key] += pair

    def report(self):
        return dict(frames=self.frames, metrics={k: (float(s/n) if n else None) for k, (s, n) in self.sums.items()},
                    denominators={k: int(n) for k, (_, n) in self.sums.items()})
