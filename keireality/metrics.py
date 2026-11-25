"""Region-wise pixel metrics with explicit coverage and episode-level aggregation."""
import math

import cv2
import numpy as np

from keisim.config import N_SEM, SEM_NAMES
from keisim.render import rays
from .dataset import load_field
from .geometry import CHASE_CAMERA, UNKNOWN, UNSEEN, VISIBLE


def regions(visibility):
    return {"all": np.ones_like(visibility, bool), "visible": visibility == VISIBLE,
            "unseen": visibility == UNSEEN, "unknown": visibility == UNKNOWN}


def local_ssim(a, b):
    # Full-image evaluation uses mirrored borders. For sparse reconstructions only fully covered windows count.
    ma, mb = cv2.blur(a, (7, 7)), cv2.blur(b, (7, 7))
    va = np.maximum(0, cv2.blur(a*a, (7, 7)) - ma*ma)
    vb = np.maximum(0, cv2.blur(b*b, (7, 7)) - mb*mb)
    cov = cv2.blur(a*b, (7, 7)) - ma*mb
    return (((2*ma*mb + .01**2) * (2*cov + .03**2))
            / ((ma*ma + mb*mb + .01**2) * (va + vb + .03**2))).mean(-1)


class ImageScores:
    def __init__(self):
        self.sums = {name: np.zeros(6, np.float64) for name in ("all", "visible", "unseen", "unknown")}

    def add(self, pred, target, masks, support=None):
        diff = pred - target
        absolute, squared = np.abs(diff).mean(-1), np.square(diff).mean(-1)
        similarity = local_ssim(pred, target)
        if support is not None:
            interior = cv2.erode(support.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        for key, region in masks.items():
            area = region if support is None else region & support
            good_ssim = region if support is None else region & interior
            self.sums[key] += [region.sum(), area.sum(), absolute[area].sum(dtype=np.float64),
                               squared[area].sum(dtype=np.float64), similarity[good_ssim].sum(dtype=np.float64),
                               good_ssim.sum()]

    def report(self):
        result = {}
        for key, (eligible, n, total, squared, similarity, ns) in self.sums.items():
            result[key] = dict(eligible=int(eligible), valid=int(n), coverage=float(n/eligible) if eligible else None,
                               l1=float(total/n) if n else None,
                               psnr=float(-10 * math.log10(squared/n)) if squared and n else None,
                               ssim=float(similarity/ns) if ns else None, ssim_pixels=int(ns))
        return result


class SemanticScores:
    def __init__(self):
        self.matrix = {key: np.zeros((N_SEM, N_SEM), np.int64) for key in ("all", "visible", "unseen", "unknown")}

    def add(self, predicted, truth, masks):
        pair = truth.astype(np.int32) * N_SEM + predicted
        for key, mask in masks.items():
            self.matrix[key] += np.bincount(pair[mask], minlength=N_SEM**2).reshape(N_SEM, N_SEM)

    def report(self):
        result = {}
        for key, m in self.matrix.items():
            union = m.sum(0) + m.sum(1) - m.diagonal()
            iou = np.divide(m.diagonal(), union, out=np.zeros(N_SEM), where=union > 0)
            result[key] = dict(miou=float(iou[union > 0].mean()) if np.any(union) else None,
                               iou={s: float(iou[i]) if union[i] else None for i, s in enumerate(SEM_NAMES)},
                               target_pixels={s: int(m[i].sum()) for i, s in enumerate(SEM_NAMES)})
        return result


def train_baseline(root, rows):
    h, w = CHASE_CAMERA.height, CHASE_CAMERA.width
    total = np.zeros((h, w, 3), np.float64)
    classes = np.zeros((N_SEM, h, w), np.int32)
    count = 0
    for row in rows:
        if row["split"] != "train":
            raise ValueError("the pixel baseline requires training data")
        total += load_field(root, row, "rgb_target")[:, 0].sum(axis=0, dtype=np.float64)
        labels = load_field(root, row, "seg_target")[:, 0]
        for i in range(N_SEM):
            classes[i] += np.count_nonzero(labels == i, axis=0).astype(np.int32)
        count += len(labels)
    return (total / (255 * count)).astype(np.float32), classes.argmax(0).astype(np.uint8)


def temporal_error(current, previous):
    """Residual RGB change on overlapping static surfaces, using GT depth and pose only for scoring."""
    pred, gt, seg, depth, pose = current
    old_pred, old_gt, old_seg, old_depth, old_pose = previous
    cam = CHASE_CAMERA
    r = rays.decode_depth(depth)
    world = rays.unproject(r, rays.camera_rays(cam), *rays.level_frame(cam, *pose))
    uv, distance, ok = rays.project(cam, world.reshape(-1, 3), *rays.level_frame(cam, *old_pose))
    idx = np.flatnonzero(ok & (r.ravel() > 0) & (seg.ravel() >= 1) & (seg.ravel() <= 7))
    u, v = np.rint(uv[idx]).astype(int).T
    previous_distance = rays.decode_depth(old_depth[v, u])
    match = (np.abs(previous_distance - distance[idx]) <= .03 * distance[idx]) & (old_seg[v, u] == seg.ravel()[idx])
    idx, u, v = idx[match], u[match], v[match]
    residual = (pred - gt).reshape(-1, 3)[idx] - (old_pred - old_gt)[v, u]
    return float(np.abs(residual).sum(dtype=np.float64)), residual.size
