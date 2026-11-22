"""Accuracy of depth images against the labels (ReconView's depth head: scripts/train_depth.py, tools/depth_eval.py;
docs/recon_view.md). Distances are metres from the camera centre along each pixel's line of sight
(keisim/render/rays.py); everything here is at the labels' half resolution."""
from __future__ import annotations

import math

import numpy as np
import torch

from keisim.config import CAMERA_RIGS, SEM
from keisim.render.rays import camera_rays

from .model import DEPTH_FAR, DEPTH_NEAR

# classes -> the groups the accuracy is reported for
GROUPS = {"road": ("road", "marking", "sidewalk"), "terrain": ("terrain",), "building": ("building",),
          "vegetation": ("vegetation",), "pole": ("pole", "tl_red", "tl_yellow", "tl_green"), "vehicle": ("vehicle",),
          "pedestrian": ("pedestrian",)}
BANDS = (10.0, 30.0, 80.0)          # [m]: 0-10, 10-30, 30-80; farther labels are not scored
GROUND_FAR = 150.0                  # the ground baseline's distance where a line of sight does not come down


def codes_to_metres(q):
    """Depth codes (int tensor) as metres (float32); 0: no surface."""
    q = q.float()
    return torch.where(q > 0, DEPTH_NEAR * torch.exp(q * (math.log(DEPTH_FAR / DEPTH_NEAR) / 65535.0)), torch.zeros_like(q))


def ground_baseline(rig="wide_narrow"):
    """The baseline "every pixel is the flat ground": (K, h, w) metres to z = 0 along every half-resolution pixel's
    line of sight; GROUND_FAR where it does not come down (or farther than that)."""
    out = []
    for cfg in CAMERA_RIGS[rig]:
        down = camera_rays(cfg, 2)[..., 1]
        out.append(np.where(down > 1e-6, np.minimum(cfg.z / np.maximum(down, 1e-9), GROUND_FAR), GROUND_FAR))
    return torch.from_numpy(np.stack(out)).float()


class DepthStats:
    """Sums of |pred - gt| / gt, of (gt / pred or pred / gt) < 1.25 and of ln(pred / gt)^2 per camera, class group and
    distance band, over the labelled pixels within BANDS[-1]; and how often "a surface" is right."""

    def __init__(self, n_cams):
        self.n = np.zeros((n_cams, len(GROUPS), len(BANDS)))
        self.rel, self.d1, self.log2 = np.zeros_like(self.n), np.zeros_like(self.n), np.zeros_like(self.n)
        self.surf_ok = self.surf_n = 0
        self._lut = None

    def add(self, pred, gt, seg, surface=None):
        """pred, gt (B, K, h, w) metres (gt 0: no surface); seg (B, K, h, w) classes; surface (B, K, h, w) bool: the
        prediction's "a surface" (None: not scored)."""
        if self._lut is None:
            lut = torch.full((len(SEM),), -1, dtype=torch.long)
            for g, names in enumerate(GROUPS.values()):
                for name in names:
                    lut[SEM[name]] = g
            self._lut = lut.to(seg.device)
        K, G, NB = self.n.shape
        grp = self._lut[seg.long()]
        band = torch.bucketize(gt, torch.tensor(BANDS, device=gt.device))        # 0..2 inside, 3 beyond
        ok = (gt > 0) & (band < NB) & (grp >= 0)
        cam = torch.arange(K, device=gt.device).view(1, K, 1, 1).expand_as(gt)
        idx = ((cam * G + grp) * NB + band)[ok]
        p, g = pred[ok].double().clamp(min=1e-3), gt[ok].double()
        size = K * G * NB
        for arr, val in ((self.n, torch.ones_like(g)), (self.rel, (p - g).abs() / g),
                         (self.d1, (torch.maximum(p / g, g / p) < 1.25).double()), (self.log2, torch.log(p / g) ** 2)):
            arr += torch.bincount(idx, weights=val, minlength=size).view(K, G, NB).cpu().numpy()
        if surface is not None:
            self.surf_ok += int((surface == (gt > 0)).sum())
            self.surf_n += gt.numel()

    def summary(self):
        """{"abs_rel", "d1", "rmse_log", "pixels"}: nested dicts group -> band -> value, cameras pooled; the bands
        "0-10", "10-30", "30-80", plus "0-30" and "all"; "cameras": the same per camera for all groups pooled;
        "surface": the share of pixels whose "a surface" is right."""
        names = [f"{int(a)}-{int(b)}" for a, b in zip((0,) + BANDS[:-1], BANDS)]
        sels = {**{nm: [i] for i, nm in enumerate(names)}, "0-30": [0, 1], "all": list(range(len(BANDS)))}

        def cell(n, rel, d1, log2):
            return None if n == 0 else {"abs_rel": float(rel / n), "d1": float(d1 / n), "rmse_log": math.sqrt(float(log2 / n)),
                                        "pixels": int(n)}

        def table(cams, groups):
            return {nm: cell(*(a[cams][:, groups][:, :, sel].sum() for a in (self.n, self.rel, self.d1, self.log2)))
                    for nm, sel in sels.items()}
        every_cam, every_grp = list(range(self.n.shape[0])), list(range(len(GROUPS)))
        out = {"groups": {g: table(every_cam, [i]) for i, g in enumerate(GROUPS)}, "all": table(every_cam, every_grp),
               "cameras": [table([k], every_grp) for k in every_cam]}
        if self.surf_n:
            out["surface"] = float(self.surf_ok / self.surf_n)
        return out
