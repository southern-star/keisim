"""ReconFill's training data (docs/recon_fill.md): the samples web/tools/fill_dump.mjs writes, and the camera they are
taken from.

A run is a directory DIR/NAME with one .npy per array (N samples, 160 x 320, top row first) and meta.json:
  in_rgb, in_dep        ReconView's picture without a fill, from its chase camera (what a fill model is given)
  loose_rgb, loose_dep  the current frame alone by a wide edge ratio
  gt_rgb, gt_dep        the town itself from there (the colours as the model frames are drawn)
  prev_rgb, prev_dep    the town of the fill before, as a mesh, drawn from this frame's chase camera
Colours are uint8 RGB; where a picture of the view has no surface its colour is BG. Depths are the 16 bit codes of
keisim/render/rays.py (0: no surface)."""
from __future__ import annotations

import json
import math
import os

import numpy as np

from keisim.config import CameraConfig
from keisim.render import rays as RY

# web/src/recon.js CHASE: 8 m behind the car's centre and 4 m up, looking at the road 12 m ahead of the car
CHASE = CameraConfig(width=320, height=160, fov_deg=math.degrees(2 * math.atan(2 * math.tan(math.radians(25.0)))),
                     x=-8.0, y=0.0, z=4.0, pitch_deg=math.degrees(math.atan2(4.0, 20.0)), name="chase")
BG = (18, 19, 28)                 # the view's background (web/src/recon.js BG, #12131c)
MAX_R = 80.0                      # the view draws no surface farther than this [m]
ARRAYS = {"in_rgb": 3, "in_dep": 0, "loose_rgb": 3, "loose_dep": 0, "gt_rgb": 3, "gt_dep": 0, "prev_rgb": 3, "prev_dep": 0}


def ground_depth(step=1):
    """The distance [m] from the chase camera to the flat ground (z = 0) along every pixel's line of sight
    (H // step, W // step); 0 where the line of sight does not come down."""
    C, R = RY.level_frame(CHASE, 0.0, 0.0, 0.0)
    dz = (RY.camera_rays(CHASE, step) @ R)[..., 2]
    return np.where(dz < -1e-6, CHASE.z / np.maximum(-dz, 1e-6), 0.0)


def load_run(path, mmap=True):
    """(arrays, meta) of one run; the arrays memory-mapped unless mmap is False."""
    meta = json.load(open(os.path.join(path, "meta.json")))
    arrays = {k: np.load(os.path.join(path, f"{k}.npy"), mmap_mode="r" if mmap else None) for k in ARRAYS}
    return arrays, meta


def runs_of(root):
    """The runs under a directory (itself, when it is one)."""
    if os.path.exists(os.path.join(root, "meta.json")):
        return [root]
    return sorted(os.path.join(root, d) for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "meta.json")))
