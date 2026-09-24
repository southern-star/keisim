"""Top-down ground texture (colour + semantics) with a mip pyramid.

The camera renders the ground by projecting this texture through the exact
ground-plane homography, so lane paint, crosswalks, curbs etc. are
perspective-correct at essentially zero per-frame cost.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import SEM
from ..roadnet import Raster, Town


def _tiled_noise(rng, shape, tile=512, sigma=1.0, octaves=((1, 1.0), (4, 0.6), (16, 0.35))):
    """Cheap multi-octave value noise in [-1,1] tiled over `shape` (H, W)."""
    H, W = shape
    acc = np.zeros((tile, tile), np.float32)
    for scale, amp in octaves:
        n = max(2, tile // scale)
        base = rng.standard_normal((n, n)).astype(np.float32)
        acc += amp * cv2.resize(base, (tile, tile), interpolation=cv2.INTER_CUBIC)
    acc = cv2.GaussianBlur(acc, (0, 0), sigma)
    acc /= (np.abs(acc).max() + 1e-6)
    reps = (int(np.ceil(H / tile)), int(np.ceil(W / tile)))
    return np.tile(acc, reps)[:H, :W]


class MapTexture:
    LEVELS = 4

    def __init__(self, town: Town, res: float | None = None, appearance_seed: int | None = None):
        res = res or town.cfg.tex_res
        self.res = res
        R = Raster(town.bbox, res)
        self.raster = R
        H, W = R.H, R.W
        rng = np.random.default_rng(town.seed * 7 + 11 if appearance_seed is None else appearance_seed)
        cfg = town.cfg

        # --- palette (BGR) with per-town variation
        grass = np.array([70, 125, 95]) * rng.uniform(0.8, 1.15, 3)
        if rng.random() < 0.3:  # dry / urban ground
            grass = np.array([95, 120, 125]) * rng.uniform(0.85, 1.1, 3)
        asphalt = np.full(3, rng.uniform(62, 95)) + rng.uniform(-4, 4, 3)
        walk = np.full(3, rng.uniform(150, 190)) + rng.uniform(-10, 10, 3)
        curb = np.full(3, 200.0)
        paint = np.array([225, 232, 235]) * rng.uniform(0.9, 1.05)
        self.palette = {"grass": grass, "asphalt": asphalt, "walk": walk, "paint": paint}

        sem = np.full((H, W), SEM["terrain"], np.uint8)
        wo, hw = cfg.walk_outer, cfg.road_half_width
        for road in town.roads:
            R.fill(sem, town.road_polygon(road, wo), SEM["sidewalk"])
        for J in town.junctions:
            R.fill(sem, J.walk_poly, SEM["sidewalk"])
        for poly in town.conn_walk_polys:
            R.fill(sem, poly, SEM["sidewalk"])
        curbmask = np.zeros((H, W), np.uint8)
        for road in town.roads:
            R.fill(curbmask, town.road_polygon(road, hw + 0.2), 1)
        for J in town.junctions:
            c = J.pos
            R.fill(curbmask, (J.road_poly - c) * ((np.linalg.norm(J.road_poly - c, axis=1, keepdims=True) + 0.2)
                                                  / (np.linalg.norm(J.road_poly - c, axis=1, keepdims=True) + 1e-6)) + c, 1)
        for poly in town.conn_curb_polys:
            R.fill(curbmask, poly, 1)
        for road in town.roads:
            R.fill(sem, town.road_polygon(road, hw), SEM["road"])
        for J in town.junctions:
            R.fill(sem, J.road_poly, SEM["road"])
        for poly in town.conn_road_polys:
            R.fill(sem, poly, SEM["road"])
        curbmask = (curbmask > 0) & (sem == SEM["sidewalk"])
        paintmask = np.zeros((H, W), np.uint8)
        for poly in town.markings:
            R.fill(paintmask, poly, 1)
        for poly in town.crosswalks:
            R.fill(paintmask, poly, 1)
        paintmask = paintmask > 0
        sem[paintmask] = SEM["marking"]

        # --- colour texture (processed in row stripes to keep memory low)
        T = 512
        noise_t = _tiled_noise(rng, (T, T))
        fine_t = _tiled_noise(rng, (T, T), tile=256, sigma=0.7, octaves=((8, 1.0), (32, 1.0)))
        tile_px = max(2, int(round(0.6 / res)))
        cols = np.arange(W) % T
        colgrid = (np.arange(W) % tile_px) == 0
        img = np.empty((H, W, 3), np.uint8)
        spec = (("grass", SEM["terrain"], 0.16, 0.10), ("walk", SEM["sidewalk"], 0.05, 0.05),
                ("asphalt", SEM["road"], 0.07, 0.09), ("paint", SEM["marking"], 0.04, 0.10))
        for r0 in range(0, H, 256):
            r1 = min(H, r0 + 256)
            rows = np.arange(r0, r1) % T
            nz = noise_t[rows][:, cols]
            fz = fine_t[rows][:, cols]
            cls = sem[r0:r1]
            out = np.empty((r1 - r0, W, 3), np.float32)
            for key, cid, amp_c, amp_f in spec:
                m = cls == cid
                shade = 1.0 + amp_c * nz[m] + amp_f * fz[m]
                out[m] = self.palette[key].astype(np.float32)[None, :] * shade[:, None]
            grid = colgrid[None, :] | ((np.arange(r0, r1) % tile_px) == 0)[:, None]
            out[(cls == SEM["sidewalk"]) & grid] *= 0.88
            cm = curbmask[r0:r1]
            out[cm] = curb.astype(np.float32)[None, :] * (1.0 + 0.03 * nz[cm])[:, None]
            img[r0:r1] = np.clip(out, 0, 255).astype(np.uint8)
        self.color = [img]
        self.sem = [sem]
        for _ in range(1, self.LEVELS):
            self.color.append(cv2.pyrDown(self.color[-1]))
            self.sem.append(np.ascontiguousarray(self.sem[-1][::2, ::2]))
        self.border = tuple(int(v) for v in grass)

    def level_affine(self, level):
        """3x3 matrix mapping texture pixel (level) -> world (x, y, 1)."""
        r = self.res * (2 ** level)
        R = self.raster
        return np.array([[r, 0.0, R.x0 + 0.5 * r], [0.0, -r, R.y1 - 0.5 * r], [0.0, 0.0, 1.0]])
