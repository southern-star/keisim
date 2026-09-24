"""Small, vectorised 2D geometry toolkit used everywhere in KeiSim.

Conventions
-----------
* World frame: x east, y north, yaw counter-clockwise from +x (radians).
* Vehicle / ego frame: x forward, y left.
"""
from __future__ import annotations

import bisect
import math

import numpy as np

TAU = 2.0 * np.pi


def wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % TAU - np.pi


def unit(theta):
    return np.array([np.cos(theta), np.sin(theta)])


def world_to_local(pts, x, y, yaw):
    """World points (...,2) -> local frame of a pose (x fwd, y left)."""
    pts = np.asarray(pts, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    dx = pts[..., 0] - x
    dy = pts[..., 1] - y
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)


def local_to_world(pts, x, y, yaw):
    pts = np.asarray(pts, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    px, py = pts[..., 0], pts[..., 1]
    return np.stack([x + c * px - s * py, y + s * px + c * py], axis=-1)


def box_corners(x, y, yaw, length, width):
    """Corners of oriented boxes, shape (...,4,2), order FL, FR, RR, RL."""
    x, y, yaw = np.asarray(x, float), np.asarray(y, float), np.asarray(yaw, float)
    hl, hw = np.asarray(length, float) / 2, np.asarray(width, float) / 2
    c, s = np.cos(yaw), np.sin(yaw)
    lx = np.stack([hl, hl, -hl, -hl], axis=-1)
    ly = np.stack([hw, -hw, -hw, hw], axis=-1) * np.ones_like(lx)
    cx = x[..., None] + c[..., None] * lx - s[..., None] * ly
    cy = y[..., None] + s[..., None] * lx + c[..., None] * ly
    return np.stack([cx, cy], axis=-1)


def obb_overlap(a, b):
    """Separating-axis test. a: (4,2) corners, b: (N,4,2) corners -> bool (N,)."""
    b = np.asarray(b, float)
    if b.ndim == 2:
        b = b[None]
    n = b.shape[0]
    if n == 0:
        return np.zeros(0, bool)
    ea0 = a[1] - a[0]
    ea1 = a[3] - a[0]
    eb0 = b[:, 1] - b[:, 0]
    eb1 = b[:, 3] - b[:, 0]
    axes = [np.broadcast_to(ea0, (n, 2)), np.broadcast_to(ea1, (n, 2)), eb0, eb1]
    hit = np.ones(n, bool)
    for ax in axes:
        pa = np.einsum("kj,nj->nk", a, ax)  # (n,4)
        pb = np.einsum("nkj,nj->nk", b, ax)  # (n,4)
        hit &= (pa.max(1) >= pb.min(1)) & (pb.max(1) >= pa.min(1))
    return hit


# ----------------------------------------------------------------------------
# Curves / polylines
# ----------------------------------------------------------------------------

def cubic_bezier(p0, p1, p2, p3, n=64):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def arc_bezier(p0, h0, p1, h1, n=64):
    """Smooth, near-circular connector from pose (p0,h0) to pose (p1,h1)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    chord = np.linalg.norm(p1 - p0)
    delta = abs(float(wrap(h1 - h0)))
    if delta < 1e-3:
        d = chord / 3.0
    else:
        radius = chord / (2.0 * np.sin(delta / 2.0))
        d = (4.0 / 3.0) * np.tan(delta / 4.0) * radius
        d = min(d, chord)  # safety for odd configurations
    return cubic_bezier(p0, p0 + d * unit(h0), p1 - d * unit(h1), p1, n)


def seg_lengths(pts):
    d = np.diff(pts, axis=0)
    return np.hypot(d[:, 0], d[:, 1])


def cumlen(pts):
    return np.concatenate([[0.0], np.cumsum(seg_lengths(pts))])


def resample(pts, ds=0.5):
    """Resample polyline at (almost) uniform arc-length spacing, keeping both ends."""
    pts = np.asarray(pts, float)
    s = cumlen(pts)
    keep = np.concatenate([[True], np.diff(s) > 1e-9])
    pts, s = pts[keep], s[keep]
    total = s[-1]
    n = max(2, int(np.ceil(total / ds)) + 1)
    si = np.linspace(0.0, total, n)
    return np.stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])], axis=-1)


def headings_of(pts):
    d = np.gradient(pts, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def normals_of(pts):
    h = headings_of(pts)
    return np.stack([-np.sin(h), np.cos(h)], axis=-1)  # left normal


def offset_polyline(pts, d):
    """Offset to the left by d (negative -> right)."""
    return pts + normals_of(pts) * d


def curvature_of(pts):
    h = np.unwrap(headings_of(pts))
    s = cumlen(pts)
    k = np.gradient(h, s, edge_order=1) if len(pts) > 2 else np.zeros(len(pts))
    return k


def polyline_project(pts, p):
    """Closest point on polyline to p. Returns (s, lateral(+left), seg_index)."""
    a = pts[:-1]
    d = pts[1:] - a
    L2 = np.maximum((d ** 2).sum(1), 1e-12)
    t = np.clip(((p - a) * d).sum(1) / L2, 0.0, 1.0)
    q = a + d * t[:, None]
    dist2 = ((q - p) ** 2).sum(1)
    i = int(np.argmin(dist2))
    L = np.sqrt(L2)
    s0 = cumlen(pts)
    cross = d[i, 0] * (p[1] - a[i, 1]) - d[i, 1] * (p[0] - a[i, 0])
    return s0[i] + t[i] * L[i], cross / L[i], i


class Polyline:
    """Arc-length parametrised polyline with fast interpolation & projection."""

    def __init__(self, pts):
        pts = np.asarray(pts, float)
        s = cumlen(pts)
        keep = np.concatenate([[True], np.diff(s) > 1e-9])
        self.pts = pts[keep]
        self.s = cumlen(self.pts)
        self.length = float(self.s[-1])
        d = np.diff(self.pts, axis=0)
        self.seg_len = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
        self.seg_dir = d / self.seg_len[:, None]
        self.seg_heading = np.arctan2(self.seg_dir[:, 1], self.seg_dir[:, 0])
        # python lists for fast scalar queries (numpy scalar ops are slow)
        self._sl = self.s.tolist()
        self._px = self.pts[:, 0].tolist()
        self._py = self.pts[:, 1].tolist()
        self._dx = self.seg_dir[:, 0].tolist()
        self._dy = self.seg_dir[:, 1].tolist()
        self._len = self.seg_len.tolist()
        self._h = self.seg_heading.tolist()
        self._nseg = len(self._len)

    def _seg1(self, s):
        if s <= 0.0:
            return 0, 0.0
        if s >= self.length:
            i = self._nseg - 1
            return i, self._len[i]
        i = bisect.bisect_right(self._sl, s) - 1
        if i >= self._nseg:
            i = self._nseg - 1
        return i, s - self._sl[i]

    def interp1(self, s):
        """Scalar fast path -> (x, y)."""
        i, d = self._seg1(s)
        return self._px[i] + self._dx[i] * d, self._py[i] + self._dy[i] * d

    def heading1(self, s):
        i, d = self._seg1(s)
        t = d / self._len[i] - 0.5
        h = self._h[i]
        if t >= 0.0:
            if i + 1 < self._nseg:
                return h + ((self._h[i + 1] - h + math.pi) % TAU - math.pi) * t
            return h
        if i > 0:
            return h + ((h - self._h[i - 1] + math.pi) % TAU - math.pi) * t
        return h

    def interp(self, s):
        s = np.minimum(np.maximum(np.asarray(s, float), 0.0), self.length)
        i = np.minimum(np.maximum(np.searchsorted(self.s, s, side="right") - 1, 0), len(self.seg_len) - 1)
        xy = self.pts[i] + self.seg_dir[i] * (s - self.s[i])[..., None]
        return xy

    def heading(self, s):
        s = np.minimum(np.maximum(np.asarray(s, float), 0.0), self.length)
        n = len(self.seg_len)
        i = np.minimum(np.maximum(np.searchsorted(self.s, s, side="right") - 1, 0), n - 1)
        # blend neighbouring segment headings for a continuous heading
        t = (s - self.s[i]) / self.seg_len[i] - 0.5
        h0 = self.seg_heading[i]
        nxt = self.seg_heading[np.minimum(i + 1, n - 1)]
        prv = self.seg_heading[np.maximum(i - 1, 0)]
        dh = np.where(t >= 0, wrap(nxt - h0), wrap(h0 - prv))
        return wrap(h0 + dh * t)

    def project(self, p, s_lo=None, s_hi=None):
        """Project point p. Optional arc-length window [s_lo, s_hi] for speed and
        for disambiguating self-overlapping routes. Returns (s, lateral)."""
        i0, i1 = 0, len(self.seg_len)
        if s_lo is not None:
            i0 = max(0, int(np.searchsorted(self.s, s_lo, side="right")) - 1)
        if s_hi is not None:
            i1 = min(len(self.seg_len), int(np.searchsorted(self.s, s_hi, side="right")) + 1)
        if i1 <= i0:
            i1 = min(len(self.seg_len), i0 + 1)
        a = self.pts[i0:i1]
        d = self.seg_dir[i0:i1]
        L = self.seg_len[i0:i1]
        rel = p - a
        t = np.clip((rel * d).sum(1), 0.0, L)
        q = a + d * t[:, None]
        dist2 = ((q - p) ** 2).sum(1)
        k = int(np.argmin(dist2))
        lat = d[k, 0] * rel[k, 1] - d[k, 1] * rel[k, 0]
        return float(self.s[i0 + k] + t[k]), float(lat)
