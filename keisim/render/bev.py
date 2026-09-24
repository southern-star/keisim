"""Ego-centric bird's-eye-view rendering for visualisation/debugging."""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..config import TL_GREEN, TL_RED, TL_YELLOW
from ..geometry import box_corners

TL_BGR = {TL_RED: (40, 40, 240), TL_YELLOW: (0, 200, 255), TL_GREEN: (60, 220, 60)}


class BEVRenderer:
    def __init__(self, tex, width=320, height=480, res=0.2, ego_row=0.72):
        self.tex, self.res = tex, res
        self.W, self.H = width, height
        self.size = max(width, height)
        self.u0 = width / 2.0
        self.v0 = height * ego_row

    def local_to_px(self, pts_local):
        pts_local = np.asarray(pts_local, float)
        u = self.u0 - pts_local[..., 1] / self.res
        v = self.v0 - pts_local[..., 0] / self.res
        return np.stack([u, v], -1)

    def world_to_px(self, pts, pose):
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        d = np.asarray(pts, float) - np.array([x, y])
        lx = c * d[..., 0] + s * d[..., 1]
        ly = -s * d[..., 0] + c * d[..., 1]
        return self.local_to_px(np.stack([lx, ly], -1))

    def base(self, pose):
        """Warp the ground texture into the ego frame."""
        x, y, yaw = pose
        tex = self.tex
        level = int(np.clip(round(math.log2(self.res / tex.res)), 0, tex.LEVELS - 1))
        A = tex.level_affine(level)            # tex px -> world
        Ainv = np.linalg.inv(A)
        c, s = math.cos(yaw), math.sin(yaw)
        # bev px -> local: x_fwd = (v0 - v) res, y_left = (u0 - u) res
        B = np.array([[0.0, -self.res, self.v0 * self.res],
                      [-self.res, 0.0, self.u0 * self.res],
                      [0.0, 0.0, 1.0]])
        # local -> world
        Wm = np.array([[c, -s, x], [s, c, y], [0, 0, 1.0]])
        M = Ainv @ Wm @ B                      # bev px -> tex px
        out = cv2.warpAffine(tex.color[level], M[:2], (self.W, self.H),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=tex.border)
        return out

    def render(self, world, pose=None, route=None, paths=(), lamp_states=None):
        """paths: iterable of (pts_local (N,2), BGR colour)."""
        e = world.ego
        pose = pose or e.pose
        img = self.base(pose)
        town = world.town
        # buildings
        for b in town.buildings:
            if abs(b["x"] - pose[0]) > 60 or abs(b["y"] - pose[1]) > 60:
                continue
            c = box_corners(b["x"], b["y"], b["yaw"], b["l"], b["w"])
            cv2.fillPoly(img, [np.round(self.world_to_px(c, pose) * 4).astype(np.int32)], (120, 120, 150), cv2.LINE_AA, shift=2)
        # route
        if route is not None:
            s0 = max(route.s_start, 0.0)
            ss = np.arange(s0, route.s_end, 1.0)
            pts = self.world_to_px(route.poly.interp(ss), pose)
            cv2.polylines(img, [np.round(pts * 4).astype(np.int32)], False, (255, 220, 120), 2, cv2.LINE_AA, shift=2)
        # stop lines coloured by state
        t = world.t
        for lane in town.lanes:
            if lane.stop_s is None:
                continue
            p = lane.poly.interp(lane.stop_s)
            if abs(p[0] - pose[0]) > 50 or abs(p[1] - pose[1]) > 50:
                continue
            h = float(lane.poly.heading(lane.stop_s))
            n = np.array([-math.sin(h), math.cos(h)])
            a, b = p + n * 1.7, p - n * 1.7
            st = town.signal_state_for_lane(lane.id, t)
            q = self.world_to_px(np.stack([a, b]), pose)
            cv2.line(img, tuple(np.round(q[0]).astype(int)), tuple(np.round(q[1]).astype(int)), TL_BGR[st], 3, cv2.LINE_AA)
        # vehicles
        tr = world.traffic
        if tr.n:
            c = box_corners(tr.xy[:, 0], tr.xy[:, 1], tr.yaw, tr.dims[:, 0], tr.dims[:, 1])
            for i in range(tr.n):
                q = self.world_to_px(c[i], pose)
                if q[:, 0].max() < 0 or q[:, 0].min() > self.W or q[:, 1].max() < 0 or q[:, 1].min() > self.H:
                    continue
                cv2.fillPoly(img, [np.round(q * 4).astype(np.int32)], (200, 120, 40), cv2.LINE_AA, shift=2)
                cv2.polylines(img, [np.round(q * 4).astype(np.int32)], True, (255, 200, 120), 1, cv2.LINE_AA, shift=2)
        pd = world.peds
        if pd.n:
            q = self.world_to_px(pd.xy, pose)
            for i in range(pd.n):
                if 0 <= q[i, 0] < self.W and 0 <= q[i, 1] < self.H:
                    cv2.circle(img, (int(q[i, 0] * 4), int(q[i, 1] * 4)), 10, (40, 60, 240) if pd.mode[i] else (40, 140, 255),
                               -1, cv2.LINE_AA, shift=2)
        # ego
        c = box_corners(e.x, e.y, e.yaw, e.LENGTH, e.WIDTH)
        q = self.world_to_px(c, pose)
        cv2.fillPoly(img, [np.round(q * 4).astype(np.int32)], (60, 200, 60), cv2.LINE_AA, shift=2)
        for pts, col in paths:
            q = self.local_to_px(pts)
            for p in q:
                cv2.circle(img, (int(p[0] * 4), int(p[1] * 4)), 9, col, -1, cv2.LINE_AA, shift=2)
        return img
