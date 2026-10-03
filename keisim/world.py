"""World state: town + ego + NPC traffic + pedestrians + signals."""
from __future__ import annotations

import math

import numpy as np

from .config import EnvConfig
from .geometry import box_corners, obb_overlap
from .render.camera import SceneState
from .traffic import Pedestrians, Traffic, vehicle_circles
from .vehicle import EgoVehicle


class World:
    def __init__(self, town, cfg: EnvConfig, rng, ego_pose, route=None):
        self.town, self.cfg, self.rng = town, cfg, rng
        self.t = float(rng.uniform(0, 200))  # random signal phase at start
        self.ego = EgoVehicle(*ego_pose)
        tc = cfg.traffic
        n_veh = int(min(tc.max_vehicles, town.total_lane_length() / rng.uniform(*tc.vehicle_spacing)))
        self.traffic = Traffic(town, tc, rng, n_veh)
        ego_xy = self.ego.xy[None]
        # a few vehicles ahead of the ego on its route -> car-following situations
        if route is not None and rng.random() < 0.7:
            first = route.lanes[0]
            s0 = route.s_start
            L = town.lanes[first].length
            if L - s0 > 30:
                self.traffic.spawn_many(1, ego_xy, avoid_r=16.0, lanes=[first], s_range=(s0 + 16.0, min(L - 3, s0 + 45.0)))
        self.traffic.spawn_many(n_veh - self.traffic.n, ego_xy, avoid_r=14.0)
        n_ped = int(min(tc.max_peds, sum(r.length for r in town.roads) / rng.uniform(*tc.ped_spacing)))
        self.peds = Pedestrians(town, tc, rng, n_ped)
        self.route = route
        self.ego_s = None                  # the ego's arc length along its route (set by KeiEnv after each step)
        self.n_heads = len(town.signal_heads)
        self._head_j = np.array([h["junction"] for h in town.signal_heads], int)
        self._head_p = np.array([h["phase"] for h in town.signal_heads], int)

    # ----------------------------------------------------------------- state
    def lamp_states(self):
        J = self.town.junctions
        return np.array([J[j].signal_state(p, self.t) for j, p in zip(self._head_j, self._head_p)], int)

    def ego_circles(self):
        e = self.ego
        return vehicle_circles(np.array([[e.x, e.y]]), np.array([e.yaw]), np.array([e.LENGTH]), np.array([e.WIDTH]))

    def scene_state(self, include_ego=False):
        tr, pd = self.traffic, self.peds
        xy, yaw, dims = tr.xy, tr.yaw, tr.dims
        color, kind = tr.color, tr.kind
        brake = (tr.acc < -0.4) | (tr.v < 0.2)
        if include_ego:
            e = self.ego
            xy = np.vstack([xy, [[e.x, e.y]]])
            yaw = np.append(yaw, e.yaw)
            dims = np.vstack([dims, [[e.LENGTH, e.WIDTH, e.HEIGHT]]])
            color = np.vstack([color, [[40, 200, 60]]])
            kind = np.append(kind, 0)
            brake = np.append(brake, e.control[2] > 0.05)
        return SceneState(xy, yaw, dims, color, kind, brake, pd.xy, pd.yaw, pd.height, pd.colors, pd.phase,
                          self.lamp_states())

    # ------------------------------------------------------------------ step
    def step(self, control, dt):
        e = self.ego
        e.step(control[0], control[1], control[2], dt)
        exy, er, _ = self.ego_circles()
        pd = self.peds
        ext_xy = np.concatenate([exy, pd.xy])
        ext_r = np.concatenate([er, np.full(pd.n, 0.35)])
        ext_o = np.concatenate([np.full(len(exy), -1), -2 - np.arange(pd.n)])
        ext_v = np.concatenate([np.full(len(exy), e.v), pd.vel])
        ext_yaw = np.concatenate([np.full(len(exy), e.yaw), pd.yaw])
        ego_info = None
        if self.traffic.has_yields and self.route is not None and self.ego_s is not None:
            k = self.route.lane_index_at(self.ego_s)
            lanes = self.route.lanes
            ego_info = (lanes[k], self.ego_s - self.route.offsets[k], lanes[k + 1] if k + 1 < len(lanes) else None,
                        e.v, e.LENGTH)
        self.traffic.step(dt, self.t, ext_xy, ext_r, ext_o, ext_v, ext_yaw, e.xy, e.v, e.yaw, ego_info)
        # pedestrians (crossing is triggered more often just ahead of the ego)
        tr = self.traffic
        veh_xy = np.vstack([tr.xy, e.xy[None]])
        veh_v = np.append(tr.v, e.v)
        veh_yaw = np.append(tr.yaw, e.yaw)
        trigger = None
        if pd.n:
            d = pd.xy - e.xy
            u = np.array([math.cos(e.yaw), math.sin(e.yaw)])
            along = d @ u
            lat = np.abs(d[:, 0] * u[1] - d[:, 1] * u[0])
            near = (along > 20.0) & (along < 45.0) & (lat < 11.0)
            trigger = near & (self.rng.random(pd.n) < self.cfg.traffic.ego_cross_rate * dt)
        pd.step(dt, veh_xy, veh_v, veh_yaw, trigger)
        self.t += dt

    # ------------------------------------------------------------ collisions
    def ego_collisions(self):
        e = self.ego
        ec = e.corners()
        hits = []
        tr = self.traffic
        if tr.n:
            d = np.hypot(*(tr.xy - e.xy).T)
            idx = np.nonzero(d < 9.0)[0]
            if len(idx):
                bc = box_corners(tr.xy[idx, 0], tr.xy[idx, 1], tr.yaw[idx], tr.dims[idx, 0], tr.dims[idx, 1])
                for i in idx[obb_overlap(ec, bc)]:
                    hits.append(("vehicle", int(i)))
        pd = self.peds
        if pd.n:
            d = np.hypot(*(pd.xy - e.xy).T)
            idx = np.nonzero(d < 4.0)[0]
            if len(idx):
                bc = box_corners(pd.xy[idx, 0], pd.xy[idx, 1], pd.yaw[idx], np.full(len(idx), 0.45), np.full(len(idx), 0.5))
                for i in idx[obb_overlap(ec, bc)]:
                    hits.append(("pedestrian", int(i)))
        sb = self.town.static_boxes
        if len(sb):
            r = 0.5 * np.hypot(sb[:, 3], sb[:, 4])
            d = np.hypot(sb[:, 0] - e.x, sb[:, 1] - e.y)
            idx = np.nonzero(d < r + 3.0)[0]
            if len(idx):
                for i in idx[obb_overlap(ec, self.town.static_corners[idx])]:
                    hits.append(("static", int(i)))
        return hits
