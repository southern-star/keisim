"""World state: town + ego + NPC traffic + pedestrians + signals."""
from __future__ import annotations

import math

import numpy as np

from .bicycles import BIKE_LEN, BIKE_R, BIKE_W, SIGNAL_D
from .config import EnvConfig
from .geometry import box_corners, obb_overlap
from .route import link_map
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
        self.parked = None
        if tc.parked > 0:       # parked cars (keisim/parked.py); no random draw without them
            from .parked import ParkedCars, parked_lanes
            n_pk = int(min(tc.max_parked, round(tc.parked * sum(town.lanes[l].length for l in parked_lanes(town))
                                                / 1000.0)))
            self.parked = ParkedCars(town, tc, rng, n_pk, avoid_xy=ego_xy, route=route)
            self.traffic.parked = self.parked
        # a few vehicles ahead of the ego on its route -> car-following situations (none with max_vehicles 0)
        if route is not None and tc.max_vehicles > 0 and rng.random() < 0.7:
            first = route.lanes[0]
            s0 = route.s_start
            L = town.lanes[first].length
            if L - s0 > 30:
                self.traffic.spawn_many(1, ego_xy, avoid_r=16.0, lanes=[first], s_range=(s0 + 16.0, min(L - 3, s0 + 45.0)))
        self.traffic.spawn_many(n_veh - self.traffic.n, ego_xy, avoid_r=14.0)
        n_ped = int(min(tc.max_peds, sum(r.length for r in town.roads) / rng.uniform(*tc.ped_spacing)))
        focus = ()
        if route is not None and tc.crosswalks > 0:     # the zebras at the exits of the route's turns
            focus = [ci for ci, kind, turn in route.zebras(town) if kind == "exit" and turn != "straight"]
        self.peds = Pedestrians(town, tc, rng, n_ped, focus_cw=focus)
        self.darts = []         # hidden children (TrafficConfig.dart_out): [ped index, route s of their line, trigger m]
        if self.parked is not None and self.parked.n:
            self.peds.set_parked(self.parked)
            if tc.dart_out > 0 and route is not None:
                self._hide_children(route)
        self.bikes = None
        if tc.bicycles > 0:     # bicycles (keisim/bicycles.py); no random draw without them
            from .bicycles import Bicycles
            n_bk = int(min(tc.max_bikes, round(tc.bicycles * sum(r.length for r in town.roads) / 1000.0)))
            cxy, cr, _ = self.traffic.circles()
            if self.parked is not None and self.parked.n:          # nor in or just behind a parked car (room to
                pk = self.parked                                    # move out round it: 8 m past its ends)
                rb = BIKE_LEN / 2 if tc.bike_hook else BIKE_R       # (Bicycles._free adds rb + 0.45)
                pr = pk.dims[:, 0] / 2 + 8.0 - (rb + 0.45)
                cxy, cr = np.concatenate([cxy, pk.xy]), np.concatenate([cr, pr])
            ahead = None
            if route is not None and rng.random() < 0.6:        # one ahead of the ego on its first road
                first = town.lanes[route.lanes[0]]
                s0 = route.s_start
                if first.kind == "road" and first.lane_index == 0 and first.length - s0 > 30:
                    ahead = (first.id, s0 + 14.0, min(first.length - 3.0, s0 + 50.0))
            self.bikes = Bicycles(town, tc, rng, n_bk, avoid_xy=ego_xy, avoid_r=10.0, car_xy=cxy, car_r=cr, ahead=ahead)
            self.traffic.bikes = self.bikes
        self.route = route
        self.ego_s = None                  # the ego's arc length along its route (set by KeiEnv after each step)
        self.ego_stood = 0.0               # s the ego has stood still (TrafficConfig.ego_patience_s)
        self.n_heads = len(town.signal_heads)
        self._head_j = np.array([h["junction"] for h in town.signal_heads], int)
        self._head_p = np.array([h["phase"] for h in town.signal_heads], int)

    def _hide_children(self, route):
        """TrafficConfig.dart_out: children hidden in front of the parked cars along the route ahead of the ego
        (keisim/parked.py), each with the route position of their line and the distance at which they run out."""
        from .parked import DART_HEIGHT, DART_SPEED, DART_TRIGGER, KID_SHIRTS, dart_spot
        town, pk, rng = self.town, self.parked, self.rng
        for k, lid in enumerate(route.lanes):
            for j in pk.by_lane.get(lid, ()):
                if k == 0 and pk.s[j] < route.s_start + 30.0:
                    continue
                if rng.random() >= self.cfg.traffic.dart_out:
                    continue
                spot = dart_spot(town, pk, j)
                if spot is None:
                    continue
                xy, to, w_to, s_to, lat_to = spot
                s_loc = float(town.lanes[lid].poly.project(xy)[0])
                s_r = float(route.offsets[k] + s_loc - route.s0[k])
                if not route.s_start + 20.0 < s_r < route.s_end:
                    continue
                shirt = KID_SHIRTS[rng.integers(len(KID_SHIRTS))]
                i = self.peds.add_hidden(xy, to, w_to, s_to, lat_to, rng.uniform(*DART_HEIGHT), rng.uniform(*DART_SPEED),
                                         shirt)
                self.darts.append([i, s_r, float(rng.uniform(*DART_TRIGGER)), False])

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
        pk = self.parked
        if pk is not None and pk.n:     # parked cars: after the NPCs (and the ego), lights off
            p_xy, p_yaw, p_dims, p_col, p_kind, p_brake = pk.scene()
            xy, yaw, dims = np.vstack([xy, p_xy]), np.append(yaw, p_yaw), np.vstack([dims, p_dims])
            color, kind, brake = np.vstack([color, p_col]), np.append(kind, p_kind), np.append(brake, p_brake)
        st = SceneState(xy, yaw, dims, color, kind, brake, pd.xy, pd.yaw, pd.height, pd.colors, pd.phase,
                        self.lamp_states())
        if self.bikes is not None and self.bikes.n:
            (st.bike_xy, st.bike_yaw, st.bike_pose, st.bike_kind, st.bike_height,
             st.bike_colors) = self.bikes.scene()
        return st

    # ------------------------------------------------------------------ step
    def step(self, control, dt):
        e = self.ego
        e.step(control[0], control[1], control[2], dt)
        self.ego_stood = self.ego_stood + dt if e.v < 0.1 else 0.0
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
            patience = self.cfg.traffic.ego_patience_s
            ego_info = (lanes[k], self.route.local_s(self.ego_s, k), lanes[k + 1] if k + 1 < len(lanes) else None,
                        e.v, e.LENGTH, lanes[k + 2] if k + 2 < len(lanes) else None, 0.0 < patience <= self.ego_stood)
        ped_margin = pass_xy = None
        if (self.town.in_road_walkways or pd.cw_on) and pd.n:   # people walking in a 路側帯 are passed closely,
            walk = pd.walking_in_road()                          # but slowly; people waiting at a zebra's kerb closely
            ped_margin = np.where(walk | pd.waiting_at_kerb(), 0.2, 0.9)
            pass_xy = pd.xy[walk]
        if self.darts:          # the children hiding in front of parked cars are beside the cars' way
            ped_margin = np.where(pd.mode == pd.HIDE, 0.0, 0.9 if ped_margin is None else ped_margin)
        self.traffic.step(dt, self.t, ext_xy, ext_r, ext_o, ext_v, ext_yaw, e.xy, e.v, e.yaw, ego_info,
                          ped_margin=ped_margin, pass_xy=pass_xy, ego_len=e.LENGTH, ego_wait=self._ego_lc_wait(),
                          peds=pd)
        # pedestrians (crossing is triggered more often just ahead of the ego)
        tr = self.traffic
        veh_xy = np.vstack([tr.xy, e.xy[None]])
        veh_v = np.append(tr.v, e.v)
        veh_yaw = np.append(tr.yaw, e.yaw)
        bk = self.bikes
        if bk is not None and bk.n:
            bk.step(dt, self.t, tr, pd, e.xy, e.v, e.yaw, (exy, er), ego_info, self._ego_turning(),
                    ego_left=self._ego_left_signal(), ego_dims=(e.LENGTH, e.WIDTH))
            mv = bk.v > 0.8                 # the people on foot watch out for the moving ones as for cars
            veh_xy = np.vstack([veh_xy, bk.xy[mv]])
            veh_v = np.append(veh_v, bk.v[mv])
            veh_yaw = np.append(veh_yaw, bk.yaw[mv])
        if self.darts and self.ego_s is not None:      # 飛び出し: a hidden child runs out ahead of the ego
            front = self.ego_s + e.LENGTH / 2
            for dt_ in self.darts:
                if not dt_[3] and e.v > 1.0 and 0.0 < dt_[1] - front <= dt_[2]:
                    pd.start_dart(dt_[0])
                    dt_[3] = True
        trigger = None
        if pd.n:
            d = pd.xy - e.xy
            u = np.array([math.cos(e.yaw), math.sin(e.yaw)])
            along = d @ u
            lat = np.abs(d[:, 0] * u[1] - d[:, 1] * u[0])
            near = (along > 20.0) & (along < 45.0) & (lat < 11.0)
            trigger = near & (self.rng.random(pd.n) < self.cfg.traffic.ego_cross_rate * dt)
        pd.step(dt, veh_xy, veh_v, veh_yaw, trigger, t=self.t)
        self.t += dt

    def _ego_left_signal(self):
        """The ego signals a left turn: its front within SIGNAL_D of a left turn of its route, or on it (the riders do
        not come up alongside it on its left: Bicycles.step)."""
        r, s = self.route, self.ego_s
        if r is None or s is None:
            return False
        k = r.lane_index_at(s)
        lanes = self.town.lanes
        cur = lanes[r.lanes[k]]
        if cur.kind == "conn":
            return cur.turn == "left"
        nxt = lanes[r.lanes[k + 1]] if k + 1 < len(r.lanes) else None
        return nxt is not None and nxt.kind == "conn" and nxt.turn == "left" and \
            cur.length - r.local_s(s, k) - self.ego.LENGTH / 2 <= SIGNAL_D

    def _ego_turning(self):
        """Where the ego will be in 0.6, 1.2 and 1.8 s along its route while it turns at a junction or is within
        12 m of the turn, moving (for the bicycles beside it: Bicycles.step), else None."""
        r, s, e = self.route, self.ego_s, self.ego
        if r is None or s is None or e.v < 0.5:
            return None
        k = r.lane_index_at(s)
        lanes = self.town.lanes
        cur = lanes[r.lanes[k]]
        if cur.kind == "conn":
            turn = cur.turn != "straight"
        else:
            nxt = lanes[r.lanes[k + 1]] if k + 1 < len(r.lanes) else None
            turn = nxt is not None and nxt.kind == "conn" and nxt.turn != "straight" \
                and cur.length - r.local_s(s, k) < 12.0
        if not turn:
            return None
        return r.poly.interp(np.minimum(s + e.v * np.array([0.6, 1.2, 1.8]), r.poly.length))

    def _ego_lc_wait(self):
        """(target lane, s along it, length) while the ego waits at the end of a lane change window (route.py): the
        NPCs coming up behind it in the target lane leave it a gap (Traffic._lane_change_gaps)."""
        r, s, e = self.route, self.ego_s, self.ego
        if r is None or s is None or not r.changes or e.v >= 1.0:
            return None
        for ch in r.changes:
            if s < ch["s_a"] - 0.3:
                if ch["a0"] >= ch["a_max"] - 1e-6 and ch["s_a"] - s < 4.0:
                    A, B = self.town.lanes[ch["from"]], self.town.lanes[ch["to"]]
                    return ch["to"], link_map(A, B, r.local_s(s, ch["k"] - 1)), e.LENGTH
                return None
        return None

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
        bk = self.bikes
        if bk is not None and bk.n:
            d = np.hypot(*(bk.xy - e.xy).T)
            idx = np.nonzero(d < 5.0)[0]
            if len(idx):
                bc = box_corners(bk.xy[idx, 0], bk.xy[idx, 1], bk.yaw[idx], np.full(len(idx), BIKE_LEN),
                                 np.full(len(idx), BIKE_W))
                for i in idx[obb_overlap(ec, bc)]:
                    hits.append(("bicycle", int(i)))
        if self.parked is not None and self.parked.n:
            hits += [("parked", j) for j in self.parked.collisions(ec, e.xy, obb_overlap)]
        sb = self.town.static_boxes
        if len(sb):
            r = 0.5 * np.hypot(sb[:, 3], sb[:, 4])
            d = np.hypot(sb[:, 0] - e.x, sb[:, 1] - e.y)
            idx = np.nonzero(d < r + 3.0)[0]
            if len(idx):
                for i in idx[obb_overlap(ec, self.town.static_corners[idx])]:
                    hits.append(("static", int(i)))
        return hits
