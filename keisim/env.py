"""Gym-style closed-loop driving environment."""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

from .config import EnvConfig, TL_RED
from .control import PlanFollower
from .expert import Expert
from .geometry import world_to_local
from .render.bev import BEVRenderer
from .render.camera import CameraRenderer, make_weather
from .render.maptex import MapTexture
from .render.meshes import StaticScene
from .roadnet import Town
from .route import random_route
from .world import World

PENALTY = {"vehicle": 0.60, "pedestrian": 0.50, "static": 0.65, "red_light": 0.70, "off_road": 0.80}


class TownAssets:
    def __init__(self, seed, cfg: EnvConfig):
        self.seed = seed
        self.cfg = cfg
        self.town = Town(seed, cfg.town)
        self._tex = self._static = self._cam = self._bev = None

    @property
    def tex(self):
        if self._tex is None:
            self._tex = MapTexture(self.town)
        return self._tex

    @property
    def camera(self):
        if self._cam is None:
            self._static = StaticScene(self.town)
            self._cam = CameraRenderer(self.town, self.tex, self._static, self.cfg.camera)
        return self._cam

    @property
    def bev(self):
        if self._bev is None:
            self._bev = BEVRenderer(self.tex)
        return self._bev


_CACHE: "OrderedDict[tuple, TownAssets]" = OrderedDict()


def get_assets(seed, cfg: EnvConfig, max_cache=3) -> TownAssets:
    key = (int(seed), id(cfg.town), id(cfg.camera))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    a = TownAssets(seed, cfg)
    _CACHE[key] = a
    while len(_CACHE) > max_cache:
        _CACHE.popitem(last=False)
    return a


class KeiEnv:
    """Closed-loop environment.

    obs = {rgb, seg, speed, command, target_point, expert{path,target_speed,tl_state}, ...}
    action = [steer(-1..1), throttle(0..1), brake(0..1)]
    """

    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        self.follower = PlanFollower()
        self.expert = Expert(self)
        self.world = None
        self._keiview = None

    def keiview(self):
        """Lazily started KeiView renderer (one headless Chrome per env)."""
        if self._keiview is None:
            from .render.keiview import KeiViewRenderer
            self._keiview = KeiViewRenderer(self.cfg.camera, quality=self.cfg.keiview_quality, gl=self.cfg.keiview_gl)
        return self._keiview

    def close(self):
        if self._keiview is not None:
            self._keiview.close()
            self._keiview = None

    # ------------------------------------------------------------------ reset
    def reset(self, town_seed=0, episode_seed=None, route_length=None, weather=None, render=True):
        cfg = self.cfg
        self.assets = get_assets(town_seed, cfg)
        town = self.assets.town
        self.town_seed, self.episode_seed = town_seed, episode_seed
        rng = np.random.default_rng(episode_seed)
        self.rng = rng
        L = route_length if route_length is not None else rng.uniform(*cfg.route_length)
        self.route = random_route(town, rng, L)
        p = self.route.poly.interp(self.route.s_start)
        h = float(self.route.poly.heading(self.route.s_start))
        self.world = World(town, cfg, rng, (p[0], p[1], h), self.route)
        self.weather = make_weather(weather or cfg.weather, rng)
        self.follower.reset()
        self.s_ego = self.route.s_start
        self.lat = 0.0
        self.progress = self.route.s_start
        self.last_progress = self.route.s_start
        self.last_progress_t = self.world.t
        self.t0 = self.world.t
        self.infractions = []
        self.distance = 0.0
        self.off_lane_dist = 0.0
        self._off_road = False
        self._hit_ids = set()
        self.done = False
        self.status = "running"
        self._render = render
        self.plan = self.expert.plan()
        if self.cfg.renderer == "keiview":
            kv = self.keiview()
            kv.load_town(town_seed)
            # lighting from its own stream so the simulation is identical to the KeiSim-rendered episode
            light_rng = np.random.default_rng([0 if episode_seed is None else int(episode_seed), 7707])
            kv.new_episode(light_rng if self.cfg.weather == "random" else None)
        return self._obs()

    # ----------------------------------------------------------------- helpers
    def _track(self):
        e = self.world.ego
        s, lat = self.route.poly.project(e.xy, self.s_ego - 4.0, self.s_ego + 25.0)
        self.s_ego, self.lat = s, lat

    @property
    def elapsed(self):
        return self.world.t - self.t0

    def command(self):
        return self.route.command(self.s_ego)

    def target_point(self, pose=None):
        pose = pose or self.world.ego.pose
        return world_to_local(self.route.target_point_world(self.s_ego), *pose)

    def render_camera(self, offset=None, want_rgb=True, want_seg=True):
        if self.cfg.renderer == "keiview":
            return self.keiview().render(self.world.ego.pose, self.world.scene_state(), self.world.t,
                                         want_rgb=want_rgb, want_seg=want_seg, offset=offset)
        cam = self.assets.camera
        return cam.render(self.world.ego.pose, self.world.scene_state(), self.weather, want_rgb=want_rgb,
                          want_seg=want_seg, offset=offset)

    def virtual_pose(self, offset):
        """Pose of a vehicle whose camera rig is displaced by `offset` (lateral, yaw)."""
        x, y, yaw = self.world.ego.pose
        dl, dyaw = offset[0], offset[1]
        return (x - math.sin(yaw) * dl, y + math.cos(yaw) * dl, yaw + dyaw)

    def render_bev(self, paths=()):
        return self.assets.bev.render(self.world, route=self.route, paths=paths)

    def _obs(self):
        e = self.world.ego
        obs = {
            "speed": e.v,
            "command": self.command(),
            "target_point": self.target_point(),
            "expert": self.plan,
            "pose": e.pose,
            "t": self.elapsed,
        }
        if self._render and (self.cfg.render_rgb or self.cfg.render_seg):
            rgb, seg = self.render_camera(want_rgb=self.cfg.render_rgb, want_seg=self.cfg.render_seg)
            obs["rgb"], obs["seg"] = rgb, seg
        return obs

    def expert_action(self):
        return self.expert.act(self.plan)

    def vector_obs(self, radius=50.0, max_agents=32, route_pts=20):
        """Object-level observation in the ego frame (for PlanT-style planners).

        vehicles: (N,7) x, y, yaw, speed, length, width, is_braking
        pedestrians: (M,5) x, y, yaw, speed, crossing
        route: (route_pts,2) route centre line every 2 m ahead
        light: state of the next relevant traffic light (0 red,1 yellow,2 green,3 none)
        """
        e = self.world.ego
        pose = e.pose
        tr, pd = self.world.traffic, self.world.peds
        out = {}
        if tr.n:
            loc = world_to_local(tr.xy, *pose)
            d = np.hypot(loc[:, 0], loc[:, 1])
            idx = np.argsort(d)[:max_agents]
            idx = idx[d[idx] < radius]
            out["vehicles"] = np.column_stack([loc[idx], np.arctan2(np.sin(tr.yaw[idx] - e.yaw), np.cos(tr.yaw[idx] - e.yaw)),
                                               tr.v[idx], tr.dims[idx, 0], tr.dims[idx, 1],
                                               (tr.acc[idx] < -0.4).astype(float)])
        else:
            out["vehicles"] = np.zeros((0, 7))
        if pd.n:
            loc = world_to_local(pd.xy, *pose)
            d = np.hypot(loc[:, 0], loc[:, 1])
            idx = np.argsort(d)[:max_agents]
            idx = idx[d[idx] < radius]
            out["pedestrians"] = np.column_stack([loc[idx], np.arctan2(np.sin(pd.yaw[idx] - e.yaw), np.cos(pd.yaw[idx] - e.yaw)),
                                                  pd.vel[idx], (pd.mode[idx] == pd.CROSS).astype(float)])
        else:
            out["pedestrians"] = np.zeros((0, 5))
        ss = self.s_ego + 2.0 * np.arange(1, route_pts + 1)
        out["route"] = world_to_local(self.route.poly.interp(np.minimum(ss, self.route.poly.length)), *pose)
        out["light"] = self.plan["tl_state"]
        out["speed"] = e.v
        out["command"] = self.command()
        return out

    # -------------------------------------------------------------------- step
    def step(self, action):
        cfg = self.cfg
        a = np.asarray(action, float)
        e = self.world.ego
        reward = 0.0
        prev_progress = self.progress
        for _ in range(cfg.action_repeat):
            s_front_prev = self.s_ego + e.LENGTH / 2
            self.world.step(a, cfg.dt)
            self._track()
            self.distance += e.v * cfg.dt
            s_front = self.s_ego + e.LENGTH / 2
            # red lights
            for s_stop, lid in self.route.stops:
                if s_front_prev < s_stop <= s_front:
                    if self.world.town.signal_state_for_lane(lid, self.world.t) == TL_RED:
                        self._infraction("red_light", lid)
            # progress
            if self.s_ego > self.progress:
                self.progress = min(self.s_ego, self.route.s_end)
            if self.progress > self.last_progress + 0.5:
                self.last_progress = self.progress
                self.last_progress_t = self.world.t
            # lane / road keeping
            if abs(self.lat) > self.world.town.cfg.lane_width / 2 + 0.3:
                self.off_lane_dist += e.v * cfg.dt
            on_road = bool(self.world.town.is_drivable(e.xy[None])[0])
            if not on_road and not self._off_road:
                self._infraction("off_road", 0)
            self._off_road = not on_road
            # collisions
            for kind, idx in self.world.ego_collisions():
                if (kind, idx) not in self._hit_ids:
                    self._hit_ids.add((kind, idx))
                    self._infraction(kind, idx)
                    if cfg.terminate_on_collision:
                        self._finish("collision_" + kind)
            if self.done:
                break
            if abs(self.lat) > cfg.route_deviation:
                self._finish("route_deviation")
                break
            if self.progress >= self.route.s_end - 0.5:
                self._finish("success")
                break
            if self.world.t - self.last_progress_t > cfg.blocked_timeout:
                self._finish("blocked")
                break
        reward += 0.05 * (self.progress - prev_progress)
        if not self.done:
            self.plan = self.expert.plan()
        obs = self._obs() if not self.done else {"speed": e.v}
        info = self.metrics()
        return obs, reward, self.done, info

    def _infraction(self, kind, idx):
        self.infractions.append({"type": kind, "t": round(self.elapsed, 2), "id": int(idx),
                                 "s": round(self.progress - self.route.s_start, 1)})

    def _finish(self, status):
        if not self.done:
            self.done = True
            self.status = status

    def metrics(self):
        rc = float(np.clip((self.progress - self.route.s_start) / max(self.route.length, 1e-6), 0, 1))
        if self.status == "success":
            rc = 1.0
        pen = 1.0
        for inf in self.infractions:
            pen *= PENALTY.get(inf["type"], 1.0)
        return {
            "status": self.status, "RC": rc, "IS": pen, "DS": rc * pen, "infractions": list(self.infractions),
            "time": round(self.elapsed, 2), "distance": round(self.distance, 1),
            "route_length": round(self.route.length, 1),
            "off_lane_ratio": self.off_lane_dist / max(self.distance, 1e-6),
        }
