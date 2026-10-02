"""CPU camera renderer: exact ground homography + painter's algorithm objects.

* The ground (asphalt, paint, sidewalks, grass) is a single top-down texture
  projected with the exact plane-induced homography, in horizontal bands that
  pick an appropriate mip level (cheap anti-aliasing).
* Objects are low-poly convex parts drawn far-to-near with cv2.fillPoly
  (back-face culling, near-plane clipping, Lambert shading, distance fog).
* The semantic label image is produced by the very same geometry, so labels are
  pixel-exact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import SEM, CameraConfig
from .maptex import MapTexture
from .meshes import (CAR_T, LAMP_COLORS_ON, LAMP_OFF, PED_T, R_BODY, R_DARK, R_GLASS, R_HAIR, R_HEAD,
                     R_PANTS, R_ROOF, R_SHIRT, R_SKIN, R_TAIL, TRUCK_T, StaticScene)


# ----------------------------------------------------------------------------
# Weather / lighting
# ----------------------------------------------------------------------------
@dataclass
class Weather:
    name: str
    sun_dir: np.ndarray        # unit vector towards the sun (world)
    ambient: float
    diffuse: float
    light: np.ndarray          # BGR multiplier
    sky_top: np.ndarray
    sky_horizon: np.ndarray
    fog_color: np.ndarray
    fog_dist: float
    exposure: float = 1.0


PRESETS = ("noon", "cloudy", "sunset", "fog")


def make_weather(name="random", rng=None) -> Weather:
    rng = rng if rng is not None else np.random.default_rng()
    if name == "random":
        name = str(rng.choice(["noon", "noon", "cloudy", "sunset", "fog"], p=[0.3, 0.2, 0.25, 0.15, 0.1]))
        jitter = True
    else:
        jitter = False
    az = rng.uniform(0, 2 * math.pi)
    j = (lambda lo, hi: rng.uniform(lo, hi)) if jitter else (lambda lo, hi: (lo + hi) / 2)

    def sun(elev_deg):
        e = math.radians(elev_deg)
        return np.array([math.cos(e) * math.cos(az), math.cos(e) * math.sin(az), math.sin(e)])

    if name == "noon":
        w = Weather(name, sun(j(40, 70)), j(0.5, 0.6), j(0.45, 0.6), np.array([1.0, 1.0, 1.0]),
                    np.array([225, 165, 105.0]), np.array([240, 222, 200.0]), np.array([232, 220, 205.0]), j(350, 600))
    elif name == "cloudy":
        w = Weather(name, sun(j(35, 60)), j(0.72, 0.85), j(0.12, 0.22), np.array([0.97, 0.97, 0.95]),
                    np.array([180, 176, 172.0]), np.array([214, 212, 210.0]), np.array([205, 203, 200.0]), j(180, 320))
    elif name == "sunset":
        w = Weather(name, sun(j(6, 16)), j(0.42, 0.52), j(0.5, 0.65), np.array([0.78, 0.9, 1.12]),
                    np.array([150, 95, 85.0]), np.array([120, 175, 245.0]), np.array([140, 170, 215.0]), j(260, 420))
    elif name == "fog":
        w = Weather(name, sun(j(30, 60)), j(0.75, 0.85), j(0.1, 0.2), np.array([0.96, 0.97, 0.98]),
                    np.array([200, 200, 198.0]), np.array([210, 210, 208.0]), np.array([205, 205, 203.0]), j(45, 80))
    else:
        raise ValueError(name)
    if jitter:
        w.exposure = float(rng.uniform(0.85, 1.12))
        w.light = w.light * rng.uniform(0.95, 1.05, 3)
    return w


# ----------------------------------------------------------------------------
# Scene state snapshot consumed by renderers
# ----------------------------------------------------------------------------
@dataclass
class SceneState:
    veh_xy: np.ndarray            # (N,2)
    veh_yaw: np.ndarray           # (N,)
    veh_dims: np.ndarray          # (N,3) length, width, height
    veh_color: np.ndarray         # (N,3)
    veh_kind: np.ndarray          # (N,) 0 car, 1 truck
    veh_brake: np.ndarray         # (N,) bool
    ped_xy: np.ndarray            # (M,2)
    ped_yaw: np.ndarray
    ped_height: np.ndarray
    ped_colors: np.ndarray        # (M,3,3): shirt, pants, skin
    ped_phase: np.ndarray         # walk cycle phase
    lamp_state: np.ndarray        # (n_heads,) TL state per signal head


def _gather_ranges(f0, f1):
    lens = (f1 - f0).astype(np.int64)
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, np.int64), lens
    starts = np.repeat(f0.astype(np.int64) - (np.cumsum(lens) - lens), lens)
    return starts + np.arange(total), lens


def _clip_near(poly, near):
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ina, inb = a[2] >= near, b[2] >= near
        if ina:
            out.append(a)
        if ina != inb:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.array(out) if len(out) >= 3 else None


def camera_pose(cfg, x, y, yaw, offset=None):
    """offset: (d_lateral(+left), d_yaw, d_pitch) augmentation of the rig."""
    dl, dyaw, dpitch = (0.0, 0.0, 0.0) if offset is None else offset
    # rigid displacement of the whole vehicle frame: shift left by dl, rotate by dyaw
    x, y = x - math.sin(yaw) * dl, y + math.cos(yaw) * dl
    cy_ = yaw + dyaw
    c, s = math.cos(cy_), math.sin(cy_)
    ox, oy = cfg.x, cfg.y
    C = np.array([x + c * ox - s * oy, y + s * ox + c * oy, cfg.z])
    p = math.radians(cfg.pitch_deg) + dpitch
    fwd = np.array([math.cos(cy_) * math.cos(p), math.sin(cy_) * math.cos(p), -math.sin(p)])
    right = np.array([math.sin(cy_), -math.cos(cy_), 0.0])
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])  # world -> camera rows
    return C, R


class CameraRenderer:
    NEAR = 0.12

    def __init__(self, town, tex: MapTexture, static: StaticScene, cfg: CameraConfig):
        self.town, self.tex, self.static, self.cfg = town, tex, static, cfg
        self.W, self.H = cfg.width, cfg.height
        self.ss = max(1, int(cfg.supersample))
        self.f = (self.W / 2.0) / math.tan(math.radians(cfg.fov_deg) / 2.0)
        self.half_fov = math.radians(cfg.fov_deg) / 2.0
        self.side = 1.0 if town.cfg.left_hand_traffic else -1.0
        # signal heads for lamp directivity: position and facing direction (housing +x points at the driver)
        heads = town.signal_heads
        self.head_pos = np.array([[*sh["head"], sh["z"]] for sh in heads], np.float64).reshape(-1, 3)
        self.head_nrm = np.array([[math.cos(sh["yaw"]), math.sin(sh["yaw"]), 0.0] for sh in heads], np.float64).reshape(-1, 3)

    # ------------------------------------------------------------------ pose
    def camera_pose(self, x, y, yaw, offset=None):
        return camera_pose(self.cfg, x, y, yaw, offset)

    def intrinsics(self, scale=1):
        f = self.f * scale
        return np.array([[f, 0, self.W * scale / 2.0], [0, f, self.H * scale / 2.0], [0, 0, 1.0]])

    def project_ground(self, pts_world, C, R, scale=1, z=0.0):
        """Project world points (N,2) at height z into the image. Returns (N,2), valid mask."""
        P = np.concatenate([pts_world, np.full((len(pts_world), 1), z)], 1)
        pc = (P - C) @ R.T
        ok = pc[:, 2] > self.NEAR
        K = self.intrinsics(scale)
        uv = (pc @ K.T)
        uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
        return uv, ok

    # ------------------------------------------------------------------ ground
    def _row_geometry(self, C, R, H, f):
        cy = H / 2.0
        v = np.arange(H, dtype=np.float64)
        a = (v - cy) / f
        # ray (camera) = [0, a, 1]; world = R^T [0,a,1] = a*R[1] + R[2]
        dx = a * R[1, 0] + R[2, 0]
        dy = a * R[1, 1] + R[2, 1]
        dz = a * R[1, 2] + R[2, 2]
        norm = np.sqrt(dx * dx + dy * dy + dz * dz)
        ground = dz < -1e-4
        t = np.where(ground, C[2] / np.maximum(-dz, 1e-9), np.inf)
        dist = t * np.sqrt(dx * dx + dy * dy)
        sinb = np.clip(-dz / norm, 1e-3, 1.0)
        return ground, dist, sinb

    def _ground(self, out, C, R, scale, sem, weather=None):
        H, W = out.shape[:2]
        f = self.f * scale
        K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
        ground, dist, sinb = self._row_geometry(C, R, H, f)
        Hg = K @ np.stack([R[:, 0], R[:, 1], -R @ C], axis=1)
        tex = self.tex
        far = dist > 380.0
        lat = dist / f
        gm = np.sqrt(lat * lat / sinb)
        level = np.clip(np.floor(np.log2(np.maximum(gm / tex.res, 1e-9))), 0, tex.LEVELS - 1).astype(int)
        level[~ground | far] = -1
        # sky rows / far rows
        if sem:
            out[~ground] = SEM["sky"]
            out[ground & far] = SEM["terrain"]
        else:
            rows = np.nonzero(~ground)[0]
            if len(rows):
                hz = rows.max() + 1
                tt = np.clip(rows / max(hz, 1), 0, 1) ** 0.7
                col = weather.sky_top[None] * (1 - tt[:, None]) + weather.sky_horizon[None] * tt[:, None]
                out[rows] = np.clip(col * weather.exposure, 0, 255).astype(np.uint8)[:, None, :]
            out[ground & far] = np.clip(weather.fog_color * weather.exposure, 0, 255).astype(np.uint8)
        # bands of equal mip level
        r = 0
        while r < H:
            lv = level[r]
            r1 = r + 1
            while r1 < H and level[r1] == lv:
                r1 += 1
            if lv >= 0:
                T = np.array([[1, 0, 0], [0, 1, -r], [0, 0, 1.0]])
                M = T @ Hg @ tex.level_affine(lv)
                if sem:
                    band = cv2.warpPerspective(tex.sem[lv], M, (W, r1 - r), flags=cv2.INTER_NEAREST,
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=int(SEM["terrain"]))
                else:
                    band = cv2.warpPerspective(tex.color[lv], M, (W, r1 - r), flags=cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=tex.border)
                out[r:r1] = band
            r = r1
        if not sem:
            g = np.nonzero(ground & ~far)[0]
            if len(g):
                sun_up = max(0.0, float(weather.sun_dir[2]))
                gain = (weather.ambient + weather.diffuse * sun_up) * weather.light * weather.exposure
                alpha = (1.0 - np.exp(-dist[g] / weather.fog_dist)).astype(np.float32)
                g0, g1 = int(g[0]), int(g[-1]) + 1          # ground rows are contiguous
                al = alpha[:, None, None]
                mul = (gain.astype(np.float32)[None, None, :] * (1.0 - al))
                add = (weather.fog_color * weather.exposure).astype(np.float32)[None, None, :] * al
                sub = out[g0:g1].astype(np.float32)
                sub *= mul
                sub += add
                np.clip(sub, 0, 255, out=sub)
                out[g0:g1] = sub.astype(np.uint8)
        return dist

    # ------------------------------------------------------------------ faces
    def _dynamic_faces(self, scene: SceneState, C, fwd_xy, max_dist):
        """Instantiate templates. Returns list of dicts of arrays."""
        out = []
        # vehicles
        if len(scene.veh_xy):
            d = scene.veh_xy - C[:2]
            dist = np.hypot(d[:, 0], d[:, 1])
            along = d @ fwd_xy
            vis = (dist < max_dist) & (along > -6.0)
            for kind, T in ((0, CAR_T), (1, TRUCK_T)):
                idx = np.nonzero(vis & (scene.veh_kind == kind))[0]
                if len(idx) == 0:
                    continue
                dims = scene.veh_dims[idx]
                loc = T["verts"][None].astype(np.float64) * dims[:, None, None, :]
                yaw = scene.veh_yaw[idx]
                c, s = np.cos(yaw)[:, None, None], np.sin(yaw)[:, None, None]
                wx = scene.veh_xy[idx, 0][:, None, None] + c * loc[..., 0] - s * loc[..., 1]
                wy = scene.veh_xy[idx, 1][:, None, None] + s * loc[..., 0] + c * loc[..., 1]
                verts = np.stack([wx, wy, loc[..., 2]], -1)
                n, F = len(idx), len(T["role"])
                role = T["role"]
                col = np.zeros((n, F, 3))
                body = scene.veh_color[idx]
                col[:, role == R_BODY] = body[:, None, :]
                col[:, role == R_ROOF] = body[:, None, :] * 0.92
                col[:, role == R_GLASS] = np.array([70, 55, 45.0])
                col[:, role == R_DARK] = np.array([30, 30, 32.0])
                col[:, role == R_HEAD] = np.array([200, 235, 245.0])
                brake = scene.veh_brake[idx][:, None, None]
                col[:, role == R_TAIL] = np.where(brake, np.array([60, 60, 255.0]), np.array([30, 30, 130.0]))
                emis = np.broadcast_to((role == R_TAIL) | (role == R_HEAD), (n, F))
                out.append({
                    "verts": verts.reshape(n * F, -1, 3), "nv": np.tile(T["nv"], n), "color": col.reshape(n * F, 3),
                    "sem": np.full(n * F, SEM["vehicle"], np.uint8), "order": np.tile(T["order"], n),
                    "emissive": emis.reshape(-1).copy(), "odist": np.repeat(dist[idx], F),
                })
                # blob shadows
                sh_l = dims[:, 0] * 0.52
                sh_w = dims[:, 1] * 0.55
                corners = np.stack([np.stack([sh_l, sh_w], -1), np.stack([sh_l, -sh_w], -1),
                                    np.stack([-sh_l, -sh_w], -1), np.stack([-sh_l, sh_w], -1)], 1)
                cc, sc = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
                sx = scene.veh_xy[idx, 0][:, None] + cc * corners[..., 0] - sc * corners[..., 1]
                sy = scene.veh_xy[idx, 1][:, None] + sc * corners[..., 0] + cc * corners[..., 1]
                out.append({"shadow": np.stack([sx, sy], -1)})
        # pedestrians
        if len(scene.ped_xy):
            d = scene.ped_xy - C[:2]
            dist = np.hypot(d[:, 0], d[:, 1])
            along = d @ fwd_xy
            idx = np.nonzero((dist < min(max_dist, 70.0)) & (along > -2.0))[0]
            if len(idx):
                T = PED_T
                h = scene.ped_height[idx]
                loc = T["verts"][None].astype(np.float64) * h[:, None, None, None]
                swing = np.sin(scene.ped_phase[idx]) * 0.12 * h
                leg = T["leg"]
                lx = loc[..., 0]
                lx[:, leg == 1, :] += swing[:, None, None]
                lx[:, leg == -1, :] -= swing[:, None, None]
                yaw = scene.ped_yaw[idx]
                c, s = np.cos(yaw)[:, None, None], np.sin(yaw)[:, None, None]
                wx = scene.ped_xy[idx, 0][:, None, None] + c * loc[..., 0] - s * loc[..., 1]
                wy = scene.ped_xy[idx, 1][:, None, None] + s * loc[..., 0] + c * loc[..., 1]
                verts = np.stack([wx, wy, loc[..., 2]], -1)
                n, F = len(idx), len(T["role"])
                role = T["role"]
                col = np.zeros((n, F, 3))
                pc = scene.ped_colors[idx]
                col[:, role == R_SHIRT] = pc[:, None, 0]
                col[:, role == R_PANTS] = pc[:, None, 1]
                col[:, role == R_SKIN] = pc[:, None, 2]
                out.append({
                    "verts": verts.reshape(n * F, -1, 3), "nv": np.tile(T["nv"], n), "color": col.reshape(n * F, 3),
                    "sem": np.full(n * F, SEM["pedestrian"], np.uint8), "order": np.tile(T["order"], n),
                    "emissive": np.zeros(n * F, bool), "odist": np.repeat(dist[idx], F),
                })
                r = 0.32 * h
                ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
                sx = scene.ped_xy[idx, 0][:, None] + r[:, None] * np.cos(ang)[None]
                sy = scene.ped_xy[idx, 1][:, None] + r[:, None] * np.sin(ang)[None]
                out.append({"shadow": np.stack([sx, sy], -1)})
        return out

    def render(self, pose, scene: SceneState, weather: Weather, want_rgb=True, want_seg=True, offset=None):
        x, y, yaw = pose
        C, R = self.camera_pose(x, y, yaw, offset)
        ss = self.ss
        W, H = self.W, self.H
        Ws, Hs = W * ss, H * ss
        cfg = self.cfg
        rgb = np.empty((Hs, Ws, 3), np.uint8) if want_rgb else None
        seg = np.empty((H, W), np.uint8) if want_seg else None
        if want_rgb:
            self._ground(rgb, C, R, ss, False, weather)
        if want_seg:
            self._ground(seg, C, R, 1, True)

        fwd_xy = R[2, :2] / (np.linalg.norm(R[2, :2]) + 1e-9)
        max_dist = cfg.max_dist
        # ---- static objects
        st = self.static
        so = st.objs
        d = so[:, :2] - C[:2]
        dist = np.hypot(d[:, 0], d[:, 1])
        along = d @ fwd_xy
        lateral = np.abs(d[:, 0] * fwd_xy[1] - d[:, 1] * fwd_xy[0])
        tanh = math.tan(min(self.half_fov + 0.15, 1.5))
        vis = (dist < max_dist + so[:, 2]) & (along > -so[:, 2]) & (lateral < np.maximum(along, 0) * tanh + so[:, 2])
        vi = np.nonzero(vis)[0]
        idx, lens = _gather_ranges(so[vi, 3], so[vi, 4])
        sf = st.faces
        odist_s = np.repeat(dist[vi], lens)
        keep = ~(sf["lod"][idx] & (odist_s > 45.0))
        idx, odist_s = idx[keep], odist_s[keep]
        verts = [sf["verts"][idx].astype(np.float64)]
        nv = [sf["nv"][idx]]
        color = [sf["color"][idx].astype(np.float64)]
        sem = [sf["sem"][idx].copy()]
        order = [sf["order"][idx]]
        emis = [sf["emissive"][idx]]
        odist = [odist_s]
        lamp = sf["lamp"][idx]
        lm = lamp >= 0
        if lm.any():
            head = lamp[lm] // 3
            slot = lamp[lm] % 3
            state = scene.lamp_state[head]
            lit = ((state == 2) & (slot == 0)) | ((state == 1) & (slot == 1)) | ((state == 0) & (slot == 2))
            # directivity: full brightness within 30 deg of the head's axis, unlit look beyond 60 deg; labelled
            # lit only within 45 deg (the same rule as KeiView's web/src/world/signals.js)
            to_cam = C[None] - self.head_pos[head]
            cosv = (to_cam * self.head_nrm[head]).sum(1) / np.maximum(np.linalg.norm(to_cam, axis=1), 1e-6)
            f = np.clip((cosv - math.cos(math.pi / 3)) / (math.cos(math.pi / 6) - math.cos(math.pi / 3)), 0.0, 1.0)
            on = np.stack([LAMP_COLORS_ON[int(s)] for s in state]) if len(state) else LAMP_OFF[None]
            cl = np.where(lit[:, None], LAMP_OFF[None] + f[:, None] * (on - LAMP_OFF[None]), LAMP_OFF[None])
            color[0][lm] = cl
            sm = np.where(lit & (cosv >= math.cos(math.pi / 4)),
                          np.array([SEM["tl_red"], SEM["tl_yellow"], SEM["tl_green"]])[np.clip(state, 0, 2)], SEM["pole"])
            sem[0][lm] = sm.astype(np.uint8)
        shadows = []
        for dyn in self._dynamic_faces(scene, C, fwd_xy, max_dist):
            if "shadow" in dyn:
                shadows.append(dyn["shadow"])
                continue
            verts.append(dyn["verts"])
            nv.append(dyn["nv"])
            color.append(dyn["color"])
            sem.append(dyn["sem"])
            order.append(dyn["order"])
            emis.append(dyn["emissive"])
            odist.append(dyn["odist"])
        verts = np.concatenate(verts)
        nv = np.concatenate(nv)
        color = np.concatenate(color)
        sem = np.concatenate(sem)
        order = np.concatenate(order)
        emis = np.concatenate(emis)
        odist = np.concatenate(odist)

        # ---- back-face culling (world space)
        v0 = verts[:, 0]
        n = np.cross(verts[:, 1] - v0, verts[:, 2] - v0)
        facing = np.einsum("ij,ij->i", n, C[None] - v0) > 1e-9
        verts, nv, color, sem, order, emis, odist, n = (a[facing] for a in (verts, nv, color, sem, order, emis, odist, n))
        # ---- to camera
        pc = (verts - C) @ R.T
        z = pc[..., 2]
        behind = (z < self.NEAR).all(1)
        keep = ~behind
        verts, nv, color, sem, order, emis, odist, n, pc, z = (
            a[keep] for a in (verts, nv, color, sem, order, emis, odist, n, pc, z))
        clip = (z < self.NEAR).any(1)
        # ---- shading + fog
        nn = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        lam = np.clip(nn @ weather.sun_dir, 0, None)
        gain = (weather.ambient + weather.diffuse * lam)[:, None] * weather.light[None] * weather.exposure
        shaded = np.where(emis[:, None], color * weather.exposure, color * gain)
        fc = verts.mean(1)
        fd = np.linalg.norm(fc - C, axis=1)
        alpha = (1.0 - np.exp(-fd / weather.fog_dist))[:, None]
        alpha = np.where(emis[:, None], alpha * 0.6, alpha)
        shaded = shaded * (1 - alpha) + (weather.fog_color * weather.exposure)[None] * alpha
        shaded = np.clip(shaded, 0, 255)
        # ---- projection
        f1 = self.f
        zc = np.maximum(z, self.NEAR)
        u = f1 * pc[..., 0] / zc + W / 2.0
        v = f1 * pc[..., 1] / zc + H / 2.0
        # screen culling for unclipped faces
        offscreen = (~clip) & ((u.max(1) < 0) | (u.min(1) > W) | (v.max(1) < 0) | (v.min(1) > H))
        # ---- painter's order: far objects first, then part order
        draw = np.nonzero(~offscreen)[0]
        o = np.lexsort((order[draw], -odist[draw]))
        draw = draw[o]

        sh_color = None
        if want_rgb:
            sh_color = tuple(float(c) for c in np.clip(self.tex.palette["asphalt"] * 0.45 * weather.exposure, 0, 255))
        # shadows first (they lie on the ground)
        for sh in shadows:
            for poly in sh:
                P = np.concatenate([poly, np.full((len(poly), 1), 0.02)], 1)
                q = (P - C) @ R.T
                if (q[:, 2] < self.NEAR).any():
                    q = _clip_near(q, self.NEAR)
                    if q is None:
                        continue
                uu = f1 * q[:, 0] / q[:, 2] + W / 2.0
                vv = f1 * q[:, 1] / q[:, 2] + H / 2.0
                if want_rgb:
                    pts = np.round(np.stack([uu, vv], -1) * ss * 8).astype(np.int32)
                    cv2.fillPoly(rgb, [pts], sh_color, lineType=cv2.LINE_AA, shift=3)

        # vectorised fixed-point coordinates (cv2 shift=3 -> 1/8 px precision)
        UV = np.stack([u, v], -1)
        np.clip(UV, -4000.0, 4000.0, out=UV)
        P_rgb = np.round(UV * (ss * 8)).astype(np.int32) if want_rgb else None
        P_seg = np.round(UV * 8).astype(np.int32) if want_seg else None
        cols = shaded.tolist()
        sems = sem.tolist()
        nvl = nv.tolist()
        clipl = clip.tolist()
        for i in draw.tolist():
            k = nvl[i]
            if clipl[i]:
                q = _clip_near(pc[i, :k], self.NEAR)
                if q is None:
                    continue
                uu = f1 * q[:, 0] / q[:, 2] + W / 2.0
                vv = f1 * q[:, 1] / q[:, 2] + H / 2.0
                Pq = np.clip(np.stack([uu, vv], -1), -4000.0, 4000.0)
                pr = np.round(Pq * (ss * 8)).astype(np.int32)
                ps = np.round(Pq * 8).astype(np.int32)
            else:
                pr = P_rgb[i, :k] if want_rgb else None
                ps = P_seg[i, :k] if want_seg else None
            if want_rgb:
                cv2.fillPoly(rgb, [pr], cols[i], lineType=cv2.LINE_8, shift=3)
            if want_seg:
                cv2.fillPoly(seg, [ps], sems[i], lineType=cv2.LINE_8, shift=3)
        if want_rgb and ss > 1:
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        return rgb, seg
