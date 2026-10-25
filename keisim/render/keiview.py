"""KeiView camera: the KeiSim ego camera rendered by KeiView (web/, three.js with anime cel
shading) in headless Chrome on the GPU, driven through web/tools/ego_server.mjs. Without a GPU,
gl="auto" (default) falls back to SwiftShader on the CPU (~2 s/frame).

Same contract as CameraRenderer.render(): returns (bgr uint8 HxWx3, semantic class ids HxW or
None) for the KeiSim camera configuration, so the expert labels, the model and every script work
unchanged. Towns are exported on demand to web/.towns/ with scripts/export_town.py.

Camera rigs (keisim/config.py CAMERA_RIGS): render_rig() draws every camera of the rig in one request. A cylindrical
camera is drawn as the pinhole faces of KeiSim's CylinderRenderer and stitched with the same maps (keisim/render/rig.py).
"""
from __future__ import annotations

import atexit
import base64
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import replace

import cv2
import numpy as np

from ..config import SEM, town_key
from .camera import camera_pose
from .rig import CylinderRenderer, cylinder_faces

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB_DIR = os.path.join(ROOT, "web")
TOWN_DIR = ".towns"

_EXPORT = None


def export_town(seed: int, town_cfg=None, out_dir: str | None = None) -> str:
    """Write web/.towns/town_<key>.json (atomic, safe with parallel workers); key = town_key(seed, town_cfg)."""
    global _EXPORT
    out_dir = out_dir or os.path.join(WEB_DIR, TOWN_DIR)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"town_{town_key(seed, town_cfg)}.json")
    if os.path.exists(path):
        return path
    if _EXPORT is None:
        spec = importlib.util.spec_from_file_location("keiview_export_town", os.path.join(ROOT, "scripts", "export_town.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXPORT = mod.export
    data = _EXPORT(int(seed), town_cfg)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, path)
    return path


def _vfov(cam_cfg):
    return math.degrees(2.0 * math.atan(math.tan(math.radians(cam_cfg.fov_deg) / 2.0) * cam_cfg.height / cam_cfg.width))


class KeiViewRenderer:
    def __init__(self, cam_cfg, quality="medium", gl="auto", node="node", max_dist=160.0, log_path=None, roads="plain",
                 cameras=None):
        """cameras: every camera of the rig (EnvConfig.cameras, cam_cfg first), for render_rig()."""
        self.cfg = cam_cfg
        self.cameras = tuple(cameras) if cameras else (cam_cfg,)
        # what each camera draws: its pinhole shots, and the maps that stitch them (None: one plain image)
        self._shots = []
        for c in self.cameras:
            if c.projection == "cylindrical":
                faces, map_x, map_y, _ = cylinder_faces(c, CylinderRenderer.FACES, CylinderRenderer.OVERSAMPLE)
                # KeiView's pixel centres are at half-integer coordinates, KeiSim's at integers
                self._shots.append((faces, (map_x - 0.5, map_y - 0.5)))
            elif c.projection == "pinhole":
                self._shots.append(([c], None))
            else:
                raise ValueError(f"unknown projection {c.projection!r}")
        self.rig = len(self.cameras) > 1 or self._shots[0][1] is not None
        # the canvas: every shot fits in it (the front camera alone: its own size)
        self.W = max(s.width for shots, _ in self._shots for s in shots)
        self.H = max(s.height for shots, _ in self._shots for s in shots)
        self.vfov = _vfov(cam_cfg)
        self.max_dist = max_dist
        self.town = None
        self._id = 0
        self._new_episode = True
        self._light = None
        self.last_ms = 0.0
        self.town_ms = 0.0
        log_path = log_path or os.environ.get("KEIVIEW_LOG")      # ego_server's stderr (timings per town / frame)
        self._log = open(log_path, "ab") if log_path else subprocess.DEVNULL
        cmd = [node, "tools/ego_server.mjs", "--w", str(self.W), "--h", str(self.H), "--q", quality, "--gl", gl,
               "--tdir", TOWN_DIR]
        if os.environ.get("KEIVIEW_SIGSCALE"):         # experiments: signal-head scale in ego mode (default 1.6)
            cmd += ["--sigscale", os.environ["KEIVIEW_SIGSCALE"]]
        if roads not in ("plain", "detail"):
            raise ValueError(f"roads must be plain or detail (got {roads!r})")
        if roads == "detail":                           # road surface detailing (web/src/world/roaddeco.js)
            cmd += ["--roads", "detail"]
        self.proc = subprocess.Popen(cmd, cwd=WEB_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log)
        hello = self._read()
        if not hello.get("ok"):
            raise RuntimeError(f"KeiView server failed: {hello}")
        self.gl, self.gpu = hello.get("gl", gl), hello.get("gpu")       # mode actually used ("auto" resolves to hw/soft)
        if self.gl == "soft":
            print(f"KeiView: software rendering (SwiftShader, no GPU found), about 2 s/frame: {self.gpu}", file=sys.stderr)
        atexit.register(self.close)

    # ------------------------------------------------------------------ protocol
    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("KeiView server exited (see its log; is `npm install` done in web/?)")
        return json.loads(line)

    def _call(self, req):
        self._id += 1
        req["id"] = self._id
        self.proc.stdin.write((json.dumps(req, separators=(",", ":")) + "\n").encode())
        self.proc.stdin.flush()
        r = self._read()
        if not r.get("ok"):
            raise RuntimeError(f"KeiView {req.get('op')}: {r.get('error')}")
        return r

    # ------------------------------------------------------------------ API
    def load_town(self, seed, town_cfg=None):
        """seed: a procedural town's seed (exported on demand), or a Town object (keisim/realworld.py): its name is the
        key, and web/.towns/town_<name>.json is written if it is not there yet (scripts/import_osm.py writes it too)."""
        from ..roadnet import Town
        if isinstance(seed, Town):
            key = seed.name
            path = os.path.join(WEB_DIR, TOWN_DIR, f"town_{key}.json")
            if not os.path.exists(path):
                spec = importlib.util.spec_from_file_location("keiview_export_town",
                                                              os.path.join(ROOT, "scripts", "export_town.py"))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                tmp = f"{path}.{os.getpid()}.tmp"
                with open(tmp, "w") as f:
                    json.dump(mod.export_town(seed), f, separators=(",", ":"))
                os.replace(tmp, path)
        else:
            key = town_key(seed, town_cfg)
        if self.town == key:
            return {"cached": True}
        if not isinstance(seed, Town):
            export_town(int(seed), town_cfg)
        r = self._call({"op": "town", "town": key})
        if r.get("errors"):
            raise RuntimeError(f"KeiView town {key}: {r['errors']}")
        self.town = key
        self.town_ms = float(r.get("ms", 0.0))
        self._new_episode = True
        return r

    def new_episode(self, rng=None):
        """Next render starts a new episode (actor pools reset). With rng, also draws a random
        lighting (sun direction, exposure, haze) for the episode; without, KeiView's default look."""
        self._new_episode = True
        self._light = None
        if rng is not None:
            self._light = {"az": float(rng.uniform(0, 360)), "el": float(rng.uniform(22, 62)),
                           "exposure": float(rng.uniform(0.88, 1.12)), "fog": float(rng.uniform(0.0014, 0.0055))}

    def _actors(self, C, scene):
        veh, ped = [], []
        r2 = self.max_dist ** 2
        if len(scene.veh_xy):
            d2 = ((scene.veh_xy - C[:2]) ** 2).sum(1)
            for i in np.nonzero(d2 < r2)[0]:
                x, y = scene.veh_xy[i]
                L, W, H = scene.veh_dims[i]
                b, g, r = scene.veh_color[i]
                veh.append([int(i), round(float(x), 3), round(float(y), 3), round(float(scene.veh_yaw[i]), 4),
                            round(float(L), 2), round(float(W), 2), round(float(H), 2), int(r), int(g), int(b),
                            int(scene.veh_kind[i]), int(bool(scene.veh_brake[i]))])
        if len(scene.ped_xy):
            d2 = ((scene.ped_xy - C[:2]) ** 2).sum(1)
            for i in np.nonzero(d2 < r2)[0]:
                x, y = scene.ped_xy[i]
                cols = scene.ped_colors[i][:, ::-1]           # (shirt, pants, skin) BGR -> RGB
                ped.append([int(i), round(float(x), 3), round(float(y), 3), round(float(scene.ped_yaw[i]), 4),
                            round(float(scene.ped_height[i]), 2)] + [int(v) for v in cols.reshape(-1)]
                           + [round(float(scene.ped_phase[i]), 3)])
        return veh, ped

    @staticmethod
    def _decode(v):
        """(bgr, class ids or None) of one image of a reply, top row first."""
        W, H = v["w"], v["h"]
        rgb = np.frombuffer(base64.b64decode(v["rgb"]), np.uint8).reshape(H, W, 3)[::-1, :, ::-1]
        seg = None
        if v.get("sem"):
            seg = np.ascontiguousarray(np.frombuffer(base64.b64decode(v["sem"]), np.uint8).reshape(H, W)[::-1])
        return np.ascontiguousarray(rgb), seg

    def render(self, pose, scene, t, want_rgb=True, want_seg=True, offset=None):
        """(bgr, class ids or None) of the first camera."""
        if self.rig:
            return self.render_rig(pose, scene, t, want_rgb, want_seg, offset, which=[0])[0]
        C, R = camera_pose(self.cfg, *pose, offset)
        veh, ped = self._actors(C, scene)
        req = {"op": "render", "cam": {"pos": [float(v) for v in C], "fwd": [float(v) for v in R[2]], "vfov": self.vfov},
               "t": float(t), "veh": veh, "ped": ped, "labels": bool(want_seg), "reset": self._new_episode}
        if self._new_episode:
            req["light"] = self._light
        self._new_episode = False
        r = self._call(req)
        self.last_ms = r.get("ms", 0.0)
        return self._decode(r)

    def render_rig(self, pose, scene, t, want_rgb=True, want_seg=True, offset=None, which=None):
        """[(bgr, class ids or None)] of every camera of the rig (or of the cameras `which`), in one request: the same
        scene state, sky and shadows for all."""
        which = range(len(self.cameras)) if which is None else which
        # the level rig places the sky dome and the sun's shadow box, and finds the actors
        C, R = camera_pose(replace(self.cameras[0], yaw_deg=0.0, pitch_deg=0.0), *pose, offset)
        veh, ped = self._actors(C, scene)
        cams, spans = [], []
        for ci in which:
            shots, maps = self._shots[ci]
            spans.append((ci, len(cams), len(shots)))
            for s in shots:
                Cs, Rs = camera_pose(s, *pose, offset)
                cams.append({"pos": [float(v) for v in Cs], "fwd": [float(v) for v in Rs[2]], "vfov": _vfov(s),
                             "w": s.width, "h": s.height, "face": maps is not None})
        req = {"op": "render", "cam": {"pos": [float(v) for v in C], "fwd": [float(v) for v in R[2]], "vfov": 60.0},
               "cams": cams, "t": float(t), "veh": veh, "ped": ped, "labels": bool(want_seg), "reset": self._new_episode}
        if self._new_episode:
            req["light"] = self._light
        self._new_episode = False
        r = self._call(req)
        self.last_ms = r.get("ms", 0.0)
        imgs = [self._decode(v) for v in r["views"]]
        out = []
        for ci, i0, n in spans:
            maps = self._shots[ci][1]
            rgbs, segs = [v[0] for v in imgs[i0:i0 + n]], [v[1] for v in imgs[i0:i0 + n]]
            if maps is None:
                out.append((rgbs[0], segs[0]))
                continue
            rgb = cv2.remap(np.concatenate(rgbs, 1), *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            seg = None
            if segs[0] is not None:
                seg = cv2.remap(np.concatenate(segs, 1), *maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=int(SEM["sky"]))
            out.append((rgb, seg))
        return out

    def close(self):
        if getattr(self, "proc", None) is None or self.proc.poll() is not None:
            return
        try:
            self._call({"op": "quit"})
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()
