"""KeiView camera: the KeiSim ego camera rendered by KeiView (web/, three.js with anime cel
shading) in headless Chrome on the GPU, driven through web/tools/ego_server.mjs.

Same contract as CameraRenderer.render(): returns (bgr uint8 HxWx3, semantic class ids HxW or
None) for the KeiSim camera configuration, so the expert labels, the model and every script work
unchanged. Towns are exported on demand to web/.towns/ with scripts/export_town.py.
"""
from __future__ import annotations

import atexit
import base64
import importlib.util
import json
import math
import os
import subprocess

import numpy as np

from .camera import camera_pose

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB_DIR = os.path.join(ROOT, "web")
TOWN_DIR = ".towns"

_EXPORT = None


def export_town(seed: int, out_dir: str | None = None) -> str:
    """Write web/.towns/town_<seed>.json (atomic, safe with parallel workers)."""
    global _EXPORT
    out_dir = out_dir or os.path.join(WEB_DIR, TOWN_DIR)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"town_{seed}.json")
    if os.path.exists(path):
        return path
    if _EXPORT is None:
        spec = importlib.util.spec_from_file_location("keiview_export_town", os.path.join(ROOT, "scripts", "export_town.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXPORT = mod.export
    data = _EXPORT(int(seed))
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, path)
    return path


class KeiViewRenderer:
    def __init__(self, cam_cfg, quality="medium", gl="hw", node="node", max_dist=160.0, log_path=None):
        self.cfg = cam_cfg
        self.W, self.H = cam_cfg.width, cam_cfg.height
        hfov = math.radians(cam_cfg.fov_deg)
        self.vfov = math.degrees(2.0 * math.atan(math.tan(hfov / 2.0) * self.H / self.W))
        self.max_dist = max_dist
        self.town = None
        self._id = 0
        self._new_episode = True
        self._light = None
        self.last_ms = 0.0
        self._log = open(log_path, "ab") if log_path else subprocess.DEVNULL
        cmd = [node, "tools/ego_server.mjs", "--w", str(self.W), "--h", str(self.H), "--q", quality, "--gl", gl,
               "--tdir", TOWN_DIR]
        self.proc = subprocess.Popen(cmd, cwd=WEB_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log)
        hello = self._read()
        if not hello.get("ok"):
            raise RuntimeError(f"KeiView server failed: {hello}")
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
    def load_town(self, seed):
        if self.town == seed:
            return {"cached": True}
        export_town(int(seed))
        r = self._call({"op": "town", "town": int(seed)})
        if r.get("errors"):
            raise RuntimeError(f"KeiView town {seed}: {r['errors']}")
        self.town = seed
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

    def render(self, pose, scene, t, want_rgb=True, want_seg=True, offset=None):
        C, R = camera_pose(self.cfg, *pose, offset)
        veh, ped = self._actors(C, scene)
        req = {"op": "render", "cam": {"pos": [float(v) for v in C], "fwd": [float(v) for v in R[2]], "vfov": self.vfov},
               "t": float(t), "veh": veh, "ped": ped, "labels": bool(want_seg), "reset": self._new_episode}
        if self._new_episode:
            req["light"] = self._light
        self._new_episode = False
        r = self._call(req)
        self.last_ms = r.get("ms", 0.0)
        W, H = r["w"], r["h"]
        rgb = np.frombuffer(base64.b64decode(r["rgb"]), np.uint8).reshape(H, W, 3)[::-1, :, ::-1]
        seg = None
        if r.get("sem"):
            seg = np.ascontiguousarray(np.frombuffer(base64.b64decode(r["sem"]), np.uint8).reshape(H, W)[::-1])
        return np.ascontiguousarray(rgb), seg

    def close(self):
        if getattr(self, "proc", None) is None or self.proc.poll() is not None:
            return
        try:
            self._call({"op": "quit"})
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()
