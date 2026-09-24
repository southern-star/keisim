"""Visualisation helpers: overlays, dashboard composition and H.264 video writing."""
from __future__ import annotations

import shutil
import subprocess

import cv2
import numpy as np

from .config import CMD_NAMES, SEM_COLORS, TL_NAMES
from .geometry import local_to_world

TL_COL = [(60, 60, 255), (0, 210, 255), (80, 230, 80), (170, 170, 170)]


def project_local_path(cam, pose, pts_local, scale=1.0, offset=None):
    """Project ego-frame ground points into the (scaled) camera image."""
    C, R = cam.camera_pose(*pose, offset)
    world = local_to_world(np.asarray(pts_local, float), *pose)
    uv, ok = cam.project_ground(world, C, R, scale=1)
    return uv * scale, ok


def draw_path(img, uv, ok, color, radius=4, thickness=2):
    pts = [tuple(np.round(p).astype(int)) for p, k in zip(uv, ok) if k]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, radius, color, -1, cv2.LINE_AA)


def text_block(img, lines, org=(8, 8), scale=0.5, color=(255, 255, 255), bg=(0, 0, 0), alpha=0.55, lh=None):
    lh = lh or int(22 * scale / 0.5)
    txt = [l[0] if isinstance(l, tuple) else l for l in lines]
    w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for l in txt) + 12
    h = lh * len(lines) + 8
    x, y = org
    sub = img[y:y + h, x:x + w]
    if sub.size:
        img[y:y + h, x:x + w] = (sub * (1 - alpha) + np.array(bg) * alpha).astype(np.uint8)
    for i, l in enumerate(lines):
        c = color
        if isinstance(l, tuple):
            l, c = l
        cv2.putText(img, l, (x + 6, y + lh * (i + 1) - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, c, 1, cv2.LINE_AA)


def colorize_seg(seg):
    return SEM_COLORS[np.clip(seg, 0, len(SEM_COLORS) - 1)]


def speed_bar(img, org, w, h, v, target, vmax=12.0):
    x, y = org
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * min(v, vmax) / vmax), y + h), (90, 200, 90), -1)
    tx = x + int(w * min(target, vmax) / vmax)
    cv2.line(img, (tx, y - 3), (tx, y + h + 3), (0, 220, 255), 2)


def compose_dashboard(rgb, bev, seg_panel, info_lines, cam=None, pose=None, paths=(), title=None):
    """rgb (160x320) -> 640x320 main view with projected paths; returns 960x480 frame."""
    H, W = rgb.shape[:2]
    s = 2
    main = cv2.resize(rgb, (W * s, H * s), interpolation=cv2.INTER_LINEAR)
    if cam is not None and pose is not None:
        for pts, col in paths:
            uv, ok = project_local_path(cam, pose, pts, scale=s)
            draw_path(main, uv, ok, col)
    if title:
        text_block(main, [title], org=(8, 8), scale=0.55)
    seg_small = cv2.resize(seg_panel, (320, 160), interpolation=cv2.INTER_NEAREST)
    info = np.full((160, 320, 3), 28, np.uint8)
    text_block(info, info_lines, org=(4, 4), scale=0.45, alpha=0.0, lh=19)
    bottom = np.concatenate([seg_small, info], 1)
    left = np.concatenate([main, bottom], 0)
    bev_r = cv2.resize(bev, (320, 480), interpolation=cv2.INTER_LINEAR) if bev.shape[:2] != (480, 320) else bev
    return np.concatenate([left, bev_r], 1)


def ffmpeg_path():
    """System ffmpeg if available, otherwise the static build shipped with imageio-ffmpeg."""
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class VideoWriter:
    """H.264 mp4 through an ffmpeg pipe (browser friendly); falls back to OpenCV."""

    def __init__(self, path, fps=10, size=None):
        self.path, self.fps, self.size = path, fps, size
        self.proc = None
        self.cv = None

    def _open(self, frame):
        h, w = frame.shape[:2]
        exe = ffmpeg_path()
        if exe:
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                   "-r", str(self.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                   "-preset", "veryfast", "-movflags", "+faststart", self.path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))

    def write(self, frame):
        if self.proc is None and self.cv is None:
            self._open(frame)
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        if self.cv is not None:
            self.cv.release()


def info_lines(v, target, cmd, tl_pred=None, tl_gt=None, reason=None, extra=()):
    lines = [f"speed  {v * 3.6:5.1f} km/h", f"target {target * 3.6:5.1f} km/h", f"command: {CMD_NAMES[cmd]}"]
    if tl_pred is not None:
        lines.append((f"light (model): {TL_NAMES[tl_pred]}", TL_COL[tl_pred]))
    if tl_gt is not None:
        lines.append((f"light (truth): {TL_NAMES[tl_gt]}", TL_COL[tl_gt]))
    if reason:
        lines.append(f"expert: {reason}")
    lines += list(extra)
    return lines
