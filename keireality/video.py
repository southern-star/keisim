"""Labeled diagnostic videos from recorded, time-aligned RGB frames."""
import os
from functools import lru_cache
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np


@lru_cache(maxsize=1)
def title_font():
    from PIL import ImageFont
    for name in ("NotoSansCJK-Regular.ttc", "meiryo.ttc", "ヒラギノ角ゴシック W3.ttc", "DroidSansFallbackFull.ttf"):
        try:
            return ImageFont.truetype(name, 15)
        except OSError:
            continue
    raise RuntimeError("Japanese video labels need a CJK font such as Noto Sans CJK")


@lru_cache(maxsize=512)
def japanese_header(title):
    from PIL import Image, ImageDraw
    header = Image.new("RGB", (384, 25), (24, 24, 27))
    ImageDraw.Draw(header).text((8, 1), title, font=title_font(), fill=(245, 245, 245))
    return np.asarray(header)


def panel(rgb, title):
    tile = cv2.resize(rgb, (384, 192), interpolation=cv2.INTER_NEAREST)
    if not title.isascii():
        return np.vstack((japanese_header(title), tile))
    tile = cv2.copyMakeBorder(tile, 25, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 27))
    cv2.putText(tile, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (245, 245, 245), 1, cv2.LINE_AA)
    return tile


def quad(tiles):
    a, b, c, d = (panel(rgb, title) for rgb, title in tiles)
    return np.vstack((np.hstack((a, b)), np.hstack((c, d))))


def write_video(path, images, fps=10):
    path = Path(path)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        dest = Path(tmp) / "video.mp4"
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-s", "768x434", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)], stdin=subprocess.PIPE)
        count = 0
        try:
            for image in images:
                if image.shape != (434, 768, 3) or image.dtype != np.uint8:
                    raise ValueError("expected RGB uint8 768x434 comparison frames")
                proc.stdin.write(image.tobytes())
                count += 1
        finally:
            proc.stdin.close()
            proc.wait()
        if proc.returncode or not count:
            raise RuntimeError("empty or failed video encode")
        os.replace(dest, path)
    return count
