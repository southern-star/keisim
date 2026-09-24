#!/usr/bin/env python3
"""Cut short highlight clips (red-light stop & go, pedestrian yield) from
evaluation videos using their per-frame event sidecars (*.json)."""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keisim.viz import ffmpeg_path  # noqa: E402


def find_windows(ev, pre=7.0, post=7.0):
    wins = []
    # red light: stopped for the light, then moving off again
    stopped_at = None
    for i, e in enumerate(ev):
        if e["reason"] == "red_light" and e["v"] < 0.1 and stopped_at is None:
            stopped_at = e["t"]
        if stopped_at is not None and e["v"] > 3.0:
            wins.append(("redlight", max(0.0, stopped_at - pre), e["t"] + 4.0))
            stopped_at = None
    # pedestrians the expert would yield to
    last = -1e9
    for e in ev:
        if e["reason"] == "pedestrian" and e["t"] - last > 15:
            wins.append(("pedestrian", max(0.0, e["t"] - pre), e["t"] + post))
            last = e["t"]
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=sorted(glob.glob("runs/videos/*.mp4")))
    ap.add_argument("--out", default="docs/clips")
    ap.add_argument("--max_len", type=float, default=32.0)
    ap.add_argument("--crf", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    made = []
    for v in args.videos:
        side = v[:-4] + ".json"
        if not os.path.exists(side):
            continue
        ev = json.load(open(side))
        if not ev:
            continue
        t0 = ev[0]["t"]
        for k, (kind, a, b) in enumerate(find_windows(ev)):
            a, b = a - t0, min(b - t0, a - t0 + args.max_len)
            name = os.path.join(args.out, f"{os.path.basename(v)[:-4]}_{kind}{k}.mp4")
            subprocess.run([ffmpeg_path(), "-y", "-loglevel", "error", "-ss", f"{a:.1f}", "-t", f"{b - a:.1f}", "-i", v,
                            "-c:v", "libx264", "-crf", str(args.crf), "-preset", "slow", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", "-an", name], check=True)
            made.append((name, kind, round(b - a, 1), os.path.getsize(name) // 1024))
    for m in made:
        print(m)


if __name__ == "__main__":
    main()
