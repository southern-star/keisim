#!/usr/bin/env python3
"""Build a self-contained, offline view of measured test frames and their estimated observation support."""
import argparse
import base64
import json
from pathlib import Path

import cv2


def build(out):
    results = json.loads((out / "metrics.json").read_text())
    episodes = sorted(results["test_frame_counts"])
    images = {}
    for ep in episodes:
        images[ep] = {}
        for angle in (-36, -18, 0, 18, 36):
            path = out / f"sample_{ep}_{angle:+d}.png"
            raw = cv2.imread(str(path))
            if raw is None or raw.shape != (434, 768, 3):
                raise ValueError(f"missing test frame {path}")
            ok, encoded = cv2.imencode(".png", raw, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            if not ok:
                raise ValueError("cannot encode sample")
            images[ep][str(angle)] = "data:image/png;base64," + base64.b64encode(encoded).decode()
    template = Path(__file__).with_name("reality_viewer.html").read_text()
    (out / "viewer.html").write_text(template.replace("/* SAMPLES */{}", json.dumps(images, separators=(",", ":"))))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    build(p.parse_args().out)
