"""Hashes of KeiView's images and labels for a few fixed frames of the wide_narrow rig (and the front camera): run it
before and after a change to web/src/ego.js or the pipeline to see that the frames are bit for bit the same.

    python tools/keiview_hash.py                      # this checkout's web/
    python tools/keiview_hash.py --web /path/to/web   # another checkout's (e.g. `git archive origin/main web`, with
                                                      # node_modules and .towns linked in)
Prints one line per image: frame, camera, sha1 of the RGB image and of the labels. Needs a GPU for the same hashes
(SwiftShader draws other pixels)."""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keisim.render.keiview as KV  # noqa: E402
from keisim.config import EnvConfig, set_camera_rig  # noqa: E402
from keisim.env import KeiEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", default=None, help="the web/ directory KeiView is served from")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--depth", type=int, default=0, help="1: ask for depth too (the images and labels must not change)")
    args = ap.parse_args()
    if args.web:
        KV.WEB_DIR = os.path.abspath(args.web)
    sha = lambda a: hashlib.sha1(a.tobytes()).hexdigest()[:16]
    for rig in ("wide_narrow", "front"):
        env = KeiEnv(set_camera_rig(EnvConfig(renderer="keiview"), rig))
        try:
            env.reset(town_seed=5, episode_seed=1, weather="noon")
            for k in range(args.frames):
                if args.depth and rig != "front":
                    views = env.render_cameras_depth()[0]
                else:
                    views = env.render_cameras()
                for cfg, (rgb, seg) in zip(env.cfg.cameras, views):
                    print(f"{rig} frame {k} {cfg.name}: rgb {sha(rgb)} labels {sha(seg)}")
                for _ in range(10):                                   # 1 s on: the expert drives
                    env.step(env.expert.act())
        finally:
            env.close()


if __name__ == "__main__":
    main()
