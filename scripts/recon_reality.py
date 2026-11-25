#!/usr/bin/env python3
"""Run native ReconView oracle, model and accumulated model on exactly the captured frames."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keipilot.agent import load_model
from keireality.dataset import file_hash, load_field, load_manifest, write_json, writer_lock


def encode(a):
    return base64.b64encode(np.ascontiguousarray(a)).decode("ascii")


@torch.inference_mode()
def reconstruct(args):
    data, out = Path(args.data), Path(args.out)
    collection = load_manifest(data)
    rows = [r for r in collection["shards"] if r["split"] == args.split]
    if not rows:
        raise ValueError("empty replay split")
    net = load_model(args.depth_ckpt, "cpu").eval().requires_grad_(False)
    if not net.aux_depth or net.rig != "wide_narrow":
        raise ValueError("ReconView model baseline requires the wide_narrow mc14 depth head")
    torch.set_num_threads(2)
    names = ("oracle", "model", "model_accum")
    command = ["node", "tools/reality_recon.mjs"] + (["--soft"] if args.gl == "soft" else [])
    fingerprint = dict(data_sha256=collection["contract_sha256"], depth_checkpoint_sha256=file_hash(args.depth_ckpt),
                       split=args.split, limit=args.limit, gl=args.gl, width=256, height=128,
                       fields=list(names), alpha="native rasterized coverage", rgb="sRGB, no ego or path overlay")
    with writer_lock(out):
        if (out / "config.json").exists() and json.loads((out / "config.json").read_text()) != fingerprint:
            raise ValueError("output directory belongs to a different replay")
        write_json(out / "config.json", fingerprint)
        proc = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1] / "web",
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        tick = time.perf_counter()
        try:
            hello = json.loads(proc.stdout.readline())
            if not hello.get("ready"):
                raise RuntimeError(f"ReconView did not initialize: {hello}")
            count, previous, files = 0, None, {}
            for row in rows:
                arrays = {k: load_field(data, row, k) for k in ("rgb_in", "depth_in", "ego_pose_world", "step")}
                frames = min(row["count"], args.limit - count) if args.limit else row["count"]
                result = {k: [] for k in names}
                for start in range(0, frames, 8):
                    rgb = arrays["rgb_in"][start:min(start + 8, frames)]
                    image = torch.from_numpy(rgb.copy()).permute(0, 1, 4, 2, 3)
                    n = len(image)
                    pred = net(image, torch.zeros(n, dtype=torch.long), torch.zeros(n, 2), with_depth=True)
                    seg = pred["seg"].argmax(1).byte().reshape(n, 2, 40, 80).numpy()
                    depths = net.depth_metres(pred["depth"]).reshape(n, 2, 80, 160).numpy()
                    for j in range(n):
                        i = start + j
                        step = int(arrays["step"][i])
                        identifier = f'{row["episode_id"]}:{step}'
                        q = dict(id=identifier, reset=previous != row["episode_id"], pose=arrays["ego_pose_world"][i].tolist(),
                                 rgb=[encode(a) for a in rgb[j]], dep=[encode(a.astype("<u2", copy=False)) for a in arrays["depth_in"][i]],
                                 seg=encode(seg[j]), predicted=encode(depths[j].astype("<f4", copy=False)))
                        proc.stdin.write(json.dumps(q) + "\n")
                        proc.stdin.flush()
                        answer = json.loads(proc.stdout.readline())
                        if answer.get("id") != identifier or "error" in answer:
                            raise RuntimeError(f'ReconView failed on {identifier}: {answer.get("error")}')
                        for name in names:
                            pixels = np.frombuffer(base64.b64decode(answer["images"][name]), np.uint8)
                            result[name].append(pixels.reshape(128, 256, 4)[::-1])
                        previous = row["episode_id"]
                dest = out / row["path"] / "recon.npz"
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                    np.savez_compressed(tmp, **{k: np.stack(v) for k, v in result.items()},
                                        step=arrays["step"][:frames])
                os.replace(tmp.name, dest)
                count += frames
                files[row["path"]] = dict(count=frames, sha256=file_hash(dest))
                if args.limit and count >= args.limit:
                    break
            write_json(out / "manifest.json", dict(**fingerprint, shards=files, gpu=hello["gpu"], frames=count,
                                                   seconds=time.perf_counter() - tick))
            print(f"Native ReconView produced {count} matched frames per baseline on {hello['gpu']}")
        finally:
            proc.stdin.close()
            proc.wait(timeout=60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "depth-ckpt", "out"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--gl", choices=("hw", "soft"), default="hw")
    parser.add_argument("--limit", type=int)
    reconstruct(parser.parse_args())
