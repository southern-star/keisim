#!/usr/bin/env python3
"""Cache detached FP16 mc14 tokens by immutable RGB shard and checkpoint identity."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, json_hash, load_field, load_manifest, write_json, writer_lock
from keireality.encoder import extract_mem, frozen_pilot, state_hash


def cache(args):
    root, dest = Path(args.data), Path(args.out)
    collection = load_manifest(root)
    if args.bs < 1:
        raise ValueError("batch size must be positive")
    ckpt_hash = file_hash(args.ckpt)
    pilot = frozen_pilot(args.ckpt, args.device)
    before = state_hash(pilot)
    tick = time.perf_counter()
    specs = dict(collection_sha256=collection["contract_sha256"], checkpoint_sha256=ckpt_hash,
                 model_sha256=before, storage="float16", shape=[500, 256], version=1)
    if args.pyramid:
        specs.update(version=2, pyramid={"c2": [2, 128, 20, 40], "c3": [2, 256, 10, 20]})
    with writer_lock(dest):
        spec_file = dest / "config.json"
        if spec_file.exists() and json.loads(spec_file.read_text()) != specs:
            raise ValueError("cache belongs to different inputs or checkpoint")
        write_json(spec_file, specs)
        rows, created = {}, 0
        profile = None
        for row in collection["shards"]:
            name = row["path"]
            out = dest / name
            out.mkdir(parents=True, exist_ok=True)
            rgb_hash = row["fields"]["rgb_in"]["sha256"]
            if file_hash(root / name / "rgb_in.npy") != rgb_hash:
                raise ValueError(f"corrupt RGB in {name}")
            key = json_hash([specs, name, rgb_hash])
            meta = out / "features.json"
            target = out / "mem.npy"
            if meta.exists():
                record = json.loads(meta.read_text())
                if key != record["key"] or file_hash(target) != record["sha256"]:
                    raise ValueError(f"invalid cached shard {name}")
                for level in specs.get("pyramid", {}):
                    if file_hash(out / (level + ".npy")) != record["pyramid"][level]:
                        raise ValueError(f"invalid cached {level} in {name}")
            else:
                rgb = load_field(root, row, "rgb_in")
                chunks = {field: [] for field in ("mem", *specs.get("pyramid", {}))}
                for i in range(0, len(rgb), args.bs):
                    batch = torch.from_numpy(np.array(rgb[i:i + args.bs])).permute(0, 1, 4, 2, 3).to(args.device)
                    result = extract_mem(pilot, batch, parity=not rows and i == 0, pyramid=args.pyramid)
                    if not rows and i == 0:
                        sample = batch[:1]
                        for _ in range(3):
                            extract_mem(pilot, sample)
                        times = []
                        for _ in range(20):
                            t = time.perf_counter()
                            extract_mem(pilot, sample)
                            if args.device == "cuda":
                                torch.cuda.synchronize()
                            times.append(1000 * (time.perf_counter() - t))
                        profile = {f"p{p}_ms": float(np.percentile(times, p)) for p in (50, 95, 99)}
                    for field, values in (result if args.pyramid else {"mem": result}).items():
                        chunks[field].append(values.to(dtype=torch.float16).cpu().numpy())
                hashes = {}
                for field, parts in chunks.items():
                    with tempfile.NamedTemporaryFile(dir=out, delete=False) as tmp:
                        np.save(tmp, np.concatenate(parts), allow_pickle=False)
                    path = out / (field + ".npy")
                    os.replace(tmp.name, path)
                    hashes[field] = file_hash(path)
                record = dict(key=key, sha256=hashes.pop("mem"), count=len(rgb), rgb_sha256=rgb_hash)
                if hashes:
                    record["pyramid"] = hashes
                write_json(meta, record)
                created += len(rgb)
            rows[name] = record
        after = state_hash(pilot)
        if after != before:
            raise RuntimeError("frozen driving parameters or BatchNorm buffers changed")
        if not created and (dest / "manifest.json").exists():
            prior = json.loads((dest / "manifest.json").read_text())
            if all(prior.get(k) == v for k, v in specs.items()) and prior.get("shards") == rows:
                print("Verified complete feature cache")
                return
        manifest = dict(**specs, model_sha256_after=after, shards=rows, seconds=time.perf_counter() - tick,
                        frames_computed_this_run=created, driving_pass_with_feature_read_batch1=profile,
                        device=args.device, torch_version=str(torch.__version__), cuda=torch.version.cuda,
                        cuda_peak_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None)
        write_json(dest / "manifest.json", manifest)
        print(f'{sum(v["count"] for v in rows.values())} frames, {len(rows)} shards, {manifest["seconds"]:.2f}s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--pyramid", action="store_true", help="also store frozen spatial c2 and c3 features")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    cache(parser.parse_args())
