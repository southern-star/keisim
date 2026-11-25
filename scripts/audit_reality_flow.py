#!/usr/bin/env python3
"""Verify G3 collection, frozen caches, static-only history labels and the previous run's artifact manifest."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import checked_shards, episode_id, file_hash, load_field, load_manifest, write_json


def read(path):
    return json.loads(Path(path).read_text())


def verified(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f"hash mismatch for {path}")


def audit(args):
    collection = load_manifest(args.data)
    plan = read(args.data / "plan.json")
    protocol = read(args.config)
    spatial = read(args.features / "manifest.json")
    labels = read(args.labels / "manifest.json")
    conditions = read(args.cache / "test" / "manifest.json")
    contract = collection["contract_sha256"]
    if (spatial["collection_sha256"] != contract or labels["collection_sha256"] != contract or
            conditions["collection"] != contract or spatial["model_sha256"] != spatial["model_sha256_after"]):
        raise ValueError("inconsistent frozen feature or collection identity")
    verified(args.features / "manifest.json", conditions["features"])
    verified(args.pilot, spatial["checkpoint_sha256"])
    verified(args.renderer, conditions["renderer"])
    towns = [e["town_seed"] for e in plan["episodes"]]
    if towns != protocol["eval_towns"] or conditions["angles"] != protocol["test_angles"]:
        raise ValueError("G3 towns or cameras differ from the protocol")
    history_towns = sorted({e["town_seed"] for prior in args.prior_data for e in read(prior / "plan.json")["episodes"]})
    if set(towns) & set(history_towns):
        raise ValueError("G3 town already present in an earlier collection")

    counts = Counter()
    rows = []
    coverage = {}
    for ep in plan["episodes"]:
        name = episode_id(ep)
        shard_rows = checked_shards(args.data, ep, contract)
        rows.extend(shard_rows)
        times = np.concatenate([load_field(args.data, r, "t_sim") for r in shard_rows])
        steps = np.concatenate([load_field(args.data, r, "step") for r in shard_rows])
        parity = collection["episodes"][name]["parity"]
        if (len(times) != protocol["frames_per_town"] or not np.allclose(np.diff(times), .1, rtol=0, atol=1e-6)
                or not (np.diff(steps) == 1).all() or any(any(v.values()) for v in parity.values())):
            raise ValueError(f"snapshot continuity or camera parity failed for {name}")
        for index, row in enumerate(shard_rows):
            where = row["path"]
            sm = spatial["shards"][where]
            if sm["count"] != row["count"] or sm["rgb_sha256"] != row["fields"]["rgb_in"]["sha256"]:
                raise ValueError("camera and frozen features are misaligned")
            for key, digest in {"mem":sm["sha256"], **sm["pyramid"]}.items():
                verified(args.features / where / (key + ".npy"), digest)
            lab = args.labels / where / "visibility.npy"
            verified(lab, labels["shards"][where]["sha256"])
            vis = np.load(lab, mmap_mode="r")
            seg = load_field(args.data, row, "seg_target")
            now = load_field(args.data, row, "visibility_now")
            past = vis == 2
            if ((past & ((seg < 1) | (seg > 7))).any() or (index == 0 and past[0].any()) or
                    not np.array_equal(vis == 1, now == 1)):
                raise ValueError(f"past-only static support contract violated in {where}")
            values = np.bincount(vis.ravel(), minlength=256)
            got = {str(k):int(v) for k, v in enumerate(values) if v}
            if got != {k:v for k,v in labels["shards"][where]["counts"].items() if v}:
                raise ValueError("support class counts differ")
            counts.update(got)
        coverage[name] = dict(frames=len(times), first_step=int(steps[0]), last_step=int(steps[-1]),
                              first_s=float(times[0]), last_s=float(times[-1]), input_parity=parity)
    if rows != collection["shards"] or [r["path"] for r in rows] != conditions["shards"]:
        raise ValueError("shard ordering differs between source and caches")
    for key, digest in conditions["arrays"].items():
        verified(args.cache / "test" / (key + ".npy"), digest)
    previous = read(args.previous_manifest)["artifacts"]
    for name, record in previous.items():
        verified(args.previous_manifest.parent / name, record["sha256"])
    directories = dict(raw=args.data, features=args.features, labels=args.labels, evaluation_conditions=args.cache / "test")
    sizes = {name:sum(p.stat().st_size for p in folder.rglob("*.npy")) for name, folder in directories.items()}
    report = dict(passed=True, protocol_sha256=file_hash(args.config), collection_sha256=contract,
                  renderer_config_sha256=collection["renderer_config_sha256"], collection_s=collection["wall_seconds"],
                  coverage=coverage, previously_recorded_towns=history_towns, total_ticks=collection["frames"],
                  synchronized_views=collection["frames"]*len(conditions["angles"]), support_counts=dict(counts),
                  shards=len(rows), raw_arrays=sum(len(r["fields"]) for r in rows), feature_arrays=len(rows)*3,
                  support_arrays=len(rows), condition_arrays=len(conditions["arrays"]), npy_file_bytes=sizes,
                  previous_manifest_sha256=file_hash(args.previous_manifest), previous_artifacts_verified=len(previous))
    write_json(args.out, report)
    print(f'{len(rows)} shards, {report["total_ticks"]} ticks, {report["synchronized_views"]} target views verified')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "data", "features", "labels", "cache", "pilot", "renderer", "previous-manifest", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--prior-data", type=Path, nargs="+", required=True)
    audit(parser.parse_args())
