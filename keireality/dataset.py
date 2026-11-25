"""Immutable collection plan and atomic, memory-mappable shards of complete consecutive frames."""
from contextlib import contextmanager
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def json_hash(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    digest = sha256()
    with open(path, "rb") as src:
        for block in iter(lambda: src.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        json.dump(obj, tmp, indent=2, ensure_ascii=False, allow_nan=False)
        tmp.write("\n")
    os.replace(tmp.name, path)


@contextmanager
def writer_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def episode_id(ep):
    return f't{ep["town_seed"]}-e{ep["episode_seed"]}'


def validate_plan(episodes):
    towns, ids = {}, set()
    if not episodes:
        raise ValueError("empty episode plan")
    for ep in episodes:
        for key in ("town_seed", "episode_seed", "frames", "start_step"):
            value = ep.get(key, 0) if key == "start_step" else ep[key]
            if type(value) is not int or value < 0:
                raise ValueError(f"expected a nonnegative integer {key}: {value}")
        name, town, split = episode_id(ep), ep["town_seed"], ep["split"]
        if split not in ("train", "val", "test") or ep["frames"] < 1 or ep.get("start_step", 0) < 0:
            raise ValueError(f"invalid episode: {ep}")
        if towns.setdefault(town, split) != split or name in ids:
            raise ValueError(f"town split overlap or duplicate episode: {name}")
        ids.add(name)


def write_shard(root, ep, frames, contract_hash):
    first = int(frames[0]["step"])
    if ([int(f["step"]) for f in frames] != list(range(first, first + len(frames)))
            or not np.allclose(np.diff([f["t_sim"] for f in frames]), .1, rtol=0, atol=1e-7)):
        raise ValueError("shards require consecutive 10 Hz control frames from a single episode")
    directory = Path("shards") / episode_id(ep) / f"{first:06d}"
    dest = root / directory
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".writing-") as tmp:
        staging = Path(tmp) / "shard"
        staging.mkdir()
        fields = {}
        for name in frames[0]:
            array = np.stack([frame[name] for frame in frames])
            file = staging / (name + ".npy")
            np.save(file, array, allow_pickle=False)
            fields[name] = dict(shape=list(array.shape), dtype=array.dtype.str, sha256=file_hash(file))
        meta = dict(episode_id=episode_id(ep), town_seed=ep["town_seed"], episode_seed=ep["episode_seed"],
                    split=ep["split"], first=first, count=len(frames), path=str(directory),
                    fields=fields, contract_sha256=contract_hash)
        write_json(staging / "shard.json", meta)
        staging.rename(dest)
    return meta


def checked_shards(root, ep, contract_hash):
    found = []
    next_step = ep.get("start_step", 0)
    for file in sorted((root / "shards" / episode_id(ep)).glob("[0-9]*/shard.json")):
        row = json.loads(file.read_text())
        if (row["contract_sha256"] != contract_hash or row["episode_id"] != episode_id(ep)
                or row["split"] != ep["split"] or row["first"] != next_step
                or row["count"] < 1 or root / row["path"] != file.parent):
            raise ValueError(f"incompatible or discontinuous shard {file}")
        for field, schema in row["fields"].items():
            path = file.parent / (field + ".npy")
            if file_hash(path) != schema["sha256"]:
                raise ValueError(f"corrupt shard field {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != schema["shape"] or array.dtype.str != schema["dtype"] or len(array) != row["count"]:
                raise ValueError(f"wrong array schema {path}")
        found.append(row)
        next_step += row["count"]
    return found


def load_manifest(root):
    root = Path(root)
    obj = json.loads((root / "manifest.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    if not obj["complete"] or obj["contract_sha256"] != json_hash(plan):
        raise ValueError(f"incomplete or mismatched collection: {root}")
    validate_plan(plan["episodes"])
    episodes = {episode_id(ep): ep for ep in plan["episodes"]}
    starts = {k: ep.get("start_step", 0) for k, ep in episodes.items()}
    for row in obj["shards"]:
        ep = episodes[row["episode_id"]]
        if (any(row[k] != ep[k] for k in ("split", "town_seed", "episode_seed"))
                or row["contract_sha256"] != obj["contract_sha256"] or row["first"] != starts[episode_id(ep)]
                or row["count"] < 1):
            raise ValueError("manifest shard does not match its planned episode")
        starts[episode_id(ep)] += row["count"]
    if any(starts[k] - ep.get("start_step", 0) != obj["episodes"][k]["count"] for k, ep in episodes.items()):
        raise ValueError("manifest episode count differs from stored shards")
    return obj


def load_field(root, row, name):
    return np.load(Path(root) / row["path"] / (name + ".npy"), mmap_mode="r", allow_pickle=False)
