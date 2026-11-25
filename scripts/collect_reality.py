#!/usr/bin/env python3
"""Record 10 Hz wide/narrow inputs and synchronized KeiView chase teachers for KeiReality."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import (checked_shards, episode_id, json_hash, load_field, load_manifest, validate_plan, write_json, write_shard,
                                writer_lock)
from keireality.snapshot import calibration, capture_env, render_snapshot


def collect_episode(env, root, ep, contract_hash, args):
    start = ep.get("start_step", 0)
    rows = checked_shards(root, ep, contract_hash)
    done_file = root / "shards" / episode_id(ep) / "done.json"
    count = sum(r["count"] for r in rows)
    if done_file.is_file():
        done = json.loads(done_file.read_text())
        if done["count"] != count or done["contract_sha256"] != contract_hash:
            raise ValueError(f'inconsistent completed episode {episode_id(ep)}')
        return rows, done
    env.reset(town_seed=ep["town_seed"], episode_seed=ep["episode_seed"], render=False)
    kv = env.keiview()
    if args.gl == "hw" and (kv.gl != "hw" or "swiftshader" in str(kv.gpu).lower() or "llvmpipe" in str(kv.gpu).lower()):
        raise RuntimeError(f"hardware rendering requested, got {kv.gpu}")
    audit_file = done_file.parent / "parity.json"
    frames, durations = [], []
    checks = json.loads(audit_file.read_text()) if rows and audit_file.exists() else {}
    recorded_states = [digest for row in rows for digest in load_field(root, row, "state_sha256")]
    previous = None
    for step in range(start + ep["frames"]):
        if step >= start:
            tick = time.perf_counter()
            frame, audit = render_snapshot(env, step, previous if step > start else None,
                                           audit=args.audit and step == start)
            if audit:
                checks = audit
                audit_file.parent.mkdir(parents=True, exist_ok=True)
                write_json(audit_file, checks)
            if step < start + count:
                # Replay the saved prefix too, so actor pools and other renderer history reach the same state.
                if frame["state_sha256"] != recorded_states[step - start]:
                    raise ValueError("resumed simulation differs from the stored episode")
            else:
                durations.append(time.perf_counter() - tick)
                frames.append(frame)
                if len(frames) == args.shard_size:
                    rows.append(write_shard(root, ep, frames, contract_hash))
                    frames.clear()
        previous = tuple(env.world.ego.pose)
        if step == start + ep["frames"] - 1:
            break
        env.step(env.expert_action())
        if env.done:
            break
    if frames:
        rows.append(write_shard(root, ep, frames, contract_hash))
    if not rows:
        raise RuntimeError(f'{episode_id(ep)} ended before the first requested frame')
    done = dict(count=sum(r["count"] for r in rows), contract_sha256=contract_hash, status=env.status,
                gl=kv.gl, gpu=kv.gpu, parity=checks, captured_this_run=len(durations),
                snapshot_seconds=sum(durations), snapshot_ms_p50=float(1000 * np.median(durations)) if durations else None,
                snapshot_ms_p95=float(1000 * np.percentile(durations, 95)) if durations else None)
    write_json(done_file, done)
    return rows, done


def collect(args):
    root = Path(args.out)
    episodes = json.loads(Path(args.plan).read_text()) if args.plan else [
        dict(town_seed=args.town, episode_seed=args.seed, split=args.split, frames=args.frames)]
    validate_plan(episodes)
    if args.shard_size < 1 or args.shard_size > 256:
        raise ValueError("shard size must be between 1 and 256")
    env = capture_env(args.gl, orbit=args.orbit)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    contract = dict(version=1, commit=commit, calibration=calibration(env), gl=args.gl, episodes=episodes,
                    shard_size=args.shard_size, audit=args.audit)
    contract_hash = json_hash(contract)
    with writer_lock(root):
        plan = root / "plan.json"
        if plan.exists() and json.loads(plan.read_text()) != contract:
            raise ValueError("output directory belongs to a different collection plan")
        if not plan.exists():
            write_json(plan, contract)
        manifest_path = root / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()).get("complete"):
            existing_manifest = load_manifest(root)
            for ep in episodes:
                checked_shards(root, ep, contract_hash)
            print(f'Verified complete collection: {existing_manifest["frames"]} frames')
            return
        # RGB, labels and uint16 depth for both inputs and the teacher, plus visibility/metadata and file headers.
        planned_frames = sum(e["frames"] for e in episodes)
        estimate = planned_frames * (860_000 + (len(env.cfg.cameras) - 3) * 232_000)
        existing = sum(p.stat().st_size for p in (root / "shards").rglob("*.npy"))
        if estimate > args.max_gib * 2**30 or estimate - existing > shutil.disk_usage(root).free:
            raise ValueError(f"collection requires up to {estimate / 2**30:.2f} GiB; check disk or --max-gib")
        rows, measured = [], {}
        manifest = dict(version=1, contract_sha256=contract_hash, renderer_config_sha256=json_hash(contract["calibration"]),
                        complete=False, shards=rows, episodes=measured)
        tick = time.perf_counter()
        with tempfile.TemporaryFile(mode="w+") as samples:
            gpu = None
            try:
                if shutil.which("nvidia-smi"):
                    gpu = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                            "-i", "0", "-lms", "250"], stdout=samples, stderr=subprocess.DEVNULL)
                for ep in episodes:
                    if planned_frames >= 1000:
                        renderer = env.keiview()
                        if renderer.gl == "soft" or any(s in str(renderer.gpu).lower() for s in ("swiftshader", "llvmpipe")):
                            raise ValueError("large collections need a hardware renderer; use a small CPU smoke sample")
                    episode_rows, result = collect_episode(env, root, ep, contract_hash, args)
                    rows.extend(episode_rows)
                    measured[episode_id(ep)] = result
                    write_json(root / "manifest.json", manifest)
                    print(f'{episode_id(ep)} {ep["split"]}: {result["count"]} frames; '
                          f'{result["snapshot_ms_p50"]} ms p50', flush=True)
            finally:
                env.close()
                if gpu is not None:
                    gpu.terminate()
                    gpu.wait()
            samples.seek(0)
            memory = [int(s.strip()) for s in samples if s.strip().isdigit()]
        manifest.update(complete=True, frames=sum(r["count"] for r in rows), wall_seconds=time.perf_counter() - tick,
                        elapsed_since_plan_s=time.time() - plan.stat().st_mtime,
                        wall_seconds_scope="completing invocation; elapsed_since_plan_s includes interruptions",
                        total_gpu_mib_start=memory[0] if memory else None, total_gpu_mib_peak=max(memory, default=None),
                        array_bytes=sum(p.stat().st_size for p in (root / "shards").rglob("*.npy")))
        write_json(root / "manifest.json", manifest)
        print(json.dumps({k: v for k, v in manifest.items() if k not in ("shards", "episodes")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--plan", help="JSON list of town_seed, episode_seed, split, frames and optional start_step")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--town", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--gl", choices=("hw", "soft", "auto"), default="hw")
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--max-gib", type=float, default=8)
    parser.add_argument("--audit", action="store_true", help="compare input RGB/labels with the native two-camera path")
    parser.add_argument("--orbit", action="store_true", help="synchronously capture chase and four orbit teachers")
    collect(parser.parse_args())
