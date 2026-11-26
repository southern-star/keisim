#!/usr/bin/env python3
"""Record four new 32-second expert drives and compare the frozen repair on paired observations."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import torch

from collect_reality_sweep import sweep
from eval_reality_disocclusion import town_report
from eval_reality_release_gate import ROOT, evaluate_town, load_release
from fit_reality_disocclusion import FIT, OUT
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.geometry import orbit_camera
from keisim.config import EnvConfig, set_camera_rig
from keisim.env import KeiEnv
from reality_closed_loop import load_background_repair
from run_reality_blur_fresh import moving_teacher


def evaluate():
    dest = OUT / "fresh"
    data = ROOT / "data/reality/p2_disocclusion_fresh"
    selected_sha = file_hash(FIT / "selection.json")
    replay = json.loads((OUT / "replay/summary.json").read_text())
    if (replay["selection_sha256"] != selected_sha
            or any(replay["towns"][str(t)]["status"] != "PASS" for t in (40, 41))):
        raise ValueError("Expected the same repair that passed the two validation towns")
    episodes = [dict(split="test", town_seed=20404+i, episode_seed=32044+i, start_step=30*i, frames=320)
                for i in range(4)]
    with writer_lock(dest):
        plan = dest / "episodes.json"
        write_json(plan, episodes)
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts/collect_reality.py"), "--gl", "hw",
                        "--orbit", "--audit", "--plan", str(plan), "--out", str(data)], cwd=ROOT, check=True)
        manifest = load_manifest(data)
        if json.loads((data / "plan.json").read_text())["episodes"] != episodes:
            raise ValueError("The new recordings differ from the four planned drives")
        contract = dict(selection=selected_sha, episodes=file_hash(plan), collection=manifest["contract_sha256"],
                        software={name: file_hash(Path(__file__).with_name(name)) for name in
                                  (Path(__file__).name, "eval_reality_disocclusion.py", "eval_reality_release_gate.py",
                                   "run_reality_blur_fresh.py")})
        models, repair = load_release(), load_background_repair()
        controls = dest / "controls"
        controls.mkdir(exist_ok=True)
        angles = sorted({sweep(t) for t in range(48)})
        cfg = set_camera_rig(EnvConfig(renderer="keiview", weather="noon", keiview_gl="hw"), "wide_narrow")
        cfg.extra_cameras = (*cfg.extra_cameras, *(orbit_camera(a) for a in angles))
        env = KeiEnv(cfg)
        summaries = {}
        try:
            for ep in episodes:
                town = ep["town_seed"]
                rows = [r for r in manifest["shards"] if r["town_seed"] == town and r["split"] == "test"]
                ticks = sum(r["count"] for r in rows)
                if ticks != ep["frames"]:
                    summaries[town] = dict(status="INCOMPLETE", ticks=ticks)
                    continue
                teacher = dest / f"town{town}-teacher.npz"
                path = dest / f"town{town}.json"
                reference = controls / path.name
                if path.is_file():
                    result = json.loads(path.read_text())
                    if (result["contract"] != contract or result["archive"] != file_hash(reference)
                            or result["teacher_sha256"] != file_hash(teacher)):
                        raise ValueError(f"Different inputs for the saved assessment of town {town}")
                    archived = json.loads(reference.read_text())
                else:
                    moving_teacher(env, angles, ep, rows, teacher, data=data)
                    archived = evaluate_town(data, rows, models, teacher, controls)
                    result = town_report(data, rows, models, repair, teacher, archived, dest, None)
                    result.update(contract=contract, archive=file_hash(reference), teacher_sha256=file_hash(teacher))
                    write_json(path, result)
                summaries[town] = dict(status=result["status"], ticks=ticks, release=archived["status"],
                                       sha256=file_hash(path))
                print("new town", town, summaries[town], flush=True)
        finally:
            env.close()
        write_json(dest / "summary.json", dict(towns=summaries, selection=selected_sha, episodes=file_hash(plan),
                   replay=file_hash(OUT / "replay/summary.json"), evaluator=file_hash(Path(__file__))))


if __name__ == "__main__":
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    evaluate()
