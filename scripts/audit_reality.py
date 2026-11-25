#!/usr/bin/env python3
"""Audit two real captures of the same episode and save the Phase 0 four-panel movie and metrics."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import checked_shards, load_field, load_manifest, write_json
from keireality.geometry import CHASE_CAMERA, UNKNOWN, VISIBLE
from keireality.video import quad, write_video
from keisim.config import SEM
from keisim.render import rays


def frame_stream(root, episode, contract):
    for row in checked_shards(root, episode, contract):
        arrays = {key: load_field(root, row, key) for key in row["fields"]}
        for i in range(row["count"]):
            yield {key: value[i] for key, value in arrays.items()}


def audit(a, b, out):
    out.mkdir(parents=True, exist_ok=True)
    ac, bc = load_manifest(a), load_manifest(b)
    pa, pb = (json.loads((p / "plan.json").read_text()) for p in (a, b))
    ea, eb = pa["episodes"][0], pb["episodes"][0]
    if (any(ea.get(k, 0) != eb.get(k, 0) for k in ("town_seed", "episode_seed", "start_step"))
            or pa["calibration"] != pb["calibration"]):
        raise ValueError("recaptures must use the same camera calibration, town, seed and start step")
    first = frame_stream(a, ea, ac["contract_sha256"])
    second = frame_stream(b, eb, bc["contract_sha256"])
    fields = ("rgb_in", "rgb_target", "seg_in", "seg_target", "depth_in", "depth_target", "visibility_now",
              "state_sha256", "step", "t_sim", "ego_pose_world")
    changed = {k: 0 for k in fields}
    visibility = np.zeros(256, np.int64)
    # The ray through the ego roof should land on the exposed ground while the ego mesh is excluded.
    uv, _, ok = rays.project(CHASE_CAMERA, [[0, 0, 1.45]], *rays.level_frame(CHASE_CAMERA, 0, 0, 0))
    assert ok[0]
    u, v = np.rint(uv[0]).astype(int)
    roof_classes = []
    timestamps = []

    def frames():
        for fa, fb in zip(first, second):
            for k in fields:
                changed[k] += int(np.count_nonzero(fa[k] != fb[k]))
            mask = fa["visibility_now"][0]
            visibility[:] += np.bincount(mask.ravel(), minlength=256)
            roof_classes.append(int(fa["seg_target"][0, v, u]))
            t = float(fa["t_sim"])
            timestamps.append(t)
            colors = np.full((*mask.shape, 3), (23, 30, 42), dtype=np.uint8)
            colors[mask == VISIBLE] = (99, 218, 165)
            colors[mask == UNKNOWN] = (118, 122, 138)
            img = quad([(fa["rgb_in"][0], f'Wide | step {fa["step"]}'), (fa["rgb_in"][1], f'Narrow | t={t:.2f}s'),
                        (fa["rgb_target"][0], "KeiView | chase teacher"),
                        (colors, "Visibility | green: seen, gray: unknown, dark: unseen")])
            if len(timestamps) == 1:
                cv2.imwrite(str(out / "snapshot.png"), img[..., ::-1])
            yield img

    n = write_video(out / "synchronized.mp4", frames())
    parity = [cam for value in ac["episodes"].values() for cam in value["parity"].values()]
    result = dict(frames=n, timestamps_10hz=bool(np.allclose(np.diff(timestamps), .1, atol=1e-7, rtol=0)),
                  independent_recapture_changed_elements=changed, rig_parity=ac["episodes"],
                  input_parity_exact=len(parity) == 2 and all(p["rgb_max"] == p["seg_changed"] == 0 for p in parity),
                  input_within_native_repeat=len(parity) == 2 and all(
                      p["seg_changed"] == 0 and p["rgb_max"] <= p["repeat_max"] and p["rgb_mae"] <= p["repeat_mae"]
                      for p in parity),
                  visibility_counts={str(i): int(visibility[i]) for i in (0, 1, 255)},
                  ego_roof_vehicle_pixels=sum(x == SEM["vehicle"] for x in roof_classes),
                  ego_roof_sample=[int(u), int(v)], ego_roof_classes=sorted(set(roof_classes)))
    write_json(out / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False))
    if (n < 16 or any(changed[k] for k in ("state_sha256", "step", "t_sim", "ego_pose_world"))
            or not result["timestamps_10hz"] or not result["input_within_native_repeat"]):
        raise SystemExit("snapshot contract or native input repeatability check failed; see audit.json")
    if any(changed.values()):
        print("Separate renderer launches produced pixel differences; inspect the per-field counts in audit.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--recapture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.data, args.recapture, args.out)
