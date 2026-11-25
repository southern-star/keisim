#!/usr/bin/env python3
"""Render synchronized G3 comparison videos and plot the measured 32-second error curves."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def comparison_frames(data, rows, start, native, arrays, angles, switched, last_image, traffic):
    from PIL import Image
    from keireality.dataset import load_field
    from keireality.flow_metrics import display_image
    from keireality.video import quad
    from keisim.config import SEM

    track = len(angles) if switched else angles.index(0)
    total = sum(row["count"] for row in rows)
    offset = 0
    peak = 0
    peak_image = None
    for row in rows:
        gt = load_field(data, row, "rgb_target")
        if last_image:
            seg = load_field(data, row, "seg_target")
            visibility = load_field(data, row, "visibility_now")
        for j in range(row["count"]):
            v = int(arrays["views"][offset, track])
            title = f"教師映像　町{row['town_seed']}　{offset/10:04.1f}秒　{angles[v]:+d}°"
            p2 = native[start + offset, v].astype(np.float32).transpose(1, 2, 0)
            tiles = [(gt[j, v], title), ((p2 * 255).round().clip(0, 255).astype(np.uint8), "Phase 2　履歴モデル")]
            for n in (1, 4):
                raw = arrays[f"nfe{n}"][offset, track].astype(np.float32).transpose(1, 2, 0)
                rgb = ((display_image(raw) + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
                tiles.append((rgb, f"MeanFlow　{n}回生成"))
            image = quad(tiles)
            if last_image:
                actors = int(np.count_nonzero((seg[j, v] >= SEM["vehicle"]) & (visibility[j, v] == 1)))
                if actors > peak:
                    peak = actors
                    peak_image = image
                    traffic.update(offset=offset, seconds=offset / 10, visible_dynamic_teacher_pixels=actors)
            if last_image and offset == total - 1:
                Image.fromarray(image).save(last_image)
            yield image
            offset += 1
    if peak_image is not None:
        traffic_path = last_image.with_name(rows[0]["episode_id"] + "_traffic.png")
        Image.fromarray(peak_image).save(traffic_path)
        traffic["file"] = traffic_path.name


def make_videos(args):
    from keireality.dataset import file_hash, load_manifest, write_json
    from keireality.video import write_video
    data = load_manifest(args.data)
    cache_dir = args.cache / "test"
    contract = json.loads((cache_dir / "manifest.json").read_text())
    rollouts = json.loads((args.rollouts / "manifest.json").read_text())
    if (contract["collection"] != data["contract_sha256"] or
            rollouts["contract"]["cache"] != file_hash(cache_dir / "manifest.json")):
        raise ValueError("comparison sources do not match")
    native = np.load(cache_dir / "native.npy", mmap_mode="r")
    start = 0
    checks = {}
    for name in data["episodes"]:
        rows = [row for row in data["shards"] if row["episode_id"] == name]
        arrays = {key: np.load(args.rollouts / name / (key + ".npy"), mmap_mode="r")
                  for key in ("views", "nfe1", "nfe4")}
        record = rollouts["episodes"][name]
        for switched in ([False, True] if start == 0 else [False]):
            path = args.out / ("view_switch.mp4" if switched else name + ".mp4")
            png = None if switched else args.out / (name + "_last.png")
            traffic = {}
            frames = comparison_frames(args.data, rows, start, native, arrays, contract["angles"], switched, png, traffic)
            count = write_video(path, frames, fps=10)
            stream = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,duration,nb_read_frames", "-of", "json", str(path)
            ]))["streams"][0]
            if (count != record["ticks"] or int(stream["nb_read_frames"]) != count or
                    float(stream["duration"]) != count / 10 or stream["r_frame_rate"] != "10/1" or
                    (stream["width"], stream["height"]) != (768, 434)):
                raise ValueError(f"video verification failed for {path}")
            checks[path.name] = dict(episode=name, frames=count, switched=switched, **stream, sha256=file_hash(path))
            if traffic:
                checks[path.name]["traffic_sample"] = traffic
            print(path, count, flush=True)
        start += record["ticks"]
    write_json(args.out / "media-check.json", checks)


def plot_trajectory(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    trace = json.loads((args.out / "trajectory.json").read_text())
    episodes = list(dict.fromkeys(row["episode"] for row in trace))
    if len(episodes) != 4:
        raise ValueError("expected four test towns")
    fonts = font_manager.findSystemFonts()
    japanese = next((p for p in fonts if p.endswith("NotoSansCJK-Regular.ttc")), None)
    if japanese:
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=japanese).get_name()
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.9), sharex=True, sharey=True)
    curves = (("phase2", "Phase 2", "#374151"), ("nfe1", "MeanFlow 1 NFE", "#0072b2"),
              ("nfe4", "MeanFlow 4 NFE", "#cc591a"))
    peak = 0.
    for name, ax in zip(episodes, axes.flat):
        rows = [r for r in trace if r["episode"] == name]
        seconds = np.asarray([r["seconds"] for r in rows])
        for key, label, color in curves:
            value = np.asarray([r["chase"][key]["all_l1"] for r in rows])
            # A trailing ten-tick mean also reports the partial window at the start.
            sums = np.r_[0., np.cumsum(value)]
            end = np.arange(1, len(value) + 1)
            begin = np.maximum(0, end - 10)
            smooth = (sums[end] - sums[begin]) / (end - begin)
            peak = max(peak, float(smooth.max()))
            ax.plot(seconds, smooth, label=label, color=color, lw=1.5)
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_xlim(0, 32)
        ax.set_xticks([0, 8, 16, 24, 32])
        ax.grid(alpha=.22)
    axes[0, 0].set_ylim(0, max(.01, 1.05 * peak))
    axes[0, 0].set_ylabel("RGB L1")
    axes[1, 0].set_ylabel("RGB L1")
    for ax in axes[1]:
        ax.set_xlabel("走行時間（秒）" if japanese else "Drive time (s)")
    title = "中央追跡視点の RGB 誤差   |   直近 1 秒の平均" if japanese else "Chase-view RGB error   |   trailing 1 s mean"
    fig.suptitle(title, x=.075, ha="left", fontsize=15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5,.93),
               ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .89))
    fig.savefig(args.out / "trajectory.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    video = sub.add_parser("video")
    for name in ("data", "cache", "rollouts", "out"):
        video.add_argument("--" + name, type=Path, required=True)
    plot = sub.add_parser("plot")
    plot.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (make_videos if args.action == "video" else plot_trajectory)(args)
