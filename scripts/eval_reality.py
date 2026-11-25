#!/usr/bin/env python3
"""Score paired frozen-feature predictions, native ReconView and training-only baselines on a held-out split."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, load_field, write_json
from keireality.learning import RealityFrames
from keireality.metrics import ImageScores, SemanticScores, regions, temporal_error, train_baseline
from keireality.renderer import SingleFrameRenderer


def timing(model, mem, view, device):
    for _ in range(8):
        model(mem, view)
    sync = torch.cuda.synchronize if device == "cuda" else lambda: None
    values = {k: [] for k in ("total", "adapter", "upsampling_and_heads", "rgb_to_cpu")}
    # The first convolution begins immediately after feature projection and cross-attention.
    middle = []
    def mark(module, inputs):
        if device == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            middle.append(event)
        else:
            middle.append(time.perf_counter())
    handle = model.base.register_forward_pre_hook(mark)
    try:
        for _ in range(50):
            sync()
            t = time.perf_counter()
            if device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
            rgb, _ = model(mem, view)
            if device == "cuda":
                stop.record()
            sync()
            finished = time.perf_counter()
            boundary = middle.pop()
            if device == "cuda":
                values["adapter"].append(start.elapsed_time(boundary))
                values["upsampling_and_heads"].append(boundary.elapsed_time(stop))
            else:
                values["adapter"].append(1000 * (boundary - t))
                values["upsampling_and_heads"].append(1000 * (finished - boundary))
            values["total"].append(1000 * (finished - t))
            t = time.perf_counter()
            rgb.cpu()
            values["rgb_to_cpu"].append(1000 * (time.perf_counter() - t))
    finally:
        handle.remove()
    return {k: {f"p{p}_ms": float(np.percentile(v, p)) for p in (50, 95, 99)} for k, v in values.items()}


def new_scores():
    return ({k: ImageScores() for k in ("generated", "mean", "oracle", "model", "model_accum", "hold_previous_generated")},
            {k: SemanticScores() for k in ("generated_head", "modal")})


def finish(scores):
    return {**{k: v.report() for k, v in scores[0].items()}, **{k: v.report() for k, v in scores[1].items()}}


@torch.inference_mode()
def evaluate(args):
    torch.set_num_threads(2)
    data, out, recon_root = Path(args.data), Path(args.out), Path(args.recon)
    out.mkdir(parents=True, exist_ok=True)
    ds = RealityFrames(data, args.features, args.split)
    checkpoint = torch.load(args.ckpt, map_location=args.device)
    if (checkpoint["config"]["data_sha256"] != ds.collection["contract_sha256"] or
            checkpoint["config"]["pilot_sha256"] != ds.cache["checkpoint_sha256"]):
        raise ValueError("renderer weights were trained against different data or pilot features")
    model = SingleFrameRenderer().to(args.device, memory_format=torch.channels_last).eval()
    model.load_state_dict(checkpoint["model"])
    recon = json.loads((recon_root / "manifest.json").read_text())
    if (recon["data_sha256"] != ds.collection["contract_sha256"] or recon["split"] != args.split
            or recon["frames"] != len(ds)):
        raise ValueError("ReconView comparison is incomplete or belongs to another manifest")
    mean, modal = train_baseline(data, [r for r in ds.collection["shards"] if r["split"] == "train"])
    global_scores, episodes, frame_stats = new_scores(), {}, []
    previous, last_j, last_ep = None, -1, None
    temporal_sum, temporal_n = 0., 0
    tests = {"correct_l1": 0., "shuffled_l1": 0., "zero_l1": 0.}
    tick = time.perf_counter()
    for batch in DataLoader(ds, batch_size=args.bs):
        mem, view = batch["mem"].to(args.device), batch["view_pose"].to(args.device)
        if last_j == -1:
            latency = timing(model, mem[:1], view[:1], args.device)
        rgb, logits = model(mem, view)
        indices = batch["index"].tolist()
        wrong = torch.stack([ds.mem_at((i + len(ds)//2) % len(ds)) for i in indices]).to(args.device)
        truth = batch["rgb"].to(args.device)
        for name, prediction in (("correct_l1", rgb), ("shuffled_l1", model(wrong, view)[0]),
                                 ("zero_l1", model(torch.zeros_like(mem), view)[0])):
            tests[name] += (prediction - truth).abs().mean((1, 2, 3)).sum().item()
        preds = rgb.permute(0, 2, 3, 1).cpu().numpy()
        semantics = logits.argmax(1).byte().cpu().numpy()
        for offset, index in enumerate(indices):
            j, i = ds.index[index]
            row = ds.rows[j]
            episode = row["episode_id"]
            if j != last_j:
                path = recon_root / row["path"] / "recon.npz"
                if file_hash(path) != recon["shards"][row["path"]]["sha256"]:
                    raise ValueError(f"corrupt ReconView data {path}")
                with np.load(path, allow_pickle=False) as stored:
                    rr = {k: stored[k] for k in stored.files}
                side = {k: load_field(data, row, k) for k in ("step", "depth_target", "ego_pose_world")}
                if not np.array_equal(rr["step"], side["step"]):
                    raise ValueError("replay frames are out of sync")
                prediction_path = out / "predictions" / row["path"]
                prediction_path.mkdir(parents=True, exist_ok=True)
                rgb_save = np.lib.format.open_memmap(prediction_path / "rgb.npy", "w+", np.uint8, (row["count"], 128, 256, 3))
                last_j = j
            if episode != last_ep:
                previous, last_ep = None, episode
                episodes[episode] = new_scores()
            pred = preds[offset]
            gt = batch["rgb"][offset].permute(1, 2, 0).numpy()
            label = batch["semantic"][offset].numpy()
            masks = regions(batch["visibility"][offset].numpy())
            rgb_save[i] = np.rint(pred * 255).astype(np.uint8)
            # Accumulate once at episode level. The global sums are combined after the last frame.
            images, segs = episodes[episode]
            for name, prediction in (("generated", pred), ("mean", mean)):
                images[name].add(prediction, gt, masks)
            for name in ("oracle", "model", "model_accum"):
                rgba = rr[name][i]
                images[name].add(rgba[..., :3].astype(np.float32) / 255, gt, masks, rgba[..., 3] > 0)
            segs["generated_head"].add(semantics[offset], label, masks)
            segs["modal"].add(modal, label, masks)
            current = (pred.copy(), gt.copy(), label.copy(), side["depth_target"][i, 0], side["ego_pose_world"][i])
            if previous is not None:
                images["hold_previous_generated"].add(previous[0], gt, masks)
                t_sum, t_n = temporal_error(current, previous)
                temporal_sum += t_sum
                temporal_n += t_n
            error = np.abs(pred - gt).mean(-1)
            dynamic = masks["visible"] & np.isin(label, [11, 12])
            frame_stats.append(dict(episode=episode, step=int(side["step"][i]), row=j, offset=i,
                                    visible_l1=float(error[masks["visible"]].mean()) if masks["visible"].any() else None,
                                    visible_dynamic_pixels=int(dynamic.sum()),
                                    dynamic_rgb_l1=float(error[dynamic].mean()) if dynamic.any() else None))
            previous = current
    for images, seg in episodes.values():
        for k, score in images.items():
            for region, values in score.sums.items():
                global_scores[0][k].sums[region] += values
        for k, score in seg.items():
            for region, matrix in score.matrix.items():
                global_scores[1][k].matrix[region] += matrix
    score = finish(global_scores)
    tests = {k: v / len(ds) for k, v in tests.items()}
    generated, baseline = score["generated"]["visible"], score["mean"]["visible"]
    iou, base_iou = score["generated_head"]["visible"]["iou"], score["modal"]["visible"]["iou"]
    def improved(label):
        return iou[label] > base_iou[label] if iou[label] is not None and base_iou[label] is not None else None
    criteria = dict(visible_rgb=generated["l1"] < baseline["l1"] or generated["ssim"] > baseline["ssim"],
                    road_iou=improved("road"), vehicle_iou=improved("vehicle"),
                    zero_and_shuffled=tests["correct_l1"] < min(tests["shuffled_l1"], tests["zero_l1"]))
    result = dict(split=args.split, frames=len(ds), scores=score, episodes={k: finish(s) for k, s in episodes.items()},
                  ablations=tests, image_and_semantic_criteria=criteria, generator_fp32_batch1=latency,
                  temporal_static_residual_l1=temporal_sum/temporal_n if temporal_n else None, temporal_rgb_samples=temporal_n,
                  elapsed_s=time.perf_counter() - tick, device=args.device,
                  cuda_peak_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                  collection_sha256=ds.collection["contract_sha256"], pilot_sha256=ds.cache["checkpoint_sha256"],
                  model_sha256=file_hash(args.ckpt))
    write_json(out / "metrics.json", result)
    write_json(out / "frames.json", frame_stats)
    print(json.dumps(dict(criteria=criteria, visible_l1=generated["l1"], mean_l1=baseline["l1"], ablations=tests)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "features", "ckpt", "recon", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluate(parser.parse_args())
