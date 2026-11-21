"""The future head on the validation frames, one look-ahead at a time (docs/future_aux.md, phase 2).

    uv run python tools/future_eval.py --ckpt runs/future_aux/fw03/last.pt [--out runs/future_aux/fw03/future_eval.json]

train.py's validation takes one look-ahead per frame (the one closest to 1 s that the frame has), so its numbers per
look-ahead are of different frames. Here every look-ahead is scored on all the validation frames that have it (the
checkpoint's data, split and look-aheads, from its args), against three predictions of the future segmentation:
  head        the future head
  persist     the frame's own segmentation label (nothing moves), at the label's resolution (80x160)
  persist_lr  the same at the head's resolution: the label averaged over 4x4 pixels per class (20x40) and raised
              again as the head's logits are. What a head that knew the present perfectly and nothing of the motion
              would score: the head beats it only by knowing where things go
  seg         (0 s only) the model's own segmentation of the frame: persist is its ceiling
IoU of vehicles and pedestrians and the mean over the classes there, for all frames and for the moving ones
(v >= 0.2 m/s).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from keipilot.data import STEP_S, DrivingDataset, future_index, load_shards  # noqa: E402
from keipilot.model import KeiPilot, channels_last  # noqa: E402
from keisim.config import N_SEM, SEM  # noqa: E402

MOVING_V = 0.2


def scores(c):
    inter = c.diag().float()
    iou = inter / (c.sum(0).float() + c.sum(1).float() - inter).clamp(min=1)
    return {"mIoU": iou[c.sum(1) > 0].mean().item(), "IoU_vehicle": iou[SEM["vehicle"]].item(),
            "IoU_pedestrian": iou[SEM["pedestrian"]].item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    if not ck["model_cfg"].get("aux_future"):
        sys.exit("a checkpoint without the future head")
    model = KeiPilot(pretrained=False, **ck["model_cfg"])
    model.load_state_dict(ck["model"])
    model = model.to(args.device).to(memory_format=torch.channels_last).eval()
    data, files = load_shards(a["data"], with_prev=False, mmap=True)
    seed = a["seed"] if a.get("split_seed") is None else a["split_seed"]
    eps = np.unique(data["episode"])
    np.random.default_rng(seed).shuffle(eps)
    is_val = np.isin(data["episode"], eps[: max(1, int(len(eps) * a["val_frac"]))]) & ~data["stale_creep"]
    fut, _ = future_index(data, files, a["future_horizons"])
    res = {"ckpt": args.ckpt, "horizons": a["future_horizons"], "val_frames": int(is_val.sum()), "per_horizon": {}}
    for j, h in enumerate(a["future_horizons"]):
        d = dict(data)
        d["fut_idx"], d["fut_dt"] = fut[:, [j]], np.array([h * STEP_S], np.float32)
        idx = np.nonzero(is_val & (fut[:, j] >= 0))[0]
        dl = DataLoader(DrivingDataset(d, idx, train=False, future=True), batch_size=args.bs, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
        conf = {}
        for b in dl:
            b = {k: v.to(args.device, non_blocking=True) for k, v in b.items()}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(channels_last(b["img"]), b["cmd"], b["tp"], speed=b["v"], stand_m=b["stand_m"],
                            stand_known=b["stand_known"], with_future=True, fut_dt=b["fut_dt"], fut_v=b["fut_v"])
            K = b["fut_seg"].shape[1] if b["fut_seg"].dim() == 4 else 1
            fs, cur = (b["fut_seg"].flatten(0, 1), b["seg"].flatten(0, 1)) if K > 1 else (b["fut_seg"], b["seg"])
            size = fs.shape[-2:]
            lr = F.avg_pool2d(F.one_hot(cur, N_SEM).permute(0, 3, 1, 2).float(), 4)
            preds = {"head": F.interpolate(out["future"].float(), size=size, mode="bilinear", align_corners=False).argmax(1),
                     "persist": cur,
                     "persist_lr": F.interpolate(lr, size=size, mode="bilinear", align_corners=False).argmax(1)}
            moving = (b["fut_v"] >= MOVING_V).repeat_interleave(K, 0)
            for who, p in preds.items():
                for name, m in (("all", torch.ones_like(moving)), ("moving", moving)):
                    c = torch.bincount((fs[m] * N_SEM + p[m]).flatten(), minlength=N_SEM * N_SEM).view(N_SEM, N_SEM)
                    conf[who, name] = conf.get((who, name), 0) + c.cpu()
            if j == 0:      # the present: the model's segmentation against the frame's own label
                p = F.interpolate(out["seg"].float(), size=size, mode="bilinear", align_corners=False).argmax(1)
                c = torch.bincount((cur * N_SEM + p).flatten(), minlength=N_SEM * N_SEM).view(N_SEM, N_SEM)
                conf["seg", "now"] = conf.get(("seg", "now"), 0) + c.cpu()
        r = {"frames": int(len(idx)), "moving_frames": int((data["v"][idx] >= MOVING_V).sum())}
        for (who, name), c in conf.items():
            r.setdefault(name, {})[who] = scores(c)
        res["per_horizon"][f"{h * STEP_S:.1f}s"] = r
        print(f"{h * STEP_S:.1f} s ({len(idx)} frames, {r['moving_frames']} moving)")
        for name in ("all", "moving"):
            print(f"  {name:6s} " + " | ".join(f"{who}: mIoU {x['mIoU']:.3f} vehicle {x['IoU_vehicle']:.3f} pedestrian "
                                              f"{x['IoU_pedestrian']:.3f}" for who, x in r[name].items()))
        if "now" in r:
            x = r["now"]["seg"]
            print(f"  the model's segmentation of the frame itself: mIoU {x['mIoU']:.3f} vehicle {x['IoU_vehicle']:.3f} "
                  f"pedestrian {x['IoU_pedestrian']:.3f}")
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "future_eval.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
