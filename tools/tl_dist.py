"""Traffic-light state accuracy of a checkpoint on held-out shards, by the distance to the light's stop line (lt_d):
which distances a camera rig loses.  python tl_dist.py ckpt data_dir [more dirs]"""
import json, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from keipilot.agent import load_model
from keipilot.data import DrivingDataset, load_shards
from keipilot.model import channels_last

ckpt, dirs = sys.argv[1], sys.argv[2:]
d, _ = load_shards(dirs)
n = len(d["cmd"])
ds = DrivingDataset(d, np.arange(n), train=False)
dl = DataLoader(ds, batch_size=128, num_workers=4)
model = load_model(ckpt).cuda().eval()
pred = []
with torch.no_grad():
    for b in dl:
        b = {k: v.cuda() for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(channels_last(b["img"]), b["cmd"], b["tp"], speed=b["v"], speed_known=b["v_known"],
                        img_prev=b["img_prev"], has_prev=b["has_prev"], stand_m=b["stand_m"],
                        stand_known=b["stand_known"])
        pred.append(out["tl_logits"].argmax(-1).cpu().numpy())
pred = np.concatenate(pred)
gt = np.asarray(d["tl"])[:n]
lt_d = np.asarray(d["lt_d"], float)[:n] if "lt_d" in d else np.full(n, np.nan)
res = {"frames": int(n), "tl_acc": float((pred == gt).mean())}
bins = [(-5, 3), (3, 10), (10, 20), (20, 35), (35, 60), (60, 1e9)]
for lo, hi in bins:
    m = (lt_d >= lo) & (lt_d < hi) & (gt < 3)
    if m.sum():
        res[f"{lo}-{hi if hi < 1e8 else 'inf'} m"] = {"n": int(m.sum()), "acc": round(float((pred[m] == gt[m]).mean()), 4),
                                    "red_as_green": int(((gt == 0) & (pred == 2) & m).sum())}
m = gt == 3
res["no light"] = {"n": int(m.sum()), "acc": round(float((pred[m] == 3).mean()), 4)}
print(json.dumps(res, indent=1))
