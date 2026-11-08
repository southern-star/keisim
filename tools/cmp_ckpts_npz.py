"""Raw speed target, stop probability and light head of several checkpoints on saved model inputs (.npz from
tools/replay_kv_freeze.py or tools/replay_hit_grid.py), with the saved speed / stand_m and with stand_m set to other
values: is a decision new to a model, and does the standing input carry it?
  python cmp_ckpts_npz.py ckpt[,ckpt...] a.npz [b.npz ...]   (fork dir)"""
import sys
import numpy as np
import torch
from keipilot.agent import KeiPilotAgent

ckpts, files = sys.argv[1].split(","), sys.argv[2:]
for c in ckpts:
    agent = KeiPilotAgent(c)
    print("##", c)
    for f in files:
        d = np.load(f)
        cells = []
        for sm in (float(d["stand_m"]), 3.0, 10.0, 30.0):
            b = torch.from_numpy(np.ascontiguousarray(d["kv"][None][..., ::-1].transpose(0, 1, 4, 2, 3))).to(agent.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=agent.amp):
                o = agent.model(b, torch.tensor([int(d["command"])], device=agent.device),
                                torch.tensor(d["target_point"][None], device=agent.device), with_seg=False,
                                speed=torch.tensor([float(d["speed"])], device=agent.device),
                                stand_m=torch.tensor([sm], device=agent.device))
            v, _ = agent.model.decode_speed(o["speed_logits"].float())
            p0 = o["speed_logits"].float().softmax(-1)[0, 0]
            cells.append(f"stand_m {sm:4.1f}: {v.item() * 3.6:5.1f} km/h p_stop {p0.item():.2f}")
        print(f"  {f.split('/')[-1]:24s} " + " | ".join(cells))
