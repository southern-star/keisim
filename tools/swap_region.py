"""Swap image regions of one saved input (.npz from tools/replay_kv_freeze.py or tools/replay_hit_grid.py) for
another's, e.g. the frame before a model's decision flipped, and rerun checkpoints: which change flips it.
Boxes are in camera pixels (x0 x1 y0 y1). Prints the raw target, the stop probability and the light head.
  python swap_region.py ckpt[,ckpt] base.npz donor.npz cam x0 x1 y0 y1 [cam x0 x1 y0 y1 ...]   (fork dir)"""
import sys
import numpy as np
import torch
from keipilot.agent import KeiPilotAgent

ckpts, base, donor = sys.argv[1].split(","), np.load(sys.argv[2]), np.load(sys.argv[3])
boxes = [tuple(int(v) for v in sys.argv[i:i + 5]) for i in range(4, len(sys.argv), 5)]


def run(agent, kv):
    b = torch.from_numpy(np.ascontiguousarray(kv[None][..., ::-1].transpose(0, 1, 4, 2, 3))).to(agent.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=agent.amp):
        o = agent.model(b, torch.tensor([int(base["command"])], device=agent.device),
                        torch.tensor(base["target_point"][None], device=agent.device), with_seg=False,
                        speed=torch.tensor([float(base["speed"])], device=agent.device),
                        stand_m=torch.tensor([float(base["stand_m"])], device=agent.device))
    v, _ = agent.model.decode_speed(o["speed_logits"].float())
    tl = o["tl_logits"].float().softmax(-1)[0].cpu().numpy()
    return v.item() * 3.6, o["speed_logits"].float().softmax(-1)[0, 0].item(), tl


for c in ckpts:
    agent = KeiPilotAgent(c)
    kv = base["kv"].copy()
    print("##", c)
    v, p, tl = run(agent, kv)
    print(f"  base          {v:5.1f} km/h p_stop {p:.2f} tl {np.round(tl, 2)}")
    for cam, x0, x1, y0, y1 in boxes:
        x = base["kv"].copy()
        x[cam, y0:y1, x0:x1] = donor["kv"][cam, y0:y1, x0:x1]
        v, p, tl = run(agent, x)
        print(f"  cam {cam} x {x0}-{x1} y {y0}-{y1} from donor: {v:5.1f} km/h p_stop {p:.2f} tl {np.round(tl, 2)}")
    v, p, tl = run(agent, donor["kv"])
    print(f"  donor         {v:5.1f} km/h p_stop {p:.2f} tl {np.round(tl, 2)}")
