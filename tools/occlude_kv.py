"""Occlusion test for a KeiView phantom, on the model inputs tools/replay_kv_freeze.py saved (out_prefix_T.npz).
Runs the rig model on the KeiView images, on KeiSim's images of the same state, on KeiView with one camera swapped
for KeiSim's, and with a grey patch slid over each KeiView image. The patches that raise the model's speed target
the most show where it sees what stops it. Prints the raw target (before the safety layers), yield_tta and the light
head's red+yellow for each case and the strongest patches; saves out.png: each KeiView image next to the same image
tinted by the target gain where it is covered (red: covering this lets the car go).
  python occlude_kv.py ckpt inputs.npz out.png [patch [stride]]   (fork dir)"""
import sys
import cv2
import numpy as np
import torch
from keipilot.agent import KeiPilotAgent

ckpt, src, out = sys.argv[1:4]
P = int(sys.argv[4]) if len(sys.argv) > 4 else 32
S = int(sys.argv[5]) if len(sys.argv) > 5 else P // 2
d = np.load(src)
agent = KeiPilotAgent(ckpt)
kv, ks = d["kv"], d["ks"]               # (K, H, W, 3) BGR
K, H, W, _ = kv.shape


@torch.no_grad()
def run(imgs):
    """imgs (N, K, H, W, 3) BGR -> (N, 3): raw speed target m/s, yield_tta, red+yellow."""
    res, dev = [], agent.device
    for i in range(0, len(imgs), 64):
        b = torch.from_numpy(np.ascontiguousarray(imgs[i:i + 64][..., ::-1].transpose(0, 1, 4, 2, 3))).to(dev)
        n = len(b)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=agent.amp):
            o = agent.model(b, torch.full((n,), int(d["command"]), device=dev),
                            torch.tensor(np.repeat(d["target_point"][None], n, 0), device=dev), with_seg=False,
                            speed=torch.full((n,), float(d["speed"]), device=dev),
                            stand_m=torch.full((n,), float(d["stand_m"]), device=dev))
        v, _ = agent.model.decode_speed(o["speed_logits"].float())
        tl = o["tl_logits"].float().softmax(-1)
        tta = o["yield_tta"].float().reshape(n) if "yield_tta" in o else torch.full((n,), float("nan"))
        res.append(torch.stack([v, tta.to(v.device), tl[:, 0] + tl[:, 1]], 1).cpu().numpy())
    return np.concatenate(res)


def show(name, r):
    print(f"{name:28s} target {r[0] * 3.6:5.1f} km/h  yield_tta {r[1]:4.1f}  red+yellow {r[2]:.2f}")


print(f"{src}: speed {float(d['speed']) * 3.6:.1f} km/h, stand_m {float(d['stand_m']):.1f}, command {int(d['command'])}, "
      f"{K} cameras {W}x{H}, patch {P} stride {S}")
base = run(kv[None])[0]
show("KeiView", base)
show("KeiSim", run(ks[None])[0])
for k in range(K):
    x = kv.copy()
    x[k] = ks[k]
    show(f"KeiView, camera {k} KeiSim's", run(x[None])[0])
rows, top = [], []
for k in range(K):
    ys, xs = range(0, H - P + 1, S), range(0, W - P + 1, S)
    fill = kv[k].reshape(-1, 3).mean(0).astype(np.uint8)
    batch = []
    for y in ys:
        for x in xs:
            im = kv.copy()
            im[k, y:y + P, x:x + P] = fill
            batch.append(im)
    r = run(np.stack(batch))
    gain, cnt = np.zeros((H, W)), np.zeros((H, W))
    for j, (y, x) in enumerate((y, x) for y in ys for x in xs):
        gain[y:y + P, x:x + P] += r[j, 0] - base[0]
        cnt[y:y + P, x:x + P] += 1
        top.append((r[j, 0] - base[0], k, x, y, r[j]))
    gain /= np.maximum(cnt, 1)
    a = np.clip(gain * 3.6 / 10.0, 0, 1)[..., None]           # 10 km/h gain = full red
    tint = (kv[k] * (1 - 0.7 * a) + np.array([0, 0, 255]) * 0.7 * a).astype(np.uint8)
    rows.append(np.concatenate([kv[k], tint], 1))
top.sort(key=lambda t: -t[0])
for g, k, x, y, r in top[:8]:
    show(f"  patch cam {k} x {x:3d} y {y:3d}", r)
im = np.concatenate(rows, 0)
cv2.imwrite(out, cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
print("saved", out)
