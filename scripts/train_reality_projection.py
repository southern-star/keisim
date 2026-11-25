#!/usr/bin/env python3
"""Train only the predicted-depth feature projection from paired consecutive training frames."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.depth_projection import DepthProjectionBranch
from keireality.geometry import ORBIT_ANGLES
from keisim.config import SEM
from prepare_reality_projection import checked_fields
from train_reality import save_checkpoint


class TrainingPairs(Dataset):
    def __init__(self, data, features, cache, cfg, protocol_hash):
        dest = cache / "train"
        meta = json.loads((dest / "manifest.json").read_text())
        self.contract = meta["contract"]
        raw = load_manifest(data)
        fm = json.loads((features / "manifest.json").read_text())
        rows = [r for r in raw["shards"] if r["split"] == "train"]
        if (self.contract["collection"] != raw["contract_sha256"] or self.contract["features"] != file_hash(features/"manifest.json")
                or self.contract["protocol"] != protocol_hash or self.contract["rows"] != [r["path"] for r in rows]
                or self.contract["angles"] != cfg["train_angles"] or self.contract["split"] != "train"):
            raise ValueError("reference and training data differ")
        for name, sha in meta["files"].items():
            if file_hash(dest / name) != sha:
                raise ValueError(f"corrupt training reference {name}")
        self.base, self.depth, self.warp = [np.load(dest / (name+".npy"), mmap_mode="r")
                                            for name in ("base", "estimated", "training_warp")]
        self.views = [ORBIT_ANGLES.index(a) for a in cfg["train_angles"]]
        self.inputs, self.index, self.previous = [], [], []
        prior = None
        for row in rows:
            arrays = checked_fields(data, row, ("step","view_pose","rgb_target","seg_target","visibility_now","rgb_in","seg_in"))
            for key in ("c2", "c3"):
                path = features / row["path"] / (key+".npy")
                if file_hash(path) != fm["shards"][row["path"]]["pyramid"][key]:
                    raise ValueError(f"feature checksum mismatch {path}")
                arrays[key] = np.load(path, mmap_mode="r")
            for j, step in enumerate(arrays["step"]):
                i = len(self.index)
                self.previous.append(i-1 if prior == (row["episode_id"], step-1) else i)
                self.index.append((len(self.inputs), j))
                prior = row["episode_id"], step
            self.inputs.append(arrays)
        if len(self.index) != len(self.base):
            raise ValueError("reference length mismatch")

    def __len__(self):
        return len(self.index) * len(self.views)

    def __getitem__(self, item):
        t, v = divmod(item, len(self.views))
        ids = [self.previous[t], t]
        view = self.views[v]
        pair = [(self.inputs[k], j) for k,j in (self.index[i] for i in ids)]
        def take(name, *key):
            return np.stack([a[name][(j, *key)] for a,j in pair])
        return dict(c2=take("c2"), c3=take("c3"), depth=self.depth[ids], base=self.base[ids, v],
                    view=take("view_pose", view), truth=take("rgb_target", view), seg=take("seg_target", view),
                    visible=take("visibility_now", view), warp=self.warp[t,v].copy(),
                    source=take("rgb_in", slice(None), slice(None,None,2), slice(None,None,2)),
                    source_seg=take("seg_in", slice(None), slice(None,None,2), slice(None,None,2)))


def masked(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def projection_loss(branch, batch, edge_threshold):
    warp = batch["warp"]
    b = {k:x.flatten(0, 1) for k,x in batch.items() if k != "warp"}
    truth = b["truth"].permute(0,3,1,2).float() / 255
    car = b["seg"] == SEM["vehicle"]
    visible = b["visible"] == 1
    encoded = branch.encode(b["c2"], b["c3"])
    pred = branch.render_encoded(b["base"], encoded, b["view"], b["depth"])
    error = pred - truth
    photo = error.abs().mean(1)
    source = b["source"].permute(0,1,4,2,3).float() / 255
    source_error = (encoded[:,:,:3].float() - source).abs().mean(2)
    source_loss = source_error.mean() + masked(source_error, b["source_seg"] == SEM["vehicle"])
    edge_loss = pred.new_zeros(())
    for axis in (-2, -1):
        n = car.size(axis)-1
        a, c = car.narrow(axis,0,n), car.narrow(axis,1,n)
        delta = truth.diff(dim=axis)
        visible_pair = visible.narrow(axis,0,n) & visible.narrow(axis,1,n)
        strong = visible_pair & (delta.abs().mean(1) >= edge_threshold)
        difference = (pred.diff(dim=axis) - delta).abs().mean(1)
        edge_loss = edge_loss + masked(difference, strong & (a ^ c)) + .35*masked(difference, strong & (a | c))
    previous, current = error.unflatten(0, (-1, 2)).unbind(1)
    prev_idx = (warp.long().flatten(1).clamp_min(1)-1)[:,None].expand(-1,3,-1)
    warped = previous.flatten(2).gather(2, prev_idx).reshape_as(current)
    flicker = masked((current - warped).abs().mean(1), warp > 0)
    vacated = masked(current.abs().mean(1), warp < 0)
    return (photo.mean() + 1.25*masked(photo, car & visible) + .45*edge_loss +
            4.5*flicker + 1.25*vacated + .45*source_loss)


def train(a):
    cfg = json.loads(a.config.read_text())
    sha = file_hash(a.config)
    torch.set_num_threads(2)
    torch.manual_seed(cfg["seed"])
    ds = TrainingPairs(a.data, a.features, a.cache, cfg, sha)
    net = DepthProjectionBranch().to(a.device, memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler(a.device, enabled=a.device=="cuda")
    steps_per_epoch = math.ceil(len(ds)/cfg["batch_pairs"])
    total = cfg["epochs"]*steps_per_epoch
    contract = dict(protocol=sha, training_cache=file_hash(a.cache/"train/manifest.json"), phase2=ds.contract["phase2"],
                    depth_head=ds.contract["depth_head"], pilot=ds.contract["pilot"],
                    architecture=file_hash(ROOT/"keireality/depth_projection.py"), trainer=file_hash(Path(__file__)))
    with writer_lock(a.out):
        checkpoint = a.out/"last.pt"
        step, history = 0, []
        if checkpoint.exists():
            ck = torch.load(checkpoint, map_location=a.device)
            report_path = a.out/"training.json"
            if report_path.exists():
                # Re-serializing identical tensor values can change the digest used by downstream evaluations.
                # A completed run keeps the trainer identity recorded at the time it actually ran.
                saved = json.loads(report_path.read_text())
                published = a.out/"model.pt"
                recorded = ck["contract"]
                compatible = all(recorded[k] == v for k,v in contract.items() if k != "trainer")
                if (not compatible or saved["contract"] != recorded or ck["step"] != total or saved["steps"] != total
                        or saved["model"] != file_hash(published)):
                    raise ValueError("published projection experiment does not match its inputs or checkpoint")
                final = torch.load(published, map_location=a.device)
                if (final["contract"] != recorded or final["step"] != total or ck["model"].keys() != final["model"].keys()
                        or any(not torch.equal(v,final["model"][k]) for k,v in ck["model"].items())):
                    raise ValueError("published candidate differs from the final training state")
                print(f"Verified completed candidate, {total} updates, sha256 {saved['model']}", flush=True)
                return
            if ck["contract"] != contract:
                raise ValueError("checkpoint and training inputs differ")
            net.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            scaler.load_state_dict(ck["scaler"])
            step, history = ck["step"], ck["history"]
        (a.out/"trainer-used.py").write_bytes(Path(__file__).read_bytes())
        write_json(a.out/"protocol.json", dict(config=cfg, contract=contract, pairs=len(ds), steps=total))
        for epoch in range(len(history), cfg["epochs"]):
            timer = time.perf_counter()
            loader = DataLoader(ds, batch_size=cfg["batch_pairs"], shuffle=True, pin_memory=a.device=="cuda",
                                generator=torch.Generator().manual_seed(cfg["seed"]+epoch))
            running = 0.
            for batch in loader:
                batch = {k:v.to(a.device, non_blocking=True) for k,v in batch.items()}
                ratio = step/max(total-1,1)
                lr = cfg["end_learning_rate"] + .5*(cfg["learning_rate"]-cfg["end_learning_rate"])*(1+math.cos(math.pi*ratio))
                opt.param_groups[0]["lr"] = lr
                opt.zero_grad(set_to_none=True)
                with torch.autocast(a.device, enabled=a.device=="cuda", dtype=torch.bfloat16):
                    loss = projection_loss(net, batch, cfg["edge_threshold"])
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite objective at {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
                scaler.step(opt)
                scaler.update()
                step += 1
                running += loss.item()
                if step % 100 == 0:
                    print(f"step {step}/{total}, loss {loss.item():.5f}", flush=True)
            history.append(dict(epoch=epoch+1, step=step, loss=running/len(loader), seconds=time.perf_counter()-timer))
            save_checkpoint(checkpoint, dict(model=net.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                                            step=step, history=history, contract=contract))
            print(history[-1], flush=True)
        save_checkpoint(a.out/"model.pt", dict(model=net.state_dict(), contract=contract, step=step))
        write_json(a.out/"training.json", dict(contract=contract, steps=step, history=history,
                   ticks=len(ds.index), views=len(ds.views), parameters=sum(p.numel() for p in net.parameters()),
                   model=file_hash(a.out/"model.pt"), cuda_peak_bytes=torch.cuda.max_memory_allocated() if a.device=="cuda" else None))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key, default in dict(config="configs/reality/p2_depth_projection.json", data="data/reality/p2",
                             features="data/reality/p2_spatial", cache="data/reality/p2_projection",
                             out="runs/reality/p2_projection/train").items():
        p.add_argument("--"+key, type=Path, default=Path(default))
    p.add_argument("--device", choices=("cuda","cpu"), default="cuda")
    train(p.parse_args())
