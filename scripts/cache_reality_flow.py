#!/usr/bin/env python3
"""Materialize a frozen Phase 2 forward pass for each training/validation or held-out tick and camera."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, load_manifest, write_json, writer_lock
from keireality.flow_data import phase2_condition
from keireality.geometry import ORBIT_ANGLES
from keireality.multiview import MultiViewFrames
from eval_reality_temporal import load_renderer


@torch.inference_mode()
def cache(args):
    torch.set_num_threads(2)
    collection = load_manifest(args.data)
    fm = json.loads((args.features / "manifest.json").read_text())
    model = load_renderer(args.renderer, fm["checkpoint_sha256"], args.device).requires_grad_(False)
    with writer_lock(args.out):
        for split in args.splits:
            angles = ORBIT_ANGLES if split=="test" else (0,-36,36)
            dest = args.out / split
            contract = dict(version=1, split=split, angles=list(angles), collection=collection["contract_sha256"],
                            renderer=file_hash(args.renderer), features=file_hash(args.features/"manifest.json"),
                            pilot=fm["checkpoint_sha256"], shards=[r["path"] for r in collection["shards"] if r["split"]==split])
            reference = None
            if split=="test" and args.spatial_reference:
                contract["chase_reference"] = file_hash(args.spatial_reference)
                reference = load_renderer(args.spatial_reference,fm["checkpoint_sha256"],args.device).requires_grad_(False)
            if dest.exists():
                m = json.loads((dest/"manifest.json").read_text())
                if any(m.get(k)!=v for k,v in contract.items()) or any(file_hash(dest/(k+".npy"))!=v for k,v in m["arrays"].items()):
                    raise ValueError("incompatible or damaged frozen condition cache")
                print("Verified", split, flush=True)
                continue
            ds = MultiViewFrames(args.data, args.features, split, tuple(ORBIT_ANGLES.index(a) for a in angles),
                                 temporal=True, teachers=False)
            start = time.perf_counter()
            with tempfile.TemporaryDirectory(dir=args.out,prefix=".writing-") as tmp:
                stage = Path(tmp)/"split"
                stage.mkdir()
                arrays = {}
                def output(name, shape):
                    f = np.lib.format.open_memmap(stage/(name+".npy"),mode="w+",dtype=np.float16,
                                                 shape=(len(ds.index),len(angles),*shape))
                    arrays[name] = f
                    return f.reshape(len(ds),*shape)
                cond = output("current",(28,64,128))
                native = output("native",(3,128,256)) if split=="test" else None
                if reference is not None:
                    chase = np.lib.format.open_memmap(stage/"chase.npy",mode="w+",dtype=np.float16,
                                                      shape=(len(ds.index),3,128,256))
                    arrays["chase"] = chase
                pos = 0
                for batch in DataLoader(ds,batch_size=8):
                    b = {k:batch[k].to(args.device) for k in ("mem","c2","c3","past","motion","valid","view_pose")}
                    current, rgb = phase2_condition(model,b)
                    if not torch.isfinite(current).all() or not torch.isfinite(rgb).all():
                        raise ValueError("invalid Phase 2 render")
                    end = pos+len(rgb)
                    cond[pos:end] = current.half().cpu().numpy()
                    if native is not None:
                        native[pos:end] = rgb.half().cpu().numpy()
                    if reference is not None:
                        selected = np.flatnonzero(np.arange(pos,end)%len(angles)==0)
                        if len(selected):
                            jj = torch.as_tensor(selected,device=args.device)
                            ref = reference(*[b[k][jj] for k in ("mem","view_pose","c2","c3")])[0]
                            chase[(pos+selected)//len(angles)] = ref.half().cpu().numpy()
                    pos = end
                if pos != len(ds):
                    raise ValueError("incomplete split")
                for f in arrays.values(): f.flush()
                hashes = {k:file_hash(stage/(k+".npy")) for k in arrays}
                write_json(stage/"manifest.json",dict(**contract,ticks=len(ds.index),arrays=hashes,seconds=time.perf_counter()-start))
                stage.rename(dest)
            print(f"{split}: {len(ds.index)} ticks, {len(ds)} conditions",flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("data","features","renderer","out"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--splits",nargs="+",choices=("train","val","test"),required=True)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--spatial-reference",type=Path,help="optional fixed-chase reference for the test split")
    cache(p.parse_args())
