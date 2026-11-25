#!/usr/bin/env python3
"""Train the conditional flow on contiguous windows, using only the network's own previous images for feedback."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keireality.dataset import file_hash, write_json, writer_lock
from keireality.flow_data import FlowSplit
from keireality.flow_model import ConditionalVelocity, FlowRollout, frame_noise, make_condition
from keireality.meanflow import broadcast_time, meanflow_terms
from keisim.config import SEM
from train_reality import save_checkpoint


def resume_checkpoint(path, device):
    # Early run logs used NumPy float64 scalars. Keep the tensor-only loader for old and new snapshots.
    try:
        from numpy._core.multiarray import scalar
    except ImportError:
        from numpy.core.multiarray import scalar
    with torch.serialization.safe_globals([scalar, np.dtype, type(np.dtype("float64"))]):
        return torch.load(path,map_location=device,weights_only=True)


def training_windows(ds, cfg):
    length = cfg["window"]
    blocks = []
    for ep in ds.episodes:
        if ep.length < length:
            raise ValueError("training window exceeds episode length")
        starts = sorted(set(range(0,ep.length-length+1,length)) | {ep.length-length})
        blocks.extend((ep.start+s,v) for s in starts for v in range(len(ds.angles)))
    return np.asarray(blocks)


def select_window(blocks, number, cfg, views):
    bs, length, seed = cfg["batch_size"],cfg["window"],cfg["seed"]
    positions = np.arange(number*bs,(number+1)*bs)
    epoch = positions // len(blocks)
    chosen = np.empty((bs,2),dtype=int)
    for e in np.unique(epoch):
        order = np.random.default_rng([seed, int(e), 31]).permutation(len(blocks))
        chosen[epoch==e] = blocks[order[positions[epoch==e] % len(blocks)]]
    angles = np.tile(chosen[:,1], (length,1))
    rng = np.random.default_rng([seed,number,33])
    for i in range(1,length):
        switch = (epoch>0) & (rng.random(bs)<cfg["switch_view_probability"])
        angles[i] = np.where(switch,rng.integers(views,size=bs),angles[i-1])
    return chosen[:,0], angles


@torch.no_grad()
def validation(field, ds, mode, nfes, seed):
    field.eval()
    device = ds.current.device
    totals = {n:torch.zeros((),device=device) for n in nfes}
    frames = 0
    for ep in ds.episodes:
        streams = {n:FlowRollout(field,n,mode) for n in nfes}
        for i in range(ep.start,ep.start+ep.length):
            noise = frame_noise(ep.seed,ds.steps[i],ds.angles,seed,device)
            for n,stream in streams.items():
                image = stream.render(ds.current[i],ds.poses[i],ds.delta[i].expand(len(noise),-1),
                                      ep.name,ds.times[i],noise)
                totals[n] += (image-ds.target[i].float()).abs().mean()*.5
            frames += 1
    field.train()
    return {str(n):float(value)/frames for n,value in totals.items()}


def lr_at(i, cfg):
    warm = cfg["warmup_steps"]
    if i<=warm:
        return cfg["learning_rate"]*i/warm
    x = (i-warm)/(cfg["updates"]-warm)
    return cfg["end_learning_rate"] + (cfg["learning_rate"]-cfg["end_learning_rate"])*(.5+.5*math.cos(math.pi*x))


def train(args):
    cfg = json.loads(args.config.read_text())
    seed, bs, length = cfg["seed"], cfg["batch_size"], cfg["window"]
    mode = args.mode
    updates = cfg["updates"] if mode=="meanflow" else cfg["flow_matching_control_updates"]
    if cfg["checkpoint_every"] % length or updates % length or cfg["validation_every"] % cfg["checkpoint_every"]:
        raise ValueError("checkpoints must end on whole training windows")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    ds,val = [FlowSplit(args.data,args.cache,s,device) for s in ("train","val")]
    if (ds.angles!=cfg["train_angles"] or val.angles!=ds.angles or ds.meta["renderer"]!=val.meta["renderer"]
            or {x.town for x in ds.episodes} & {x.town for x in val.episodes}):
        raise ValueError("invalid training/validation pairing")
    net = ConditionalVelocity(cfg["prior_sigma"]).to(device=device,memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    contract = dict(protocol=file_hash(args.config),mode=mode,updates=updates,renderer=ds.meta["renderer"],
                    train=file_hash(ds.cache/"manifest.json"),val=file_hash(val.cache/"manifest.json"),
                    parameters=sum(p.numel() for p in net.parameters()),torch=str(torch.__version__),device=str(device))
    blocks = training_windows(ds,cfg)
    with writer_lock(args.out):
        last = args.out/"last.pt"
        config_file = args.out/"config.json"
        if config_file.exists() and json.loads(config_file.read_text()) != contract:
            raise ValueError("incompatible training run")
        write_json(config_file,contract)
        step, best, best_step, elapsed = 0, float("inf"), 0, 0.
        log = []
        seen = np.zeros((len(ds.current),len(ds.angles)),bool)
        if last.exists():
            ck = resume_checkpoint(last,device)
            if ck["contract"] != contract:
                raise ValueError("incompatible training checkpoint")
            net.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            step,best,best_step,log,elapsed = (ck[k] for k in ("step","best","best_step","log","elapsed"))
            seen[:] = ck["seen"].cpu().numpy()
            del ck
            if step==updates and (args.out/"training.json").exists() and (args.out/"best.pt").exists():
                print("Verified finished training run",flush=True)
                return
        tick = time.perf_counter()
        sums = np.zeros(3)
        for k in range(step,updates):
            pos = k%length
            if pos==0:
                starts, cameras = select_window(blocks,k//length,cfg,len(ds.angles))
                previous = torch.zeros(bs,3,64,128,device=device)
                old_view = torch.zeros(bs,11,device=device)
            ii, vv = starts+pos,cameras[pos]
            seen[ii,vv] = True
            idx = torch.as_tensor(ii,device=device)
            vi = torch.as_tensor(vv,device=device)
            rng = torch.Generator(device=device).manual_seed(seed*1000003+k)
            good = (torch.rand(bs,device=device,generator=rng)>=cfg["previous_dropout"]) & (pos!=0)
            view = ds.poses[idx,vi]
            cond = make_condition(ds.current[idx,vi],view,previous,old_view,ds.delta[idx],view.new_full((bs,),.1),good)
            x = ds.target[idx,vi].float()
            noise = torch.randn(x.shape,device=device,generator=rng)
            pair = torch.rand(2,bs,device=device,generator=rng)
            r,t = pair.amin(0),pair.amax(0)
            draw = torch.rand(bs,device=device,generator=rng)
            r = torch.where(draw<cfg["equal_times_probability"],t,r)
            endpoint = draw>1-cfg["endpoint_probability"]
            r,t = torch.where(endpoint,0.,r),torch.where(endpoint,1.,t)
            if mode=="meanflow":
                pred,target,_ = meanflow_terms(net,x,noise,r,t,cond)
            else:
                target = noise-x
                pred = net(x+broadcast_time(t,x)*target,t,t,cond)
            flow_loss = (pred-target).square().mean()
            fresh = torch.randn(x.shape,device=device,generator=rng)
            one = torch.ones_like(t)
            image = fresh-net(fresh,torch.zeros_like(t) if mode=="meanflow" else one,one,cond)
            weight = torch.where(ds.seg[idx,vi]>=SEM["vehicle"],cfg["endpoint_dynamic_pixel_weight"],1.)
            fidelity = ((image-x).abs().mean(1)*weight).sum()/weight.sum()
            loss = flow_loss+cfg["endpoint_l1_weight"]*fidelity
            opt.param_groups[0]["lr"] = lr_at(k+1,cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(),cfg["gradient_clip"],error_if_nonfinite=True)
            opt.step()
            # The prediction for this exact tick becomes the following tick's condition, detached and display-clamped.
            previous,old_view = image.detach().clamp(-1,1),view.detach()
            sums += [float(flow_loss.detach()),float(fidelity.detach()),float(grad)]
            step = k+1
            if step%100==0:
                print(f"{mode} {step}/{updates}, flow {sums[0]/100:.4f}, endpoint {sums[1]/100:.4f}",flush=True)
                log.append(dict(step=step,flow=float(sums[0]/100),endpoint=float(sums[1]/100),
                                gradient_norm=float(sums[2]/100)))
                sums[:] = 0
            if step%cfg["validation_every"]==0 or step==updates:
                nfes = cfg["control_validation_nfe"] if step==cfg["flow_matching_control_updates"] else cfg["test_nfe"]
                scores = validation(net,val,mode,nfes,cfg["validation_seed"])
                log[-1]["validation_l1"] = scores
                score = (scores["1"]+scores["4"])/2
                state = dict(model=net.state_dict(),step=step,contract=contract,sigma=cfg["prior_sigma"])
                if score<best:
                    best,best_step=score,step
                    save_checkpoint(args.out/"best.pt",state)
                if step==cfg["flow_matching_control_updates"]:
                    save_checkpoint(args.out/"matched.pt",state)
                print("validation",step,scores,flush=True)
            if step%cfg["checkpoint_every"]==0 or step==updates:
                duration = elapsed+time.perf_counter()-tick
                save_checkpoint(last,dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step,best=best,
                                         best_step=best_step,elapsed=duration,contract=contract,log=log,
                                         seen=torch.from_numpy(seen.copy())))
                write_json(args.out/"progress.json",dict(step=step,elapsed_s=duration,log=log,seen=int(seen.sum())))
        report = dict(**contract,selected_step=best_step,validation_l1=best,train_ticks=len(ds.current),
                      unique_pairs_seen=int(seen.sum()),available_pairs=seen.size,examples=updates*bs,
                      elapsed_s=elapsed+time.perf_counter()-tick,selected_sha256=file_hash(args.out/"best.pt"),
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else None,
                      validation=[r for r in log if "validation_l1" in r])
        write_json(args.out/"training.json",report)
        print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("config","data","cache","out"): p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--mode",choices=("meanflow","flow_matching"),default="meanflow")
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    train(p.parse_args())
