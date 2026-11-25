import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from keireality.dataset import file_hash, json_hash, write_json, write_shard
from keireality.flow_data import Episode, FlowSplit
from keireality.flow_metrics import g3_decision, projected_scores
from keireality.flow_model import ConditionalVelocity, FlowRollout, frame_noise, make_condition
from keireality.meanflow import sample_meanflow
from keireality.multiview import MultiViewFrames
from keireality.rgb_probe import probe_scores
from keireality.temporal_metrics import Totals
from keisim.config import N_SEM, SEM


class EchoField(nn.Module):
    def forward(self,z,r,t,c):
        return z-(c[:,:3]*.3+c[:,28:31]*.4+c[:,31:32]*.05)


def test_stream_feedback_resets_rewinds_and_multiple_views():
    stream=FlowRollout(EchoField(),1)
    current=torch.ones(1,28,64,128)
    camera=torch.ones(1,11)
    delta=torch.zeros(1,3)
    noise=frame_noise(8,0,[0],77,"cpu")
    first=stream.render(current,camera,delta,"a",10.,noise)
    assert not stream.used_previous
    second=stream.render(current,camera,delta,"a",10.1,noise)
    assert stream.used_previous
    torch.testing.assert_close(second,first*.4+.35)
    repeat=stream.render(current,-camera,delta,"a",10.1,noise)
    torch.testing.assert_close(repeat,second)
    for ep,t in (("a",10.4),("b",10.5),("b",8.)):
        torch.testing.assert_close(stream.render(current,camera,delta,ep,t,noise),first)
        assert not stream.used_previous
    mixed=frame_noise(8,0,[18,0],77,"cpu")
    assert torch.equal(noise[0],mixed[1]) and not torch.equal(mixed[0],mixed[1])
    assert not torch.equal(noise,frame_noise(8,1,[0],77,"cpu"))


def test_invalid_previous_never_reaches_field():
    cur=torch.randn(2,28,8,16); view=torch.randn(2,11)
    prev=torch.randn(2,3,8,16); old=torch.randn(2,11); delta=torch.randn(2,3)
    keep=torch.tensor([False,True])
    a=make_condition(cur,view,prev,old,delta,torch.full((2,),.1),keep)
    b=make_condition(cur,view,prev+5,old+20,delta+4,torch.full((2,),.6),keep)
    assert a.shape==(2,59,8,16) and torch.equal(a[0],b[0]) and not torch.equal(a[1],b[1])


@pytest.mark.parametrize("nfe",[1,4])
@torch.no_grad()
def test_gaussian_velocity_prior_is_exact_for_every_interval(nfe):
    torch.set_num_threads(2)
    model=ConditionalVelocity(.18).eval()
    c=torch.randn(1,59,8,16); z=torch.randn(1,3,8,16)
    out=sample_meanflow(model,z,c,nfe)
    torch.testing.assert_close(out,c[:,:3]+.18*z,atol=2e-5,rtol=2e-5)


def test_window_schedule_covers_every_pair_and_stays_within_episode():
    import sys
    from types import SimpleNamespace
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from train_reality_flow import select_window,training_windows
    ds=SimpleNamespace(episodes=[Episode("a",0,0,0,27),Episode("b",1,1,27,27)],angles=[0,36,-36])
    cfg=dict(window=10,batch_size=3,seed=17,switch_view_probability=.5)
    blocks=training_windows(ds,cfg)
    seen=set()
    for k in range(len(blocks)//3):
        start,angles=select_window(blocks,k,cfg,3)
        for lane,s in enumerate(start):
            assert s//27==(s+9)//27
            seen.update((s+i,v) for i,v in enumerate(angles[:,lane]))
    assert seen=={(i,v) for i in range(54) for v in range(3)}


def test_inference_split_loads_without_teacher_arrays(tmp_path):
    data,cache=tmp_path/"data",tmp_path/"cache"
    data.mkdir(); (cache/"test").mkdir(parents=True)
    ep=dict(split="test",town_seed=9,episode_seed=4,frames=2)
    plan={"episodes":[ep]}; digest=json_hash(plan)
    write_json(data/"plan.json",plan)
    frames=[dict(step=np.int32(i),t_sim=np.float64(i/10),view_pose=np.ones((1,11),np.float32),
                 rgb_in=np.zeros((2,2,4,3),np.uint8),delta_pose=np.zeros(3,np.float32)) for i in range(2)]
    row=write_shard(data,ep,frames,digest)
    write_json(data/"manifest.json",dict(complete=True,contract_sha256=digest,shards=[row],episodes={row["episode_id"]:{"count":2}}))
    target=cache/"test"/"current.npy"
    np.save(target,np.zeros((2,1,28,64,128),np.float16))
    write_json(cache/"test"/"manifest.json",dict(collection=digest,split="test",angles=[0],shards=[row["path"]],
                                               arrays={"current":file_hash(target)}))
    ds=FlowSplit(data,cache,"test",teachers=False)
    assert len(ds.current)==2 and not hasattr(ds,"target") and not hasattr(ds,"seg")
    features = tmp_path / "pilot"
    where = features / row["path"]
    where.mkdir(parents=True)
    for key, shape in (("mem", (2,500,256)), ("c2",(2,2,2,2,4)), ("c3",(2,2,4,1,2))):
        np.save(where / (key + ".npy"), np.ones(shape, np.float16))
    pairing = dict(rgb_sha256=row["fields"]["rgb_in"]["sha256"], count=2, sha256=file_hash(where / "mem.npy"),
                   pyramid={key:file_hash(where / (key + ".npy")) for key in ("c2", "c3")})
    write_json(features / "manifest.json", dict(collection_sha256=digest, shards={row["path"]:pairing}))
    features_only = MultiViewFrames(data, features, "test", temporal=True, teachers=False)
    batch = features_only[1]
    assert set(batch) == {"mem", "c2", "c3", "view_pose", "past", "motion", "valid", "index"}
    assert batch["valid"].tolist() == [True] + [False] * 6


def test_vacated_actor_scores_come_from_output_pixels():
    gt=np.ones((1,2,3),np.float32); old=np.zeros_like(gt)
    pred=gt.copy(); pred[0,0]=0
    label=np.array([[SEM["vehicle"],SEM["road"]]])
    indices=np.array([1]),np.array([1])
    vacated=np.array([0]),np.array([0])
    s=projected_scores((pred-gt,gt,label),(old-old,old,label),(indices,vacated))
    assert s["warped_residual_l1"]==(0,1) and s["vacated_rgb_l1"]==(1,1)
    assert s["vacated_old_color_rate"]==(1,1) and s["vacated_actor_rate"]==(1,1)
    m=np.zeros((N_SEM,N_SEM),int); m[11,12]=8; m[1,11]=2
    scores=probe_scores(m)
    assert scores["dynamic_iou"]==.8 and scores["mean_iou"]==0


def test_g3_requires_quality_guards_and_real_rgb_calibration():
    cfg=json.loads((Path(__file__).resolve().parents[1]/"configs/reality/p3_full.json").read_text())
    base=dict(all_l1=.2,all_ssim=.4,native_l1=.2,warped_residual_l1=.04,vacated_rgb_l1=.1,phantom_fpr=.01)
    totals=Totals()
    totals.add({k:(v,1) for k,v in base.items()})
    scores={g:{k:totals.report() for k in ("phase2","nfe1","nfe4")} for g in ("fixed","quarter0","quarter3")}
    p={"methods":{k:{"total_ms":{"p50":10,"p95":12}} for k in ("phase2","nfe1","nfe4")}}
    scores["fixed"]["nfe1"]["metrics"]["all_l1"]=.19
    out=g3_decision(cfg,scores,p,True,True)
    json.dumps(out,allow_nan=False)
    assert out["nfe1"]["status"]=="PASS" and out["nfe4"]["status"]=="FAIL"
    assert g3_decision(cfg,scores,p,False,True)["nfe1"]["status"]=="FAIL"
    scores["fixed"]["nfe1"]["metrics"]["warped_residual_l1"]=.06
    assert not g3_decision(cfg,scores,p,True,True)["nfe1"]["criteria"]["flicker"]
