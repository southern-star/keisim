import json
from pathlib import Path

import numpy as np
import torch

from keipilot.model import KeiPilot
from keireality.encoder import extract_mem
from keireality.geometry import CHASE_CAMERA, view_pose
from keireality.scene_detail import SceneDetail
from keireality.spatial import ray_points
from keisim.config import CAMERA_RIGS
from keisim.render import rays


ROOT = Path(__file__).resolve().parents[1]


def detail():
    return SceneDetail(json.loads((ROOT/"configs/reality/p2_scene_detail.json").read_text()))


@torch.inference_mode()
def test_scene_taps_same_driving_stem_and_driver_outputs():
    torch.set_num_threads(2)
    pilot=KeiPilot(pretrained=False,n_cams=2).eval()
    image=torch.randint(256,(1,2,3,160,320),dtype=torch.uint8)
    actual=[]
    hook=pilot.stem[2].register_forward_hook(lambda mod,args,result:actual.append(result.detach()))
    try:
        observed=extract_mem(pilot,image,parity=True,pyramid=True,scene=True)
        regular=extract_mem(pilot,image,pyramid=True)
    finally:
        hook.remove()
    assert len(actual)==3
    assert torch.equal(observed["stem"][0],actual[1]) and torch.equal(actual[1],actual[2])
    assert regular.keys()=={"mem","c2","c3"}
    for k,v in regular.items():
        assert torch.equal(observed[k],v)
    assert observed["source_logits"].shape==(1,2,13,40,80)


def test_ground_correspondence_matches_source_pixel_and_rejects_occlusion():
    torch.set_num_threads(2)
    model=detail()
    view=torch.tensor(view_pose(CHASE_CAMERA))[None]
    points=ray_points(view,128,256,())[:,0]
    yy,xx=torch.meshgrid(torch.arange(160)/159,torch.arange(320)/319,indexing="ij")
    colors=torch.stack((xx,yy,xx*0))[None,None].repeat(1,2,1,1,1)
    probs=torch.zeros(1,2,3,40,80)
    probs[:,1,1]=1  # only the narrow camera supplies this ground hypothesis
    dd=[]
    for cam in CAMERA_RIGS["wide_narrow"]:
        center,rot=rays.level_frame(cam,0.,0.,0.)
        d=rays.camera_rays(cam,2)@rot
        r=np.divide(-center[2],d[:,:,2],out=np.zeros(d.shape[:2]),where=d[:,:,2]<-.01)
        dd.append(torch.tensor(r,dtype=torch.float32))
    depth=torch.stack(dd)[None]
    rgb,confidence=model.directional_color(points,colors,probs,depth,sky=False)
    good=(confidence[0,0]>.95).numpy()
    assert good.sum()>100
    narrow=CAMERA_RIGS["wide_narrow"][1]
    uv,_,_=rays.project(narrow,points[0].numpy()[good],*rays.level_frame(narrow,0.,0.,0.))
    expected=uv.clip(min=0,max=[319,159])/[319,159]
    np.testing.assert_allclose(rgb[0,:2].permute(1,2,0).numpy()[good],expected,atol=2e-5)
    _,blocked=model.directional_color(points,colors,probs,depth*.2,sky=False)
    assert not blocked[0,0].numpy()[good].any()


@torch.inference_mode()
def test_temporal_static_correspondence_and_surface_change():
    torch.set_num_threads(2)
    m=detail()
    v=torch.tensor(view_pose(CHASE_CAMERA))[None]
    direction=ray_points(v,128,256,(1.,))[:,1]-v[:,None,None,:3]
    rgb=torch.full((1,3,128,256),.36)
    depth=torch.full((1,1,128,256),20.)
    static=torch.ones_like(depth)
    old=rgb-.06
    delta=torch.zeros(3)
    blended=m.temporal(rgb,depth,static,direction,v,delta,(old,depth,static,v))
    assert torch.all(blended<rgb) and torch.all(blended>old)
    for d,s in ((depth*.5,static),(depth,static*0)):
        assert torch.equal(m.temporal(rgb,depth,static,direction,v,delta,(old,d,s,v)),rgb)
    assert torch.equal(m.temporal(rgb,depth,static,direction,v,delta,None),rgb)


def test_full_scene_metric_includes_every_pixel_and_class_boundary(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    from eval_reality_scene_detail import SceneScores
    t=np.zeros((10,16,3),np.float32)
    t[:,8:]=.90
    seg=np.ones(t.shape[:2],np.uint8)
    seg[:,8:]=11
    scores=SceneScores()
    scores.add(np.stack([t*0,t*.5,t]),t,seg,np.ones(seg.shape,bool),.1,None,None)
    r=scores.report()
    assert r["counts"]["all_l1"]==160 and r["counts"]["scene_edges"]==10
    for k in ("scene_edges","silhouette","ssim"):
        assert abs(r["C"][k]-1.)<1e-6
    assert r["C"]["all_l1"]==0 and r["classes"]["road"]["pixels"]==80
    assert abs(r["classes"]["vehicle"]["edge"]["B"]-.5)<1e-6
