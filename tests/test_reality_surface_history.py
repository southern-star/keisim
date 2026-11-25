from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import torch

from keireality.background_history import BackgroundHistoryDisplay, Observation, old_camera, previous_grid
from keireality.geometry import orbit_camera, view_pose
from keireality.scene_detail import SceneDetail
from keireality.spatial import ray_points
from keireality.surface_observation import ObservedDetail
from keisim.config import SEM


def test_paired_alpha_intervention_preserves_observed_colors(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from diagnose_reality_surface_gaps import pair_sums, summarize
    labels = np.repeat([SEM["building"], SEM["vegetation"]], 4)
    before = np.tile([0., 0., .6, .6], 2)
    after = np.tile([0., .6, 0., .6], 2)
    color = np.ones((len(labels),3))*.4
    zero = np.zeros_like(color)
    old = zero, before[:,None]*color, before, color
    now = zero, after[:,None]*color, after, color
    totals = np.zeros((1,2,4,4))
    pair_sums(totals,[.25],np.arange(1,len(labels)+1),labels,now,old)
    for groups in summarize(totals,[.25])["0.25"].values():
        assert groups["switching"]["pairs"] == 2
        assert abs(groups["switching"]["C"] - .24) < 1e-8
        assert groups["switching"]["equal_alpha"] == 0.
        assert groups["supports_gap_hypothesis"]
    partial=before.copy()
    partial[[1,5]]=.05
    totals.fill(0)
    pair_sums(totals,[.25],np.arange(1,len(labels)+1),labels,now,
              (zero,partial[:,None]*color,partial,color),exact_gaps=True)
    for groups in summarize(totals,[.25])["0.25"].values():
        assert groups["lost"]["pairs"]==1 and groups["gained"]["pairs"]==0


def test_current_camera_in_old_ego_and_changed_display_view():
    angle = .21
    motion = torch.tensor([[1.5,-2.,np.sin(angle),np.cos(angle),-.1]],dtype=torch.float32)
    current = torch.tensor(view_pose(orbit_camera(18)))[None]
    past = torch.tensor(view_pose(orbit_camera(-18)))[None]
    translated = old_camera(current,motion)
    direction = ray_points(current,12,24,(1.,))[:,1]-current[:,None,None,:3]
    depth = torch.full((1,1,12,24),32.)
    grid,_,z = previous_grid(translated,past,depth,direction,current)
    points = current[:,None,None,:3] + 32.*direction / (direction*current[:,None,None,3:6]).sum(-1,keepdim=True)
    x,y,h = points[0,6,12].numpy()
    ca,sa = np.cos(angle),np.sin(angle)
    previous_point = np.array([ca*(x-1.5)+sa*(y+2),-sa*(x-1.5)+ca*(y+2),h])
    origin,forward,up = past[0,:9].numpy().reshape(3,3)
    relative = previous_point-origin
    optical = relative@forward
    slope = np.tan(np.deg2rad(float(past[0,9]))/2)
    expected=[relative@np.cross(forward,up)/(optical*slope*float(past[0,10])), -relative@up/(optical*slope)]
    np.testing.assert_allclose(grid[0,6,12],expected,atol=1e-5)
    assert abs(float(z[0,0,6,12])-optical)<1e-5


def test_observed_history_fills_a_gap_and_rejects_occlusion(monkeypatch):
    h,w=12,24
    c=torch.zeros(1,3,h,w)
    scalar=torch.ones(1,1,h,w)
    view=torch.tensor(view_pose(orbit_camera(0)))[None]
    direction=ray_points(view,h,w,(1.,))[:,1]-view[:,None,None,:3]
    frame=SimpleNamespace(optical_depth=scalar*10000.,cover=scalar*0.,direction=direction,moving=scalar*0.)
    obs=Observation(c,scalar,scalar)
    display=SimpleNamespace(past=[obs],detail=None,reference=SimpleNamespace(
        motion=torch.tensor([[[0.,0.,0.,1.,-.1]]])))
    def project(self,colors,probs,v,depth):
        return torch.cat((c+1,scalar),1),scalar,scalar*30.,scalar*0.
    monkeypatch.setattr(SceneDetail,"surfaces",project)
    apply=lambda mask: BackgroundHistoryDisplay.blend(display,c,c,frame,view,0,mask)
    assert (apply(scalar)>0).all()
    assert torch.equal(apply(scalar*0),c)
    frame.optical_depth,frame.cover=scalar*8.,scalar
    assert torch.equal(apply(scalar),c)
    logits=torch.full((1,13,h,w),-15.)
    logits[:,SEM["building"]]=0.
    logits[:,SEM["vehicle"],6,12]=15.
    mask=BackgroundHistoryDisplay.background_mask(logits,frame)
    assert mask[0,0,6,12]==0 and mask[0,0,6,15]==0 and mask[0,0,0,0]==1


def test_new_gate_checks_both_contours_independently(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from eval_reality_surface_history import decision
    a=dict(all_l1=.2,ssim=.3,scene_edges=.1,silhouette=.2,static_flicker=.05,vacated_l1=.1)
    d=dict(all_l1=.1,ssim=.4,scene_edges=.2,silhouette=.4,static_flicker=.04,vacated_l1=.09)
    row=dict(A=a,D=d,C=dict(silhouette=.4),classes=dict(marking=dict(edge=dict(C=.5,D=.5))))
    assert decision(row)["status"]=="PASS"
    row["classes"]["marking"]["edge"]["D"]=.49
    assert decision(row)["six_vs_A"]["status"]=="PASS" and decision(row)["status"]=="HOLD"
    row["C"]["silhouette"]=.41
    assert decision(row)["contours_vs_C"]==dict(marking=False,vehicle=False)


def test_history_is_three_observations_and_resets_on_gaps_and_episode_changes():
    torch.set_num_threads(2)
    phase2=torch.nn.Identity()
    phase2.history=True
    cfg=Path(__file__).resolve().parents[1]/"configs/reality/p2_scene_detail.json"
    display=BackgroundHistoryDisplay(phase2,ObservedDetail(json.loads(cfg.read_text())))
    features=dict(mem=torch.zeros(1,500,256),stem=torch.zeros(1,2,64,80,160),
                  source_logits=torch.zeros(1,2,13,40,80),estimated_depth=torch.ones(1,2,80,160))
    for i in range(6):
        display.observe(features,"one",i/10,np.zeros(3))
        assert len(display.past)==min(i,3)
    display.observe(features,"one",.8,np.zeros(3))
    assert not display.past
    display.observe(features,"one",.9,np.zeros(3))
    display.observe(features,"two",1.,np.zeros(3))
    assert not display.past


def test_source_guard_protects_ground_rays_but_not_the_sky():
    view=torch.tensor(view_pose(orbit_camera(0)))[None]
    direction=ray_points(view,64,128,(1.,))[:,1]-view[:,None,None,:3]
    display=SimpleNamespace(source_guard=torch.ones(1,2,1,40,80))
    blocked=BackgroundHistoryDisplay.foreground_rays(display,view,direction)
    assert blocked[0,0,0,64]==0 and blocked[0,0,39,64]==1
    display.source_guard.zero_()
    assert torch.count_nonzero(BackgroundHistoryDisplay.foreground_rays(display,view,direction))==0
