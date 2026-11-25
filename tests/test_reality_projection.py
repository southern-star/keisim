from pathlib import Path

import numpy as np
import pytest
import torch

from keireality.depth_projection import DepthProjectionBranch, DepthProjector
from keireality.geometry import CHASE_CAMERA, orbit_camera, view_pose
from keireality.spatial import SpatialRenderer
from keireality.stream import TemporalDisplay
from keisim.config import CAMERA_RIGS
from keisim.render import rays


@pytest.mark.parametrize("angle", [0, -18, 36])
def test_subpixel_projection_matches_calibrated_source_rays(angle):
    torch.set_num_threads(2)
    camera = orbit_camera(angle)
    view = torch.tensor(view_pose(camera))[None]
    depth = torch.zeros(1,2,80,160)
    depth[0,:,36,85] = 24
    grids, mass, inv = DepthProjector()(depth, view)
    y,x = np.indices((64,128))
    for k,src in enumerate(CAMERA_RIGS["wide_narrow"]):
        origin, rotation = rays.level_frame(src,0,0,0)
        point = origin + (rays.camera_rays(src,2)[36,85] @ rotation)*24
        uv, _, ok = rays.project(camera,point[None],*rays.level_frame(camera,0,0,0))
        assert ok[0]
        weights = mass[0,k].numpy()
        np.testing.assert_allclose(weights.sum(),1,atol=1e-6)
        np.testing.assert_allclose([(weights*x).sum(),(weights*y).sum()], (uv[0]+.5)/2-.5, atol=1e-5)
        supported = weights>0
        expected = [(2*85+1)/160-1, (2*36+1)/80-1]
        np.testing.assert_allclose(grids[0,k].numpy()[supported], np.tile(expected,(supported.sum(),1)), atol=1e-7)
        np.testing.assert_allclose(inv[0,k].numpy()[supported], 1/np.dot(point-view[0,:3].numpy(),view[0,3:6].numpy()), atol=1e-6)


def test_nearer_splats_hide_far_surface():
    view = torch.tensor(view_pose(CAMERA_RIGS["wide_narrow"][1]))[None]
    depth = torch.zeros(1,2,80,160)
    depth[0,1,40,80] = 5.
    project = DepthProjector()
    first = project(depth,view,size=(16,32))
    depth[0,1,40,81] = 35.
    both = project(depth,view,size=(16,32))
    for a,b in zip(first,both):
        torch.testing.assert_close(a,b)


@torch.inference_mode()
def test_stream_projection_fallback_and_rejected_tick():
    torch.set_num_threads(2)
    torch.manual_seed(409)
    base = SpatialRenderer(support=True,temporal=True).eval()
    branch = DepthProjectionBranch().eval()
    display = TemporalDisplay(base,branch)
    control = TemporalDisplay(base)
    features = dict(mem=torch.randn(1,500,256).half(),c2=torch.randn(1,2,128,20,40).half(),
                    c3=torch.randn(1,2,256,10,20).half())
    view = torch.tensor(view_pose(CHASE_CAMERA))[None]
    with pytest.raises(ValueError,match="estimated-depth"):
        display.observe(features,"one",0.,np.zeros(3))
    control.observe(features,"one",0.,np.zeros(3))
    depth = torch.full((1,2,80,160),16.)
    display.observe(dict(**features,estimated_depth=depth),"one",0.,np.zeros(3))
    b,sem,support,a = display.render(view,return_reference=True)
    ref = control.render(view)
    for actual,want in zip((a,sem,support),ref):
        assert torch.equal(actual,want)
    assert not torch.equal(a,b)
    with pytest.raises(ValueError,match="finite"):
        display.observe(dict(features,c2=features["c2"]*0,estimated_depth=depth),"one",np.nan,np.zeros(3))
    assert torch.equal(b,display.render(view)[0])
    display.observe(dict(**features,estimated_depth=depth*0),"two",0.,np.zeros(3))
    b,_,_,a = display.render(view,return_reference=True)
    assert torch.equal(a,b) and not display.valid.any()


def test_gate_uses_silhouette_and_both_independent_stability_checks(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from eval_reality_projection import Scores, decision
    truth = np.zeros((3,4,3),np.float32)
    truth[:,2:] = 1
    seen = np.ones((3,4),bool)
    scores = Scores()
    scores.edges(np.stack([truth*0,truth]),truth,truth[...,0]>0,seen,.1)
    report = scores.report()
    assert report["counts"]["silhouette"]==3 and report["B"]["silhouette"]==1.
    assert decision(report)["status"] == "INCONCLUSIVE"
    for key in ("static_flicker","vacated_l1"):
        report["A"][key] = report["B"][key] = .12
    assert decision(report)["status"] == "PASS"
    report["B"]["vacated_l1"] += .00001
    assert decision(report)["status"] == "FAIL"
