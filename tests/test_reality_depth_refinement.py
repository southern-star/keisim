from pathlib import Path

import pytest
import torch

from keireality.depth_refinement import SourceDepthRefiner, aligned_features


def test_lower_resolution_feature_centers_match_the_calibrated_source_rays():
    source = torch.tensor([[[[1.,2.,3.],[4.,5.,6.]]]])
    aligned = aligned_features(source, 8, 12)
    torch.testing.assert_close(aligned[:,:,::4,::4], source)
    assert aligned[0,0,2,2].item()==3.
    assert aligned[0,0,-1,-1].item()==6.


def test_source_depth_prediction_uses_only_sensor_features_and_is_bounded():
    torch.set_num_threads(2)
    net=SourceDepthRefiner().eval()
    inputs=dict(stem=torch.zeros(1,2,64,80,160),c2=torch.zeros(1,2,128,20,40),
                source_logits=torch.zeros(1,2,13,40,80),estimated_depth=torch.full((1,2,80,160),12.))
    inputs["estimated_depth"][:,0,:8]=0
    with torch.no_grad():
        result=net.metres(inputs)
    assert result.shape==inputs["estimated_depth"].shape
    assert result.isfinite().all()
    assert (result[:,0,:8]==0).all()
    torch.testing.assert_close(result[:,1], torch.full_like(result[:,1],12.),atol=2e-5,rtol=0)


def test_swing_covers_both_extremes_and_returns_to_center(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from collect_reality_sweep import sweep
    angles=[sweep(i) for i in range(97)]
    assert angles[::12]==[0,36,0,-36,0,36,0,-36,0]
    assert all(abs(a-b)==3 for a,b in zip(angles,angles[1:]))


@pytest.mark.parametrize("failed", ["vehicle","marking"])
def test_chase_pass_cannot_release_a_failed_turning_contour(monkeypatch,failed):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from eval_reality_depth_refiner import gate
    a=dict(all_l1=.2,ssim=.3,scene_edges=.1,silhouette=.2,static_flicker=.05,vacated_l1=.1)
    e=dict(all_l1=.1,ssim=.4,scene_edges=.2,silhouette=.4,static_flicker=.04,vacated_l1=.09)
    def row():
        return dict(A=dict(a),D=dict(e),depth=dict(e),classes=dict(marking=dict(edge=dict(D=.50,depth=.50))))
    tracks=dict(chase=row(),sweep=row())
    assert gate(tracks,"depth")["status"]=="PASS"
    if failed=="vehicle":
        tracks["sweep"]["depth"]["silhouette"]-=.005
    else:
        tracks["sweep"]["classes"]["marking"]["edge"]["depth"]-=.005
    result=gate(tracks,"depth")
    assert result["six_vs_A"]["status"]=="PASS" and result["status"]=="HOLD"
    assert not result["contours_vs_D"]["sweep"][failed]


def test_inverse_surface_rejects_an_occluder_on_a_past_ray():
    from keireality.inverse_surface import camera_hit
    from keisim.config import CAMERA_RIGS
    cam=CAMERA_RIGS["wide_narrow"][1]
    origin=torch.tensor([cam.x,cam.y,cam.z]).reshape(1,1,1,3)
    direction=torch.tensor([1.,0.,0.]).reshape(1,1,1,3)
    distance=torch.tensor([[[20.]]])
    rgba=torch.ones(1,4,160,320)
    for surface,expected in ((20.,1.),(8.,0.),(0.,0.)):
        depth=torch.full((1,80,160),surface)
        rgb,weight,_=camera_hit(cam,origin,direction,distance,depth,rgba,solve=False)
        assert weight.item()==pytest.approx(expected,abs=2e-5)
        assert rgb.shape==(1,3,1,1)
