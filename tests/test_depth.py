"""ReconView's depth head (docs/recon_view.md S2): shards with depth images, the head on a frozen driving model,
its loss and accuracy, train_depth.py and the ONNX export (run: python -m pytest tests/test_depth.py -q).
The invariants of G2: a model with the head drives as the one without it (1), training the head leaves every other
tensor as it was (2), the exported driving outputs are the released file's (3)."""
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from keipilot.data import LABEL_KEYS, DrivingDataset, ShardWriter, load_shards, write_meta  # noqa: E402
from keipilot.depthstats import BANDS, GROUPS, DepthStats, codes_to_metres, ground_baseline  # noqa: E402
from keipilot.model import DEPTH_LOSS_MAX, DEPTH_NEAR, KeiPilot, depth_loss  # noqa: E402
from keisim.config import CAMERA_RIGS, SEM  # noqa: E402
from keisim.render import rays as RY  # noqa: E402

CFG = dict(pretrained=False, speed_input=True, mem_input=True, aux_yield=True, n_cams=2)
MC14 = os.path.join(ROOT, "runs", "keipilot_mc14", "last.pt")


def labels(i, episode=0):
    lab = {k: (np.zeros(2, np.float32) if k == "tp" else np.zeros((10, 2), np.float32) if k == "path" else 0)
           for k in LABEL_KEYS}
    lab.update(step=np.int32(2 * i), episode=np.int64(episode), town=np.int32(1), virtual=np.bool_(False))
    return lab


def write_set(out, n=8, depth=True, per_shard=5, seed=0, episodes=4):
    """A tiny wide_narrow data set of random pictures; returns the full-resolution depth codes [frame][camera]."""
    rng = np.random.default_rng(seed)
    w = ShardWriter(str(out), "expert_w00", frames_per_shard=per_shard)
    deps = []
    for i in range(n):
        cams = [rng.integers(0, 256, (160, 320, 3), dtype=np.uint8) for _ in range(2)]
        segs = [rng.integers(0, 13, (160, 320), dtype=np.uint8) for _ in range(2)]
        dep = [rng.integers(0, 65536, (160, 320)).astype(np.uint16) for _ in range(2)]
        deps.append(dep)
        w.add(cams[0], segs[0], extra=[(cams[1], segs[1])], **({"dep": dep} if depth else {}), **labels(i, i % episodes))
    w.flush()
    write_meta(str(out), rig="wide_narrow", renderer="keiview", label_version=7, **({"depth": 1} if depth else {}))
    return deps


# ---------------------------------------------------------------------- shards
def test_shards_without_depth_are_as_before(tmp_path):
    write_set(tmp_path / "a", depth=False)
    z = np.load(tmp_path / "a" / "expert_w00_0000.npz")
    assert not any(k.startswith("dep") for k in z.files)
    assert {"jpg", "jpg_off", "seg", "seg_off", "jpg_c1", "jpg_c1_off", "seg_c1", "seg_c1_off"} <= set(z.files)
    data, _ = load_shards([str(tmp_path / "a")])
    assert "depth" not in data and not any(k.startswith("dep") for k in data)
    assert "dep" not in DrivingDataset(data, np.arange(8), train=False)[0]


def test_shards_with_depth(tmp_path):
    """dep / dep_c1 hold every camera's codes at the labels' resolution: the pixels (2i, 2j), as the labels' own
    halving keeps them. Everything else of the samples is what it is without depth; mmap reads the same."""
    deps = write_set(tmp_path / "d", depth=True)
    write_set(tmp_path / "a", depth=False)
    z = np.load(tmp_path / "d" / "expert_w00_0000.npz")
    assert {"dep", "dep_off", "dep_c1", "dep_c1_off"} <= set(z.files)
    assert set(z.files) - {"dep", "dep_off", "dep_c1", "dep_c1_off"} == set(np.load(tmp_path / "a" / "expert_w00_0000.npz").files)
    data, files = load_shards([str(tmp_path / "d")])
    plain, _ = load_shards([str(tmp_path / "a")])
    assert data["depth"] is True and len(files) == 2
    ds, dp = DrivingDataset(data, np.arange(8), train=False), DrivingDataset(plain, np.arange(8), train=False)
    mapped = DrivingDataset(load_shards([str(tmp_path / "d")], mmap=True)[0], np.arange(8), train=False)
    for i in (0, 4, 7):
        s, p = ds[i], dp[i]
        assert s["dep"].shape == (2, 80, 160) and s["dep"].dtype == torch.int32
        for c in range(2):
            assert np.array_equal(s["dep"][c].numpy(), deps[i][c][::2, ::2].astype(np.int32))
        assert set(s) == set(p) | {"dep"}
        assert all(torch.equal(s[k], p[k]) or (s[k] != s[k]).all() for k in p)          # (yield_tta is NaN: no label)
        assert torch.equal(mapped[i]["dep"], s["dep"])
    assert DrivingDataset(data, np.arange(8), train=True)[1]["dep"].shape == (2, 80, 160)
    # directories with and without depth together: read as data without depth
    both, _ = load_shards([str(tmp_path / "d"), str(tmp_path / "a")])
    assert "depth" not in both and len(both["cmd"]) == 16
    # a shard has depth for all of its frames or for none
    w = ShardWriter(str(tmp_path / "x"), "expert_w00")
    img, seg = np.zeros((160, 320, 3), np.uint8), np.zeros((160, 320), np.uint8)
    w.add(img, seg, extra=[(img, seg)], **labels(0))
    with pytest.raises(AssertionError):
        w.add(img, seg, extra=[(img, seg)], dep=[np.zeros((160, 320), np.uint16)] * 2, **labels(1))


def test_collect_depth_needs_keiview():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import collect
    assert collect.parse_args([]).depth == 0
    with pytest.raises(SystemExit):
        collect.parse_args(["--depth", "1"])
    with pytest.raises(SystemExit):
        collect.parse_args(["--depth", "1", "--renderer", "keiview", "--history", "0.4"])
    assert collect.parse_args(["--depth", "1", "--renderer", "keiview", "--rig", "wide_narrow"]).depth == 1


def test_collect_with_depth_in_keiview(tmp_path):
    """collect.py --depth 1 on KeiView's rig: every kept frame has both cameras' depth, and its codes agree with the
    labels (no surface where the label is sky). Skipped without KeiView."""
    if not os.path.isdir(os.path.join(ROOT, "web", "node_modules", "puppeteer-core")):
        pytest.skip("KeiView needs `npm install` in web/")
    out = tmp_path / "kv"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "collect.py"), "--out", str(out), "--renderer", "keiview",
                        "--rig", "wide_narrow", "--virtual_prob", "0", "--depth", "1", "--frames", "24", "--workers", "1",
                        "--shard", "24"], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0 and "KeiView" in r.stdout + r.stderr:
        pytest.skip("KeiView did not start")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert json.load(open(out / "meta.json"))["depth"] == 1
    data, _ = load_shards([str(out)])
    assert data["depth"] and len(data["cmd"]) == 24
    s = DrivingDataset(data, np.arange(24), train=False)[5]
    assert s["dep"].shape == (2, 80, 160)
    assert ((s["dep"] > 0) == (s["seg"] != SEM["sky"])).float().mean() > 0.995
    r_m = codes_to_metres(s["dep"])
    road = (s["seg"] == SEM["road"]) & (r_m > 0)
    assert road.sum() > 500 and float(r_m[road].min()) > 1.5 and float(r_m[road].median()) < 40


# ---------------------------------------------------------------------- the head
def test_model_without_the_head_is_unchanged():
    """(1) The head is created last and only when asked for; a model with it gives the driving outputs of the model
    without it, bit for bit, whether the depth is asked for or not."""
    torch.manual_seed(0)
    plain = KeiPilot(**CFG).eval()
    state = torch.random.get_rng_state()
    torch.manual_seed(0)
    again = KeiPilot(aux_depth=False, **CFG)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), again.state_dict().values()))
    assert not any(k.startswith("aux_depth") for k in plain.state_dict())
    torch.manual_seed(0)
    dm = KeiPilot(aux_depth=True, **CFG).eval()
    assert all(torch.equal(v, dm.state_dict()[k]) for k, v in plain.state_dict().items())     # the same draws before it
    new = dm.load_compatible(plain.state_dict())
    assert new and all(k.startswith("aux_depth_") for k in new)
    assert not any(isinstance(m, (torch.nn.BatchNorm2d, torch.nn.Dropout)) for n, m in dm.named_modules() if n.startswith("aux_depth"))
    n_new = sum(p.numel() for n, p in dm.named_parameters() if n.startswith("aux_depth_"))
    assert n_new == sum(p.numel() for p in dm.parameters()) - sum(p.numel() for p in plain.parameters()) == 68538
    assert dm.aux_depth_rays.shape == (2, 3, 40, 80)
    assert torch.allclose(dm.aux_depth_rays[1].permute(1, 2, 0), torch.from_numpy(RY.camera_rays(CAMERA_RIGS["wide_narrow"][1], 4)).float())
    B = 2
    img = torch.randint(0, 256, (B, 2, 3, 160, 320), dtype=torch.uint8)
    cmd, tp = torch.tensor([0, 2]), torch.tensor([[18.0, -3.0], [30.0, 6.0]])
    kw = dict(speed=torch.tensor([0.0, 7.0]), speed_known=torch.tensor([True, False]),
              stand_m=torch.tensor([0.0, 10.0]), stand_known=torch.tensor([True, True]))
    with torch.no_grad():
        for with_seg in (True, False):
            a = plain(img, cmd, tp, with_seg=with_seg, **kw)
            b = dm(img, cmd, tp, with_seg=with_seg, **kw)
            c = dm(img, cmd, tp, with_seg=with_seg, with_depth=True, **kw)
            d = plain(img, cmd, tp, with_seg=with_seg, with_depth=True, **kw)      # no head: the request is ignored
            assert set(a) == set(b) == set(d) and set(c) == set(a) | {"depth"}
            assert all(torch.equal(a[k], b[k]) and torch.equal(a[k], c[k]) and torch.equal(a[k], d[k]) for k in a)
            assert c["depth"].shape == (B * 2, 2, 80, 160)
            assert torch.equal(c["depth"], dm.forward_depth(img))
    # a fresh head says 8 m and "a surface" everywhere, roughly; metres are 0 where it says "none"
    m = KeiPilot.depth_metres(c["depth"])
    assert m.shape == (4, 80, 160) and 4.0 < float(m.median()) < 16.0
    y = torch.tensor([[[[math.log(10 / DEPTH_NEAR), 0.0, 99.0]], [[1.0, -1.0, 1.0]]]])
    assert torch.allclose(KeiPilot.depth_metres(y), torch.tensor([[[10.0, 0.0, 400.0]]]), rtol=1e-5)


def test_depth_loss():
    """L1 of ln(r) over the pixels with a surface within DEPTH_LOSS_MAX, cross-entropy of "a surface" over all;
    vehicles and people count four times."""
    r = torch.tensor([[[10.0, 20.0, 200.0, 0.0]]])
    q = torch.from_numpy(RY.encode_depth(r.numpy()).astype(np.int32))
    seg = torch.tensor([[[SEM["road"], SEM["vehicle"], SEM["building"], SEM["sky"]]]])
    t = torch.log(codes_to_metres(q).clamp(min=DEPTH_NEAR) / DEPTH_NEAR)
    y = torch.zeros(1, 2, 1, 4)
    y[0, 0] = t + torch.tensor([0.1, 0.2, 5.0, 3.0])                     # the far and the empty pixel do not count
    y[0, 1] = torch.tensor([20.0, 20.0, 20.0, -20.0])                    # "a surface" right everywhere
    loss, logs = depth_loss(y, q, seg)
    assert logs["depth_l1"] == pytest.approx((0.1 * 1 + 0.2 * 4) / 5, rel=1e-5) and logs["surface"] < 1e-6
    assert float(loss) == pytest.approx(logs["depth_l1"], rel=1e-4) and DEPTH_LOSS_MAX == 150.0
    y[0, 1, 0, 3] = 20.0                                                 # a surface seen in the sky
    assert depth_loss(y, q, seg)[1]["surface"] == pytest.approx(20.0 / 7, rel=1e-4)
    assert float(depth_loss(y, torch.zeros_like(q), seg)[1]["depth_l1"]) == 0.0


def test_depth_stats_and_the_ground_baseline():
    g = ground_baseline("wide_narrow")
    assert g.shape == (2, 80, 160)
    for k, cfg in enumerate(CAMERA_RIGS["wide_narrow"]):                  # the plane z = 0 along the half-resolution rays
        C, R = RY.level_frame(cfg, 0.0, 0.0, 0.0)
        d = RY.camera_rays(cfg, 2)
        z = RY.unproject(g[k].numpy().astype(np.float64), d, C, R)[..., 2]
        near = g[k].numpy() < 100
        assert near.sum() > 3000 and np.abs(z[near]).max() < 1e-3 and (g[k].numpy()[~near] <= 150.0).all()
    st = DepthStats(2)
    gt = torch.tensor([5.0, 20.0, 50.0, 90.0, 0.0, 20.0]).view(1, 1, 1, 6).repeat(1, 2, 1, 1)
    seg = torch.tensor([SEM["road"], SEM["vehicle"], SEM["building"], SEM["road"], SEM["sky"], SEM["sky"]]).view(1, 1, 1, 6).repeat(1, 2, 1, 1)
    pred = gt * torch.tensor([1.1, 0.5, 1.0, 3.0, 1.0, 1.0])
    st.add(pred, gt, seg, surface=gt > 0)
    s = st.summary()
    assert s["groups"]["road"]["0-10"]["abs_rel"] == pytest.approx(0.1) and s["groups"]["road"]["0-10"]["pixels"] == 2
    assert s["groups"]["road"]["30-80"] is None                              # 90 m: beyond the last band
    assert s["groups"]["vehicle"]["10-30"]["abs_rel"] == pytest.approx(0.5) and s["groups"]["vehicle"]["10-30"]["d1"] == 0.0
    assert s["groups"]["building"]["all"]["d1"] == 1.0 and s["groups"]["pedestrian"]["all"] is None
    assert s["all"]["0-30"]["abs_rel"] == pytest.approx(0.3) and s["all"]["all"]["pixels"] == 6     # sky is in no group
    assert s["cameras"][1]["0-30"]["pixels"] == 2 and s["surface"] == 1.0
    assert list(GROUPS)[0] == "road" and BANDS == (10.0, 30.0, 80.0)


# ---------------------------------------------------------------------- training
def test_train_depth_leaves_the_driving_model_alone(tmp_path):
    """(2) A few steps of scripts/train_depth.py on the CPU: every tensor that is not the head's is the checkpoint's,
    bit for bit, the batch-norm statistics and their counters included; the head has moved; the result loads as a
    driving model and plans as the checkpoint does. In train() the statistics would have moved."""
    write_set(tmp_path / "d", n=12, per_shard=6)
    torch.manual_seed(1)
    init = KeiPilot(**CFG)
    with torch.no_grad():                               # statistics that are not the fresh zeros and ones
        for m in init.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.normal_(0, 0.1)
                m.running_var.uniform_(0.5, 1.5)
                m.num_batches_tracked.fill_(7)
    cfg = {k: v for k, v in CFG.items() if k != "pretrained"}
    torch.save({"model": init.state_dict(), "model_cfg": cfg, "rig": "wide_narrow"}, tmp_path / "init.pt")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train_depth.py"), "--init", str(tmp_path / "init.pt"),
                        "--data", str(tmp_path / "d"), "--out", str(tmp_path / "out"), "--epochs", "1", "--bs", "2",
                        "--workers", "0", "--max_steps", "3", "--device", "cpu", "--val_frac", "0.25"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "tensors are the ones of" in r.stdout
    ck = torch.load(tmp_path / "out" / "last.pt", map_location="cpu")
    assert ck["model_cfg"]["aux_depth"] is True and ck["rig"] == "wide_narrow"
    sd0 = init.state_dict()
    assert set(ck["model"]) - set(sd0) and all(k.startswith("aux_depth_") for k in set(ck["model"]) - set(sd0))
    assert all(torch.equal(ck["model"][k], v) for k, v in sd0.items())
    assert int(ck["model"]["stem.1.num_batches_tracked"]) == 7
    torch.manual_seed(0)
    fresh = KeiPilot(aux_depth=True, **CFG).state_dict()
    assert not torch.equal(ck["model"]["aux_depth_h.2.weight"], fresh["aux_depth_h.2.weight"])
    log = json.load(open(tmp_path / "out" / "log.json"))
    assert log[0]["step"] == 3 and "abs_rel" in log[0]["val"]["all"]["all"]
    from keipilot.agent import load_model
    m = load_model(str(tmp_path / "out" / "last.pt"), "cpu")
    assert m.aux_depth and m.rig == "wide_narrow"
    img = torch.randint(0, 256, (1, 2, 3, 160, 320), dtype=torch.uint8)
    args = (img, torch.tensor([1]), torch.tensor([[18.0, -3.0]]))
    kw = dict(speed=torch.tensor([5.0]), stand_m=torch.tensor([3.0]))
    with torch.no_grad():
        a, b = init.eval()(*args, **kw), m(*args, with_depth=True, **kw)
    assert all(torch.equal(a[k], b[k]) for k in a)
    # the reason for eval(): in train() a forward pass alone moves the trunk's statistics
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import train_depth
    m.train()
    with torch.no_grad():
        m.forward_depth(img)
    assert "stem.1.running_mean" in train_depth.unchanged(m, sd0)


@pytest.mark.skipif(not os.path.exists(MC14), reason="runs/keipilot_mc14/last.pt is not here")
def test_mc14_with_the_head_drives_as_mc14():
    ck = torch.load(MC14, map_location="cpu")
    plain = KeiPilot(pretrained=False, **ck["model_cfg"]).eval()
    plain.load_state_dict(ck["model"])
    dm = KeiPilot(pretrained=False, aux_depth=True, **ck["model_cfg"]).eval()
    new = dm.load_compatible(ck["model"])
    assert new and all(k.startswith("aux_depth_") for k in new)
    img = torch.randint(0, 256, (1, 2, 3, 160, 320), dtype=torch.uint8)
    args = (img, torch.tensor([1]), torch.tensor([[18.0, -3.0]]))
    kw = dict(speed=torch.tensor([5.0]), stand_m=torch.tensor([3.0]))
    with torch.no_grad():
        a, b = plain(*args, **kw), dm(*args, with_depth=True, **kw)
    assert all(torch.equal(a[k], b[k]) for k in a) and b["depth"].shape == (2, 2, 80, 160)


def test_export_onnx_with_the_head(tmp_path):
    """(3) scripts/export_onnx.py of mc14 with a depth head: the driving outputs are the released keipilot_mc.onnx's,
    bit for bit, and the last output is the head's depth in metres (KEIPILOT_MC_ONNX: the released file; needs onnx
    and onnxruntime)."""
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    ref = os.environ.get("KEIPILOT_MC_ONNX")
    if not ref or not os.path.exists(ref) or not os.path.exists(MC14):
        pytest.skip("KEIPILOT_MC_ONNX (the released keipilot_mc.onnx) or runs/keipilot_mc14/last.pt is not here")
    ck = torch.load(MC14, map_location="cpu")
    cfg = dict(ck["model_cfg"], aux_depth=True)
    torch.manual_seed(0)
    dm = KeiPilot(pretrained=False, **cfg).eval()
    dm.load_compatible(ck["model"])
    torch.save({"model": dm.state_dict(), "model_cfg": cfg, "rig": ck["rig"]}, tmp_path / "mc14d.pt")
    dst = str(tmp_path / "mc14d.onnx")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "export_onnx.py"), str(tmp_path / "mc14d.pt"), dst],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    a, b = ort.InferenceSession(ref), ort.InferenceSession(dst)
    assert [(i.name, i.shape) for i in a.get_inputs()] == [(i.name, i.shape) for i in b.get_inputs()]
    assert [o.name for o in b.get_outputs()] == [o.name for o in a.get_outputs()] + ["depth"]
    rng = np.random.default_rng(3)
    for _ in range(3):
        feed = {i.name: (rng.integers(0, 256, i.shape).astype(np.float32) if i.name.startswith("im")
                         else rng.normal(size=i.shape).astype(np.float32)) for i in a.get_inputs()}
        out = b.run(None, feed)
        for x, y in zip(a.run(None, feed), out):
            assert np.array_equal(x, y)
        with torch.no_grad():
            want = dm.depth_metres(dm.forward_depth(torch.from_numpy(feed["imgs"]))).numpy()
        assert out[-1].shape == (2, 80, 160) and out[-1].dtype == np.float32
        assert np.abs(out[-1] - want).max() < 2e-2 * max(1.0, float(want.max()))
