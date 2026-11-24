"""ReconFill's data (docs/recon_fill.md): the chase camera in Python against the page's, the ground's depth, and the
training samples the page writes (run: python -m pytest tests/test_fill.py -q). The last test drives the page in a
headless Chrome and is skipped without `npm install` in web/, a Chrome or web/models/keipilot_mc_depth.onnx."""
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from keipilot.filldata import ARRAYS, BG, CHASE, ground_depth, load_run  # noqa: E402
from keisim.render import rays as RY  # noqa: E402


def test_chase_camera_is_the_pages():
    """keipilot/filldata.py CHASE and BG are web/src/recon.js's (the text of its definitions, so a change there fails
    here)."""
    js = open(os.path.join(ROOT, "web", "src", "recon.js")).read()
    m = re.search(r"export const CHASE = \{([^}]*)\}", js)
    assert m, "web/src/recon.js has no CHASE"
    text = " ".join(m.group(1).split())
    assert "projection: 'pinhole', x: -8, y: 0, z: 4, yaw: 0, pitch: Math.atan2(4, 20) / DEG" in text
    assert "fov: 2 * Math.atan(2 * Math.tan(25 * DEG)) / DEG, width: 320, height: 160" in text
    assert (CHASE.x, CHASE.y, CHASE.z, CHASE.yaw_deg, CHASE.width, CHASE.height) == (-8.0, 0.0, 4.0, 0.0, 320, 160)
    assert CHASE.pitch_deg == pytest.approx(math.degrees(math.atan2(4, 20)))
    assert CHASE.fov_deg == pytest.approx(math.degrees(2 * math.atan(2 * math.tan(math.radians(25)))))
    assert "new THREE.Color('#12131c')" in js and BG == (0x12, 0x13, 0x1c)


def test_ground_depth_is_the_flat_ground():
    """ground_depth() along the chase camera's lines of sight ends on z = 0, at full and half resolution; the lines of
    sight above the horizon have none."""
    pose = (12.5, -7.25, 0.6)
    C, R = RY.level_frame(CHASE, *pose)
    for step in (1, 2):
        gd, rays = ground_depth(step), RY.camera_rays(CHASE, step)
        assert gd.shape == (CHASE.height // step, CHASE.width // step)
        P = RY.unproject(gd, rays, C, R)
        down = gd > 0
        assert np.abs(P[down][:, 2]).max() < 1e-9
        assert down[-1].all() and not down[0].any()            # the bottom row sees the ground, the top row the sky
        # the image centre looks at the road 12 m ahead of the car: 20 m ahead of the camera, 4 m below it
        if step == 1:
            c = gd[CHASE.height // 2 - 1: CHASE.height // 2 + 1, CHASE.width // 2 - 1: CHASE.width // 2 + 1].mean()
            assert c == pytest.approx(math.hypot(20.0, 4.0), rel=0.01)


def test_fill_dump(tmp_path):
    """web/tools/fill_dump.mjs writes samples numpy reads, with the view's background exactly where its depth has no
    surface and the town's ground at height 0 (tools/fill_data.py check)."""
    node = shutil.which("node")
    web = os.path.join(ROOT, "web")
    if node is None or not os.path.isdir(os.path.join(web, "node_modules", "puppeteer-core")):
        pytest.skip("the page needs node and `npm install` in web/")
    if not os.path.exists(os.path.join(web, "models", "keipilot_mc_depth.onnx")):
        pytest.skip("no web/models/keipilot_mc_depth.onnx")
    res = subprocess.run([node, "tools/fill_dump.mjs", "--town", "1000", "--sec", "5", "--every", "1", "--out", str(tmp_path)],
                         cwd=web, capture_output=True, text=True, timeout=600)
    if res.returncode != 0 and "No Chrome found" in res.stderr + res.stdout:
        pytest.skip("no Chrome")
    assert res.returncode == 0, res.stdout + res.stderr
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from fill_data import check_run
    arrays, meta = load_run(str(tmp_path / "1000"))
    n = len(meta["samples"])
    assert n >= 3 and all(arrays[k].shape[0] == n for k in ARRAYS)
    assert all(s["prevFrames"] == 2 for s in meta["samples"][1:])          # the fill before: FILL.every model frames earlier
    r = check_run(str(tmp_path / "1000"))
    assert not r["bad"] and not any(r["mismatch"].values()), r
    assert all(b["n"] > 100 and abs(b["median"]) < 0.005 for b in r["road_z"][:2]), r["road_z"]
    assert 0.02 < r["holes"] / r["omega"] < 0.6                            # the view has holes, and not only holes


def test_fillnet_first_guess():
    """Before training the net answers its first guess: the view's own colour and depth where it has a surface, the
    flat ground's depth where it has none; with and without the loose rule's picture."""
    import torch
    from keipilot.fillnet import CODE_MAX_R, FillNet, fill_loss
    torch.manual_seed(0)
    rgb, dep = torch.rand(2, 3, CHASE.height, CHASE.width), torch.rand(2, 1, CHASE.height, CHASE.width)
    dep[:, :, :, :100] = 0.0                                 # a hole: the left of the picture
    for loose in (True, False):
        net = FillNet(loose=loose).eval()
        with torch.no_grad():
            out_rgb, code, surf = net(rgb, dep, rgb, dep)
        ground = net.ground.expand_as(dep)
        assert torch.equal(out_rgb, rgb * (dep > 0))
        assert torch.equal(code[dep > 0], dep[dep > 0])
        hole = (dep == 0) & (ground > 0)
        assert torch.equal(code[hole], ground[hole]) and hole.any()
        assert torch.all(code[(dep == 0) & (ground == 0)] == CODE_MAX_R)
        assert torch.all(surf == 2.0)
        batch = {"in_rgb": rgb, "in_dep": dep, "loose_rgb": rgb, "loose_dep": dep, "gt_rgb": rgb, "gt_dep": dep.clamp(min=0.1)}
        loss, parts = fill_loss(net.train(), batch)
        loss.backward()
        assert torch.isfinite(loss) and set(parts) == {"surface", "depth", "rgb", "grad"}


def test_fillgen_keeps_the_shape_and_learns_only_the_colour():
    """P2 (keipilot/fillgen.py): the depth and "a surface here" are P1's own; a fresh colour net answers P1's colour
    plus the fixed noise; with r = t the mean flow's target is the plain velocity; only the colour net gets gradients."""
    import torch
    from keipilot import fillgen
    from keipilot.fillnet import FillNet
    torch.manual_seed(0)
    shape = FillNet(loose=False)
    torch.nn.init.normal_(shape.head.weight, std=0.05)
    gen = fillgen.FillGen(shape, scale=0.2).eval()
    rgb, dep = torch.rand(2, 3, CHASE.height, CHASE.width), torch.rand(2, 1, CHASE.height, CHASE.width)
    dep[:, :, :, :100] = 0.0
    with torch.no_grad():
        want = shape(rgb, dep)
        got = gen(rgb, dep)
    assert torch.equal(got[1], want[1]) and torch.equal(got[2], want[2])
    assert torch.allclose(got[0], want[0].clamp(0, 1) + 0.2 * gen.noise, atol=1e-6)
    assert gen.config()["kind"] == "gen" and fillgen.FillGen(**{k: v for k, v in gen.config().items() if k != "kind"}).colour.width == gen.colour.width

    t, r = fillgen.sample_times(4000, -2.0, 2.0, 0.75, "cpu")
    assert torch.all(t >= r) and 0.70 < float((t > r).float().mean()) < 0.80
    batch = {"in_rgb": rgb, "in_dep": dep, "loose_rgb": rgb, "loose_dep": dep, "gt_rgb": torch.rand_like(rgb), "gt_dep": dep.clamp(min=0.1)}
    torch.nn.init.normal_(gen.colour.head.weight, std=0.05)
    gen.train()
    assert not shape.training
    loss, parts = fillgen.meanflow_loss(gen, batch)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None for p in shape.parameters()) and gen.colour.head.weight.grad.abs().sum() > 0
    # r = t everywhere: plain flow matching, the loss's error is u - (e - x); the same draws give the same number
    torch.manual_seed(1)
    _, fm = fillgen.meanflow_loss(gen, batch, ratio=0.0)
    assert "mse_mf" not in fm and fm["mse"] == fm["mse_fm"]
    # param "x": a fresh net answers P1's colour itself; the velocity is (z - f) / t
    gx = fillgen.FillGen(shape, scale=0.2, param="x").eval()
    with torch.no_grad():
        assert torch.equal(gx(rgb, dep)[0], want[0].clamp(0, 1))
        z, t = torch.randn(2, 3, CHASE.height, CHASE.width), torch.full((2, 1, 1, 1), 0.5)
        cond = gx.condition(rgb, dep)[0]
        assert torch.allclose(gx.velocity(z, t, t, cond), z / 0.5)
    loss, _ = fillgen.meanflow_loss(gx.train(), batch)
    assert torch.isfinite(loss)


def test_fillgan_keeps_the_shape_and_the_measures_tell_noise_from_lines():
    """The other way to the colours (docs/recon_fill.md 6.10): param "g" has no noise, a fresh net answers P1's colour,
    the depth is P1's; the discriminator's loss reaches the colour net alone. tools/fill_eval.py's "lines" is high for
    stripes and low for noise."""
    import torch
    from keipilot import fillgen
    from keipilot.fillnet import FillNet
    from tools.fill_eval import coherence
    torch.manual_seed(0)
    shape = FillNet(loose=False)
    torch.nn.init.normal_(shape.head.weight, std=0.05)
    gen = fillgen.FillGen(shape, scale=0.2, param="g").eval()
    rgb, dep = torch.rand(2, 3, CHASE.height, CHASE.width), torch.rand(2, 1, CHASE.height, CHASE.width)
    dep[:, :, :, :100] = 0.0
    with torch.no_grad():
        want, got = shape(rgb, dep), gen(rgb, dep)
    assert torch.equal(got[1], want[1]) and torch.equal(got[2], want[2]) and torch.equal(got[0], want[0].clamp(0, 1))
    again = fillgen.FillGen(**{k: v for k, v in gen.config().items() if k != "kind"})
    assert again.param == "g" and again.colour.noise is False

    disc = fillgen.PatchDisc(11)
    batch = {"in_rgb": rgb, "in_dep": dep, "loose_rgb": rgb, "loose_dep": dep, "gt_rgb": torch.rand_like(rgb), "gt_dep": dep.clamp(min=0.1)}
    torch.nn.init.normal_(gen.colour.head.weight, std=0.05)
    gen.train()
    cond, fake, comp, w = fillgen.gan_picture(gen, batch)
    assert disc(comp, cond).shape[1:] == (1, 38, 78)
    ld, _ = fillgen.disc_loss(disc, cond, comp, batch["gt_rgb"], w)
    ld.backward()
    assert gen.colour.head.weight.grad is None and disc.net[0].parametrizations.weight.original.grad.abs().sum() > 0
    lg, parts = fillgen.gen_loss(disc, cond, fake, comp, batch["gt_rgb"], w)
    lg.backward()
    assert torch.isfinite(lg) and all(p.grad is None for p in shape.parameters()) and gen.colour.head.weight.grad.abs().sum() > 0
    # outside the target pixels the discriminator sees the town's own picture
    target = (batch["gt_dep"] > 0) & (batch["gt_dep"] <= fillgen.CODE_MAX_R)
    assert torch.equal(comp[(~target).expand_as(comp)], batch["gt_rgb"][(~target).expand_as(comp)])

    region = torch.ones(1, 1, 64, 64, dtype=torch.bool)
    stripes = (torch.arange(64)[None, None, None, :] // 4 % 2).float().expand(1, 3, 64, 64)
    ratio = lambda x: (lambda c, e: c / e)(*coherence(x, region))
    assert ratio(stripes) > 0.95 and ratio(torch.rand(1, 3, 64, 64)) < 0.45


def test_fillgan_with_the_fill_before():
    """P2c (docs/recon_fill.md 6.14): with prev=True and no fill before, a net started from one without it answers the
    same; the depth stays P1's; the degraded fill before has no surface where it had none; dropped samples have none."""
    import torch
    from keipilot import fillgen
    from keipilot.fillnet import FillNet
    torch.manual_seed(0)
    shape = FillNet(loose=False)
    torch.nn.init.normal_(shape.head.weight, std=0.05)
    old = fillgen.FillGen(shape, scale=0.2, param="g").eval()
    torch.nn.init.normal_(old.colour.head.weight, std=0.05)
    gen = fillgen.FillGen(shape, scale=0.2, param="g", prev=True).eval()
    state, first = old.colour.state_dict(), "enc.0.0.weight"
    w = torch.zeros_like(gen.colour.state_dict()[first])
    w[:, :state[first].shape[1]] = state[first]
    gen.colour.load_state_dict({**state, first: w})
    rgb, dep = torch.rand(2, 3, CHASE.height, CHASE.width), torch.rand(2, 1, CHASE.height, CHASE.width)
    dep[:, :, :, :100] = 0.0
    prgb, pdep = torch.rand_like(rgb), torch.rand_like(dep)
    pdep[:, :, :40] = 0.0
    with torch.no_grad():
        want, none, with_prev = old(rgb, dep), gen(rgb, dep), gen(rgb, dep, prev_rgb=prgb, prev_dep=pdep)
    assert torch.equal(none[0], want[0]) and torch.equal(with_prev[0], want[0])          # the new planes' weights are zero
    assert torch.equal(with_prev[1], want[1]) and torch.equal(with_prev[2], want[2])
    cfg = gen.config()
    assert cfg["prev"] is True and fillgen.FillGen(**{k: v for k, v in cfg.items() if k != "kind"}).ncond == 16

    drgb, ddep = fillgen.degrade_prev(prgb, pdep)
    assert torch.all(ddep[pdep == 0] == 0) and torch.all(drgb[(ddep == 0).expand_as(drgb)] == 0) and float(drgb.max()) <= 1.0
    batch = {"in_rgb": rgb, "in_dep": dep, "loose_rgb": rgb, "loose_dep": dep, "gt_rgb": torch.rand_like(rgb), "gt_dep": dep.clamp(min=0.1), "prev_rgb": prgb, "prev_dep": pdep}
    nrgb, ndep = fillgen.training_prev(gen, batch, p_drop=1.0)
    assert float(nrgb.abs().sum()) == 0 and float(ndep.abs().sum()) == 0
    srgb, sdep = fillgen.training_prev(gen, batch, p_drop=0.0, p_self=1.0)
    assert srgb.shape == rgb.shape and torch.all(sdep[pdep == 0] == 0)
    gen.train()
    disc = fillgen.PatchDisc(gen.ncond)
    cond, fake, comp, wgt = fillgen.gan_picture(gen, batch, prev=(srgb, sdep))
    assert cond.shape[1] == 16
    lg, _ = fillgen.gen_loss(disc, cond, fake, comp, batch["gt_rgb"], wgt)
    lg.backward()
    assert gen.colour.enc[0][0].weight.grad[:, 11:].abs().sum() > 0


def test_fillgan_condition_in_two_parts_is_the_same_net():
    """The ONNX export of a net with the fill before gives the colour net its condition in two parts (WebGPU's limit
    of buffers per shader): the same answer as with the condition in one."""
    import torch
    from keipilot import fillgen
    from keipilot.fillnet import FillNet
    torch.manual_seed(0)
    shape = FillNet(loose=False)
    gen = fillgen.FillGen(shape, scale=0.2, param="g", prev=True).eval()
    for p in gen.colour.parameters():
        torch.nn.init.normal_(p, std=0.05)
    rgb, dep = torch.rand(1, 3, CHASE.height, CHASE.width), torch.rand(1, 1, CHASE.height, CHASE.width)
    prgb, pdep = torch.rand_like(rgb), torch.rand_like(dep) * (torch.rand_like(dep) > 0.3)
    with torch.no_grad():
        one, one0 = gen(rgb, dep, prev_rgb=prgb, prev_dep=pdep)[0], gen(rgb, dep)[0]
        gen.split = True
        two, two0 = gen(rgb, dep, prev_rgb=prgb, prev_dep=pdep)[0], gen(rgb, dep)[0]
    assert torch.allclose(one, two, atol=1e-5) and torch.allclose(one0, two0, atol=1e-5) and not torch.allclose(one, one0, atol=1e-3)
