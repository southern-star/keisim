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
