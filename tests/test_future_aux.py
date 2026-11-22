"""The auxiliary future head (docs/future_aux.md; run: python -m pytest tests -q): off by default, and a model,
a data set and a training run without it are as they were."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from keipilot.data import (FUTURE_HORIZONS, STEP_S, DrivingDataset, ShardWriter, future_index,  # noqa: E402
                           load_shards, write_meta)
from keipilot.model import FUTURE_MOVING_WEIGHT, KeiPilot, compute_loss, future_loss  # noqa: E402
from keisim.config import N_SEM, SEM  # noqa: E402

MC14 = os.path.join(ROOT, "runs", "keipilot_mc14", "last.pt")
# the samples of a data set without the future head (keipilot.data.DrivingDataset.__getitem__ before it)
KEYS = {"img", "seg", "cmd", "tp", "path", "speed", "tl", "v", "v_known", "stand_m", "stand_known", "img_prev",
        "has_prev", "yield_tta"}
CFG = dict(pretrained=False, speed_input=True, mem_input=True, aux_yield=True)


def seg_class(step, cam=0):
    """The one class a test frame's segmentation is filled with: tells the frames (and cameras) apart."""
    return (step // 2 + 5 * cam) % N_SEM


def write_set(out, episodes, cams=2, prefix="expert_w00", per_shard=1000):
    """A tiny data set: episodes = [(town, episode, [(step, virtual), ...]), ...], random pictures."""
    rng = np.random.default_rng(0)
    w = ShardWriter(str(out), prefix, frames_per_shard=per_shard)
    for town, ep, frames in episodes:
        for step, virtual in frames:
            views = [(rng.integers(0, 256, (160, 320, 3), dtype=np.uint8),
                      np.full((160, 320), seg_class(step, c), np.uint8)) for c in range(cams)]
            w.add(*views[0], extra=views[1:], cmd=np.int8(1), tp=np.array([20.0, 0.0], np.float32),
                  path=np.stack([np.arange(1, 11) * 2.0, np.zeros(10)], 1).astype(np.float32), speed=np.float32(5.0),
                  tl=np.int8(3), v=np.float32(0.0 if step < 4 else 4.0), reason=np.int8(0), town=np.int32(town),
                  episode=np.int64(ep), step=np.int32(step), weather=np.int8(0), virtual=np.bool_(virtual))
    w.flush()
    write_meta(str(out), label_version=7, renderer="keisim", **({"rig": "wide_narrow"} if cams > 1 else {}))


def brute_force(data, files, horizons):
    """future_index, one frame at a time."""
    dirs = [os.path.dirname(os.path.abspath(f)) for f in files]
    key = [(dirs[s], int(t), int(e)) for s, t, e in zip(data["src"], data["town"], data["episode"])]
    n = len(key)
    seen, twice = set(), set()
    for i in range(n):
        k = key[i] + (int(data["step"][i]),)
        if k in seen:
            twice.add(key[i])
        seen.add(k)
    fut = np.full((n, len(horizons)), -1, np.int64)
    for i in range(n):
        if data["virtual"][i] or key[i] in twice:
            continue
        for j, h in enumerate(horizons):
            for k in range(n):
                if key[k] == key[i] and not data["virtual"][k] and data["step"][k] == data["step"][i] + h:
                    fut[i, j] = k
    return fut, len(twice)


def test_future_index_virtual_frames_repeated_steps_and_twins():
    rows = []       # (file, town, episode, step, virtual)
    rows += [(0, 1, 7, s, s in (4, 10)) for s in range(0, 32, 2)]        # a/: steps 4 and 10 from a virtual camera
    rows += [(2, 1, 7, s, False) for s in range(0, 12, 2)]               # b/: a twin of the same town and episode
    rows += [(0, 1, 8, s, False) for s in (0, 2, 4, 4, 6, 8)]            # a step recorded twice
    rows += [(1, 2, 7, s, False) for s in range(0, 40, 4)]               # the same episode seed in another town;
    rows += [(0, 1, 9, s, False) for s in (0, 20)]                       # its frames in a second shard of a/
    rng = np.random.default_rng(1)
    rows = [rows[i] for i in rng.permutation(len(rows))]                 # in any order
    files = ["a/expert_w00_0000.npz", "a/expert_w00_0001.npz", "b/expert_w00_0000.npz"]
    data = {"src": np.array([r[0] for r in rows], np.int32), "town": np.array([r[1] for r in rows], np.int32),
            "episode": np.array([r[2] for r in rows], np.int64), "step": np.array([r[3] for r in rows], np.int32),
            "virtual": np.array([r[4] for r in rows])}
    fut, info = future_index(data, files, FUTURE_HORIZONS)
    want, twice = brute_force(data, files, FUTURE_HORIZONS)
    assert np.array_equal(fut, want)
    assert info["episodes"] == 5 and info["episodes_repeated_steps"] == twice == 1 and info["frames_repeated_steps"] == 6

    def at(file, town, ep, step):
        return rows.index(next(r for r in rows if r[:4] == (file, town, ep, step)))
    j4, j10 = FUTURE_HORIZONS.index(4), FUTURE_HORIZONS.index(10)
    assert fut[at(0, 1, 7, 0), j4] == -1 and fut[at(0, 1, 7, 0), j10] == -1       # both ahead are virtual frames
    assert fut[at(0, 1, 7, 2), j4] == at(0, 1, 7, 6) and fut[at(0, 1, 7, 2), j10] == at(0, 1, 7, 12)
    assert (fut[at(0, 1, 7, 4)] == -1).all()                                      # a virtual frame has no future
    assert fut[at(0, 1, 7, 26), j4] == at(0, 1, 7, 30) and fut[at(0, 1, 7, 26), j10] == -1     # the episode ends
    assert fut[at(2, 1, 7, 0), j4] == at(2, 1, 7, 4) and fut[at(2, 1, 7, 2), j10] == -1        # the twin stays in b/
    assert all((fut[at(0, 1, 8, s)] == -1).all() for s in (0, 2, 6, 8))           # the whole episode is left out
    assert fut[at(1, 2, 7, 0), j4] == at(1, 2, 7, 4)                              # across a/'s shards, by town
    assert fut[at(0, 1, 9, 0), FUTURE_HORIZONS.index(20)] == at(0, 1, 9, 20)
    assert (future_index(data, files, (5, 15))[0] == -1).all()                    # odd look-aheads: nothing recorded


def test_future_index_of_no_frames():
    data = {k: np.zeros(0, t) for k, t in (("src", np.int32), ("town", np.int32), ("episode", np.int64),
                                           ("step", np.int32), ("virtual", bool))}
    fut, info = future_index(data, ["a/x_0000.npz"], FUTURE_HORIZONS)
    assert fut.shape == (0, len(FUTURE_HORIZONS)) and info["episodes"] == 0


@pytest.mark.parametrize("cams", [1, 2])
def test_model_without_the_head_is_unchanged(cams):
    torch.manual_seed(0)
    plain = KeiPilot(n_cams=cams, **CFG).eval()
    state = torch.random.get_rng_state()
    torch.manual_seed(0)
    again = KeiPilot(n_cams=cams, aux_future=False, **CFG)
    assert torch.equal(torch.random.get_rng_state(), state)        # the same random draws: the same initial model
    assert all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), again.state_dict().values()))
    assert not any(k.startswith("aux_future") for k in plain.state_dict())
    fut = KeiPilot(n_cams=cams, aux_future=True, **CFG).eval()
    new = fut.load_compatible(plain.state_dict())
    assert new and all(k.startswith("aux_future_") for k in new)
    n_new = sum(p.numel() for n, p in fut.named_parameters() if n.startswith("aux_future_"))
    assert n_new == sum(p.numel() for p in fut.parameters()) - sum(p.numel() for p in plain.parameters())
    assert 0.15e6 < n_new < 0.3e6
    B = 2
    img = torch.randint(0, 256, (B, 3, 160, 320) if cams == 1 else (B, cams, 3, 160, 320), dtype=torch.uint8)
    cmd, tp = torch.tensor([0, 2]), torch.tensor([[18.0, -3.0], [30.0, 6.0]])
    kw = dict(speed=torch.tensor([0.0, 7.0]), speed_known=torch.tensor([True, False]),
              stand_m=torch.tensor([0.0, 10.0]), stand_known=torch.tensor([True, True]))
    with torch.no_grad():
        for with_seg in (True, False):
            a = plain(img, cmd, tp, with_seg=with_seg, **kw)
            b = fut(img, cmd, tp, with_seg=with_seg, **kw)
            assert set(a) == set(b) and "future" not in b
            assert all(torch.equal(a[k], b[k]) for k in a)
            # asking for the future changes nothing else, and a model without the head ignores the request
            c = fut(img, cmd, tp, with_seg=with_seg, with_future=True, fut_dt=torch.tensor([0.4, 2.0]),
                    fut_v=torch.tensor([0.0, 7.0]), **kw)
            assert set(c) == set(a) | {"future"} and all(torch.equal(a[k], c[k]) for k in a)
            assert c["future"].shape == (B * cams, N_SEM, 20, 40)
            d = plain(img, cmd, tp, with_seg=with_seg, with_future=True, **kw)
            assert set(d) == set(a) and all(torch.equal(a[k], d[k]) for k in a)


def test_future_head_reads_the_look_ahead_and_the_speed():
    torch.manual_seed(0)
    m = KeiPilot(n_cams=2, aux_future=True, **CFG).eval()
    img = torch.randint(0, 256, (1, 2, 3, 160, 320), dtype=torch.uint8)
    cmd, tp = torch.tensor([1]), torch.tensor([[18.0, -3.0]])

    def fut(dt, v):
        with torch.no_grad():
            return m(img, cmd, tp, with_future=True, fut_dt=torch.tensor([dt]), fut_v=torch.tensor([v]))["future"]
    assert torch.equal(fut(0.4, 0.0), fut(2.0, 8.0))               # the FiLM starts at zero: no modulation yet
    with torch.no_grad():
        torch.nn.init.normal_(m.aux_future_film[2].weight, std=0.1)
    assert not torch.equal(fut(0.4, 5.0), fut(2.0, 5.0)) and not torch.equal(fut(1.0, 0.0), fut(1.0, 8.0))
    with torch.no_grad():   # without the two it defaults to 1 s at standstill
        assert torch.equal(m(img, cmd, tp, with_future=True)["future"], fut(1.0, 0.0))


@pytest.mark.parametrize("cams", [1, 2])
def test_future_loss_counts_only_frames_with_a_label_and_stays_out_of_the_logged_loss(cams):
    torch.manual_seed(0)
    B = 3
    m = KeiPilot(n_cams=cams, aux_future=True, **CFG)
    shape = (B, 80, 160) if cams == 1 else (B, cams, 80, 160)
    batch = {"img": torch.randint(0, 256, (B, 3, 160, 320) if cams == 1 else (B, cams, 3, 160, 320), dtype=torch.uint8),
             "seg": torch.randint(0, N_SEM, shape), "cmd": torch.tensor([0, 1, 2]), "tp": torch.randn(B, 2) * 10,
             "path": torch.randn(B, 10, 2), "speed": torch.rand(B) * 8, "tl": torch.tensor([0, 3, 2]),
             "yield_tta": torch.tensor([float("nan"), 2.0, 9.0]),
             "fut_seg": torch.randint(0, N_SEM, shape), "fut_ok": torch.tensor([True, False, True]),
             "fut_dt": torch.tensor([0.4, 1.0, 2.0]), "fut_v": torch.tensor([0.0, 3.0, 6.0])}
    out = m(batch["img"], batch["cmd"], batch["tp"], with_future=True, fut_dt=batch["fut_dt"], fut_v=batch["fut_v"])
    off, logs_off = compute_loss(m, out, batch, w_aux=0.5)
    on, logs_on = compute_loss(m, out, batch, w_aux=0.5, w_future=0.3)
    assert "future" not in logs_off and logs_on["loss"] == logs_off["loss"] == pytest.approx(off.item())
    assert {k: v for k, v in logs_on.items() if k != "future"} == logs_off
    assert on.item() == pytest.approx(off.item() + 0.3 * logs_on["future"], rel=1e-5)
    # the label of a frame without a future frame does not count
    other = dict(batch, fut_seg=batch["fut_seg"].clone())
    other["fut_seg"][1] = (other["fut_seg"][1] + 1) % N_SEM
    assert compute_loss(m, out, other, w_future=0.3)[1]["future"] == logs_on["future"]
    other["fut_seg"][0] = (other["fut_seg"][0] + 1) % N_SEM
    assert compute_loss(m, out, other, w_future=0.3)[1]["future"] != logs_on["future"]
    none = dict(batch, fut_ok=torch.zeros(B, dtype=torch.bool))
    loss, logs = compute_loss(m, out, none, w_future=0.3)
    assert logs["future"] == 0.0 and loss.item() == pytest.approx(off.item() - 0.5 * logs_off.get("aux", 0.0), rel=1e-5)
    # the weighted mean: vehicles and pedestrians count FUTURE_MOVING_WEIGHT times
    logits = torch.zeros(1, N_SEM, 20, 40)
    logits[:, SEM["road"]] = 2.0
    lab = torch.full((1, 80, 160), SEM["road"])
    lab[:, :, :40] = SEM["pedestrian"]
    ce = torch.nn.functional.cross_entropy(logits, lab[:, ::4, ::4], reduction="none")
    road, ped = ce[lab[:, ::4, ::4] == SEM["road"]].mean(), ce[lab[:, ::4, ::4] == SEM["pedestrian"]].mean()
    want = (0.75 * road + 0.25 * FUTURE_MOVING_WEIGHT * ped) / (0.75 + 0.25 * FUTURE_MOVING_WEIGHT)
    assert future_loss(logits, lab, torch.tensor([True])).item() == pytest.approx(want.item(), rel=1e-5)


EPISODES = [(1, 70, [(s, s in (8, 14)) for s in range(0, 44, 2)]),      # steps 8 and 14 from a virtual camera
            (1, 71, [(s, False) for s in range(0, 24, 2)]),
            (2, 72, [(s, False) for s in range(0, 24, 4)])]


@pytest.mark.parametrize("cams", [1, 2])
def test_dataset_returns_the_future_frame(tmp_path, cams):
    write_set(tmp_path / "a", EPISODES, cams, per_shard=15)             # the first episode runs over two shards
    for mmap in (False, True):
        data, files = load_shards([str(tmp_path / "a")], with_prev=False, mmap=mmap)
        n = len(data["step"])
        plain = DrivingDataset(data, np.arange(n), train=False)
        assert set(plain[0]) == KEYS
        data["fut_idx"], _ = future_index(data, files, FUTURE_HORIZONS)
        data["fut_dt"] = np.asarray(FUTURE_HORIZONS, np.float32) * np.float32(STEP_S)
        assert set(DrivingDataset(data, np.arange(n), train=True)[0]) == KEYS     # still off unless asked for
        val = DrivingDataset(data, np.arange(n), train=False, future=True)
        tr = DrivingDataset(data, np.arange(n), train=True, future=True)
        seen = set()
        for i in range(n):
            ep, step, virtual = int(data["episode"][i]), int(data["step"][i]), bool(data["virtual"][i])
            frames = dict(next(f for _, e, f in EPISODES if e == ep))
            have = [h for h in FUTURE_HORIZONS if frames.get(step + h) is False and not virtual]
            s, p = val[i], plain[i]
            assert set(s) == KEYS | {"fut_seg", "fut_ok", "fut_dt", "fut_v"}
            assert all(torch.equal(s[k], p[k]) or (k == "yield_tta" and torch.isnan(s[k])) for k in KEYS)
            assert s["fut_seg"].shape == s["seg"].shape == ((80, 160) if cams == 1 else (2, 80, 160))
            assert s["fut_seg"].dtype == torch.int64 and s["fut_v"].item() == pytest.approx(float(data["v"][i]))
            assert bool(s["fut_ok"]) == bool(have)
            if not have:
                assert not s["fut_seg"].any() and s["fut_dt"].item() == 1.0
                assert not bool(tr[i]["fut_ok"])
                continue
            h = min(have, key=lambda h: (abs(h - 10), h))               # closest to 1 s, the shorter of two
            assert s["fut_dt"].item() == pytest.approx(h * 0.1)
            for c in range(cams):
                fs = s["fut_seg"] if cams == 1 else s["fut_seg"][c]
                assert (fs == seg_class(step + h, c)).all()
            for _ in range(6):                                          # training: any of the look-aheads it has
                t = tr[i]
                h = round(t["fut_dt"].item() / 0.1)
                assert h in have and bool(t["fut_ok"])
                assert ((t["fut_seg"] if cams == 1 else t["fut_seg"][0]) == seg_class(step + h)).all()
                seen.add((i, h))
        assert len({h for _, h in seen}) == len(FUTURE_HORIZONS)


@pytest.mark.skipif(not os.path.exists(MC14), reason="runs/keipilot_mc14/last.pt is not here")
def test_mc14_loads_into_a_model_with_the_head():
    ck = torch.load(MC14, map_location="cpu")
    assert "aux_future" not in ck["model_cfg"]
    plain = KeiPilot(pretrained=False, **ck["model_cfg"]).eval()
    plain.load_state_dict(ck["model"])                                  # strict: nothing new without the head
    fut = KeiPilot(pretrained=False, aux_future=True, **ck["model_cfg"]).eval()
    new = fut.load_compatible(ck["model"])
    assert new and all(k.startswith("aux_future_") for k in new)
    K = ck["model_cfg"]["n_cams"]
    img = torch.randint(0, 256, (1, K, 3, 160, 320), dtype=torch.uint8)
    args = (img, torch.tensor([1]), torch.tensor([[18.0, -3.0]]))
    kw = dict(speed=torch.tensor([5.0]), stand_m=torch.tensor([3.0]))
    with torch.no_grad():
        a, b = plain(*args, **kw), fut(*args, with_future=True, **kw)
    assert all(torch.equal(a[k], b[k]) for k in a) and b["future"].shape == (K, N_SEM, 20, 40)


def run_train(data, out, *extra):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--data", str(data), "--out", str(out),
           "--epochs", "1", "--bs", "4", "--workers", "1", "--max_iters", "2", "--device", "cpu", "--speed_input",
           "--mem_input", "--aux_weight", "0.5", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout, torch.load(os.path.join(out, "last.pt"), map_location="cpu"), \
        json.load(open(os.path.join(out, "history.json")))


def test_train_runs_with_and_without_the_head(tmp_path):
    """A few iterations on the CPU. Without --future_weight: no head, no future metrics, the samples' keys as
    before (test_dataset_returns_the_future_frame). With it: the head, its loss and its metrics."""
    eps = [(1, 70 + k, [(s, s % 10 == 6) for s in range(0, 60, 2)]) for k in range(6)]
    write_set(tmp_path / "d", eps, cams=2)
    log, ck, hist = run_train(tmp_path / "d", tmp_path / "off")
    assert "future" not in log and "aux_future" not in ck["model_cfg"] and ck["args"]["future_weight"] == 0.0
    assert not any(k.startswith("aux_future") for k in ck["model"])
    assert not any(k.startswith("fut") for k in hist[0]) and "future" not in hist[0]["train"]
    base_keys = set(hist[0])
    log, ck, hist = run_train(tmp_path / "d", tmp_path / "on", "--future_weight", "0.5",
                              "--init", str(tmp_path / "off" / "last.pt"))
    assert "future labels ([4, 10, 16, 20] steps ahead)" in log and "aux_future_film.0.weight" in log
    assert ck["model_cfg"]["aux_future"] is True and any(k.startswith("aux_future_head") for k in ck["model"])
    h = hist[0]
    assert set(h) == base_keys | {"future", "fut_frames", "fut_mIoU", "fut_IoU_vehicle", "fut_IoU_pedestrian",
                                  "fut_persist_mIoU", "fut_persist_IoU_vehicle", "fut_persist_IoU_pedestrian",
                                  "fut_detail"}
    assert "future" in h["train"] and h["fut_frames"] > 0 and {"all", "moving", "1.0s"} <= set(h["fut_detail"])
    # every test frame is one class, another one each step: persistence scores nothing
    assert h["fut_persist_mIoU"] == 0.0 and h["fut_detail"]["all"]["head"]["pixels"] == h["fut_frames"] * 2 * 80 * 160
    # the logged loss is the one without the future term, as in the run without the head
    assert h["loss"] == pytest.approx(h["path"] + h["speed"] + 0.5 * h["tl"] + 0.5 * h["seg"], rel=1e-4)
    # a model with the head drives as any other: the agent loads it and plans without the head
    from keipilot.agent import load_model
    m = load_model(os.path.join(tmp_path, "on", "last.pt"), "cpu")
    assert m.aux_future


def test_export_onnx_of_mc14_is_unchanged(tmp_path):
    """scripts/export_onnx.py is untouched and calls the heads one by one: the file it writes for mc14 gives the
    released keipilot_mc.onnx's outputs (KEIPILOT_MC_ONNX: that file; needs onnx and onnxruntime)."""
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    ref = os.environ.get("KEIPILOT_MC_ONNX")
    if not ref or not os.path.exists(ref) or not os.path.exists(MC14):
        pytest.skip("KEIPILOT_MC_ONNX (the released keipilot_mc.onnx) or runs/keipilot_mc14/last.pt is not here")
    dst = str(tmp_path / "mc14.onnx")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "export_onnx.py"), MC14, dst],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rng = np.random.default_rng(3)
    a, b = ort.InferenceSession(ref), ort.InferenceSession(dst)
    assert [(i.name, i.shape) for i in a.get_inputs()] == [(i.name, i.shape) for i in b.get_inputs()]
    assert [o.name for o in a.get_outputs()] == [o.name for o in b.get_outputs()]
    for _ in range(3):
        feed = {i.name: (rng.integers(0, 256, i.shape).astype(np.float32) if i.name.startswith("im")
                         else rng.normal(size=i.shape).astype(np.float32)) for i in a.get_inputs()}
        for x, y in zip(a.run(None, feed), b.run(None, feed)):
            assert np.array_equal(x, y)
