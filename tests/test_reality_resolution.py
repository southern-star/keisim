from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from keireality.resolution import CHANNELS, SIZE, ResolutionDrive, ResolutionHead
from keisim.config import SEM


def test_new_head_starts_at_the_existing_output_on_the_native_grid():
    model = ResolutionHead().eval()
    inputs = torch.rand(1, CHANNELS, *SIZE)
    with torch.inference_mode():
        torch.testing.assert_close(model(inputs), inputs[:, :3])


def test_projection_history_only_crosses_consecutive_ticks_of_one_drive():
    display = ResolutionDrive(SimpleNamespace(observe=lambda *args: None))
    features = {"source_logits": torch.zeros(1, 2, max(SEM.values()) + 1, 4, 8)}
    display.observe(features, "first", 0., None)
    display.current[0] = "first observation"
    display.observe(features, "first", .1, None)
    assert display.prior == {0: "first observation"}
    assert display.current == {}
    display.current[0] = "second observation"
    display.observe(features, "first", .3, None)
    assert display.prior == {}
    display.current[0] = "third observation"
    display.observe(features, "next", .4, None)
    assert display.prior == {}


def test_white_line_scoring_uses_the_actual_paint_color(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from score_reality_resolution import Scores

    # Only the two edges of the white stripe should contribute. Yellow paint has no prediction.
    labels = np.array([[SEM["road"], SEM["marking"], SEM["road"], SEM["marking"], SEM["road"]]])
    truth = np.array([[[0, 0, 0], [240, 240, 240], [0, 0, 0], [240, 200, 0], [0, 0, 0]]], np.uint8)
    pred = truth.copy()
    pred[0, 3] = 0
    score = Scores(["pred"])
    score.frame(pred[None], truth, labels, np.ones_like(labels, dtype=bool), np.full_like(labels, -1))
    result = score.results()
    assert result["counts"]["white_line_gradient_mae"] == 2
    assert result["methods"]["pred"]["white_line_gradient_mae"] == 0.
    assert result["methods"]["pred"]["white_line_contrast"] == 1.
