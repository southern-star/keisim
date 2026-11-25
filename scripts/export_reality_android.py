#!/usr/bin/env python3
"""Export the trained 384×192 display for a single phone observation and a fixed chase view."""
import argparse
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.onnx import symbolic_helper

from diagnose_reality_blur import ROOT, frozen
from keireality.dataset import file_hash, write_json
from keireality.encoder import extract_mem
from keireality.geometry import orbit_camera, view_pose
from keireality.refined_display import RefinedDrive
from keireality.resolution import ResolutionDrive, ResolutionHead


class PhoneFrame(nn.Module):
    def __init__(self):
        super().__init__()
        self.pilot, self.depth, self.renderer, self.detail = frozen("cpu")
        fit = ROOT / "runs/reality/p2_resolution/fit"
        selection = json.loads((fit / "selection.json").read_text())
        if (file_hash(fit / "head.pt") != selection["checkpoint"] or
                file_hash(ROOT / "keireality/resolution.py") != selection["model"]):
            raise ValueError("The selected 384 display has changed")
        self.strength = selection["strength"]
        self.head = ResolutionHead()
        self.head.load_state_dict(torch.load(fit / "head.pt", map_location="cpu")["model"])
        self.register_buffer("view", torch.from_numpy(view_pose(orbit_camera(0)))[None])

    def forward(self, rig, visible):
        features = extract_mem(self.pilot, rig, pyramid=True, scene=True)
        for name in ("c2", "c3", "mem"):
            features[name] = features[name].half()
        depth = self.depth.depth_metres(self.depth.forward_depth(rig)).reshape(1, 2, 80, 160)
        features["estimated_depth"] = depth * visible.float()
        features["source_logits"] = features["source_logits"] * visible[:, :, None, ::2, ::2]
        baseline = RefinedDrive(self.renderer, self.detail, None, "ray_aa_strong")
        display = ResolutionDrive(baseline, self.head, self.strength)
        # A handheld camera supplies no vehicle odometry. Start a fresh observation for each frame.
        display.observe(features, 0, 0., np.zeros(3, np.float32))
        _, inputs = display.prepare(self.view)
        rgb = self.head(inputs.float(), self.strength)
        return (rgb * 255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).contiguous()


def example(town, frame, mask_edges):
    path = ROOT / f"runs/reality/p2_resolution/capture/town{town}/cameras.npy"
    bgr = np.load(path, mmap_mode="r")[frame]
    rgb = torch.from_numpy(bgr[..., ::-1].copy()).permute(0, 3, 1, 2)[None]
    valid = torch.ones(1, 2, 80, 160, dtype=torch.uint8)
    if mask_edges:
        valid[:, 0, :, :43] = valid[:, 0, :, -43:] = 0
        rgb[:, 0, :, :, :86] = rgb[:, 0, :, :, -86:] = torch.tensor([124,116,104])[:,None,None]
    return rgb, valid


def export(dest):
    torch.set_num_threads(3)
    torch.backends.mha.set_fastpath_enabled(False)
    def radians(graph, value):
        factor = graph.op("Constant", value_t=torch.tensor(math.pi / 180, dtype=torch.float32))
        return graph.op("Mul", value, factor)

    def hypot(graph, x, y):
        return graph.op("Sqrt", graph.op("Add", graph.op("Mul", x, x), graph.op("Mul", y, y)))

    def atan2(graph, y, x):
        # Constant-folded camera rays can be promoted to double by the exporter.
        # The mobile CPU kernels use float coordinates, as in the eager renderer.
        def scalar(value):
            return graph.op("Constant", value_t=torch.tensor(value, dtype=torch.float32))
        x, y = graph.op("Cast", x, to_i=1), graph.op("Cast", y, to_i=1)
        ax, ay = graph.op("Abs", x), graph.op("Abs", y)
        r = graph.op("Div", graph.op("Min", ax, ay), graph.op("Max", graph.op("Max", ax, ay), scalar(1e-20)))
        angle = graph.op("Atan", r)
        angle = graph.op("Where", graph.op("Greater", ay, ax), graph.op("Sub", scalar(math.pi/2), angle), angle)
        angle = graph.op("Where", graph.op("Less", x, scalar(0)), graph.op("Sub", scalar(math.pi), angle), angle)
        return graph.op("Where", graph.op("Less", y, scalar(0)), graph.op("Neg", angle), angle)

    @symbolic_helper.parse_args("v","v","i","i","b")
    def grid_sample(graph, source, grid, mode, padding, align):
        return graph.op("GridSample", source, graph.op("Cast", grid, to_i=1),
                        mode_s=("bilinear","nearest","bicubic")[mode],
                        padding_mode_s=("zeros","border","reflection")[padding], align_corners_i=int(align))

    @symbolic_helper.parse_args("v","i","v","v","s","b")
    def scatter_reduce(graph, source, axis, index, values, reduction, include_self):
        if reduction!="amin" or not include_self:
            raise ValueError("Expected the depth projector's nearest-surface reduction")
        return graph.op("ScatterElements", source, index, values, axis_i=axis, reduction_s="min")

    torch.onnx.register_custom_op_symbolic("aten::deg2rad", radians, 18)
    torch.onnx.register_custom_op_symbolic("aten::hypot", hypot, 18)
    torch.onnx.register_custom_op_symbolic("aten::atan2", atan2, 18)
    torch.onnx.register_custom_op_symbolic("aten::grid_sampler", grid_sample, 18)
    torch.onnx.register_custom_op_symbolic("aten::scatter_reduce", scatter_reduce, 18)
    model = PhoneFrame().eval().requires_grad_(False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fixtures = [example(6, 15, True), example(7, 36, False), example(6, 82, True)]
    with tempfile.TemporaryDirectory(dir=dest.parent) as local:
        path = Path(local) / "keireality.onnx"
        with torch.inference_mode():
            torch.onnx.export(model, fixtures[0], str(path), input_names=["rig","visible"], output_names=["rgb"],
                              opset_version=18, dynamo=False, external_data=False)
        onnx.checker.check_model(str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 3
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        rows = []
        for rgb, valid in fixtures:
            with torch.inference_mode():
                expected = model(rgb, valid).numpy()
            start = time.perf_counter()
            output, = session.run(None, {"rig":rgb.numpy(),"visible":valid.numpy()})
            error = np.abs(output.astype(np.int16)-expected.astype(np.int16))
            if output.shape != (1,192,384,3) or output.std()<5 or error.mean()>.5:
                raise ValueError(f"Conversion mismatch: shape={output.shape}, mean error={error.mean()}")
            rows.append(dict(mean_rgb8=float(error.mean()), max_rgb8=int(error.max()),
                             host_cpu_ms=1000*(time.perf_counter()-start)))
        path.replace(dest)
    stats = ROOT / "runs/reality/android/export.json"
    stats.parent.mkdir(parents=True,exist_ok=True)
    write_json(stats,dict(sha256=file_hash(dest),bytes=dest.stat().st_size,parity=rows))
    print(dest, rows, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"android/app/src/main/assets/keireality.onnx")
    export(parser.parse_args().out)
