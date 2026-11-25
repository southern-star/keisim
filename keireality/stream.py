"""Streaming temporal display. Observe once per driving tick, then render any number of ego-relative views."""
import torch

from .history import HistoryWindow


class TemporalDisplay:
    def __init__(self, renderer, projection=None):
        if renderer.history is None:
            raise ValueError("a temporal renderer is required")
        self.model = renderer.eval()
        self.projection = projection.eval() if projection is not None else None
        self.projected_source = None
        self.window = HistoryWindow()
        self.features = None
        self.past = self.motion = self.valid = None

    @torch.inference_mode()
    def observe(self, features, episode, t, delta):
        mem = features["mem"]
        if mem.shape != (1, 500, 256):
            raise ValueError("streaming observations require one driving tick")
        source = None
        if self.projection is not None:
            if "estimated_depth" not in features or features["estimated_depth"].shape != (1, 2, 80, 160):
                raise ValueError("depth projection requires a synchronized estimated-depth grid")
            source = self.projection.encode(features["c2"], features["c3"])
        fine = mem[:, :400].reshape(2, 10, 20, 256)[:, ::2, ::2]
        coarse = mem[:, 400:].reshape(2, 5, 10, 256)
        packed = (.5 * fine.float() + .5 * coarse.float()).permute(0, 3, 1, 2).half()
        previous, motion = self.window.push(packed, episode, t, delta)
        self.past = packed.new_zeros(1, 7, *packed.shape)
        if previous:
            self.past[0, :len(previous)] = torch.stack(previous)
        self.valid = (torch.arange(7, device=mem.device) < len(previous))[None]
        self.motion = torch.as_tensor(motion, device=mem.device)[None]
        self.features = features
        self.projected_source = source

    @torch.inference_mode()
    def render(self, view, *, return_reference=False):
        if self.features is None:
            raise ValueError("observe a driving tick first")
        rgb, semantic, support = self.model(self.features["mem"], view, self.features["c2"], self.features["c3"],
                                            self.past, self.motion, self.valid)
        result = rgb if self.projection is None else self.projection.render_encoded(
            rgb, self.projected_source, view, self.features["estimated_depth"])
        return (result, semantic, support, rgb) if return_reference else (result, semantic, support)
