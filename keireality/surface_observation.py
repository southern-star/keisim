"""Trace the actual weight of the projected image through the frozen C compositor."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .scene_detail import SceneDetail


@dataclass
class SurfaceFrame:
    rgb: torch.Tensor
    depth: torch.Tensor
    static: torch.Tensor
    direction: torch.Tensor
    alpha: torch.Tensor
    surface_rgb: torch.Tensor
    cover: torch.Tensor
    optical_depth: torch.Tensor
    moving: torch.Tensor


class ObservedDetail(SceneDetail):
    """A fourth color channel tags a projection with 1 and all other layers with 0.

    Its composited value is the exact RGB mixture weight after ground and sky
    occlusion. The three original color channels follow the same operations as C.
    """
    def surfaces(self, colors, probs, view, depth):
        rgb, cover, near, moving = super().surfaces(colors, probs, view, depth)
        self.last_surface = rgb, cover, near, moving
        return torch.cat((rgb, torch.ones_like(cover)), 1), cover, near, moving

    def directional_color(self, points, colors, probs, depth, *, sky):
        rgb, weight = super().directional_color(points, colors, probs, depth, sky=sky)
        return F.pad(rgb, (0, 0, 0, 0, 0, 1)), weight

    def inspect(self, base, encoded, view, estimated):
        rgba, radial, static, direction = super().frame(
            F.pad(base, (0, 0, 0, 0, 0, 1)), encoded, view, estimated)
        surface_rgb, cover, near, moving = self.last_surface
        return SurfaceFrame(rgba[:, :3], radial, static, direction, rgba[:, 3:4],
                            surface_rgb, cover, near, moving)
