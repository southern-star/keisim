"""Causal background repair around moving actors, layered over the released 384 display."""
from math import isclose

import torch
from torch import nn
from torch.nn import functional as F

from keisim.config import SEM
from .background_history import old_camera, previous_grid
from .resolution import SIZE
from .scene_detail import sample
from .spatial import ray_points


class ActorContext:
    """Track estimated actor footprints per view slot, in the coordinate system of the background."""
    def __init__(self):
        self.episode = self.time = None
        self.previous, self.current = {}, {}

    def observe(self, episode, t):
        continuous = self.episode == episode and self.time is not None and isclose(t - self.time, .1, abs_tol=1e-6)
        self.previous, self.current = self.current if continuous else {}, {}
        self.episode, self.time = episode, t

    def features(self, native, view, slot):
        control = native.baseline.control
        coarse, logits, _ = control.reference.render(view)
        target = logits.float().softmax(1)
        actor = target[:, [SEM["vehicle"], SEM["pedestrian"]]].sum(1, keepdim=True)
        _, coverage, _, moving = control.detail.last_surface
        actor = torch.maximum(actor, coverage * moving).clamp(0, 1)
        old_actor = torch.zeros_like(actor)
        if slot in self.previous:
            previous_actor, previous_view = self.previous[slot]
            height, width = actor.shape[-2:]
            points = ray_points(view, height, width, (1.,))
            origin = view[:, None, None, :3]
            direction = points[:, 1] - origin
            distance = ((points[:, 0] - origin) * view[:, None, None, 3:6]).sum(-1)[:, None]
            estimated_z = F.interpolate(native.current[slot][1], (height, width), mode="nearest")
            background_z = torch.where((points[:, 0, :, :, 2].abs() < .03)[:, None], distance, estimated_z)
            previous = old_camera(view, control.reference.motion[0, :1])
            grid, inside, _ = previous_grid(previous, previous_view, background_z, direction, view)
            old_actor = sample(previous_actor, grid) * inside
        self.current[slot] = actor, view
        maps = torch.cat((coarse, actor, old_actor), dim=1)
        return F.interpolate(maps, SIZE, mode="bilinear", align_corners=False)


def actor_support(context):
    # Include the narrow band around uncertain silhouettes. Keep the interior of confident actors.
    current, previous = context[:, 3:4], context[:, 4:5]
    vicinity = F.max_pool2d(torch.maximum(current, previous), 13, 1, 6)
    return ((vicinity - .035) / .45).clamp(0, 1) * (1 - current.square())


class BackgroundRepair(nn.Module):
    """A bounded local correction conditioned on current color and estimated actor visibility."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(25, 20, 3, padding=1, padding_mode="replicate"), nn.SiLU(),
            nn.Conv2d(20, 20, 3, padding=2, dilation=2, padding_mode="replicate"), nn.SiLU(),
            nn.Conv2d(20, 3, 3, padding=1, padding_mode="replicate"),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, release, inputs, context, strength=1.):
        signals = torch.cat((release, inputs, context[:, :3] - release, context[:, 3:]), dim=1)
        correction = .30 * self.layers(signals.contiguous(memory_format=torch.channels_last)).float().tanh()
        return (release + strength * actor_support(context) * correction).clamp(0, 1)


class DisocclusionDrive:
    def __init__(self, native, repair, strength):
        self.native = native
        self.repair = repair
        self.strength = strength
        self.context = ActorContext()

    def observe(self, features, episode, t, delta):
        self.native.observe(features, episode, t, delta)
        self.context.observe(episode, t)

    @torch.inference_mode()
    def paired(self, view, slot=0):
        _, inputs = self.native.prepare(view, slot)
        context = self.context.features(self.native, view, slot)
        with torch.autocast(inputs.device.type, dtype=torch.float16, enabled=inputs.is_cuda):
            release = self.native.head(inputs.contiguous(memory_format=torch.channels_last), self.native.strength)
            if self.repair is None:
                repaired = torch.lerp(release, context[:, :3], self.strength * actor_support(context))
            else:
                repaired = self.repair(release, inputs, context, self.strength)
        return release, repaired

    def render(self, view, slot=0):
        return self.paired(view, slot)[1]
