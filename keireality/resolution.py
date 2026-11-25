"""Native 384×192 detail from the frozen driving stem, current depth and prior camera observations."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from keisim.config import CAMERA_RIGS, SEM
from .background_history import old_camera, previous_grid
from .inverse_surface import surface_seed
from .scene_detail import sample
from .spatial import camera_grid, ray_points


SIZE = (192, 384)
CHANNELS = 17


class ResolutionHead(nn.Module):
    """A small residual decoder trained against independent native-resolution target images."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(CHANNELS, 24, 3, padding=1, padding_mode="replicate")
        self.context = nn.Sequential(nn.SiLU(), nn.Conv2d(24, 24, 3, padding=2, dilation=2),
                                     nn.SiLU(), nn.Conv2d(24, 24, 3, padding=1))
        self.rgb = nn.Sequential(nn.SiLU(), nn.Conv2d(24, 3, 3, padding=1))
        nn.init.zeros_(self.rgb[-1].weight)
        nn.init.zeros_(self.rgb[-1].bias)

    def forward(self, inputs, strength=1.):
        base = inputs[:, :3].float()
        stem = self.stem(inputs)
        correction = self.rgb(stem + self.context(stem)).float()
        return (base + strength * .40 * correction.tanh()).clamp(0, 1)


class ResolutionDrive:
    def __init__(self, baseline, head=None, strength=1.):
        self.baseline = baseline
        self.head = head
        self.strength = strength
        self.episode = self.time = None
        self.prior, self.current = {}, {}

    @torch.inference_mode()
    def observe(self, features, episode, t, delta):
        contiguous = self.episode == episode and self.time is not None and math.isclose(t-self.time, .1, abs_tol=1e-6)
        self.prior, self.current = self.current if contiguous else {}, {}
        self.episode, self.time = episode, t
        self.baseline.observe(features, episode, t, delta)
        probs = features["source_logits"].float().softmax(2)
        marking = probs[:, :, SEM["marking"]]
        ground = probs[:, :, SEM["road"]] + probs[:, :, SEM["terrain"]] + probs[:, :, SEM["sidewalk"]] + marking
        self.classes = torch.stack((ground, marking, probs[:, :, SEM["vehicle"]]), 2)

    def projected(self, points, *, road):
        """Read continuous color at true output pixel centers and reject inconsistent source ranges."""
        display = self.baseline.control
        colors = display.encoded[0]
        total = points.new_zeros((1, 3, *SIZE))
        mass = torch.zeros_like(total[:, :1])
        label_sum = torch.zeros_like(mass)
        for k, cam in enumerate(CAMERA_RIGS["wide_narrow"]):
            grid, inside = camera_grid(points, cam, 1)
            pixel = grid.new_tensor([1/cam.width, 1/cam.height])
            z = sample(display.depth[:, k:k+1], grid + pixel)
            classes = sample(self.classes[:, k], grid + 3*pixel)
            distance = (points - points.new_tensor((cam.x, cam.y, cam.z))).norm(dim=-1)[:, None]
            agreement = (1 - (z-distance).abs()/(.8 + .18*distance)).clamp(0, 1)
            weight = inside[:, None] * agreement * (2.5 if k else 1.)
            if road:
                weight = weight * classes[:, :1] * (points[..., 2].abs() < .03)[:, None]
            total += sample(colors[:, k], grid) * weight
            label_sum += classes[:, 1:2] * weight if road else classes[:, 2:3] * weight
            mass += weight
        denom = mass.clamp_min(1e-6)
        return total/denom, mass.clamp(max=1.), label_sum/denom

    @torch.inference_mode()
    def prepare(self, view, slot=0):
        low = self.baseline.render(view, slot)
        base = F.interpolate(low.float(), size=SIZE, mode="bicubic", align_corners=False).clamp(0, 1)
        display = self.baseline.control
        hypotheses = ray_points(view, *SIZE, (1.,))
        origin = view[:, None, None, :3]
        direction = hypotheses[:, 1] - origin
        cosine = (direction*view[:, None, None, 3:6]).sum(-1)[:, None].clamp_min(.05)
        z, valid = surface_seed(display.observation.displays[slot][3])
        optical = F.interpolate(z.reciprocal(), size=SIZE, mode="bilinear", align_corners=False).reciprocal()
        valid = F.interpolate(valid.float(), size=SIZE, mode="nearest")
        xyz = origin + direction * (optical/cosine).permute(0, 2, 3, 1)
        surface, hit, car = self.projected(xyz, road=False)
        hit = hit * valid
        road, ground, marking = self.projected(hypotheses[:, 0], road=True)
        road_z = (hypotheses[:, 0]-origin).norm(dim=-1)[:, None]*cosine
        ground = ground * (1-hit * (optical < .8*road_z).float())
        far = torch.where(ground > .25, road_z, torch.where(direction[..., 2][:,None] > .12, 3000*cosine, 80*cosine))
        z = torch.where(hit > .1, optical, far)
        guide = torch.lerp(torch.lerp(base, surface, .7*hit), road, .8*ground)
        static = 1 - F.max_pool2d(car*hit, 7, 1, 3)
        past_weight = torch.zeros_like(hit)
        past_color = base
        if slot in self.prior:
            old_rgb, old_z, old_static, old_view = self.prior[slot]
            previous = old_camera(view, display.reference.motion[0, :1])
            grid, inside, expected = previous_grid(previous, old_view, z, direction, view)
            measured = sample(old_z, grid, mode="nearest")
            close = (1-(measured-expected).abs()/(1.2+.14*expected.clamp_min(0))).clamp(0,1)
            past_weight = inside * close * static * sample(old_static, grid)
            past_color = sample(old_rgb, grid)
        self.current[slot] = (guide, z, static, view)
        inputs = torch.cat((base, (surface-base)*hit, hit, car*hit, (road-base)*ground, ground, marking*ground,
                            (past_color-base)*past_weight, past_weight), 1)
        return low, inputs

    @torch.inference_mode()
    def render(self, view, slot=0):
        _, inputs = self.prepare(view, slot)
        with torch.autocast(device_type=inputs.device.type, dtype=torch.float16, enabled=inputs.is_cuda):
            return self.head(inputs.contiguous(memory_format=torch.channels_last), self.strength)
