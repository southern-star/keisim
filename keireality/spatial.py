"""Sample frozen driving features along a requested camera's rays. Distances are hypotheses in metres."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from keisim.config import CAMERA_RIGS
from .renderer import SingleFrameRenderer


def ray_points(view, height, width, distances):
    """Target cell centres in the latest ego frame, with a flat-ground hypothesis and fixed radial shells."""
    v = view.float()
    y, x = torch.meshgrid((torch.arange(height, device=v.device) + .5) * (2 / height) - 1,
                          (torch.arange(width, device=v.device) + .5) * (2 / width) - 1, indexing="ij")
    pos, forward, up = [v[:, i:i+3, None, None] for i in (0, 3, 6)]
    right = torch.linalg.cross(forward, up, dim=1)
    slope = torch.tan(torch.deg2rad(v[:, 9, None, None, None]) / 2)
    direction = F.normalize(forward + slope * (x * v[:, 10, None, None, None] * right - y * up), dim=1)
    ground = (-pos[:, 2] / direction[:, 2].clamp(max=-1e-5)).clamp(.25, 160)
    depth = torch.stack([ground, *[torch.full_like(ground, d) for d in distances]], 1)
    return (pos[:, None] + direction[:, None] * depth[:, :, None]).permute(0, 1, 3, 4, 2)


def camera_grid(points, camera, stride):
    """Map ego-space points to a ResNet grid, whose cell i is centred at input pixel stride*i."""
    x, y, z = points.unbind(-1)
    forward, right, down = x - camera.x, camera.y - y, camera.z - z
    pitch = math.radians(camera.pitch_deg)
    if camera.projection == "cylindrical":
        f = camera.width / math.radians(camera.fov_deg)
        horizontal = torch.hypot(forward, right)
        u = camera.width / 2 + f * torch.atan2(right, forward)
        v = camera.height / 2 - f * math.tan(pitch) + f * down / horizontal.clamp_min(1e-5)
        valid = horizontal > 1e-5
    else:
        f = (camera.width / 2) / math.tan(math.radians(camera.fov_deg) / 2)
        depth = down * math.sin(pitch) + forward * math.cos(pitch)
        u = camera.width / 2 - .5 + f * right / depth.clamp_min(1e-5)
        v = camera.height / 2 - .5 + f * (down * math.cos(pitch) - forward * math.sin(pitch)) / depth.clamp_min(1e-5)
        valid = depth > 1e-5
    valid = valid & (u > -.5) & (u < camera.width - .5) & (v > -.5) & (v < camera.height - .5)
    grid = torch.stack(((2*u + stride) / camera.width - 1, (2*v + stride) / camera.height - 1), -1)
    return grid.masked_fill(~valid[..., None], 2), valid


def sample_rig(features, points, stride):
    # Flatten depth hypotheses into the output height to sample all of them in one pass per camera.
    b, planes, h, w, _ = points.shape
    samples, masks = [], []
    for i, camera in enumerate(CAMERA_RIGS["wide_narrow"]):
        grid, valid = camera_grid(points, camera, stride)
        sampled = F.grid_sample(features[:, i].float(), grid.reshape(b, planes*h, w, 2),
                                padding_mode="border", align_corners=False)
        sampled = sampled.reshape(b, -1, planes, h, w) * valid[:, None]
        samples.extend([sampled.flatten(1, 2), valid.float()])
        masks.append(valid.any(dim=1))
    return torch.cat(samples, 1), torch.stack(masks, 1).any(dim=1)


class SpatialRenderer(SingleFrameRenderer):
    def __init__(self, support=False, temporal=False):
        super().__init__()
        self.c2_proj = nn.Conv2d(128, 12, 1)
        self.c3_proj = nn.Conv2d(256, 8, 1)
        # 2 cameras, 4 hypotheses, and a validity indicator for each feature level.
        self.fuse = nn.Sequential(nn.Conv2d(2*4*(12+1+8+1), 96, 1), nn.GroupNorm(8, 96), nn.GELU())
        # Unconfirmed, current, past-only static, unknown (includes sky and ambiguous edges).
        self.support = nn.Conv2d(24, 4, 1) if support else None
        self.history = HistoryAdapter() if temporal else None

    def decode_features(self, mem, view_pose, c2, c3, past=None, motion=None, valid=None):
        x = self.base(self.latent_grid(mem.float(), view_pose))
        x = self.stages[:5](x)
        if self.history is not None:
            x = x + self.history(view_pose, past, motion, valid).to(x.dtype)
        x = self.stages[5:10](x)
        pts = ray_points(view_pose, *x.shape[-2:], (12., 28., 80.))
        parts = []
        for features, proj, stride in ((c2, self.c2_proj, 8), (c3, self.c3_proj, 16)):
            feat = proj(features.flatten(0, 1).float()).unflatten(0, (-1, 2))
            parts.append(sample_rig(feat, pts, stride)[0])
        return self.stages[10:](x + self.fuse(torch.cat(parts, 1)).to(x.dtype))

    def forward(self, mem, view_pose, c2, c3, past=None, motion=None, valid=None):
        x = self.decode_features(mem, view_pose, c2, c3, past, motion, valid)
        return self.rgb(x).sigmoid(), self.semantic(x), self.support(x) if self.support is not None else None


class HistoryAdapter(nn.Module):
    """Ego-compensated, per-pixel attention over at most seven previous compact feature grids."""
    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv2d(256, 12, 1)
        self.pose = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 16))
        self.encode = nn.Sequential(nn.Conv2d(2*2*(12+1) + 16, 32, 1), nn.GroupNorm(8, 32), nn.GELU())
        self.score = nn.Conv2d(32, 1, 1)
        self.output = nn.Conv2d(32, 128, 1, bias=False)

    def forward(self, view, past, motion, valid):
        b, times = valid.shape
        h, w = 16, 32
        pts = ray_points(view, h, w, (30.,))[:, None]
        dx, dy, sn, cs, _ = motion.float().unbind(-1)
        dx, dy, sn, cs = [a[..., None, None, None] for a in (dx, dy, sn, cs)]
        # Past ego pose is expressed in the latest ego frame. Invert it to project into that camera.
        x, y = pts[..., 0] - dx, pts[..., 1] - dy
        old = torch.stack((cs*x + sn*y, -sn*x + cs*y, pts[..., 2].expand_as(x)), -1)
        reduced = self.reduce(past.flatten(0, 2).float()).unflatten(0, (b*times, 2))
        sampled, inside = sample_rig(reduced, old.flatten(0, 1), 32)
        pose = self.pose(motion.flatten(0, 1)).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        values = self.encode(torch.cat([sampled, pose], 1))
        logits = self.score(values).reshape(b, times, h, w).float()
        mask = valid[:, :, None, None] & inside.reshape(b, times, h, w)
        weights = logits.masked_fill(~mask, -1e4).softmax(1) * mask
        merged = (values.reshape(b, times, 32, h, w) * weights[:, :, None]).sum(1)
        return self.output(merged)
