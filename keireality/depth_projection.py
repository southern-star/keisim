"""Forward-project frozen driving features using the model's radial depth estimates."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from keisim.config import CAMERA_RIGS
from keisim.render import rays


class DepthProjector(nn.Module):
    def __init__(self):
        super().__init__()
        origins, directions = [], []
        for cam in CAMERA_RIGS["wide_narrow"]:
            center, rotation = rays.level_frame(cam, 0., 0., 0.)
            origins.append(center)
            directions.append(rays.camera_rays(cam, 2) @ rotation)
        self.register_buffer("origins", torch.tensor(np.stack(origins), dtype=torch.float32))
        self.register_buffer("directions", torch.tensor(np.stack(directions), dtype=torch.float32))

    @torch.no_grad()
    def forward(self, depth, view, size=(64, 128)):
        """Return one source sampling grid, coverage and inverse depth per camera. Depth zero means missing."""
        b = len(view)
        if depth.shape != (b, 2, 80, 160) or view.shape != (b, 11):
            raise ValueError("expected per-tick estimated radial depth [B,2,80,160] and camera pose [B,11]")
        h, w = size
        center, forward, up = (view[:, None, None, i:i+3].float() for i in (0, 3, 6))
        right = torch.linalg.cross(forward, up, dim=-1)
        pos = self.origins[:, None, None] + depth.float()[..., None] * self.directions
        pts = pos.flatten(2, 3) - center
        z = (pts * forward).sum(-1)
        slope = torch.tan(torch.deg2rad(view[:, 9:10].float()) / 2)[:, :, None]
        u = ((pts * right).sum(-1) / (z.clamp_min(.01) * slope * view[:, 10, None, None]) + 1) * w / 2 - .5
        v = (1 - (pts * up).sum(-1) / (z.clamp_min(.01) * slope)) * h / 2 - .5
        inside = (depth.flatten(2) > 0) & (z > .01) & (u > -1) & (u < w) & (v > -1) & (v < h)
        u, v = u.clamp(-1, w), v.clamp(-1, h)
        x, y = u.floor().long(), v.floor().long()
        dx, dy = u - x, v - y
        indices, weights = [], []
        for ox, oy, weight in ((0,0,(1-dx)*(1-dy)), (1,0,dx*(1-dy)), (0,1,(1-dx)*dy), (1,1,dx*dy)):
            xx, yy = x+ox, y+oy
            valid = inside & (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h) & (weight > 1e-5)
            indices.append((yy*w + xx).clamp(0, h*w-1))
            weights.append(torch.where(valid, weight, 0.))
        index = torch.cat(indices, -1)
        weight = torch.cat(weights, -1)
        z = z.repeat(1, 1, 4)
        out = z.new_zeros(b, 2, h*w)
        nearest = torch.full_like(out, torch.inf)
        nearest.scatter_reduce_(2, index, torch.where(weight > 0, z, torch.inf), reduce="amin")
        # Ignore splats from behind the nearest surface. Smooth subpixel weights reduce point-grid flicker.
        front = nearest.gather(2, index)
        weight = weight * (z <= front * 1.04 + .15)
        mass = out.scatter_add(2, index, weight)
        denom = mass.clamp_min(1e-8)
        yy, xx = torch.meshgrid(torch.arange(80, device=z.device), torch.arange(160, device=z.device), indexing="ij")
        coords = []
        for coord, length in ((xx, 160), (yy, 80)):
            normalized = (2 * coord.flatten().float() + 1) / length - 1
            avg = out.scatter_add(2, index, weight * normalized.repeat(4)) / denom
            coords.append(torch.where(mass > 0, avg, 2.).reshape(b, 2, h, w))
        inv = out.scatter_add(2, index, weight / z.clamp_min(.1)) / denom
        return torch.stack(coords, -1), mass.reshape(b, 2, h, w).clamp(max=1), inv.reshape(b, 2, h, w)


class DepthProjectionBranch(nn.Module):
    """Decode camera-space features and combine their projected detail with native Phase 2 RGB."""
    def __init__(self):
        super().__init__()
        self.projector = DepthProjector()
        self.fine = nn.Sequential(nn.Conv2d(128, 48, 1), nn.GroupNorm(6, 48), nn.SiLU())
        self.coarse = nn.Sequential(nn.Conv2d(256, 24, 1), nn.GroupNorm(4, 24), nn.SiLU())
        self.source = nn.Sequential(nn.Conv2d(72, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
                                    nn.Conv2d(64, 12*16, 3, padding=1), nn.PixelShuffle(4))
        self.fuse = nn.Sequential(nn.Conv2d(2*(12+2)+3, 40, 3, padding=1), nn.GroupNorm(5,40), nn.SiLU(),
                                  nn.Conv2d(40, 32, 3, padding=1), nn.GroupNorm(4,32), nn.SiLU(),
                                  nn.Conv2d(32, 4, 3, padding=1),
                                  nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))

    def encode(self, c2, c3):
        fine = self.fine(c2.flatten(0, 1).float())
        coarse = self.coarse(c3.flatten(0, 1).float())
        feat = self.source(torch.cat((fine, F.interpolate(coarse, size=fine.shape[-2:], mode="bilinear",
                                                          align_corners=False)), 1))
        # The first three learned channels are supervised on synchronized camera colors during training.
        return torch.cat((feat[:, :3].sigmoid(), feat[:, 3:]), 1).unflatten(0, (-1, 2))

    def render_encoded(self, base, encoded, view, estimated_depth):
        grid, cover, inv = self.projector(estimated_depth, view)
        features = F.grid_sample(encoded.flatten(0,1).float(), grid.flatten(0,1),
                                 mode="bilinear", padding_mode="zeros", align_corners=False).unflatten(0, (-1, 2))
        available = (cover > 0).float()
        projected = torch.cat((features * available[:, :, None], cover[:, :, None], (5*inv).clamp(max=2)[:, :, None]), 2)
        base_low = F.avg_pool2d(base.float(), 2)
        logits = self.fuse(torch.cat((projected.flatten(1,2), base_low), 1)).float()
        # Blend continuously near unfilled cells. Outside the small projected neighborhood A and B agree exactly.
        footprint = F.max_pool2d(cover.amax(1, keepdim=True), 3, 1, 1)
        alpha = logits[:, 3:].sigmoid() * F.interpolate(footprint, scale_factor=2, mode="bilinear", align_corners=False)
        return torch.lerp(base.float(), logits[:, :3].sigmoid(), alpha)

    def forward(self, base, c2, c3, view, estimated_depth):
        return self.render_encoded(base, self.encode(c2, c3), view, estimated_depth)
