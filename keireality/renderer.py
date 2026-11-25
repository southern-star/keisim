"""Single-frame chase renderer. Inference accepts only driving tokens and an ego-relative camera."""
import torch
from torch import nn
from torch.nn import functional as F

from keisim.config import N_SEM


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
                                    nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width))

    def forward(self, x):
        return F.gelu(x + self.layers(x))


class SingleFrameRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128))
        self.queries = nn.Parameter(torch.randn(1, 8 * 16, 128) * .02)
        self.view = nn.Sequential(nn.Linear(11, 128), nn.GELU(), nn.Linear(128, 128))
        self.register_buffer("pose_scale", torch.tensor([10., 10., 10., 1., 1., 1., 1., 1., 1., 90., 2.]))
        self.attn = nn.MultiheadAttention(128, 4, batch_first=True)
        self.base = nn.Sequential(nn.Conv2d(128, 256, 1), ResidualBlock(256))
        stages = []
        for a, b in ((256, 128), (128, 96), (96, 48), (48, 24)):
            stages.extend([nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(a, b, 3, padding=1),
                           nn.GroupNorm(8, b), nn.GELU(), ResidualBlock(b)])
        self.stages = nn.Sequential(*stages)
        self.rgb = nn.Conv2d(24, 3, 1)
        self.semantic = nn.Conv2d(24, N_SEM, 1)

    def latent_grid(self, mem, view_pose):
        kv = self.project(mem)
        q = self.queries + self.view(view_pose / self.pose_scale)[:, None]
        read, _ = self.attn(q, kv, kv, need_weights=False)
        return (q + read).transpose(1, 2).reshape(-1, 128, 8, 16)

    def forward(self, mem, view_pose):
        x = self.stages(self.base(self.latent_grid(mem, view_pose)))
        return self.rgb(x).sigmoid(), self.semantic(x)
