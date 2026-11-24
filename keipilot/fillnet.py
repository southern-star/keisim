"""ReconFill's model (docs/recon_fill.md): a small U-Net that fills in what ReconView leaves dark.

Input, all from ReconView's chase camera at 320x160 (keipilot/filldata.py):
  the view without a fill: colour (3), depth (1), "a surface here" (1); the distance to the flat ground along every
  pixel's line of sight (1, a constant: the town is flat); and, with loose=True, the current frame alone by the
  loose triangle rule: colour, depth, "a surface here" (5).
Output: colour (3), ln(depth / DEPTH_NEAR) (1) and the logit of "a surface within 80 m" (1), all at 320x160.
Depths go in and out as codes / 65535 (the 16 bit codes of keisim/render/rays.py: 0..1 is 0.25..400 m, log scale).
The net predicts corrections of a first guess: the view's own colour and depth where it has a surface, the ground's
depth where it has none.

No BatchNorm (P2 differentiates through the net with torch.func.jvp): plain convolutions, or GroupNorm."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from keisim.render import rays as RY

from .filldata import CHASE, MAX_R, ground_depth

LOG_RANGE = math.log(RY.DEPTH_FAR / RY.DEPTH_NEAR)
CODE_MAX_R = math.log(MAX_R / RY.DEPTH_NEAR) / LOG_RANGE          # the code (0..1) of 80 m


def _block(cin, cout, norm, dilation=1):
    layers = [nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation)]
    if norm == "gn":
        layers.append(nn.GroupNorm(min(8, cout // 4), cout))
    layers.append(nn.ReLU(inplace=True))
    return layers


class FillNet(nn.Module):
    def __init__(self, width=(16, 32, 64, 128), loose=True, norm="none"):
        super().__init__()
        self.width, self.loose, self.norm = tuple(width), loose, norm
        cin = 6 + (5 if loose else 0)
        w = self.width
        self.enc = nn.ModuleList()
        for i, c in enumerate(w):
            self.enc.append(nn.Sequential(*_block(cin if i == 0 else w[i - 1], c, norm), *_block(c, c, norm)))
        # the coarsest level sees further: two dilated convolutions
        self.wide = nn.Sequential(*_block(w[-1], w[-1], norm, 2), *_block(w[-1], w[-1], norm, 4))
        self.lat = nn.ModuleList(nn.Conv2d(w[i + 1], w[i], 1) for i in range(len(w) - 1))
        self.dec = nn.ModuleList(nn.Sequential(*_block(w[i], w[i], norm), *_block(w[i], w[i], norm)) for i in range(len(w) - 1))
        self.head = nn.Conv2d(w[0], 5, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[4] = 2.0                         # "a surface": most pixels of the town within 80 m have one
        g = RY.encode_depth(ground_depth()).astype(np.float32) / 65535.0
        self.register_buffer("ground", torch.from_numpy(g)[None, None])        # (1, 1, H, W) codes / 65535; 0: no ground

    def inputs(self, in_rgb, in_dep, loose_rgb=None, loose_dep=None):
        """The net's input from a batch's images: colours (B, 3, H, W) in 0..1, depths (B, 1, H, W) as codes / 65535
        (0: no surface). Returns (x, base_rgb, base_code)."""
        valid = (in_dep > 0).to(in_rgb.dtype)
        ground = self.ground.to(in_rgb.dtype).expand_as(in_dep)
        x = [in_rgb * valid, in_dep, valid, ground]
        if self.loose:
            lv = (loose_dep > 0).to(in_rgb.dtype)
            x += [loose_rgb * lv, loose_dep, lv]
        # the first guess of the depth: the view's own, else the ground's, else (above the horizon) far away
        base = torch.where(in_dep > 0, in_dep, torch.where(ground > 0, ground, torch.full_like(in_dep, CODE_MAX_R)))
        return torch.cat(x, 1), in_rgb * valid, base

    def forward(self, in_rgb, in_dep, loose_rgb=None, loose_dep=None):
        """Returns (rgb (B, 3, H, W) in 0..1, not clamped; code (B, 1, H, W): the depth as code / 65535; surface
        (B, 1, H, W): the logit of "a surface within 80 m")."""
        x, base_rgb, base_code = self.inputs(in_rgb, in_dep, loose_rgb, loose_dep)
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = self.wide(x)
        for i in range(len(self.width) - 2, -1, -1):
            x = self.dec[i](F.interpolate(self.lat[i](x), scale_factor=2.0, mode="nearest") + skips[i])
        y = self.head(x)
        return base_rgb + y[:, :3], base_code + y[:, 3:4], y[:, 4:5]

    def config(self):
        return {"width": list(self.width), "loose": self.loose, "norm": self.norm}


def code_to_metres(code):
    """Distances [m] of codes / 65535."""
    return RY.DEPTH_NEAR * torch.exp(code.clamp(0.0, 1.0) * LOG_RANGE)


def macs(model):
    """Multiply-adds of one forward pass at 320x160 (convolutions only)."""
    total = [0]

    def hook(m, _inp, out):
        total[0] += out.shape[2] * out.shape[3] * m.weight.numel()
    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    dev = next(model.parameters()).device
    z = torch.zeros(1, 3, CHASE.height, CHASE.width, device=dev), torch.zeros(1, 1, CHASE.height, CHASE.width, device=dev)
    with torch.no_grad():
        model(z[0], z[1], z[0], z[1])
    for h in hooks:
        h.remove()
    return total[0]


def load_fillnet(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = FillNet(**ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval()


# ---------------------------------------------------------------------- the loss
def fill_loss(model, batch, w_surface=0.2, w_depth=1.0, w_rgb=1.0, w_grad=1.0, w_seen=0.5):
    """batch: in_rgb, loose_rgb, gt_rgb (B, 3, H, W) in 0..1; in_dep, loose_dep, gt_dep (B, 1, H, W) codes / 65535.
    The target pixels: the town has a surface within 80 m. Holes (the view has nothing) weigh 1, the rest w_seen.
    Returns (loss, parts)."""
    rgb, code, surf = model(batch["in_rgb"], batch["in_dep"], batch["loose_rgb"], batch["loose_dep"])
    gt = batch["gt_dep"]
    target = ((gt > 0) & (gt <= CODE_MAX_R)).float()
    w = torch.where(batch["in_dep"] > 0, w_seen, 1.0)
    n = (target * w).sum().clamp(min=1.0)
    l_surf = (F.binary_cross_entropy_with_logits(surf.float(), target, reduction="none") * w).sum() / w.sum()
    # in ln(r): codes / 65535 times the log range
    l_depth = ((code.float() - gt).abs() * LOG_RANGE * target * w).sum() / n
    l_rgb = ((rgb.float() - batch["gt_rgb"]).abs().mean(1, keepdim=True) * target * w).sum() / n
    # the picture's gradients, where both pixels are targets: against blur
    d = rgb.float() - batch["gt_rgb"]
    tw = target * w
    gx = ((d[..., :, 1:] - d[..., :, :-1]).abs().mean(1, keepdim=True) * tw[..., :, 1:] * target[..., :, :-1]).sum()
    gy = ((d[..., 1:, :] - d[..., :-1, :]).abs().mean(1, keepdim=True) * tw[..., 1:, :] * target[..., :-1, :]).sum()
    l_grad = (gx + gy) / (2.0 * n)
    loss = w_surface * l_surf + w_depth * l_depth + w_rgb * l_rgb + w_grad * l_grad
    return loss, {"surface": l_surf.item(), "depth": l_depth.item(), "rgb": l_rgb.item(), "grad": l_grad.item()}
