"""ReconFill P2 (docs/recon_fill.md 6): the fill's colours by one-step generation, on top of P1's shape.

P1's net (keipilot/fillnet.py) stays as it is and is not trained: its depth and "a surface here" are the fill's. A
second U-Net of the same kind makes the colours: it generates x = (the town's colour - P1's colour) / scale, the
detail P1's regression blurs away, by a conditional mean flow (Geng et al. 2025, "Mean Flows for One-step Generative
Modeling"; eq. 6, 11, 12 and algorithms 1, 2):
    z_t = (1 - t) x + t e,   v = e - x,   e ~ N(0, 1)
    the net gives the average velocity u(z_t, r, t | condition); its target is v - (t - r) du/dt with
    du/dt = v du/dz + du/dt (a Jacobian-vector product), no gradient through the target
    one step:  x = e - u(e, r = 0, t = 1 | condition)
The condition: P1's input (the view's colour, depth, "a surface here", the ground's depth) and P1's output (colour,
depth, the probability of a surface). The times go in as two constant planes, t and t - r (the nets have no
normalisation layers a time embedding could modulate).

FillGen is called like FillNet and answers like it: (rgb, code, surface logit), with one fixed noise image (the
browser's ONNX holds it as a constant: the same input gives the same picture)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .filldata import CHASE
from .fillnet import CODE_MAX_R, FillNet, _block

X_CLAMP = 4.0                     # the colour difference over the scale, clamped (training targets)


class ColourNet(nn.Module):
    """u(z, t, t - r | condition): (B, 3, H, W)."""

    def __init__(self, cond, width=(16, 32, 64, 128), norm="none"):
        super().__init__()
        self.width = w = tuple(width)
        cin = 3 + 2 + cond
        self.enc = nn.ModuleList(nn.Sequential(*_block(cin if i == 0 else w[i - 1], c, norm), *_block(c, c, norm)) for i, c in enumerate(w))
        self.wide = nn.Sequential(*_block(w[-1], w[-1], norm, 2), *_block(w[-1], w[-1], norm, 4))
        self.lat = nn.ModuleList(nn.Conv2d(w[i + 1], w[i], 1) for i in range(len(w) - 1))
        self.dec = nn.ModuleList(nn.Sequential(*_block(w[i], w[i], norm), *_block(w[i], w[i], norm)) for i in range(len(w) - 1))
        self.head = nn.Conv2d(w[0], 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z, t, dt, cond):
        """z (B, 3, H, W); t, dt (B, 1, 1, 1): the time and t - r; cond (B, C, H, W)."""
        hw = z.shape[2:]
        x = torch.cat([z, t.expand(-1, 1, *hw), dt.expand(-1, 1, *hw), cond], 1)
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = self.wide(x)
        for i in range(len(self.width) - 2, -1, -1):
            x = self.dec[i](F.interpolate(self.lat[i](x), scale_factor=2.0, mode="nearest") + skips[i])
        return self.head(x)


class FillGen(nn.Module):
    def __init__(self, shape, width=(16, 32, 64, 128), scale=0.15, seed=0):
        """shape: a FillNet or its configuration (P1's net; frozen). scale: the standard deviation of the town's colour
        minus P1's at the holes (colours in 0..1)."""
        super().__init__()
        self.shape = shape if isinstance(shape, FillNet) else FillNet(**shape)
        for p in self.shape.parameters():
            p.requires_grad_(False)
        self.loose, self.scale, self.seed = self.shape.loose, float(scale), seed
        self.colour = ColourNet((11 if self.loose else 6) + 5, width)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("noise", torch.randn(1, 3, CHASE.height, CHASE.width, generator=g))

    @property
    def ground(self):
        return self.shape.ground

    def train(self, mode=True):
        super().train(mode)
        self.shape.eval()                                   # P1's net never trains
        return self

    def condition(self, in_rgb, in_dep, loose_rgb=None, loose_dep=None):
        """(cond, P1's colour clamped to 0..1, code, surface logit); nothing of it carries a gradient."""
        with torch.no_grad():
            x, _rgb0, _code0 = self.shape.inputs(in_rgb, in_dep, loose_rgb, loose_dep)
            rgb, code, surf = self.shape(in_rgb, in_dep, loose_rgb, loose_dep)
            rgb = rgb.clamp(0.0, 1.0)
            cond = torch.cat([x, rgb, code, torch.sigmoid(surf)], 1)
        return cond, rgb, code, surf

    def forward(self, in_rgb, in_dep, loose_rgb=None, loose_dep=None, noise=None):
        """As FillNet: (rgb, code, surface logit). One step from the fixed noise image (or the one given)."""
        cond, rgb, code, surf = self.condition(in_rgb, in_dep, loose_rgb, loose_dep)
        e = (self.noise if noise is None else noise).to(rgb.dtype).expand_as(rgb)
        one = torch.ones(rgb.shape[0], 1, 1, 1, dtype=rgb.dtype, device=rgb.device)
        x = e - self.colour(e, one, one, cond)
        return rgb + self.scale * x, code, surf

    def config(self):
        return {"kind": "gen", "shape": self.shape.config(), "width": list(self.colour.width), "scale": self.scale, "seed": self.seed}


def sample_times(n, mu, sigma, ratio, device):
    """(t, r), each (n, 1, 1, 1): two logit-normal draws, the larger is t; r = t for a share 1 - ratio of the samples
    (those are plain flow matching)."""
    a = torch.sigmoid(torch.randn(n, 2, device=device) * sigma + mu)
    t, r = a.max(1).values, a.min(1).values
    r = torch.where(torch.rand(n, device=device) < ratio, r, t)
    return t.view(n, 1, 1, 1), r.view(n, 1, 1, 1)


def meanflow_loss(gen, batch, mu=-2.0, sigma=2.0, ratio=0.75, p=0.75, w_seen=0.5, c=1e-3):
    """batch as keipilot/fillnet.py fill_loss. The pixels that count are P1's: the town has a surface within 80 m;
    holes weigh 1, the rest w_seen. Per sample the weighted mean squared error L of u against the target, times the
    adaptive weight 1 / (L + c)^p (no gradient through the weight). Returns (loss, parts)."""
    cond, rgb1, _code, _surf = gen.condition(batch["in_rgb"], batch["in_dep"], batch["loose_rgb"], batch["loose_dep"])
    gt = batch["gt_dep"]
    target = ((gt > 0) & (gt <= CODE_MAX_R)).float()
    w = target * torch.where(batch["in_dep"] > 0, w_seen, 1.0)
    # outside the target there is no colour to learn: x = 0 there (z is noise alone, as at the one step's start)
    x = ((batch["gt_rgb"] - rgb1) / gen.scale).clamp(-X_CLAMP, X_CLAMP) * target
    n = x.shape[0]
    t, r = sample_times(n, mu, sigma, ratio, x.device)
    e = torch.randn_like(x)
    z, v = (1.0 - t) * x + t * e, e - x
    one = torch.ones_like(t)
    u, dudt = torch.func.jvp(lambda z_, t_, d_: gen.colour(z_, t_, d_, cond), (z, t, t - r), (v, one, one))
    tgt = (v - (t - r) * dudt.float()).detach()
    err = ((u.float() - tgt) ** 2).mean(1, keepdim=True)
    L = (err * w).flatten(1).sum(1) / w.flatten(1).sum(1).clamp(min=1.0)
    loss = (L / (L.detach() + c) ** p).mean()
    same = (t == r).flatten()
    parts = {"mse": L.mean().item()}
    if same.any():
        parts["mse_fm"] = L[same].mean().item()             # the samples with r = t: plain flow matching
    if (~same).any():
        parts["mse_mf"] = L[~same].mean().item()
    return loss, parts


def colour_scale(gen, loader, device, batches=40):
    """The standard deviation of the town's colour minus P1's at the holes of some training batches."""
    s, n = 0.0, 0
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= batches:
                break
            b = {k: v.to(device) for k, v in b.items()}
            _cond, rgb1, _c, _s = gen.condition(b["in_rgb"], b["in_dep"], b["loose_rgb"], b["loose_dep"])
            hole = ((b["gt_dep"] > 0) & (b["gt_dep"] <= CODE_MAX_R) & (b["in_dep"] == 0)).expand_as(rgb1)
            d = (b["gt_rgb"] - rgb1)[hole]
            s, n = s + float((d.double() ** 2).sum()), n + d.numel()
    return (s / max(1, n)) ** 0.5
