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
browser's ONNX holds it as a constant: the same input gives the same picture).

The other way to the colours (docs/recon_fill.md 6.10; param "g"): the same colour net without noise and times gives
the difference straight from the condition, trained by an L1 loss and against a patch discriminator (PatchDisc,
gan_losses below). With prev=True (6.14) the condition also holds the fill before this one, drawn from this frame's
chase camera (colour, depth, "a surface here"; all zero where there is none): what was painted there before is
painted again."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .filldata import CHASE
from .fillnet import CODE_MAX_R, FillNet, _block

X_CLAMP = 4.0                     # the colour difference over the scale, clamped (training targets)
T_MIN = 0.05                      # param "x": no training time below this (u = (z - f) / t)


class ColourNet(nn.Module):
    """u(z, t, t - r | condition): (B, 3, H, W). With noise=False the input is the condition alone."""

    def __init__(self, cond, width=(16, 32, 64, 128), norm="none", noise=True):
        super().__init__()
        self.width, self.noise = tuple(width), noise
        w = self.width
        cin = (3 + 2 if noise else 0) + cond
        self.enc = nn.ModuleList(nn.Sequential(*_block(cin if i == 0 else w[i - 1], c, norm), *_block(c, c, norm)) for i, c in enumerate(w))
        self.wide = nn.Sequential(*_block(w[-1], w[-1], norm, 2), *_block(w[-1], w[-1], norm, 4))
        self.lat = nn.ModuleList(nn.Conv2d(w[i + 1], w[i], 1) for i in range(len(w) - 1))
        self.dec = nn.ModuleList(nn.Sequential(*_block(w[i], w[i], norm), *_block(w[i], w[i], norm)) for i in range(len(w) - 1))
        self.head = nn.Conv2d(w[0], 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z, t, dt, cond):
        """z (B, 3, H, W); t, dt (B, 1, 1, 1): the time and t - r; cond (B, C, H, W)."""
        hw = cond.shape[2:]
        x = torch.cat([z, t.expand(-1, 1, *hw), dt.expand(-1, 1, *hw), cond], 1) if self.noise else cond
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = self.wide(x)
        for i in range(len(self.width) - 2, -1, -1):
            x = self.dec[i](F.interpolate(self.lat[i](x), scale_factor=2.0, mode="nearest") + skips[i])
        return self.head(x)


class FillGen(nn.Module):
    def __init__(self, shape, width=(16, 32, 64, 128), scale=0.15, seed=0, param="u", prev=False):
        """shape: a FillNet or its configuration (P1's net; frozen). scale: the standard deviation of the town's colour
        minus P1's at the holes (colours in 0..1). param: what the colour net's output is: "u", the average velocity
        itself (the paper's form; the one step is e - u, so the net has to reproduce the noise image exactly to take
        it out), or "x": the net gives f and u = (z - f) / t (for r = t, f is the clean image; the one step from
        t = 1 is f itself, and a fresh net answers P1's colour); or "g": no noise and no times, the net gives the
        difference from the condition alone (trained against a discriminator: gan_losses)."""
        super().__init__()
        self.shape = shape if isinstance(shape, FillNet) else FillNet(**shape)
        for p in self.shape.parameters():
            p.requires_grad_(False)
        assert param in ("u", "x", "g")
        self.loose, self.scale, self.seed, self.param, self.prev = self.shape.loose, float(scale), seed, param, prev
        self.ncond = (11 if self.loose else 6) + 5 + (5 if prev else 0)
        self.colour = ColourNet(self.ncond, width, noise=param != "g")
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("noise", torch.randn(1, 3, CHASE.height, CHASE.width, generator=g))

    @property
    def ground(self):
        return self.shape.ground

    def train(self, mode=True):
        super().train(mode)
        self.shape.eval()                                   # P1's net never trains
        return self

    def condition(self, in_rgb, in_dep, loose_rgb=None, loose_dep=None, prev_rgb=None, prev_dep=None):
        """(cond, P1's colour clamped to 0..1, code, surface logit); nothing of it carries a gradient. prev_rgb,
        prev_dep: the fill before, from this frame's camera (models with prev; None: there is none)."""
        with torch.no_grad():
            x, _rgb0, _code0 = self.shape.inputs(in_rgb, in_dep, loose_rgb, loose_dep)
            rgb, code, surf = self.shape(in_rgb, in_dep, loose_rgb, loose_dep)
            rgb = rgb.clamp(0.0, 1.0)
            cond = [x, rgb, code, torch.sigmoid(surf)]
            if self.prev:
                if prev_rgb is None:
                    cond.append(torch.zeros_like(rgb[:, :1]).expand(-1, 5, -1, -1))
                else:
                    v = (prev_dep > 0).to(rgb.dtype)
                    cond += [prev_rgb * v, prev_dep, v]
            cond = torch.cat(cond, 1)
        return cond, rgb, code, surf

    def velocity(self, z, t, dt, cond):
        """u(z, r, t | cond) with dt = t - r."""
        out = self.colour(z, t, dt, cond)
        return out if self.param == "u" else (z - out) / t

    def forward(self, in_rgb, in_dep, loose_rgb=None, loose_dep=None, noise=None, prev_rgb=None, prev_dep=None):
        """As FillNet: (rgb, code, surface logit). One step from the fixed noise image (or the one given)."""
        cond, rgb, code, surf = self.condition(in_rgb, in_dep, loose_rgb, loose_dep, prev_rgb, prev_dep)
        if self.param == "g":
            return rgb + self.scale * self.colour(None, None, None, cond), code, surf
        e = (self.noise if noise is None else noise).to(rgb.dtype).expand_as(rgb)
        one = torch.ones(rgb.shape[0], 1, 1, 1, dtype=rgb.dtype, device=rgb.device)
        out = self.colour(e, one, one, cond)
        x = e - out if self.param == "u" else out           # e - u(e, 0, 1); with param "x" that is f itself
        return rgb + self.scale * x, code, surf

    def config(self):
        return {"kind": "gen", "shape": self.shape.config(), "width": list(self.colour.width), "scale": self.scale, "seed": self.seed,
                "param": self.param, "prev": self.prev}


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
    if gen.param == "x":
        t, r = t.clamp(min=T_MIN), r.clamp(min=T_MIN)
    e = torch.randn_like(x)
    z, v = (1.0 - t) * x + t * e, e - x
    one = torch.ones_like(t)
    u, dudt = torch.func.jvp(lambda z_, t_, d_: gen.velocity(z_, t_, d_, cond), (z, t, t - r), (v, one, one))
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


# ---------------------------------------------------------------------- the other way: L1 and a patch discriminator
class PatchDisc(nn.Module):
    """Is this patch of the picture the town's, given the condition? One logit per 34 x 34 pixel patch (every 4
    pixels). Spectral normalisation, no normalisation layers."""

    def __init__(self, cond, width=(32, 64, 128)):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm
        c, layers = 3 + cond, []
        for i, w in enumerate(width):
            layers += [sn(nn.Conv2d(c, w, 4, stride=2 if i < 2 else 1, padding=1)), nn.LeakyReLU(0.2, inplace=True)]
            c = w
        layers.append(sn(nn.Conv2d(c, 1, 4, padding=1)))
        self.net = nn.Sequential(*layers)

    def forward(self, rgb, cond):
        return self.net(torch.cat([rgb, cond], 1))


def degrade_prev(rgb, dep, blur=1.0, noise=0.04, cuts=3):
    """What the page's picture of the fill before lacks against the town's: per sample a Gaussian blur (sigma up to
    `blur` pixels, over the pixels with a surface), noise (sigma up to `noise`) and up to `cuts` rectangles without a
    surface. Returns (rgb, dep)."""
    n, _, h, w = rgb.shape
    v = (dep > 0).to(rgb.dtype)
    sig = torch.rand(n, device=rgb.device) * blur
    r = torch.arange(-2, 3, device=rgb.device, dtype=torch.float32)
    k = torch.exp(-(r[None] ** 2) / (2.0 * sig.clamp(min=1e-3)[:, None] ** 2))
    k = (k / k.sum(1, keepdim=True)).to(rgb.dtype)
    k2 = (k[:, :, None] * k[:, None, :])                      # (n, 5, 5)
    conv = lambda a: F.conv2d(a.reshape(1, -1, h, w), k2.repeat_interleave(a.shape[1], 0)[:, None], padding=2, groups=n * a.shape[1]).reshape(a.shape)
    rgb = conv(rgb * v) / conv(v).clamp(min=1e-3)
    rgb = (rgb + torch.randn_like(rgb) * (torch.rand(n, 1, 1, 1, device=rgb.device) * noise).to(rgb.dtype)).clamp(0.0, 1.0)
    keep = torch.ones_like(v)
    for i in range(n):
        for _ in range(int(torch.randint(0, cuts + 1, (1,)))):
            ch, cw = int(torch.randint(8, 61, (1,))), int(torch.randint(8, 81, (1,)))
            y, x = int(torch.randint(0, h - ch + 1, (1,))), int(torch.randint(0, w - cw + 1, (1,)))
            keep[i, :, y:y + ch, x:x + cw] = 0.0
    return rgb * v * keep, dep * keep


def training_prev(ema, batch, p_drop=0.25, p_self=1.0 / 3.0):
    """The "fill before" of a training batch (6.14): the town's own, degraded; for a share p_drop of the samples none
    at all; for a share p_self the net's own answer (the moving average ema, no gradient) to the degraded town's,
    degraded again. Returns (prev_rgb, prev_dep)."""
    with torch.no_grad():
        rgb, dep = degrade_prev(batch["prev_rgb"], batch["prev_dep"])
        u = torch.rand(rgb.shape[0], device=rgb.device)
        own = (u >= p_drop) & (u < p_drop + p_self)
        if own.any():
            b = {k: v[own] for k, v in batch.items()}
            out = ema(b["in_rgb"], b["in_dep"], b["loose_rgb"], b["loose_dep"], prev_rgb=rgb[own], prev_dep=dep[own])[0].clamp(0.0, 1.0)
            rgb = rgb.clone()
            rgb[own], _ = degrade_prev(out.to(rgb.dtype), dep[own], cuts=0)
        keep = (u >= p_drop).float().view(-1, 1, 1, 1)
        return rgb.float() * keep, dep.float() * keep


def gan_picture(gen, batch, w_seen=0.5, prev=None):
    """(cond, the generator's colour (B, 3, H, W) with its gradient, the same put into the town's picture outside the
    target pixels, the pixels' weights w (B, 1, H, W): 0 outside the target, holes 1, the rest w_seen). prev: (rgb,
    dep) of the fill before, for a generator with prev."""
    cond, rgb1, _code, _surf = gen.condition(batch["in_rgb"], batch["in_dep"], batch["loose_rgb"], batch["loose_dep"], *(prev or (None, None)))
    gt = batch["gt_dep"]
    target = (gt > 0) & (gt <= CODE_MAX_R)
    w = target.float() * torch.where(batch["in_dep"] > 0, w_seen, 1.0)
    fake = rgb1 + gen.scale * gen.colour(None, None, None, cond)
    return cond, fake, torch.where(target, fake.to(batch["gt_rgb"].dtype), batch["gt_rgb"]), w


def _patch_mean(d, w):
    """The mean of the discriminator's map d, every patch weighted by the mean pixel weight in its place."""
    wd = F.adaptive_avg_pool2d(w, d.shape[2:])
    return (d.float() * wd).sum() / wd.sum().clamp(min=1e-6)


def disc_loss(disc, cond, comp, gt_rgb, w):
    """The hinge loss of the discriminator: the town's picture against the generator's (no gradient into it)."""
    real, fake = disc(gt_rgb, cond), disc(comp.detach(), cond)
    return _patch_mean(F.relu(1.0 - real.float()), w) + _patch_mean(F.relu(1.0 + fake.float()), w), {"d_real": _patch_mean(real, w).item(), "d_fake": _patch_mean(fake, w).item()}


def gen_loss(disc, cond, fake, comp, gt_rgb, w, lam=100.0):
    """lam times the colour's L1 (colours in 0..1) minus the discriminator's mean answer to the generator's picture."""
    l1 = ((fake.float() - gt_rgb).abs().mean(1, keepdim=True) * w).sum() / w.sum().clamp(min=1.0)
    adv = -_patch_mean(disc(comp, cond), w)
    return lam * l1 + adv, {"l1": 255.0 * l1.item(), "adv": adv.item()}
