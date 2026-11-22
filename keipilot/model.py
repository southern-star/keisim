"""KeiPilot: a lightweight camera-only end-to-end driving model.

image (3x160x320) [+ the frame `history_dt` s earlier] + navigation command + target point (+ ego speed)
  -> ResNet-18 trunk
  -> (aux) FPN semantic segmentation            [what the model "sees"]
  -> tiny transformer decoder with learned queries over multi-scale tokens
       * 10 path queries  -> waypoints every 2 m (ego frame)
       * speed query      -> target speed distribution (two-hot bins)
       * light query      -> relevant traffic light state
The path + target speed are turned into controls by the same pure-pursuit / PI
controllers the privileged expert uses.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

N_PATH = 10
DEPTH_NEAR, DEPTH_FAR = 0.25, 400.0      # the depth head's range [m] (keisim/render/rays.py)
PATH_STEP = 2.0
SPEED_BINS = torch.arange(0.0, 12.0, 1.0)       # 0..11 m/s
N_TL = 4
N_CMD = 3
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def channels_last(img):
    """A batch of images in channels-last memory; camera rigs' (B,K,3,H,W) batches are converted inside forward."""
    return img.contiguous(memory_format=torch.channels_last) if img.dim() == 4 else img


def _resnet18_from_local_r34():
    """ResNet-18 initialised from the first two blocks of each stage of a locally
    cached ImageNet ResNet-34 (no download). Falls back to random init."""
    net = torchvision.models.resnet18(weights=None)
    ckdir = os.path.join(torch.hub.get_dir(), "checkpoints")
    cands = [os.path.join(ckdir, f) for f in ("resnet18-f37072fd.pth", "resnet34-b627a593.pth", "resnet34-43635321.pth")]
    for path in cands:
        if not os.path.exists(path):
            continue
        sd = torch.load(path, map_location="cpu")
        if "resnet18" in path:
            net.load_state_dict(sd)
            return net, path
        own = net.state_dict()
        new = {}
        for k in own:
            if k in sd and sd[k].shape == own[k].shape:
                new[k] = sd[k]
        missing = [k for k in own if k not in new]
        own.update(new)
        net.load_state_dict(own)
        return net, f"{path} (truncated, {len(new)}/{len(own)} tensors, missing {len(missing)})"
    return net, "random init"


class KeiPilot(nn.Module):
    def __init__(self, n_sem=13, d=256, n_layers=3, n_heads=8, pretrained=True, img_hw=(160, 320), speed_input=False,
                 history=False, history_dt=0.4, history_mode="frame", aux_yield=False, mem_input=False, n_cams=1,
                 aux_future=False, aux_depth=False, depth_width=(48, 32, 16, 24), depth_rig="wide_narrow"):
        super().__init__()
        # camera rigs (keisim CAMERA_RIGS): every camera's image goes through the same trunk; its tokens get the
        # camera's own position embeddings (pos3[k], pos4[k]), and the decoder reads the tokens of all of them
        self.n_cams = n_cams
        assert not (history and n_cams > 1), "history models have one camera"
        if pretrained:
            r, self.init_info = _resnet18_from_local_r34()
        else:
            r, self.init_info = torchvision.models.resnet18(weights=None), "random init"
        # optional previous frame (early fusion: 3 more input channels). Their weights start at zero, so a
        # single-frame checkpoint loads unchanged (load_compatible); a missing frame is fed as zeros.
        # history_mode "frame": the extra channels are the earlier frame; "diff": the change since then
        # (current - earlier, normalised). Static things cancel and moving pedestrians / cars stand out, which a
        # stem initialised from a single-frame model learns to use far more easily than two near-identical frames.
        self.history, self.history_dt, self.history_mode = history, history_dt, history_mode
        if history:
            conv = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
            with torch.no_grad():
                conv.weight.zero_()
                conv.weight[:, :3] = r.conv1.weight
            r.conv1 = conv
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        H, W = img_hw
        # --- segmentation (FPN-lite)
        self.lat = nn.ModuleList([nn.Conv2d(c, 96, 1) for c in (64, 128, 256, 512)])
        self.seg_head = nn.Sequential(nn.Conv2d(96, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96),
                                      nn.ReLU(inplace=True), nn.Conv2d(96, n_sem, 1))
        # --- tokens
        self.proj3 = nn.Conv2d(256, d, 1)
        self.proj4 = nn.Conv2d(512, d, 1)
        self.pos3 = nn.Parameter(torch.randn(n_cams, d, H // 16, W // 16) * 0.02)
        self.pos4 = nn.Parameter(torch.randn(n_cams, d, H // 32, W // 32) * 0.02)
        self.queries = nn.Parameter(torch.randn(N_PATH + 2, d) * 0.02)
        self.cond = nn.Sequential(nn.Linear(N_CMD + 4, d), nn.GELU(), nn.Linear(d, d))
        # optional ego speed condition. Its output layer starts at zero, so a camera-only checkpoint loaded into
        # this model behaves exactly as before; `speed_unknown` stands in when the speed is dropped out in
        # training (against the inertia problem) or not given
        self.speed_input = speed_input
        if speed_input:
            self.speed_enc = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
            nn.init.zeros_(self.speed_enc[2].weight)
            nn.init.zeros_(self.speed_enc[2].bias)
            self.speed_unknown = nn.Parameter(torch.zeros(d))
        # optional memory: how far the ego has driven since it last stood (keipilot.data.stand_update), the one
        # thing a single frame cannot show at a 止まれ line (still to stop, or moving off after the stop). Same
        # zero start as the speed input
        self.mem_input = mem_input
        if mem_input:
            self.mem_enc = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
            nn.init.zeros_(self.mem_enc[2].weight)
            nn.init.zeros_(self.mem_enc[2].bias)
            self.mem_unknown = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerDecoderLayer(d, n_heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.path_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        self.speed_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Linear(128, len(SPEED_BINS)))
        self.tl_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, N_TL))
        # optional auxiliary head on the speed token: earliest arrival [s] of a car with priority at the next right
        # turn that gives way (the quantity the expert's wait / go decision is made from; needs the oncoming speed)
        self.aux_yield = aux_yield
        if aux_yield:
            self.aux_tta_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        # optional auxiliary head, used in training only (docs/future_aux.md): each camera's segmentation `fut_dt` s
        # ahead (0.4-2 s), from the FPN features of the c2 stage (1/8 resolution). FiLM tells it the look-ahead and
        # where the car is going: the command, the target point and the recorded ego speed. Not the planner's `cond`,
        # whose speed is a counterfactual or hidden one in most training samples (--cf_prob, --speed_drop) while the
        # future frame was driven at the recorded speed; nor the decoder's tokens: the plan does not depend on the
        # predicted future. The FiLM output starts at zero (no modulation). Created last, and only when asked for,
        # so a model without it draws the same random numbers as before
        self.aux_future = aux_future
        if aux_future:
            self.aux_future_film = nn.Sequential(nn.Linear(N_CMD + 4 + 3 + 2, 128), nn.GELU(), nn.Linear(128, 2 * 96))
            nn.init.zeros_(self.aux_future_film[2].weight)
            nn.init.zeros_(self.aux_future_film[2].bias)
            self.aux_future_head = nn.Sequential(
                nn.Conv2d(96, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
                nn.Conv2d(96, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
                nn.Conv2d(96, n_sem, 1))
        # optional depth head (ReconView, docs/recon_view.md): every camera's distance image at half resolution
        # (keisim/render/rays.py: metres from the camera centre along each pixel's line of sight), read from the FPN's
        # 1/4 features and the stem's 1/2 features of a frozen driving model (scripts/train_depth.py). It gets the
        # lines of sight of the rig's cameras as an input (aux_depth_rays), which tell the two cameras apart. No
        # normalisation layers: the model stays in eval() while the head trains, so their statistics would never move.
        # Created last, and only when asked for, so a model without it draws the same random numbers as before
        self.aux_depth = aux_depth
        if aux_depth:
            from keisim.config import CAMERA_RIGS
            from keisim.render.rays import camera_rays
            cams = CAMERA_RIGS[depth_rig]
            assert len(cams) == n_cams and all((c.height, c.width) == (H, W) for c in cams), "the depth head's rig"
            a, b, k, c = depth_width
            self.depth_width, self.depth_rig = tuple(depth_width), depth_rig
            self.aux_depth_q = nn.Sequential(nn.Conv2d(96 + 3, a, 3, padding=1), nn.ReLU(), nn.Conv2d(a, b, 3, padding=1), nn.ReLU())
            self.aux_depth_skip = nn.Conv2d(64, k, 1)
            self.aux_depth_h = nn.Sequential(nn.Conv2d(b + k, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, 2, 3, padding=1))
            with torch.no_grad():                          # starts at 8 m everywhere (ln(8 / 0.25)), "a surface"
                self.aux_depth_h[2].bias.copy_(torch.tensor([math.log(8.0 / DEPTH_NEAR), 1.0]))
            rays = np.stack([camera_rays(cfg, 4) for cfg in cams])                 # (K, H/4, W/4, 3)
            self.register_buffer("aux_depth_rays", torch.from_numpy(rays).float().permute(0, 3, 1, 2).contiguous())
        prior = torch.stack([torch.arange(1, N_PATH + 1) * PATH_STEP, torch.zeros(N_PATH)], -1)
        self.register_buffer("path_prior", prior)
        self.register_buffer("speed_bins", SPEED_BINS.clone())
        self.register_buffer("mean", IMAGENET_MEAN.clone())
        self.register_buffer("std", IMAGENET_STD.clone())

    @staticmethod
    def encode_cond(cmd, tp):
        """cmd (B,) long, tp (B,2) metres in ego frame."""
        onehot = F.one_hot(cmd.long(), N_CMD).float()
        dist = torch.linalg.norm(tp, dim=-1, keepdim=True)
        feat = torch.cat([onehot, (tp / 40.0).clamp(-3, 3), torch.log1p(dist) / 4.0,
                          torch.atan2(tp[:, 1:2], tp[:, 0:1]) / 3.1416], -1)
        return feat

    @staticmethod
    def encode_speed(v):
        """v (B,) ego speed in m/s."""
        v = v.float().clamp(0.0, 20.0)[:, None]
        return torch.cat([v / 10.0, (v / 10.0) ** 2, torch.log1p(v) / 2.5], -1)

    @staticmethod
    def encode_mem(m):
        """m (B,) metres driven since the ego last stood (saturates at 10)."""
        m = m.float().clamp(0.0, 10.0)[:, None]
        return torch.cat([m / 10.0, torch.exp(-m / 1.5), torch.exp(-m / 5.0)], -1)

    @staticmethod
    def encode_dt(dt):
        """dt (B,) look-ahead of the future head in s (up to 2)."""
        dt = dt.float()[:, None] / 2.0
        return torch.cat([dt, dt ** 2], -1)

    def _depth(self, p, half):
        """The depth head: p the FPN's 1/4 features (B*K, 96, H/4, W/4), half the stem's 1/2 features (B*K, 64, H/2,
        W/2). Returns (B*K, 2, H/2, W/2): ln(r / DEPTH_NEAR) and the logit of "a surface within DEPTH_FAR"."""
        rays = self.aux_depth_rays.to(p.dtype).repeat(p.shape[0] // self.n_cams, 1, 1, 1)
        x = self.aux_depth_q(torch.cat([p, rays], 1))
        x = F.interpolate(x, size=half.shape[-2:], mode="bilinear", align_corners=False)
        return self.aux_depth_h(torch.cat([x, self.aux_depth_skip(half)], 1))

    @staticmethod
    def depth_metres(y):
        """The depth head's output (N, 2, h, w) as distances [m] (N, h, w); 0: no surface."""
        y = y.float()
        r = DEPTH_NEAR * torch.exp(y[:, 0].clamp(0.0, math.log(DEPTH_FAR / DEPTH_NEAR)))
        return torch.where(y[:, 1] > 0, r, torch.zeros_like(r))

    def forward_depth(self, img):
        """The depth head alone (training and evaluation of the head: no decoder): img as in forward(); returns
        (B*n_cams, 2, H/2, W/2) as _depth."""
        if self.n_cams > 1:
            img = img.flatten(0, 1).contiguous(memory_format=torch.channels_last)
        x = (img.float() / 255.0 - self.mean) / self.std
        half = self.stem[2](self.stem[1](self.stem[0](x)))
        c1 = self.layer1(self.stem[3](half))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        p = self.lat[3](c4)
        p = self.lat[2](c3) + F.interpolate(p, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p = self.lat[1](c2) + F.interpolate(p, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p = self.lat[0](c1) + F.interpolate(p, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        return self._depth(p, half)

    def forward(self, img, cmd, tp, with_seg=True, speed=None, speed_known=None, img_prev=None, has_prev=None,
                stand_m=None, stand_known=None, with_future=False, fut_dt=None, fut_v=None, with_depth=False):
        """img: (B,3,H,W) uint8/float RGB in [0,255]; (B,n_cams,3,H,W) for camera rigs (out["seg"] is then
        (B*n_cams, ...)). speed (B,) m/s is used by speed-input models only;
        speed_known (B,) bool hides it per sample (False -> the learned 'unknown' embedding).
        img_prev / has_prev (history models): the frame `history_dt` s earlier; missing frames enter as zeros.
        stand_m / stand_known (memory-input models): metres since the ego last stood, hidden like the speed.
        with_future (models with the future head, training): out["future"] (B*n_cams, n_sem, H/8, W/8) are the
        logits of every camera's segmentation fut_dt (B,) s ahead (1 s if None), given the recorded ego speed
        fut_v (B,) m/s (0 if None).
        with_depth (models with the depth head): out["depth"] (B*n_cams, 2, H/2, W/2), see _depth and depth_metres;
        the driving outputs are computed by the same operations as without it."""
        B = img.shape[0]
        if self.n_cams > 1:
            img = img.flatten(0, 1).contiguous(memory_format=torch.channels_last)
        x = (img.float() / 255.0 - self.mean) / self.std
        if self.history:
            if img_prev is None:
                xp = torch.zeros_like(x)
            else:
                xp = (img_prev.float() / 255.0 - self.mean) / self.std
                if self.history_mode == "diff":
                    xp = x - xp
                if has_prev is not None:
                    xp = xp * has_prev.to(xp.dtype)[:, None, None, None]
            x = torch.cat([x, xp], 1)
        with_depth = with_depth and self.aux_depth
        if with_depth:          # the stem's layers one by one (the same operations), to keep its 1/2 features
            half = self.stem[2](self.stem[1](self.stem[0](x)))
            x = self.stem[3](half)
        else:
            x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        out = {}
        with_future = with_future and self.aux_future
        if with_seg or with_future or with_depth:
            p = self.lat[3](c4)
            p = self.lat[2](c3) + F.interpolate(p, size=c3.shape[-2:], mode="bilinear", align_corners=False)
            p = self.lat[1](c2) + F.interpolate(p, size=c2.shape[-2:], mode="bilinear", align_corners=False)
            if with_future:
                dt = torch.ones(B, device=p.device) if fut_dt is None else fut_dt
                v = torch.zeros(B, device=p.device) if fut_v is None else fut_v
                f = torch.cat([self.encode_cond(cmd, tp), self.encode_speed(v), self.encode_dt(dt)], -1)
                g, b = self.aux_future_film(f).to(p.dtype).repeat_interleave(self.n_cams, 0).chunk(2, -1)
                out["future"] = self.aux_future_head(p * (1 + g[:, :, None, None]) + b[:, :, None, None])
            if with_seg or with_depth:
                p = self.lat[0](c1) + F.interpolate(p, size=c1.shape[-2:], mode="bilinear", align_corners=False)
                if with_seg:
                    out["seg"] = self.seg_head(p)
                if with_depth:
                    out["depth"] = self._depth(p, half)
        if self.n_cams > 1:     # (B*K, d, h, w) -> (B, K*h*w, d), each camera with its own positions
            t3 = (self.proj3(c3).unflatten(0, (B, self.n_cams)) + self.pos3).flatten(3).transpose(2, 3).flatten(1, 2)
            t4 = (self.proj4(c4).unflatten(0, (B, self.n_cams)) + self.pos4).flatten(3).transpose(2, 3).flatten(1, 2)
        else:
            t3 = (self.proj3(c3) + self.pos3).flatten(2).transpose(1, 2)
            t4 = (self.proj4(c4) + self.pos4).flatten(2).transpose(1, 2)
        mem = torch.cat([t3, t4], 1)
        cond = self.cond(self.encode_cond(cmd, tp))
        if self.speed_input:
            unknown = self.speed_unknown.to(cond.dtype)[None].expand_as(cond)
            if speed is None:
                cond = cond + unknown
            else:
                s = self.speed_enc(self.encode_speed(speed)).to(cond.dtype)
                if speed_known is not None:
                    s = torch.where(speed_known[:, None], s, unknown)
                cond = cond + s
        if self.mem_input:
            unknown = self.mem_unknown.to(cond.dtype)[None].expand_as(cond)
            if stand_m is None:
                cond = cond + unknown
            else:
                s = self.mem_enc(self.encode_mem(stand_m)).to(cond.dtype)
                if stand_known is not None:
                    s = torch.where(stand_known[:, None], s, unknown)
                cond = cond + s
        q = self.queries[None].expand(B, -1, -1) + cond[:, None]
        h = self.decoder(q, mem)
        out["path"] = self.path_prior[None] + self.path_head(h[:, :N_PATH]).float() * 2.0
        out["speed_logits"] = self.speed_head(h[:, N_PATH]).float()
        out["tl_logits"] = self.tl_head(h[:, N_PATH + 1]).float()
        if self.aux_yield:
            out["yield_tta"] = self.aux_tta_head(h[:, N_PATH]).float().squeeze(-1)
        return out

    def load_compatible(self, sd):
        """Load a checkpoint from a model with fewer inputs (no speed / memory / history): new parameters keep their
        neutral init (zero speed / memory encoder output, zero weights for the previous frame). Returns the new keys."""
        own = self.state_dict()
        sd = dict(sd)
        for k in ("pos3", "pos4"):                  # one camera's positions -> every camera of a rig starts with them
            if k in sd and sd[k].shape[0] == 1 and own[k].shape[0] > 1:
                sd[k] = sd[k].expand_as(own[k]).clone()
        w = sd.get("stem.0.weight")
        if w is not None and w.shape != own["stem.0.weight"].shape:       # single-frame stem -> history stem
            pad = torch.zeros_like(own["stem.0.weight"])
            pad[:, :w.shape[1]] = w
            sd["stem.0.weight"] = pad
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith(("speed_", "mem_", "aux_")) for k in missing), (missing, unexpected)
        return missing

    def decode_speed(self, logits):
        p = logits.softmax(-1)
        v = (p * self.speed_bins).sum(-1)
        stop = p[:, 0] > 0.5
        return torch.where(stop, torch.zeros_like(v), v), p


def two_hot(v, bins):
    v = v.clamp(float(bins[0]), float(bins[-1]))
    step = float(bins[1] - bins[0])
    idx = ((v - bins[0]) / step).floor().long().clamp(0, len(bins) - 2)
    w_hi = (v - bins[idx]) / step
    t = torch.zeros(v.shape[0], len(bins), device=v.device)
    t.scatter_(1, idx[:, None], (1 - w_hi)[:, None])
    t.scatter_add_(1, (idx + 1)[:, None], w_hi[:, None])
    return t


TL_WEIGHTS = torch.tensor([2.0, 3.0, 1.5, 0.5])


YIELD_TTA_MAX = 8.0
# the future head's class weights: vehicles and pedestrians (keisim.config.SEM 11, 12) x4. Most of a future picture
# follows from the ego's own motion; the ones that move by themselves are what the head is there for
FUTURE_MOVING = (11, 12)
FUTURE_MOVING_WEIGHT = 4.0


def future_loss(logits, lab, ok):
    """Cross-entropy of the future head: logits (N, C, h, w) raised to the labels' resolution (as the segmentation
    loss does), lab (N, H, W) the segmentation of the future frame, ok (N,) bool the frames that have one. Weighted
    mean over the pixels of the ok frames; 0 if there is none."""
    w = torch.ones(logits.shape[1], device=logits.device)
    w[list(FUTURE_MOVING)] = FUTURE_MOVING_WEIGHT
    lab = lab.long()
    pred = F.interpolate(logits.float(), size=lab.shape[-2:], mode="bilinear", align_corners=False)
    ce = F.cross_entropy(pred, lab, weight=w, reduction="none")
    m = ok[:, None, None].to(ce.dtype)
    return (ce * m).sum() / (w[lab] * m).sum().clamp(min=1.0)


# the depth head's loss (scripts/train_depth.py): pixels of vehicles and people x4, as the future head's
DEPTH_LOSS_MAX = 150.0      # [m]: farther surfaces count as surfaces, but not in the distance term


def depth_loss(y, q, seg, w_surface=0.2):
    """y (N, 2, h, w) the depth head's output; q (N, h, w) the depth codes of the labels (keisim/render/rays.py, 0: no
    surface); seg (N, h, w) the labels' classes. Returns (loss, logs): L1 of ln(r) over the pixels with a surface
    within DEPTH_LOSS_MAX, and the cross-entropy of "a surface", both weighted means (vehicles and people x4)."""
    y = y.float()
    w = torch.ones(seg.shape, device=y.device)
    for c in FUTURE_MOVING:
        w = torch.where(seg == c, torch.full_like(w, FUTURE_MOVING_WEIGHT), w)
    has = q > 0
    t = q.float() * (math.log(DEPTH_FAR / DEPTH_NEAR) / 65535.0)                # ln(r / DEPTH_NEAR) of the code
    m = (has & (t <= math.log(DEPTH_LOSS_MAX / DEPTH_NEAR))).float() * w
    l1 = ((y[:, 0] - t).abs() * m).sum() / m.sum().clamp(min=1.0)
    bce = (F.binary_cross_entropy_with_logits(y[:, 1], has.float(), reduction="none") * w).sum() / w.sum()
    return l1 + w_surface * bce, {"depth_l1": l1.item(), "surface": bce.item()}


def compute_loss(model, out, batch, w_seg=0.5, w_tl=0.5, w_aux=0.0, w_future=0.0):
    """Returns (loss, logs). w_future > 0 adds the future head's term to the loss; logs["loss"] stays without it
    (the validation loss that picks best.pt is then the same quantity with and without the head)."""
    path_l = F.l1_loss(out["path"], batch["path"])
    tgt = two_hot(batch["speed"], model.speed_bins)
    speed_l = -(tgt * out["speed_logits"].log_softmax(-1)).sum(-1).mean()
    tl_l = F.cross_entropy(out["tl_logits"], batch["tl"].long(), weight=TL_WEIGHTS.to(out["tl_logits"].device))
    loss = path_l + speed_l + w_tl * tl_l
    logs = {"path": path_l.item(), "speed": speed_l.item(), "tl": tl_l.item()}
    if w_aux > 0 and "yield_tta" in out and "yield_tta" in batch:
        lab = batch["yield_tta"]
        m = torch.isfinite(lab)
        if m.any():
            aux_l = F.smooth_l1_loss(out["yield_tta"][m], lab[m].clamp(0.0, YIELD_TTA_MAX))
            loss = loss + w_aux * aux_l
            logs["aux"] = aux_l.item()
    if "seg" in out and "seg" in batch:
        lab = batch["seg"].flatten(0, 1) if batch["seg"].dim() == 4 else batch["seg"]     # rigs: (B, K, h, w)
        seg = F.interpolate(out["seg"].float(), size=lab.shape[-2:], mode="bilinear", align_corners=False)
        seg_l = F.cross_entropy(seg, lab.long())
        loss = loss + w_seg * seg_l
        logs["seg"] = seg_l.item()
    logs["loss"] = loss.item()
    if w_future > 0 and "future" in out and "fut_seg" in batch:
        rig = batch["fut_seg"].dim() == 4                                                   # rigs: (B, K, h, w)
        lab = batch["fut_seg"].flatten(0, 1) if rig else batch["fut_seg"]
        ok = batch["fut_ok"].repeat_interleave(batch["fut_seg"].shape[1], 0) if rig else batch["fut_ok"]
        fut_l = future_loss(out["future"], lab, ok)
        loss = loss + w_future * fut_l
        logs["future"] = fut_l.item()
    return loss, logs
