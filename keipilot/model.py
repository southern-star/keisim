"""KeiPilot: a lightweight camera-only end-to-end driving model.

image (3x160x320) + navigation command + target point
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

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

N_PATH = 10
PATH_STEP = 2.0
SPEED_BINS = torch.arange(0.0, 12.0, 1.0)       # 0..11 m/s
N_TL = 4
N_CMD = 3
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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
    def __init__(self, n_sem=13, d=256, n_layers=3, n_heads=8, pretrained=True, img_hw=(160, 320)):
        super().__init__()
        if pretrained:
            r, self.init_info = _resnet18_from_local_r34()
        else:
            r, self.init_info = torchvision.models.resnet18(weights=None), "random init"
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
        self.pos3 = nn.Parameter(torch.randn(1, d, H // 16, W // 16) * 0.02)
        self.pos4 = nn.Parameter(torch.randn(1, d, H // 32, W // 32) * 0.02)
        self.queries = nn.Parameter(torch.randn(N_PATH + 2, d) * 0.02)
        self.cond = nn.Sequential(nn.Linear(N_CMD + 4, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerDecoderLayer(d, n_heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.path_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        self.speed_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Linear(128, len(SPEED_BINS)))
        self.tl_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, N_TL))
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

    def forward(self, img, cmd, tp, with_seg=True):
        """img: (B,3,H,W) uint8/float RGB in [0,255]."""
        x = (img.float() / 255.0 - self.mean) / self.std
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        out = {}
        if with_seg:
            p = self.lat[3](c4)
            p = self.lat[2](c3) + F.interpolate(p, size=c3.shape[-2:], mode="bilinear", align_corners=False)
            p = self.lat[1](c2) + F.interpolate(p, size=c2.shape[-2:], mode="bilinear", align_corners=False)
            p = self.lat[0](c1) + F.interpolate(p, size=c1.shape[-2:], mode="bilinear", align_corners=False)
            out["seg"] = self.seg_head(p)
        t3 = (self.proj3(c3) + self.pos3).flatten(2).transpose(1, 2)
        t4 = (self.proj4(c4) + self.pos4).flatten(2).transpose(1, 2)
        mem = torch.cat([t3, t4], 1)
        cond = self.cond(self.encode_cond(cmd, tp))
        q = self.queries[None].expand(img.shape[0], -1, -1) + cond[:, None]
        h = self.decoder(q, mem)
        out["path"] = self.path_prior[None] + self.path_head(h[:, :N_PATH]).float() * 2.0
        out["speed_logits"] = self.speed_head(h[:, N_PATH]).float()
        out["tl_logits"] = self.tl_head(h[:, N_PATH + 1]).float()
        return out

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


def compute_loss(model, out, batch, w_seg=0.5, w_tl=0.5):
    path_l = F.l1_loss(out["path"], batch["path"])
    tgt = two_hot(batch["speed"], model.speed_bins)
    speed_l = -(tgt * out["speed_logits"].log_softmax(-1)).sum(-1).mean()
    tl_l = F.cross_entropy(out["tl_logits"], batch["tl"].long(), weight=TL_WEIGHTS.to(out["tl_logits"].device))
    loss = path_l + speed_l + w_tl * tl_l
    logs = {"path": path_l.item(), "speed": speed_l.item(), "tl": tl_l.item()}
    if "seg" in out and "seg" in batch:
        seg = F.interpolate(out["seg"].float(), size=batch["seg"].shape[-2:], mode="bilinear", align_corners=False)
        seg_l = F.cross_entropy(seg, batch["seg"].long())
        loss = loss + w_seg * seg_l
        logs["seg"] = seg_l.item()
    logs["loss"] = loss.item()
    return loss, logs
