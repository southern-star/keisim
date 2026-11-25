"""Edge-guided refinement of frozen driving-camera radial depths on their calibrated rays."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from keisim.config import CAMERA_RIGS, N_SEM
from keisim.render import rays


def aligned_features(x, h, w):
    # Source cell i lies on ray stride*i. The padded endpoint makes that spacing exact.
    padded = F.pad(x, (0,1,0,1), mode="replicate")
    return F.interpolate(padded, size=(h+1,w+1), mode="bilinear", align_corners=True)[:,:,:h,:w]


class SourceDepthRefiner(nn.Module):
    """Pick between nearby depths and correct their log range with frozen camera features."""
    def __init__(self):
        super().__init__()
        direction = np.stack([rays.camera_rays(c, 2) for c in CAMERA_RIGS["wide_narrow"]])
        self.register_buffer("rays", torch.tensor(direction,dtype=torch.float32).permute(0,3,1,2))
        self.stem = nn.Conv2d(64,16,1)
        self.context = nn.Conv2d(128,12,1)
        self.fuse = nn.Sequential(nn.Conv2d(16+12+N_SEM+5,32,3,padding=1),nn.SiLU(),
                                  nn.Conv2d(32,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,11,1))
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, stem, c2, scores, estimated):
        b,k,h,w=estimated.shape
        valid = (estimated>0).flatten(0,1)[:,None].float()
        log_z = torch.where(estimated>0,estimated,120.).clamp_min(.25).log().reshape(b*k,1,h,w)
        ray = self.rays.repeat(b,1,1,1)
        # The stem and the depth rays are at 2*i. c2/semantic cells have strides 8 and 4.
        context = self.context(c2.flatten(0,1).float())
        x = torch.cat((self.stem(stem.flatten(0,1).float()),aligned_features(context,h,w),
                       aligned_features(scores.flatten(0,1).float().softmax(1),h,w),ray,log_z/4,valid),1)
        pred = self.fuse(x).float()
        neighbours = F.unfold(F.pad(log_z,(1,1,1,1),mode="replicate"),3).reshape(b*k,9,h,w)
        prior = torch.zeros_like(pred[:,:9])
        prior[:,4] = 3.5
        log_new = ((pred[:,:9]+prior).softmax(1)*neighbours).sum(1) + pred[:,9].tanh()*1.25
        surface = pred[:,10] + 1.5*(2*valid[:,0]-1)
        return log_new.reshape(b,k,h,w), surface.reshape(b,k,h,w)

    def metres(self, features):
        log_z, valid = self(features["stem"],features["c2"],features["source_logits"],features["estimated_depth"])
        return torch.where(valid>0,log_z.clamp(-1.38629436,5.99146455).exp(),0.)
