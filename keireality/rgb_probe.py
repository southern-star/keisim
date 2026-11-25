"""Independent semantic probe for scoring what a generated RGB image actually depicts."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from keisim.config import N_SEM, SEM, SEM_NAMES
from .renderer import ResidualBlock


class RGBProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(nn.Conv2d(3,24,3,padding=1),ResidualBlock(24))
        self.down1 = nn.Sequential(nn.Conv2d(24,48,3,2,1),ResidualBlock(48))
        self.down2 = nn.Sequential(nn.Conv2d(48,96,3,2,1),ResidualBlock(96),ResidualBlock(96))
        self.up1 = nn.Sequential(nn.Conv2d(144,48,3,padding=1),ResidualBlock(48))
        self.up0 = nn.Sequential(nn.Conv2d(72,24,3,padding=1),ResidualBlock(24),nn.Conv2d(24,N_SEM,1))

    def forward(self, image):
        a = self.first(image)
        b = self.down1(a)
        c = self.down2(b)
        mid = self.up1(torch.cat((F.interpolate(c,size=b.shape[-2:],mode="nearest"),b),1))
        return self.up0(torch.cat((F.interpolate(mid,size=a.shape[-2:],mode="nearest"),a),1))


def probe_scores(matrix):
    m = np.asarray(matrix,dtype=np.int64)
    union = m.sum(1)+m.sum(0)-m.diagonal()
    present = union > 0
    mean_iou = float((m.diagonal()[present] / union[present]).mean()) if present.any() else None
    dynamic = SEM["vehicle"]
    tp = m[dynamic:,dynamic:].sum()
    dyn_union = m[dynamic:].sum()+m[:,dynamic:].sum()-tp
    return dict(mean_iou=mean_iou, dynamic_iou=float(tp/dyn_union) if dyn_union else None,
                dynamic_pixels=int(m[dynamic:].sum()),pixels=int(m.sum()),
                class_iou={name:float(m[i,i]/union[i]) if union[i] else None for i,name in enumerate(SEM_NAMES)},
                class_pixels={name:int(m[i].sum()) for i,name in enumerate(SEM_NAMES)},matrix=m.tolist())
