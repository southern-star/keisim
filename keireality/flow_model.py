"""Conditional U-Net and causal state for a single 10 Hz MeanFlow frame."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .meanflow import broadcast_time, sample_meanflow
from .renderer import ResidualBlock


class FlowUNet(nn.Module):
    """Two downsampling stages, conditional RGB/support and a previous-image slot (invalid on an initial frame)."""
    def __init__(self, condition_channels=8):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(5+condition_channels, 112, 3, padding=1), ResidualBlock(112))
        self.down1 = nn.Sequential(nn.Conv2d(112, 224, 3, stride=2, padding=1), ResidualBlock(224))
        self.down2 = nn.Sequential(nn.Conv2d(224, 448, 3, stride=2, padding=1), ResidualBlock(448), ResidualBlock(448))
        self.up1 = nn.Sequential(nn.Conv2d(448+224, 224, 3, padding=1), ResidualBlock(224))
        self.up0 = nn.Sequential(nn.Conv2d(224+112, 112, 3, padding=1), ResidualBlock(112), nn.Conv2d(112, 3, 1))

    def forward(self, z, r, t, condition):
        ts = [broadcast_time(ti, z).expand(-1, 1, *z.shape[2:]) for ti in (r, t)]
        a = self.stem(torch.cat([z, condition, *ts], 1))
        b = self.down1(a)
        c = self.down2(b)
        x = self.up1(torch.cat([F.interpolate(c, size=b.shape[-2:], mode="nearest"), b], 1))
        return self.up0(torch.cat([F.interpolate(x, size=a.shape[-2:], mode="nearest"), a], 1))


def make_condition(current, view, previous, previous_view, delta, elapsed, valid):
    """Pack only inference inputs. Empty history masks the image, its camera, ego displacement and age."""
    n, _, h, w = current.shape
    scale = view.new_tensor([10,10,10,1,1,1,1,1,1,90,2])
    keep = valid.reshape(n,1)
    dx, dy, yaw = delta.unbind(1)
    motion = torch.stack((dx/2, dy/2, yaw.sin(), yaw.cos(), elapsed/.1), 1)
    previous_info = torch.cat([previous_view/scale, motion], 1)
    previous_info = torch.where(keep, previous_info, 0.)
    metadata = torch.cat([keep.float(), view/scale, previous_info], 1)[:, :, None, None].expand(-1,-1,h,w)
    image = torch.where(keep[:,:,None,None], previous, 0.)
    return torch.cat([current.float(), image, metadata], 1)


class ConditionalVelocity(FlowUNet):
    def __init__(self, sigma=.18):
        # Phase 2 RGB/support/features (28), previous RGB (3), validity + two cameras + motion and dt (28).
        super().__init__(condition_channels=59)
        self.sigma = sigma
        nn.init.zeros_(self.up0[-1].weight)
        nn.init.zeros_(self.up0[-1].bias)

    def forward(self, z, r, t, condition):
        # Analytic average velocity for a Gaussian image prior centred on Phase 2, plus a learnt correction.
        # The rationalized slope is defined also when r=t. Noise is standard normal at t=1 in both paths.
        sigma2 = self.sigma**2
        sr = (r.square() + sigma2*(1-r).square()).sqrt()
        st = (t.square() + sigma2*(1-t).square()).sqrt()
        slope = ((t+r)*(1+sigma2)-2*sigma2) / ((st+sr)*st)
        mean = condition[:, :3]
        prior = broadcast_time(slope,z)*(z-broadcast_time(1-t,z)*mean)-mean
        return prior + super().forward(z,r,t,condition)


@torch.no_grad()
def generate(field, noise, condition, nfe, mode="meanflow"):
    if mode == "meanflow":
        return sample_meanflow(field, noise, condition, nfe).clamp(-1,1)
    if mode != "flow_matching" or type(nfe) is not int or nfe < 1:
        raise ValueError("invalid flow sampler")
    z = noise
    for i in range(nfe,0,-1):
        t = z.new_full((len(z),),i/nfe)
        z = z - field(z,t,t,condition)/nfe
    return z.clamp(-1,1)


def frame_noise(episode_seed, step, views, seed, device):
    # One independent draw per (episode, physical tick, requested view). Batch order and NFE do not enter the seed.
    import hashlib
    samples = []
    for angle in views:
        key = f"{seed}/{episode_seed}/{step}/{angle}".encode()
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63-1)
        rng = torch.Generator(device=device).manual_seed(value)
        samples.append(torch.randn(3,64,128,device=device,generator=rng))
    return torch.stack(samples)


class FlowRollout:
    """Each independent video feeds its own previous generated image back to the field."""
    def __init__(self, field, nfe, mode="meanflow"):
        self.field, self.nfe, self.mode = field, nfe, mode
        self.last = self.before = None
        self.used_previous = False

    @torch.no_grad()
    def render(self, current, view, delta, episode, t, noise):
        past = self.last
        if past is not None and past[0] == episode and abs(t-past[1]) < 1e-7:
            past = self.before
        valid = (past is not None and past[0] == episode and math.isclose(t-past[1],.1,abs_tol=1e-6)
                 and past[2].shape == noise.shape)
        self.used_previous = valid
        previous, old_view = (past[2],past[3]) if valid else (torch.zeros_like(noise),torch.zeros_like(view))
        cond = make_condition(current,view,previous,old_view,delta,
                              view.new_full((len(view),),t-past[1] if valid else 0.),
                              torch.full((len(view),),valid,device=view.device,dtype=torch.bool))
        image = generate(self.field,noise,cond,self.nfe,self.mode)
        self.before = past if valid else None
        self.last = (episode,t,image.detach(),view.detach().clone())
        return image
