"""Native-resolution display of colors read from the frozen driving stem on supported scene surfaces."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from keisim.config import CAMERA_RIGS, SEM
from .depth_projection import DepthProjector
from .spatial import camera_grid, ray_points
from .stream import TemporalDisplay


def sample(image, grid, mode="bilinear"):
    return F.grid_sample(image.float(),grid,mode=mode,padding_mode="border",align_corners=False)


def combine(colors, weights):
    """Prefer the denser narrow camera when both cameras support a point."""
    priority = weights.new_tensor([1.,3.])[None,:,None,None]
    weighted = weights * priority
    return (colors * weighted[:,:,None]).sum(1) / weighted.sum(1,keepdim=True).clamp_min(1e-8)


class SceneDetail(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.camera_color = nn.Conv2d(64,12,3,padding=1,padding_mode="replicate")
        self.projector = DepthProjector()
        self.surface_strength = cfg["splat_strength"]
        self.ground_strength = cfg["ground_strength"]
        self.sky_strength = cfg["sky_strength"]
        self.history_weight = cfg["previous_static_weight"]

    def encode(self, stem, logits):
        color = F.pixel_shuffle(self.camera_color(stem.flatten(0,1).float()),2).clamp(0,1).unflatten(0,(-1,2))
        probs = logits.float().softmax(2)
        ground = probs[:,:, [SEM[k] for k in ("road","marking","sidewalk","terrain")]].sum(2)
        moving = probs[:,:,SEM["vehicle"]] + probs[:,:,SEM["pedestrian"]]
        return color, torch.stack((probs[:,:,SEM["sky"]],ground,moving),2)

    def surfaces(self, colors, probs, view, depth):
        grids, mass, inverse = self.projector(depth,view,size=(128,256))
        # The half-grid depth cell (i,j) is on source RGB pixel (2*i,2*j). Align the three resolutions explicitly.
        pixel = grids.new_tensor([1/320,1/160])
        color = sample(colors.flatten(0,1), (grids-pixel).flatten(0,1)).unflatten(0,(-1,2))
        moving = sample(probs[:,:,2:3].flatten(0,1),(grids+2*pixel).flatten(0,1)).unflatten(0,(-1,2))[:,:,0]
        z = inverse.clamp_min(1e-5).reciprocal()
        near = torch.where(mass>.015,z,10000.).amin(1,keepdim=True)
        same = (1-(z-near).clamp_min(0)/(.6+.12*near)).clamp(0,1)
        weights = mass*same
        color = combine(color,weights)
        conf = weights.amax(1,keepdim=True)
        # Interpolate across small gaps in the projected lattice using premultiplied colors.
        nearby = F.avg_pool2d(conf,3,1,1)
        color = (color*conf + .40*F.avg_pool2d(color*conf,3,1,1)) / (conf+.40*nearby).clamp_min(1e-8)
        dynamic = (moving*weights).sum(1,keepdim=True) / weights.sum(1,keepdim=True).clamp_min(1e-8)
        return color, torch.maximum(conf,nearby*.8), near, dynamic

    def directional_color(self, points, colors, probs, depth, *, sky):
        samples, weights = [], []
        for k,cam in enumerate(CAMERA_RIGS["wide_narrow"]):
            grid,inside = camera_grid(points,cam,1)
            offset = grid.new_tensor([1/cam.width,1/cam.height])
            color = sample(colors[:,k],grid)
            label = sample(probs[:,k,0:1] if sky else probs[:,k,1:2],grid+3*offset)[:,0]
            weight = ((label-.30)/.65).clamp(0,1)*inside
            if not sky:
                estimated = sample(depth[:,k:k+1],grid+offset)[:,0]
                expected = (points-self.projector.origins[k]).norm(dim=-1)
                relative = (estimated/expected.clamp_min(.01)-1).abs()
                weight = weight*((.42-relative)/.27).clamp(0,1)
            weights.append(weight)
            samples.append(color)
        weights = torch.stack(weights,1)
        return combine(torch.stack(samples,1),weights), weights.amax(1,keepdim=True)

    def frame(self, base, encoded, view, estimated):
        colors, probs = encoded
        surface, cover, near, dynamic = self.surfaces(colors,probs,view,estimated)
        hypotheses = ray_points(view,128,256,(1.,))
        origin = view[:,:3,None,None].permute(0,2,3,1)
        direction = hypotheses[:,1]-origin
        cos = (direction*view[:,None,None,3:6]).sum(-1)[:,None]
        ground,ground_conf = self.directional_color(hypotheses[:,0],colors,probs,estimated,sky=False)
        ground_range = (hypotheses[:,0]-origin).norm(dim=-1)[:,None]
        ground_z = ground_range*cos
        ground_conf = ground_conf * (hypotheses[:,0,:,:,2].abs()[:,None]<.03)
        ground_conf = ground_conf * ((near-ground_z*.80+.2)/(.12*ground_z+.3)).clamp(0,1)
        sky,sky_conf = self.directional_color(origin+direction*4000,colors,probs,estimated,sky=True)
        sky_conf = sky_conf*(direction[:,:,:,2][:,None]>-.03)*((near-70)/35).clamp(0,1)
        rgb = torch.lerp(base,surface,self.surface_strength*cover)
        rgb = torch.lerp(rgb,ground,self.ground_strength*ground_conf)
        rgb = torch.lerp(rgb,sky,self.sky_strength*sky_conf)
        radial = near/cos.clamp_min(.01)
        radial = torch.where(ground_conf>.55,ground_range,radial)
        radial = torch.where(sky_conf>.55,4000.,radial)
        static = torch.maximum(cover*(1-dynamic),torch.maximum(ground_conf,sky_conf))
        return rgb,radial,static,direction

    def temporal(self, rgb, radial, static, direction, view, delta, previous):
        """Use the immediately previous observation only when its estimated static surface matches."""
        if previous is None:
            return rgb
        old_rgb,old_depth,old_static,old_view = previous
        origin = view[:,None,None,:3]
        xyz = origin+direction*radial.permute(0,2,3,1)
        dx,dy,yaw = delta.reshape(-1,1,1,3).unbind(-1)
        x,y,z = xyz.unbind(-1)
        world = torch.stack((dx+yaw.cos()*x-yaw.sin()*y,dy+yaw.sin()*x+yaw.cos()*y,z),-1)
        pt = world-old_view[:,None,None,:3]
        forward,up = old_view[:,None,None,3:6],old_view[:,None,None,6:9]
        right = torch.linalg.cross(forward,up,dim=-1)
        optical = (pt*forward).sum(-1)
        slope = torch.tan(torch.deg2rad(old_view[:,9])/2)[:,None,None]
        grid = torch.stack(((pt*right).sum(-1)/(optical.clamp_min(.01)*slope*old_view[:,10,None,None]),
                            -(pt*up).sum(-1)/(optical.clamp_min(.01)*slope)),-1)
        inside = (grid.abs()<1).all(-1)[:,None] & (optical[:,None]>.01)
        old = sample(old_rgb,grid)
        measured = sample(old_depth,grid,mode="nearest")
        expected = pt.norm(dim=-1)[:,None]
        close = (1-(measured-expected).abs()/(.35+.07*expected)).clamp(0,1)
        compatible = (1-(old-rgb).abs().mean(1,keepdim=True)/.35).clamp(0,1)
        weight = self.history_weight*inside*close*compatible*static*sample(old_static,grid)
        return torch.lerp(rgb,old,weight)


class SceneDetailDisplay:
    """Observe a driving tick once. Each display slot can request its own camera path."""
    def __init__(self, phase2, detail, comparison_branch=None):
        self.reference = TemporalDisplay(phase2,comparison_branch)
        self.detail = detail.eval()
        self.episode = self.time = None
        self.previous, self.current = {}, {}

    @torch.inference_mode()
    def observe(self, features, episode, t, delta):
        estimated = features["estimated_depth"]
        if estimated.shape != (1,2,80,160):
            raise ValueError("expected one current two-camera depth estimate")
        encoded = self.detail.encode(features["stem"],features["source_logits"])
        self.reference.observe(features,episode,t,delta)
        contiguous = self.episode == episode and self.time is not None and math.isclose(t-self.time,.1,abs_tol=1e-6)
        self.previous,self.current = self.current if contiguous else {}, {}
        self.episode,self.time,self.encoded,self.depth = episode,t,encoded,estimated
        self.delta = torch.tensor(delta,device=estimated.device,dtype=torch.float32)

    @torch.inference_mode()
    def render(self, view, slot=0, *, comparisons=False):
        b,semantic,support,a = self.reference.render(view,return_reference=True)
        rgb,depth,static,direction = self.detail.frame(a,self.encoded,view,self.depth)
        c = self.detail.temporal(rgb,depth,static,direction,view,self.delta,self.previous.get(slot))
        self.current[slot] = rgb,depth,static,view.clone()
        return (a,b,c) if comparisons else (c,semantic,support)
