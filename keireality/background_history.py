"""Short, odometry-aligned history of observed background surfaces for the C display."""
from collections import deque
from dataclasses import dataclass, field
import math

import torch
from torch.nn import functional as F

from keisim.config import CAMERA_RIGS, SEM
from .scene_detail import SceneDetail, SceneDetailDisplay, sample
from .spatial import camera_grid


def old_camera(view, motion):
    """Express today's virtual camera in each previous ego frame (old pose in current ego)."""
    dx, dy, sn, cs = motion[:, :4].unbind(1)
    vectors = view[:, :9].expand(len(motion), -1).reshape(-1,3,3)
    offset = torch.stack((dx, dy, torch.zeros_like(dx)), 1)
    position = vectors[:,0] - offset
    vectors = torch.cat((position[:,None], vectors[:,1:]), 1)
    x,y,z = vectors.unbind(2)
    rotated = torch.stack((cs[:,None]*x + sn[:,None]*y, cs[:,None]*y - sn[:,None]*x, z), 2)
    return torch.cat((rotated.flatten(1), view[:,9:].expand(len(motion), -1)), 1)


def previous_grid(camera, previous_views, optical_depth, direction, current):
    """Back-project an estimated optical depth in the current view into older display images."""
    radial = optical_depth / (direction * current[:,None,None,3:6]).sum(-1)[:,None].clamp_min(.01)
    height, width = direction.shape[1:3]
    y,x = torch.meshgrid((torch.arange(height,device=current.device)+.5)*2/height-1,
                         (torch.arange(width,device=current.device)+.5)*2/width-1,indexing="ij")
    forward,up = camera[:,None,None,3:6], camera[:,None,None,6:9]
    slope = torch.tan(torch.deg2rad(camera[:,9])/2)[:,None,None,None]
    ray = F.normalize(forward + slope * (x[...,None]*camera[:,None,None,10:11]
                      *torch.linalg.cross(forward,up,dim=-1) - y[...,None]*up), dim=-1)
    point = camera[:,None,None,:3] + ray * radial.permute(0,2,3,1)
    relative = point - previous_views[:,None,None,:3]
    f,u = previous_views[:,None,None,3:6], previous_views[:,None,None,6:9]
    z = (relative*f).sum(-1)
    scale = z.clamp_min(.01)*torch.tan(torch.deg2rad(previous_views[:,9])/2)[:,None,None]
    grid = torch.stack(((relative*torch.linalg.cross(f,u,dim=-1)).sum(-1) /
                        (scale*previous_views[:,None,None,10]), -(relative*u).sum(-1)/scale), -1)
    return grid, ((grid.abs()<1).all(-1) & (z>.01))[:,None], z[:,None]


@dataclass
class Observation:
    colors: torch.Tensor
    probs: torch.Tensor
    depth: torch.Tensor
    displays: dict = field(default_factory=dict)


class BackgroundHistoryDisplay(SceneDetailDisplay):
    """Keep original camera observations and C frames for up to three ticks, never the filled output."""
    def __init__(self, phase2, detail):
        super().__init__(phase2, detail)
        self.past = deque(maxlen=3)
        self.observation = None

    @torch.inference_mode()
    def observe(self, features, episode, t, delta):
        contiguous = self.episode == episode and self.time is not None and math.isclose(t-self.time,.1,abs_tol=1e-6)
        if contiguous:
            self.past.appendleft(self.observation)
        else:
            self.past.clear()
        super().observe(features, episode, t, delta)
        source = features["source_logits"].float().softmax(2)
        self.source_guard = source[:,:,[SEM[n] for n in ("marking","vehicle","pedestrian")]].sum(2,keepdim=True)
        bg = source[:,:,[SEM["building"],SEM["vegetation"]]].sum(2,keepdim=True)
        color, probs = self.encoded
        bg = F.interpolate(bg.flatten(0,1), size=color.shape[-2:], mode="bilinear",
                           align_corners=False).unflatten(0, color.shape[:2])
        self.observation = Observation(torch.cat((color,bg),2), probs, self.depth)

    def foreground_rays(self, view, ray):
        """Exclude lines of sight to camera-detected markings and actors at road/vehicle heights."""
        height = view.new_tensor([0.,1.,2.2])[None,:,None,None]
        distance = (height-view[:,None,None,None,2]) / ray[:,None,:,:,2].clamp(max=-1e-5)
        valid = (ray[:,None,:,:,2]<0) & (distance>.1) & (distance<170.)
        points = view[:,None,None,None,:3] + distance[...,None]*ray[:,None]
        b,planes,h,w,_ = points.shape
        points = points.flatten(1,2)
        guard = torch.zeros_like(distance)
        for k,cam in enumerate(CAMERA_RIGS["wide_narrow"]):
            grid,inside = camera_grid(points,cam,4)
            p = sample(self.source_guard[:,k],grid)[:,0]*inside
            guard = torch.maximum(guard,p.reshape(b,planes,h,w))
        positive = ((guard>.08) & valid).any(1,keepdim=True).float()
        return F.max_pool2d(positive,7,1,3)

    @staticmethod
    def background_mask(semantic, frame):
        probs = semantic.float().softmax(1)
        bg = probs[:,[SEM["building"],SEM["vegetation"]]].sum(1,keepdim=True)
        lights = probs[:,[SEM[n] for n in ("tl_red","tl_yellow","tl_green")]].sum(1,keepdim=True)
        objects = probs[:,[SEM[n] for n in ("vehicle","pedestrian","marking")]].sum(1,keepdim=True)
        upward = frame.direction[...,2][:,None] > .008
        # The level simulator and elevated virtual camera put pavement below the
        # horizon. Keep ambiguous treetop/sky boundaries eligible for observed colors.
        blocked = (lights>.035) | (objects>torch.where(upward,.20,.035)) | ((frame.moving>.10) & (frame.cover>.08))
        blocked = F.max_pool2d(blocked.float(),13,1,6) > 0
        return ((bg+.75*upward*probs[:,SEM["sky"]:SEM["sky"]+1]-.12)/.40).clamp(0,1) * ~blocked

    def blend(self, base, c, frame, view, slot, background):
        # Historical camera colors, semantics and depth all predate the current frame.
        records = list(self.past)
        count = len(records)
        if not count:
            return c
        cameras = old_camera(view, self.reference.motion[0,:count])
        surface, cover, z, moving = SceneDetail.surfaces(
            self.detail, torch.cat([p.colors for p in records]), torch.cat([p.probs for p in records]),
            cameras, torch.cat([p.depth for p in records]))
        prev_bg = ((surface[:,3:4]-.12)/.45).clamp(0,1)
        evidence = cover * prev_bg * (moving<.12) * (z<180.)
        now_z = frame.optical_depth
        now_known = (frame.cover>.12) & (now_z<180.)
        # Use current geometry when available, then recent projected surfaces. Interpolate
        # depth across small holes only for warping. This never interpolates the stored RGB.
        inv_sum = (evidence/z.clamp_min(.1)).sum(0,keepdim=True)
        mass = evidence.sum(0,keepdim=True)
        nearby = F.avg_pool2d(mass,13,1,6)
        inv_nearby = F.avg_pool2d(inv_sum,13,1,6)
        estimate = (mass+.7*nearby)/(inv_sum+.7*inv_nearby).clamp_min(1e-7)
        depth = torch.where(now_known,now_z,estimate)
        valid_z = now_known | ((mass+.7*nearby)>.06)
        compatible = ((z-depth).abs() < .9+.18*depth) & valid_z
        direct = evidence * compatible
        # These colors can also enter a newly requested view with no previous target image.
        weighted = (surface[:,:3]*direct).sum(0,keepdim=True)
        total = direct.sum(0,keepdim=True)
        average = (c+weighted)/(1+total)
        selected = [i for i,p in enumerate(records) if slot in p.displays]
        usable = valid_z
        if selected:
            images, layers, masks, zs, alphas, old_views = zip(*(records[i].displays[slot] for i in selected))
            old_views = torch.cat(old_views)
            working_z = torch.where(valid_z,depth,60.)
            grid, inside, expected = previous_grid(cameras[selected],old_views,working_z,frame.direction,view)
            # If a plausible depth interval moves the historical sample by at most 1.5
            # pixels, even a depth hole has a well-defined warp (exactly so at a stop).
            lo = torch.where(valid_z,.65*depth,6.)
            hi = torch.where(valid_z,1.50*depth,160.)
            lo_grid,lo_inside,_ = previous_grid(cameras[selected],old_views,lo,frame.direction,view)
            hi_grid,hi_inside,_ = previous_grid(cameras[selected],old_views,hi,frame.direction,view)
            scale = grid.new_tensor([c.shape[-1]/2,c.shape[-2]/2])
            low_parallax = (((lo_grid-hi_grid)*scale).norm(dim=-1)[:,None]<1.5) & lo_inside & hi_inside
            old_z = sample(torch.cat(zs),grid,mode="nearest")
            old_known = old_z<180.
            good_depth = ~old_known | ((old_z-expected).abs() < 1.+.18*expected) | (low_parallax & ~now_known)
            observed_nearby = F.max_pool2d(evidence[selected],11,1,5)>.08
            old_alpha = sample(torch.cat(alphas),grid)
            local_alpha = sample(F.max_pool2d(torch.cat(alphas),9,1,4),grid)
            observed_nearby = observed_nearby | (local_alpha>.035) | (frame.alpha>.05)
            w = sample(torch.cat(masks),grid)*inside*(valid_z | low_parallax)*good_depth*observed_nearby
            usable = valid_z | (w.amax(0,keepdim=True)>0)
            # Accumulate the fallback image and its projected RGB correction separately.
            # In particular an old, observed weak mixture contributes a small correction
            # even if the current tick has a strong splat, and vice versa.
            strength = self.detail.surface_strength
            attenuation = torch.where(frame.cover>.002, frame.alpha/(strength*frame.cover).clamp_min(1e-6),1.)
            current_layer = frame.alpha*(frame.surface_rgb-base)
            old_layer = sample(torch.cat(layers),grid)
            direct_layer = strength*attenuation*direct[selected]*(surface[selected,:3]-base)
            source_layer = .65*old_layer+.35*direct_layer
            past_weight = 1.5*w
            denom = 1+past_weight.sum(0,keepdim=True)
            fallback = (c-current_layer+((sample(torch.cat(images),grid)-old_layer)*past_weight).sum(0,keepdim=True))/denom
            projection = (current_layer+(source_layer*past_weight).sum(0,keepdim=True))/denom
            votes = (frame.alpha/.20).clamp(max=1) + (w*(torch.maximum(old_alpha,strength*direct[selected])/.20).clamp(max=1)).sum(0,keepdim=True)
            reliability = 1-.50*(1-(votes/2).clamp(max=1))*w.amax(0,keepdim=True)
            fused = fallback + reliability*projection
            average = torch.where(w.sum(0,keepdim=True)>0,fused,average)
        return torch.lerp(c,average,background*usable).clamp(0,1)

    @torch.inference_mode()
    def render(self, view, slot=0, *, comparisons=False):
        a,semantic,support = self.reference.render(view)
        frame = self.detail.inspect(a,self.encoded,view,self.depth)
        c = self.detail.temporal(frame.rgb,frame.depth,frame.static,frame.direction,view,
                                 self.delta,self.previous.get(slot))
        self.current[slot] = frame.rgb,frame.depth,frame.static,view.clone()
        background = self.background_mask(semantic,frame)*(1-self.foreground_rays(view,frame.direction))
        d = self.blend(a,c,frame,view,slot,background)
        layer = frame.alpha*(frame.surface_rgb-a)
        self.observation.displays[slot] = (c,layer,background,frame.optical_depth,frame.alpha,view.clone())
        return (a,c,d) if comparisons else (d,semantic,support)
