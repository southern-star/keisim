"""Display candidates with learned source depths and optional static-surface stabilization."""
import torch
from torch.nn import functional as F

from .background_history import BackgroundHistoryDisplay, old_camera, previous_grid
from .inverse_surface import sample_surfaces
from .scene_detail import sample
from .spatial import ray_points

DEPTH_VARIANTS = ("depth", "stable", "guarded", "guarded_stable")
RAY_VARIANTS = ("ray_current", "ray_history", "ray_fallback", "ray_consensus", "ray_balanced", "ray_detail",
                "ray_aa", "ray_aa_strong", "ray_aa_sharp")
VARIANTS = (*DEPTH_VARIANTS, *RAY_VARIANTS)


def stabilize(display, rgb, view, slot):
    """Reproject unfiltered colors from the last three observations, with foreground and visibility guards."""
    _, radial, _, _ = display.current[slot]
    record = display.observation.displays[slot]
    background = record[2]
    record_key = ("unfiltered", slot)
    display.observation.displays[record_key] = rgb
    selected = [(i,p) for i,p in enumerate(display.past) if record_key in p.displays]
    if not selected:
        return rgb
    indices, observations = zip(*selected)
    cameras = old_camera(view,display.reference.motion[0,list(indices)])
    old_views = torch.cat([p.displays[slot][5] for p in observations])
    rays = ray_points(view,*rgb.shape[-2:],(1.,))[:,1] - view[:,None,None,:3]
    cosine = (rays*view[:,None,None,3:6]).sum(-1)[:,None]
    optical = radial*cosine
    measured = (record[3]<180.) & (record[3]>.1)
    denom = F.avg_pool2d(measured.float(),9,1,4)
    inv = F.avg_pool2d(measured / record[3].clamp_min(.1),9,1,4)
    near = torch.where(denom>.04,denom/inv.clamp_min(1e-6),60.)
    z = torch.where(measured,optical,near)
    grid, inside, expected = previous_grid(cameras,old_views,z,rays,view)
    old_z = sample(torch.cat([p.displays[slot][3] for p in observations]),grid,mode="nearest")
    relative_error = (expected-old_z).abs()/(1.+.20*expected.clamp_min(.1))
    confidence = torch.where(old_z<180.,(1-relative_error).clamp(0,1),.55)
    # If there is no surface for either view, only use correspondences insensitive to depth.
    close_grid,_,_ = previous_grid(cameras,old_views,torch.where(measured,z*.8,8.),rays,view)
    far_grid,_,_ = previous_grid(cameras,old_views,torch.where(measured,z*1.2,150.),rays,view)
    uncertainty = ((close_grid-far_grid)*grid.new_tensor([rgb.shape[-1]/2,rgb.shape[-2]/2])).norm(dim=-1)[:,None]
    confidence *= torch.where(measured,1.,(1-uncertainty/3).clamp(0,1))
    old_color = sample(torch.cat([p.displays[record_key] for p in observations]),grid)
    old_bg = sample(torch.cat([p.displays[slot][2] for p in observations]),grid)
    low = -F.max_pool2d(-rgb,3,1,1)-.018
    high = F.max_pool2d(rgb,3,1,1)+.018
    clipped = old_color.clamp(min=low,max=high)
    decay = rgb.new_tensor([.86**i for i in indices])[:,None,None,None]
    weights = 1.6*inside*confidence*old_bg*decay
    diff = (weights*(clipped-rgb)).sum(0,keepdim=True)/(1+weights.sum(0,keepdim=True))
    return (rgb+background*diff).clamp(0,1)


class RefinedDrive:
    """Observe the two live camera features once. Multiple render calls may share that observation."""
    def __init__(self, model, detail, refiner, variant="depth", *, compare=False):
        if variant not in VARIANTS:
            raise ValueError("unknown depth display")
        self.variant = variant
        self.compare = compare
        self.all_variants = compare == "all"
        self.corrected = BackgroundHistoryDisplay(model,detail) if self.all_variants or variant in DEPTH_VARIANTS else None
        needs_control = compare or variant.startswith("guarded") or variant in RAY_VARIANTS
        self.control = BackgroundHistoryDisplay(model,detail) if needs_control else None
        self.refiner = refiner

    @torch.inference_mode()
    def observe(self,features,episode,t,delta):
        if self.control is not None:
            self.control.observe(features,episode,t,delta)
        if self.corrected is not None:
            corrected = dict(features,estimated_depth=self.refiner.metres(features))
            self.corrected.observe(corrected,episode,t,delta)

    @torch.inference_mode()
    def render(self,view,slot=0):
        images = {}
        if self.control is not None:
            images["A"],_,images["D"] = self.control.render(view,slot,comparisons=True)
        if self.all_variants or self.variant in RAY_VARIANTS:
            images.update(sample_surfaces(self.control,images["A"],images["D"],view,slot,
                                         None if self.all_variants else self.variant))
        if self.corrected is None:
            return images if self.compare else images[self.variant]
        rgb = self.corrected.render(view,slot)[0]
        images["depth"] = rgb
        if self.all_variants or self.variant.endswith("stable"):
            images["stable"] = stabilize(self.corrected,rgb,view,slot)
        if self.all_variants or self.variant.startswith("guarded"):
            # Restore D around predicted vehicles and road markings; both views must agree on background.
            mask = torch.minimum(self.control.observation.displays[slot][2],self.corrected.observation.displays[slot][2])
            for variant in ("depth","stable"):
                if variant in images:
                    name = "guarded" if variant=="depth" else "guarded_stable"
                    images[name]=torch.lerp(images["D"],images[variant],mask)
        return images if self.compare else images[self.variant]
