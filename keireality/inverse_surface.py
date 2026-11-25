"""Sample camera texture on a continuous target ray where its measured source-depth surface agrees."""
import torch
from torch.nn import functional as F

from keisim.config import CAMERA_RIGS, SEM
from .background_history import old_camera, previous_grid
from .scene_detail import sample
from .spatial import camera_grid, ray_points


def surface_seed(z):
    """Seed empty target rays from local inverse depths without averaging through an established edge."""
    valid = (z>.1) & (z<180.)
    inv = valid/z.clamp_min(.1)
    mass = F.avg_pool2d(valid.float(),9,1,4)
    mean = mass/F.avg_pool2d(inv,9,1,4).clamp_min(1e-7)
    seeded = torch.where(valid,z,mean)
    return seeded.clamp(.5,180.), valid | (mass>.025)


def camera_hit(cam, origin, direction, distance, z, rgba, *, solve):
    """Intersect a ray with the source range image; reject occlusion and mixed foreground/background cells."""
    h,w=z.shape[-2:]
    offset=z.new_tensor([cam.x,cam.y,cam.z])
    relative=origin-offset
    along=(relative*direction).sum(-1)
    perpendicular=(relative.square().sum(-1)-along.square()).clamp_min(0)
    valid=(z>.15).float()
    inverse=valid/z.clamp_min(.15)
    moments=torch.stack((valid,inverse,inverse.square()),1)
    s=distance
    for iteration in range(3 if solve else 1):
        point=origin+s[...,None]*direction
        grid,inside=camera_grid(point,cam,2)
        mass,mean,square=sample(moments,grid).unbind(1)
        mean=mean/mass.clamp_min(1e-7)
        std=(square/mass.clamp_min(1e-7)-mean.square()).clamp_min(0).sqrt()
        observed=mean.clamp_min(1e-6).reciprocal()
        expected=(point-offset).norm(dim=-1)
        if solve and iteration<2:
            intersection=-along+(observed.square()-perpendicular).clamp_min(0).sqrt()
            consistent=(mass>.65) & inside & ((observed-expected).abs()<1.5+.30*expected)
            s=torch.where(consistent,.25*s+.75*intersection.clamp(.5,220.),s)
    weight=(1-(observed-expected).abs()/(.7+.12*expected)).clamp(0,1)
    weight*=inside*((mass-.45)/.50).clamp(0,1)*(1-std/(.12*mean+.0005)).clamp(0,1)
    native,_=camera_grid(point,cam,1)
    color=sample(rgba,native)
    weight*=((color[:,3]-.10)/.55).clamp(0,1)
    return color[:,:3],weight[:,None],s


def temporal_layers(display, a, d, view, slot, views, directions, z, seeded):
    """Average the base and its observed correction separately at a consistent static surface."""
    key=("observed_layers",slot)
    residual=d-a
    display.observation.displays[key]=torch.cat((a,residual),1)
    selected=[(i,p) for i,p in enumerate(display.past) if key in p.displays]
    if not selected:
        return a,residual
    indices,records=zip(*selected)
    previous_cameras=views[1:][list(indices)]
    old_views=torch.cat([p.displays[slot][5] for p in records])
    resolved=torch.where(seeded,z,60.)
    grid,inside,expected=previous_grid(previous_cameras,old_views,resolved,directions[:1],view)
    lo,_,_=previous_grid(previous_cameras,old_views,torch.where(seeded,resolved*.80,10.),directions[:1],view)
    hi,_,_=previous_grid(previous_cameras,old_views,torch.where(seeded,resolved*1.25,150.),directions[:1],view)
    ambig=((hi-lo)*grid.new_tensor([d.shape[-1]/2,d.shape[-2]/2])).norm(dim=-1)[:,None]
    previous_z=sample(torch.cat([p.displays[slot][3] for p in records]),grid,mode="nearest")
    old_mask=sample(torch.cat([p.displays[slot][2] for p in records]),grid)
    occluded=(previous_z<180.) & (previous_z<.76*expected-.5)
    weight=inside*(~occluded)*old_mask*(1-ambig/4.0).clamp(0,1)
    layers=sample(torch.cat([p.displays[key] for p in records]),grid)
    a_old,r_old=layers.split(3,dim=1)
    denom=1+weight.sum(0,keepdim=True)
    a_mean=(a+(a_old*weight).sum(0,keepdim=True))/denom
    r_mean=(residual+(r_old*weight).sum(0,keepdim=True))/denom
    return a_mean,r_mean


def aligned_observations(display, d, detail, view, slot, views, directions, z, seeded):
    """Antialias depth-guided detail on static surfaces, including roads and distant sky."""
    key=("static_samples",slot)
    rays=directions[:1]
    _,semantic,_=display.reference.render(view)
    objects=semantic.float().softmax(1)[:,[SEM[n] for n in ("vehicle","pedestrian","marking")]].sum(1,keepdim=True)
    target_guard=F.max_pool2d((objects>.035).float(),15,1,7)
    static=(1-target_guard)*(1-display.foreground_rays(view,rays))
    # Keep the most recent unfiltered sample. No filtered output feeds back into the history.
    observed=(d+detail).clamp(0,1)
    radial=display.current[slot][1]
    cosine=(rays*view[:,None,None,3:6]).sum(-1)[:,None]
    ground=(-view[:,2,None,None,None]/rays[...,2][:,None].clamp(max=-1e-5))
    far=torch.where(rays[...,2][:,None]>.14,3000.,torch.where(ground<90,ground,80.))
    optical=torch.where(radial<220.,radial*cosine,torch.where(seeded,z,far*cosine))
    display.observation.displays[key]=(observed,static,optical)
    selected=[(i,p.displays[key]) for i,p in enumerate(display.past) if key in p.displays]
    if not selected:
        return observed,observed,observed
    indices,records=zip(*selected)
    old_rgb,old_static,old_depth=map(torch.cat,zip(*records))
    old_views=torch.cat([display.past[i].displays[slot][5] for i in indices])
    grid,inside,expected=previous_grid(views[1:][list(indices)],old_views,optical,rays,view)
    old_z=sample(old_depth,grid,mode="nearest")
    old=sample(old_rgb,grid)
    occlusion=(old_z<.70*expected-.65) & (old_z<180.)
    delta=(old-observed).abs().mean(1,keepdim=True)
    weight=inside*(~occlusion)*sample(old_static,grid)*(1-((delta-.12)/.25).clamp(0,1))
    low=-F.max_pool2d(-observed,3,1,1)-.045
    high=F.max_pool2d(observed,3,1,1)+.045
    diff=(weight*(old.clamp(min=low,max=high)-observed)).sum(0,keepdim=True)
    mass=weight.sum(0,keepdim=True)
    normal=(observed+static*diff/(.8+mass)).clamp(0,1)
    strong=(observed+static*diff/(.32+mass)).clamp(0,1)
    sharper=(normal+static*.40*detail).clamp(0,1)
    return normal,strong,sharper


def sample_surfaces(display, a, d, view, slot, variant=None):
    """Fuse current and past real camera observations at the same 3-D point, once per display pixel."""
    record=display.observation.displays[slot]
    mask=record[2]
    z,seeded=surface_seed(record[3])
    past=list(display.past)
    views=view
    if past:
        views=torch.cat((view,old_camera(view,display.reference.motion[0,:len(past)])))
    origin=views[:,None,None,:3]
    dirs=ray_points(views,*d.shape[-2:],(1.,))[:,1]-origin
    cosine=(dirs[:1]*view[:,None,None,3:6]).sum(-1)
    initial=z[:,0]/cosine.clamp_min(.05)
    observations=[display.observation,*past]
    current_sum=torch.zeros_like(d)
    current_mass=torch.zeros_like(z)
    s_sum=torch.zeros_like(initial)
    for k,cam in enumerate(CAMERA_RIGS["wide_narrow"]):
        rgb,w,s=camera_hit(cam,origin[:1],dirs[:1],initial,observations[0].depth[:,k],
                           observations[0].colors[:,k],solve=True)
        w=w*seeded*(2.5 if k else 1.)
        current_sum+=w*rgb
        current_mass+=w
        s_sum+=w[:,0]*s
    current=current_sum/current_mass.clamp_min(1e-7)
    now_alpha=.82*current_mass.clamp(0,1)
    outputs={"ray_current":torch.lerp(d,current,mask*now_alpha)}
    if variant=="ray_current":
        return outputs
    solved=torch.where(current_mass[:,0]>.04,s_sum/current_mass[:,0].clamp_min(1e-7),initial)
    all_sum=current_sum.clone()
    all_mass=current_mass.clone()
    past_square=torch.zeros_like(d)
    if past:
        decay=d.new_tensor([.78**(i+1) for i in range(len(past))])[:,None,None,None]
        for k,cam in enumerate(CAMERA_RIGS["wide_narrow"]):
            rgb,w,_=camera_hit(cam,origin[1:],dirs[1:],solved,torch.cat([p.depth[:,k] for p in past]),
                               torch.cat([p.colors[:,k] for p in past]),solve=False)
            w=w*decay*seeded*(2.5 if k else 1.)
            all_sum+=(rgb*w).sum(0,keepdim=True)
            all_mass+=w.sum(0,keepdim=True)
            past_square+=(w*(rgb-current).square()).sum(0,keepdim=True)
    history=all_sum/all_mass.clamp_min(1e-7)
    history_alpha=.88*(all_mass/1.25).clamp(0,1)
    outputs.update(ray_history=torch.lerp(d,history,mask*history_alpha),
                   ray_fallback=torch.lerp(d,torch.lerp(a,history,history_alpha),mask))
    if variant in ("ray_history","ray_fallback"):
        return outputs
    past_mass=(all_mass-current_mass).clamp_min(0)
    deviation=(past_square/past_mass.clamp_min(1e-6)).mean(1,keepdim=True).clamp_min(0).sqrt()
    repeated=(current_mass.clamp(0,1)*past_mass.clamp(0,1)*(1-deviation/.16).clamp(0,1))
    # Band-limit the correction to avoid an exposure or color jump when a projection appears.
    difference=(history-d)*repeated
    low=F.avg_pool2d(F.pad(difference,(2,2,2,2),mode="replicate"),5,1)
    detail=(difference-low).clamp(-.16,.16)
    outputs["ray_detail"]=(d+mask*.85*detail).clamp(0,1)
    if variant=="ray_detail":
        return outputs
    if variant is None or variant.startswith("ray_aa"):
        normal,heavy,sharper=aligned_observations(display,d,mask*.85*detail,view,slot,views,dirs,z,seeded)
        outputs.update(ray_aa=normal,ray_aa_strong=heavy,ray_aa_sharp=sharper)
        if variant is not None:
            return outputs
    a_mean,r_mean=temporal_layers(display,a,d,view,slot,views,dirs,z,seeded)
    strong=torch.lerp(d,a_mean+(.18+.50*repeated)*r_mean+1.25*detail,mask).clamp(0,1)
    moderate=torch.lerp(d,a_mean+(.48+.45*repeated)*r_mean+1.0*detail,mask).clamp(0,1)
    outputs.update(ray_consensus=strong,ray_balanced=moderate)
    return outputs
