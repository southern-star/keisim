"""Native-resolution boundary contrast and motion-compensated temporal error on paired outputs."""
from collections import defaultdict

import numpy as np
from PIL import Image
import torch

from collect_reality_resolution import CFG, OUT, TRACKS, open_fields
from keisim.config import SEM


class Scores:
    def __init__(self, names):
        self.names = names
        self.sums = defaultdict(lambda: np.zeros(len(names), np.float64))
        self.denom = defaultdict(float)
        self.counts = defaultdict(int)
        self.old_error = None

    def add(self, key, values, denom, count):
        self.sums[key] += values
        self.denom[key] += float(denom)
        self.counts[key] += int(count)

    def frame(self, images, gt, seg, seen, warp):
        gt = gt.astype(np.float32)/255.
        images = images.astype(np.float32)/255.
        err = images - gt
        self.add("rgb_mae", np.abs(err).sum(axis=(1,2,3),dtype=np.float64), gt.size, seg.size)
        marking = seg == SEM["marking"]
        # Daytime white paint may be shaded. Reject yellow lines on the paint side of the boundary.
        white = marking & (gt.mean(-1) >= .40) & (np.ptp(gt,axis=-1) <= .16)
        for a,b in ((np.s_[:-1,:],np.s_[1:,:]),(np.s_[:,:-1],np.s_[:,1:])):
            g = gt[b]-gt[a]
            common = seen[a] & seen[b] & (np.abs(g).mean(-1) >= CFG["edge_threshold"])
            masks = {"vehicle": (seg[a]==SEM["vehicle"]) ^ (seg[b]==SEM["vehicle"]),
                     "white_line": (white[a] & ~marking[b]) | (white[b] & ~marking[a])}
            for cls,mask in masks.items():
                mask &= common
                truth = g[mask]
                pred = images[(slice(None),*b)][:,mask]-images[(slice(None),*a)][:,mask]
                count = len(truth)
                self.add(cls+"_contrast", (truth*pred).sum((1,2),dtype=np.float64),
                         np.square(truth).sum(dtype=np.float64),count)
                self.add(cls+"_gradient_mae",np.abs(pred-truth).sum((1,2),dtype=np.float64),3*count,count)
        flat = err.reshape(len(images),-1,3)
        if self.old_error is not None:
            indices = np.flatnonzero(warp.ravel() >= 0)
            prev = warp.ravel()[indices]
            residual = np.abs(flat[:,indices]-self.old_error[:,prev]).mean(-1)
            self.add("static_flicker",residual.sum(1,dtype=np.float64),len(indices),len(indices))
            for cls in ("building","vegetation"):
                keep = seg.ravel()[indices] == SEM[cls]
                self.add(cls+"_flicker",residual[:,keep].sum(1,dtype=np.float64),keep.sum(),keep.sum())
        self.old_error = flat

    def results(self):
        return dict(counts=dict(self.counts), methods={
            m:{k:float(v[i]/self.denom[k]) if self.denom[k] else None for k,v in self.sums.items()}
            for i,m in enumerate(self.names)})


def bytes_rgb(tensor):
    return tensor.clamp(0,1).mul(255).round().byte().permute(0,2,3,1).cpu().numpy()


@torch.inference_mode()
def score_town(town, head, strengths, movie_dir=None):
    """The head receives cached driving features only. Labels and positions enter the scorer afterwards."""
    raw = open_fields(OUT/"capture"/f"town{town}")
    cache = open_fields(OUT/"features"/f"town{town}")
    names = ["current", *[str(s) for s in strengths]]
    n = len(cache["inputs"])
    reports, movies = {}, {}
    if movie_dir is not None and len(strengths)!=1:
        raise ValueError("Publish exactly one frozen candidate")
    for slot,track in enumerate(TRACKS):
        scores, sheets, frames = Scores(names), [], []
        for t in range(n):
            x = torch.from_numpy(cache["inputs"][t,slot].copy()).permute(2,0,1)[None].to("cuda").float()
            with torch.autocast("cuda",dtype=torch.float16):
                values = [x[:,:3], *[head(x.contiguous(memory_format=torch.channels_last),s) for s in strengths]]
            predicted = bytes_rgb(torch.cat(values))
            scores.frame(predicted,raw["rgb"][t,slot],raw["labels"][t,slot],cache["visible"][t,slot],cache["warp"][t,slot])
            if movie_dir is not None:
                frames.append(np.concatenate(predicted, axis=1))
                if len(frames)==SHEET or t==n-1:
                    name = f"town{town}-{track}-{t-len(frames)+1:03d}.png"
                    Image.fromarray(np.concatenate(frames,axis=0)).save(movie_dir/name,compress_level=2)
                    sheets.append(name)
                    frames.clear()
        reports[track] = scores.results()
        movies[f"town{town}-{track}"] = sheets
    return reports,movies


SHEET = 8


def changes(row, method):
    base = row["methods"]["current"]
    new = row["methods"][method]
    return {k:float(new[k]/v-1) if v is not None and v!=0 and new[k] is not None else None for k,v in base.items()}


def quality_check(row, method):
    change = changes(row,method)
    return {k:change[k] is not None and (change[k]>0 if "contrast" in k else change[k]<=0)
            for k in ("vehicle_contrast","white_line_contrast","vehicle_gradient_mae","white_line_gradient_mae",
                      "static_flicker","building_flicker","vegetation_flicker")}
