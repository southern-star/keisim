"""Read the exact tokens used by the frozen mc14 driving decoder."""
from hashlib import sha256

import torch

from keipilot.agent import load_model


def frozen_pilot(checkpoint, device):
    pilot = load_model(checkpoint, device).eval().requires_grad_(False)
    if pilot.rig != "wide_narrow" or pilot.n_cams != 2:
        raise ValueError(f"expected the wide_narrow checkpoint; got {pilot.rig}, {pilot.n_cams} cameras")
    return pilot


def state_hash(model):
    digest = sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def extract_mem(pilot, rgb_in, parity=False, pyramid=False, *, scene=False):
    if rgb_in.dtype != torch.uint8 or rgb_in.shape[1:] != (2, 3, 160, 320):
        raise ValueError("mc14 expects uint8 RGB [B, wide/narrow, 3, 160, 320]")
    if pilot.training:
        raise ValueError("freeze the pilot in eval mode before extracting features")
    if scene and not pyramid:
        raise ValueError("scene detail requires a feature dictionary")
    batch = len(rgb_in)
    # Navigation inputs affect only the driving queries after mem has been formed.
    cmd = torch.zeros(batch, dtype=torch.long, device=rgb_in.device)
    tp = torch.zeros(batch, 2, device=rgb_in.device)
    extra = {"with_reality_stem": True} if scene else {}
    out = pilot(rgb_in, cmd, tp, with_seg=scene, with_reality_features=True, with_reality_pyramid=pyramid, **extra)
    mem = out.pop("reality_mem")
    features = {"mem": mem}
    if pyramid:
        for name, shape in (("c2", (2, 128, 20, 40)), ("c3", (2, 256, 10, 20))):
            value = out.pop("reality_" + name)
            if value.shape != (batch, *shape) or not torch.isfinite(value).all():
                raise ValueError(f"invalid {name} features")
            features[name] = value
    if mem.shape != (batch, 500, 256) or not torch.isfinite(mem).all():
        raise ValueError(f"invalid mc14 features {tuple(mem.shape)}")
    if scene:
        features["stem"] = out.pop("reality_stem")
        features["source_logits"] = out["seg"].unflatten(0, (batch, 2))
        if features["stem"].shape != (batch,2,64,80,160):
            raise ValueError("unexpected driving stem grid")
    if parity:
        plain = pilot(rgb_in, cmd, tp, with_seg=scene)
        if set(plain) != set(out) or any(not torch.equal(plain[k], out[k]) for k in plain):
            raise RuntimeError("enabling KeiReality features changed driving outputs")
    return features if pyramid else mem
