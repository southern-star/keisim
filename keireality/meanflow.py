"""Conditional MeanFlow training terms and interval sampling, using data at time zero and noise at one."""
import torch


def broadcast_time(t, image):
    return t.reshape(-1, *([1] * (image.ndim - 1)))


def meanflow_terms(field, data, noise, r, t, condition):
    """Return u, the detached regression target and the measured total derivative (Geng et al., Eq. 11)."""
    if data.shape != noise.shape or r.shape != t.shape or t.shape != (len(data),):
        raise ValueError("expected equal data/noise shapes and one time pair per example")
    if not ((0 <= r) & (r <= t) & (t <= 1)).all():
        raise ValueError("expected 0 <= r <= t <= 1")
    velocity = noise - data
    z = data + broadcast_time(t, data) * velocity
    # Closing over condition fixes it along the transport path. r also has zero tangent.
    output, derivative = torch.func.jvp(lambda z_, r_, t_: field(z_, r_, t_, condition),
                                        (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t)))
    target = velocity - broadcast_time(t-r, data) * derivative
    return output, target.detach(), derivative.detach()


def integrate_meanflow(field, noise, condition, nfe=1, *, trace=None):
    """Integrate the same intervals in training and inference. A diagnostic trace receives each raw state."""
    if type(nfe) is not int or nfe < 1:
        raise ValueError("nfe must be a positive integer")
    z = noise
    for i in range(nfe, 0, -1):
        t = noise.new_full((len(noise),), i/nfe)
        r = noise.new_full((len(noise),), (i-1)/nfe)
        z = z - broadcast_time(t-r, z) * field(z, r, t, condition)
        if trace is not None:
            trace.append(z.detach())
    return z


@torch.no_grad()
def sample_meanflow(field, noise, condition, nfe=1):
    return integrate_meanflow(field, noise, condition, nfe)
