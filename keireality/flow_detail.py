"""Supervise the final image of a complete transport and preserve the native decoder's fine detail."""
import torch
from torch.nn import functional as F


def image_losses(image, target, pixel_weight, edge_threshold, strong_edge_weight):
    """RGB and oriented adjacent-pixel errors. Image and target use the training range [-1, 1]."""
    color = ((image - target).abs().mean(1) * pixel_weight).sum() / pixel_weight.sum()
    gradient = image.new_zeros(())
    for axis in (-2, -1):
        truth = target.diff(dim=axis)
        delta = image.diff(dim=axis)
        weights = torch.where(truth.abs().mean(1) > 2 * edge_threshold, strong_edge_weight, 1.)
        gradient = gradient + ((delta - truth).abs().mean(1) * weights).sum() / weights.sum()
    return color, gradient * .5


def native_display(image, phase2):
    """Both tensors are unit RGB. Only the native Phase 2 detail above the flow grid is added."""
    size = phase2.shape[-2:]
    low = F.adaptive_avg_pool2d(phase2, image.shape[-2:])
    correction = F.interpolate(image - low, size=size, mode="bilinear", align_corners=False)
    return (phase2 + correction).clamp(0, 1)


def subtract_source_component(rgb, noise, kernel):
    """Subtract only the fitted response to the known source noise from unit RGB, after generation."""
    return (rgb - F.conv2d(noise, kernel, padding=kernel.shape[-1] // 2)).clamp(0, 1)


def highpass(x):
    center = x[..., 1:-1, 1:-1]
    neighbors = x[..., :-2, 1:-1] + x[..., 2:, 1:-1] + x[..., 1:-1, :-2] + x[..., 1:-1, 2:]
    return center - .25 * neighbors


def detail_sums(images, truth, flat_threshold, edge_threshold):
    """Return additive numerators and denominators per method, using RGB in [0, 1]."""
    # images: method x view x RGB x H x W. The two masks depend only on the teacher.
    error = images - truth
    methods = len(images)
    dims = (1, 2, 3, 4)
    span = F.max_pool2d(truth, 3, 1) + F.max_pool2d(-truth, 3, 1)
    flat = span.amax(1, keepdim=True) <= flat_threshold
    band = highpass(error)
    edge_dot = images.new_zeros(methods)
    edge_energy = images.new_zeros(())
    for axis in (-2, -1):
        change = truth.diff(dim=axis)
        mask = change.abs().mean(1, keepdim=True) >= edge_threshold
        edge_dot += (images.diff(dim=axis) * change * mask).sum(dims)
        edge_energy += (change.square() * mask).sum()
    numerator = torch.stack((error.abs().sum(dims), (band.abs() * flat).sum(dims), edge_dot), 1)
    denominator = torch.stack((images.new_tensor(truth.numel()), flat.sum() * truth.shape[1], edge_energy))
    return numerator.double(), denominator.double()
