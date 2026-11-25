import torch
from torch.nn import functional as F

from keireality.flow_detail import detail_sums, image_losses, native_display, subtract_source_component
from keireality.meanflow import integrate_meanflow, sample_meanflow


def test_four_step_training_differentiates_through_every_interval():
    weights = torch.tensor([.1, .2, .3, .4], requires_grad=True)
    z = torch.ones(1, 1)
    def field(current, r, t, condition):
        return weights[(t * 4).long() - 1] * current
    trace = []
    endpoint = integrate_meanflow(field, z, None, 4, trace=trace)
    torch.testing.assert_close(endpoint, torch.prod(1 - weights / 4).reshape_as(z))
    torch.testing.assert_close(endpoint, sample_meanflow(field, z, None, 4))
    assert len(trace) == 4 and all(not state.requires_grad for state in trace)
    endpoint.sum().backward()
    assert (weights.grad < 0).all()


def test_native_display_preserves_exact_reference_detail_and_low_frequency_shift():
    base = torch.linspace(.2, .7, 2*3*16*32).reshape(2, 3, 16, 32)
    low = F.avg_pool2d(base, 2)
    torch.testing.assert_close(native_display(low, base), base)
    torch.testing.assert_close(native_display(low + .1, base), base + .1)


def test_grain_and_edge_measures_distinguish_noise_from_contrast():
    truth = torch.full((1, 3, 12, 16), .2)
    truth[..., 8:] = .8
    flat = torch.full_like(truth, .5)
    noisy = truth.clone()
    noisy[..., ::2, ::2] += .08
    n, d = detail_sums(torch.stack([truth, flat, noisy]), truth, .025, .1)
    metrics = n / d
    torch.testing.assert_close(metrics[0], torch.tensor([0., 0., 1.], dtype=metrics.dtype))
    assert metrics[1, 1] < 1e-6 and metrics[1, 2] == 0
    assert metrics[2, 1] > .01
    rgb, edge = image_losses(2*truth-1, 2*truth-1, torch.ones(1, 12, 16), .1, 4)
    assert rgb == 0 and edge == 0


def test_known_source_response_subtracts_noise_and_preserves_scene_contrast():
    rgb = torch.full((1, 3, 8, 12), .2)
    rgb[..., 6:] = .8
    noise = torch.randn_like(rgb)
    kernel = torch.zeros(3, 3, 5, 5)
    kernel[0, 1, 2, 1] = .04
    kernel[1, 0, 3, 3] = -.03
    observed = rgb + F.conv2d(noise, kernel, padding=2)
    torch.testing.assert_close(subtract_source_component(observed, noise, kernel), rgb)
