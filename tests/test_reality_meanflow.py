import pytest
import torch

from keireality.meanflow import broadcast_time, meanflow_terms, sample_meanflow


def test_conditional_total_derivative_and_detached_target():
    data = torch.tensor([[.1, -.7], [.3, .5]])
    noise, c = data.flip(0), torch.tensor([[.3], [.8]])
    r, t = torch.tensor([.1, .2]), torch.tensor([.4, .7])
    scale = torch.tensor(.8, requires_grad=True)
    def field(z, start, end, condition):
        return scale*z.square() + condition*z*end[:, None] + (start*end)[:, None]
    u, target, derivative = meanflow_terms(field, data, noise, r, t, c)
    v = noise-data
    z = data + broadcast_time(t, data)*v
    expected = (2*scale*z + c*t[:, None])*v + c*z + r[:, None]
    torch.testing.assert_close(derivative, expected)
    torch.testing.assert_close(target, v-(t-r)[:, None]*expected)
    assert not target.requires_grad and u.requires_grad
    (u-target).square().mean().backward()
    assert torch.isfinite(scale.grad) and scale.grad != 0
    with pytest.raises(ValueError, match="r <= t"):
        meanflow_terms(field, data, noise, t, r, c)


@pytest.mark.parametrize("nfe", [1, 4, 8])
def test_noise_to_data_interval_sign_and_calls(nfe):
    x, eps = torch.tensor([[.7]]), torch.tensor([[-.2]])
    calls = []
    def field(z, r, t, cond):
        calls.append((r.item(), t.item()))
        return eps-x
    torch.testing.assert_close(sample_meanflow(field, eps, None, nfe), x)
    assert len(calls) == nfe and calls[0][1] == 1 and calls[-1][0] == 0
    with pytest.raises(ValueError, match="positive"):
        sample_meanflow(field, eps, None, 0)
