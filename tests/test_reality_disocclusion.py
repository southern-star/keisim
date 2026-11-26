import torch

from keireality.disocclusion import ActorContext, BackgroundRepair, actor_support


def test_only_immediate_same_episode_view_history_survives():
    context = ActorContext()
    context.observe("a", 0.)
    context.current[7] = "left"
    context.current[11] = "right"
    context.observe("a", .1)
    assert context.previous == {7: "left", 11: "right"}
    assert context.current == {}
    context.current[11] = "right again"
    context.observe("a", .2)
    assert context.previous == {11: "right again"}
    context.current[7] = "before dropped frame"
    context.observe("a", .4)
    assert context.previous == {}
    context.current[7] = "before reset"
    context.observe("b", .5)
    assert context.previous == {}


@torch.no_grad()
def test_correction_is_local_bounded_and_preserves_confident_foreground():
    model = BackgroundRepair().eval()
    model.layers[-1].bias.fill_(5.)
    release = torch.full((1, 3, 32, 48), .5)
    inputs = torch.zeros(1, 17, 32, 48)
    context = torch.zeros(1, 5, 32, 48)
    assert torch.equal(model(release, inputs, context, .25), release)
    context[0, 4, 15, 25] = 1.
    context[0, 3, 15, 22] = 1.
    before = release.clone()
    repaired = model(release, inputs, context, .25)
    support = actor_support(context)
    assert support[0, 0, 15, 25] == 1.
    assert support[0, 0, 15, 22] == 0.
    assert support[0, 0, 0, 0] == 0.
    assert torch.equal(repaired * (support == 0), release * (support == 0))
    assert 0. < (repaired - release).max().item() <= .075001
    assert torch.equal(release, before)
    assert torch.equal(model(release, inputs, context, 0.), release)
