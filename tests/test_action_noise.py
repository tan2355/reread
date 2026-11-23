from dataclasses import replace

import pytest
import torch

from reread.config import ReReadConfig
from reread.noise import perturb_actions


def test_exact_action_corruption_preserves_grippers_labels_and_source():
    config = replace(ReReadConfig(), action_perturb_probability=1.0)
    actions = torch.randn(2, 32, 14, requires_grad=True)
    original = actions.detach().clone()
    valid = torch.ones(2, 32, dtype=torch.bool)
    valid[0, 20:] = False
    supplied_mask = torch.ones_like(valid)
    candidate, detail = perturb_actions(
        actions, valid, config, noise=torch.ones_like(actions), mask=supplied_mask
    )
    assert torch.equal(actions, original)
    assert not candidate.requires_grad
    assert candidate.dtype == torch.float32
    assert torch.equal(detail["mask"], valid)
    assert supplied_mask.all()
    assert torch.equal(candidate[..., [6, 13]][valid], original[..., [6, 13]][valid])
    assert not detail["offset"][..., [6, 13]].any()
    assert not detail["offset"][~valid].any()
    assert torch.equal(candidate[~valid], original[~valid])
    joint = [i for i in range(14) if i not in (6, 13)]
    torch.testing.assert_close(
        detail["offset"][..., joint][valid], torch.full((52, 12), 0.05)
    )


def test_standard_deviation_is_absolute_normalized_units_not_relative_rms():
    config = replace(ReReadConfig(), action_perturb_probability=1.0)
    valid = torch.ones(1, 32, dtype=torch.bool)
    small = torch.ones(1, 32, 14)
    large = small * 1000
    _, left = perturb_actions(small, valid, config, noise=small)
    _, right = perturb_actions(large, valid, config, noise=small)
    torch.testing.assert_close(left["offset"], right["offset"], atol=0, rtol=0)


def test_local_generator_replays_without_advancing_global_rng():
    config = ReReadConfig()
    actions = torch.zeros(3, 32, 14)
    valid = torch.ones(3, 32, dtype=torch.bool)
    rng_state = torch.random.get_rng_state().clone()
    a, d1 = perturb_actions(
        actions, valid, config, generator=torch.Generator().manual_seed(10)
    )
    b, d2 = perturb_actions(
        actions, valid, config, generator=torch.Generator().manual_seed(10)
    )
    assert torch.equal(a, b)
    assert torch.equal(d1["mask"], d2["mask"])
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert d1["mask"].any() and not d1["mask"].all()


@pytest.mark.parametrize(
    "changes", [dict(action_perturb_probability=0.0), dict(action_perturb_std=0.0)]
)
def test_disabled_is_exact_and_does_not_draw(changes):
    config = replace(ReReadConfig(), **changes)
    action = torch.randn(1, 32, 14).bfloat16()
    valid = torch.ones(1, 32, dtype=torch.bool)
    state = torch.random.get_rng_state().clone()
    candidate, detail = perturb_actions(action, valid, config)
    assert torch.equal(candidate, action.float())
    assert not detail["offset"].any() and not detail["mask"].any()
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize("row", [0, 20])
def test_full_native_chunk_must_be_finite_even_where_labels_are_missing(row):
    action = torch.zeros(1, 32, 14)
    valid = torch.ones(1, 32, dtype=torch.bool)
    valid[:, 10:] = False
    action[:, row, 1] = float("nan")
    with pytest.raises(FloatingPointError):
        perturb_actions(action, valid, ReReadConfig())


def test_label_validity_never_blanks_finite_base_input_positions():
    action = torch.randn(2, 32, 14)
    action[:, 24, 2] = -0.0
    valid = torch.zeros(2, 32, dtype=torch.bool)
    valid[0, :3] = True
    config = replace(ReReadConfig(), action_perturb_probability=1.0)
    candidate, details = perturb_actions(
        action, valid, config, noise=torch.ones_like(action)
    )
    assert torch.equal(candidate[~valid], action[~valid])
    assert torch.equal(
        candidate[~valid].view(torch.int32), action[~valid].view(torch.int32)
    )
    assert not details["offset"][~valid].any()
    assert torch.equal(candidate, action + details["offset"])
    all_invalid, details = perturb_actions(action, torch.zeros_like(valid), config)
    assert torch.equal(all_invalid, action)
    assert not details["offset"].any()


@pytest.mark.parametrize("mode", ["valid", "mask", "noise"])
def test_shape_contract(mode):
    action = torch.zeros(1, 32, 14)
    valid = torch.ones(1, 32, dtype=torch.bool)
    kwargs = {}
    if mode == "valid":
        valid = valid.float()
    else:
        kwargs[mode] = torch.ones(1)
    with pytest.raises(ValueError):
        perturb_actions(action, valid, ReReadConfig(), **kwargs)
