from dataclasses import replace

import pytest
import torch

from reread.config import ReReadConfig
from reread.model import ReRead


def small_config(**changes):
    return replace(
        ReReadConfig(),
        proprio_dim=3,
        horizon=4,
        action_tokens=4,
        video_tokens=6,
        kv_dim=12,
        layers=(1, 2),
        width=24,
        heads=4,
        depth=2,
        ff_dim=48,
        inference_steps=3,
        decision_stride=2,
        history_dropout=0,
        activation_checkpointing=False,
        video_grid=(3, 4),
        head_rows=1,
        wrist_split=2,
        camera_pools=((1, 2), (1, 2), (1, 2)),
        **changes,
    )


def fixture(batch=2, **changes):
    torch.set_num_threads(2)
    torch.manual_seed(33)
    model = ReRead(small_config(**changes))
    shape = (batch, 3, 2, 10, 12)
    memory = dict(
        k=torch.randn(shape),
        v=torch.randn(shape),
        states=torch.randn(batch, 3, 3),
        valid=torch.ones(batch, 3, dtype=torch.bool),
        timestamps=torch.tensor([[0, 2, 4]]).repeat(batch, 1),
        executed_steps=torch.tensor([[2, 2, 0]]).repeat(batch, 1),
    )
    return model, memory, torch.randn(batch, 4, 14)


def activate(model):
    with torch.no_grad():
        model.output.weight.normal_(std=0.03)


def test_zero_head_exact_candidate_parity_and_aux_visibility():
    model, memory, action = fixture()
    before = {name: value.clone() for name, value in memory.items()}
    result = model(action, memory, return_aux=True)
    assert torch.equal(result["actions"], action)
    assert torch.equal(result["history_visible"], memory["valid"])
    assert all(torch.equal(before[name], memory[name]) for name in before)
    with pytest.raises(TypeError):
        model(action, memory, predictor_memory={})


def test_kv_is_not_perturbed_and_forward_never_injects_action_noise():
    model, memory, action = fixture()
    activate(model)
    reference = model(action, memory)
    for _ in range(4):
        torch.testing.assert_close(model(action, memory), reference, atol=0, rtol=0)
    model.eval()
    torch.testing.assert_close(model(action, memory), reference, atol=0, rtol=0)


def test_frozen_inputs_and_zero_head_then_internal_gradients():
    model, memory, action = fixture()
    action.requires_grad_()
    for name in ("k", "v", "states"):
        memory[name].requires_grad_()
    target = action.detach() + 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    model.loss(action, memory, target).backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert model.key_proj.weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad()
    model.loss(action, memory, target).backward()
    assert model.key_proj.weight.grad.abs().sum() > 0
    assert model.reader[0].cross_attn.in_proj_weight.grad.abs().sum() > 0
    assert action.grad is None
    assert all(memory[name].grad is None for name in ("k", "v", "states"))


def test_checkpointing_matches_forward_and_all_gradients():
    model, memory, action = fixture()
    activate(model)
    other = ReRead(replace(model.config, activation_checkpointing=True))
    other.load_state_dict(model.state_dict())
    target = action + 0.1
    model.loss(action, memory, target).backward()
    other.loss(action, memory, target).backward()
    for lhs, rhs in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(lhs.grad, rhs.grad)


def test_cold_start_nested_dropout_and_padding_are_preserved():
    model, memory, action = fixture()
    model.config = replace(model.config, history_dropout=1.0)
    result = model(action, memory, return_aux=True)
    assert not result["history_visible"][:, :2].any()
    assert result["history_visible"][:, -1].all()
    for name in ("k", "v", "states"):
        memory[name][:, :2] = float("nan")
    assert torch.isfinite(model(action, memory)).all()
    assert memory["valid"].all()


def test_full_action_projection_retains_every_horizon_position():
    model, memory, action = fixture()
    activate(model)
    original = model(action, memory)
    changed = action.clone()
    changed[:, -1, 0] += 0.2
    assert not torch.equal(model(changed, memory), original)
    assert model.action_proj.in_features == 14
    assert model.action_position.num_embeddings == 4
