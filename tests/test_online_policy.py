from dataclasses import replace

import pytest
import torch

from reread.config import ReReadConfig
from reread.model import ReRead
from reread.policy import TemporalMemoryPolicy


def tiny_config():
    return replace(
        ReReadConfig(),
        kv_dim=12,
        layers=(1,),
        video_tokens=6,
        width=24,
        heads=4,
        depth=2,
        ff_dim=48,
        video_grid=(3, 4),
        head_rows=1,
        wrist_split=2,
        camera_pools=((1, 2), (1, 2), (1, 2)),
    )


class Frozen:
    def __init__(self, config):
        self.config, self.calls = config, []

    def extract(self, **observation):
        assert set(observation) == {"proprio", "input_image"}
        c = self.config
        marker = float(observation["proprio"][0, 0])
        self.calls.append(marker)
        feature = torch.full(
            (1, len(c.layers), c.video_tokens + c.action_tokens, c.kv_dim), marker
        )
        return torch.full((1, c.horizon, c.action_dim), marker), dict(
            k=feature, v=feature * 0.3
        )


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    config = tiny_config()
    frozen, model = Frozen(config), ReRead(config)
    with torch.no_grad():
        model.output.bias.fill_(0.25)
    policy = TemporalMemoryPolicy(
        frozen,
        model,
        bf16_corrector=False,
        runtime=dict(
            vae_encoding="native",
            patch_projection="native",
            timestep_cache=False,
            fused_visible_attention=False,
        ),
    )
    return policy, model, frozen


def act(policy, decision, *, step=None, episode="a"):
    return policy.act(
        episode_id=episode,
        decision_index=decision,
        observation_step=decision * 20 if step is None else step,
        proprio=torch.full((1, 14), float(decision + 1)),
        input_image=torch.zeros(1, 3, 4, 4),
    )


def test_real_online_policy_corrects_all_first_three_with_progressive_real_history():
    policy, model, frozen = fixture()
    memories = []
    handle = model.register_forward_pre_hook(lambda _m, args: memories.append(args[1]))
    for decision in range(3):
        result = act(policy, decision)
        torch.testing.assert_close(
            result["action"] - result["base_action"],
            torch.full_like(result["action"], 0.25),
        )
        assert len(memories) == decision + 1
    handle.remove()
    assert frozen.calls == [1.0, 2.0, 3.0]
    assert memories[0]["valid"].tolist() == [[False, False, True]]
    assert memories[1]["valid"].tolist() == [[False, True, True]]
    assert memories[2]["timestamps"].tolist() == [[0, 20, 40]]
    assert memories[2]["valid"].all()
    assert [memories[2]["k"][0, index].unique().item() for index in range(3)] == [
        1.0,
        2.0,
        3.0,
    ]
    assert len(policy.bank) == 2


def test_episode_change_and_explicit_reset_correct_first_decision_with_empty_history():
    policy, model, _ = fixture()
    calls = []
    handle = model.register_forward_pre_hook(lambda *_: calls.append(True))
    for episode in ("a", "b"):
        for decision in range(3):
            out = act(policy, decision, episode=episode)
            assert not torch.equal(out["action"], out["base_action"])
        assert len(calls) == (3 if episode == "a" else 6)
    policy.reset()
    assert len(policy.bank) == 0
    for decision in range(3):
        out = act(policy, decision, episode="b")
        assert not torch.equal(out["action"], out["base_action"])
    handle.remove()
    assert len(calls) == 9


@pytest.mark.parametrize("bad_step", [0, 33, 40])
def test_invalid_executed_prefix_rejected_before_frozen_or_corrector(bad_step):
    policy, model, frozen = fixture()
    calls = []
    handle = model.register_forward_pre_hook(lambda *_: calls.append(True))
    act(policy, 0)
    with pytest.raises(ValueError, match="prefix|decision_stride"):
        act(policy, 1, step=bad_step)
    handle.remove()
    assert frozen.calls == [1.0] and calls == [True]
    assert policy.decision_index == 0 and policy.observation_step == 0
    assert len(policy.bank) == 1


def test_nonfinite_third_correction_does_not_advance_online_history_and_retry_works(
    monkeypatch,
):
    policy, model, frozen = fixture()
    for decision in range(2):
        act(policy, decision)
    forward = model.forward
    monkeypatch.setattr(
        model, "forward", lambda action, memory: torch.full_like(action, float("nan"))
    )
    with pytest.raises(ValueError, match="not committed"):
        act(policy, 2)
    assert policy.decision_index == 1 and policy.observation_step == 20
    assert [record.timestamp for record in policy.bank.records] == [0, 20]
    monkeypatch.setattr(model, "forward", forward)
    out = act(policy, 2)
    assert torch.isfinite(out["action"]).all()
    assert not torch.equal(out["action"], out["base_action"])
    assert frozen.calls == [1.0, 2.0, 3.0, 3.0]
