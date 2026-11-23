import pytest
import torch
from torch import nn

from reread.config import ReReadConfig
from reread.policy import TemporalMemoryPolicy


def config():
    return ReReadConfig(
        kv_dim=3,
        layers=(1, 2),
        action_dim=2,
        proprio_dim=2,
        horizon=4,
        action_tokens=4,
        decision_stride=2,
        width=12,
        heads=3,
        ff_dim=16,
    )


class Frozen:
    def __init__(self):
        self.config = config()
        self.calls = []

    def extract(self, **observation):
        self.calls.append(observation)
        c = self.config
        value = float(observation["proprio"][0, 0])
        feature = torch.full(
            (1, len(c.layers), c.video_tokens + c.action_tokens, c.kv_dim), value
        )
        return torch.full((1, c.horizon, c.action_dim), value), {
            "k": feature,
            "v": feature + 10,
        }


class Corrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config()
        self.memory = []
        self.fail = False
        self.nan = False

    def forward(self, action, memory):
        self.memory.append(memory)
        if self.fail:
            raise RuntimeError("simulated correction failure")
        return action + (float("nan") if self.nan else 1)


def act(policy, index, step, value=0, episode="a"):
    return policy.act(
        episode_id=episode,
        decision_index=index,
        observation_step=step,
        proprio=torch.full((1, 2), float(value)),
        input_image=torch.zeros(1, 3, 4, 4),
    )


def test_one_frozen_call_per_decision_three_times_and_episode_reset():
    frozen, corrector = Frozen(), Corrector()
    policy = TemporalMemoryPolicy(frozen, corrector)
    for index, step in enumerate([0, 2, 3, 5, 7]):
        result = act(policy, index, step, value=index)
        assert result["action"].shape == (4, 2)
        assert result["action"].unique().tolist() == [index + 1]
        assert result["base_action"].unique().tolist() == [index]
        assert len(frozen.calls) == index + 1
        assert set(frozen.calls[-1]) == {"proprio", "input_image"}
    assert corrector.memory[-1]["timestamps"].tolist() == [[3, 5, 7]]
    assert corrector.memory[-1]["states"][0, :, 0].tolist() == [2, 3, 4]
    act(policy, 0, 0, episode="b")
    assert corrector.memory[-1]["valid"].tolist() == [[False, False, True]]
    assert len(policy.bank) == 1


def test_correction_failure_transactional_and_retry_allowed():
    frozen, corrector = Frozen(), Corrector()
    policy = TemporalMemoryPolicy(frozen, corrector)
    act(policy, 0, 0)
    records = policy.bank.records
    corrector.fail = True
    with pytest.raises(RuntimeError, match="failure"):
        act(policy, 1, 2, value=1)
    assert policy.bank.records is records
    assert policy.decision_index == 0 and policy.observation_step == 0
    assert records[-1].executed_steps == 0
    corrector.fail = False
    act(policy, 1, 2, value=1)
    assert policy.bank.records[0].executed_steps == 2


def test_nan_result_cannot_advance_memory():
    frozen, corrector = Frozen(), Corrector()
    policy = TemporalMemoryPolicy(frozen, corrector)
    corrector.nan = True
    with pytest.raises(ValueError, match="not committed"):
        act(policy, 0, 0)
    assert policy.decision_index is None and len(policy.bank) == 0


@pytest.mark.parametrize(
    "index,step", [(0, 2), (2, 2), (1, 0), (1, 5), (True, 1), (1, 1.5)]
)
def test_invalid_decisions_do_not_run_backbone(index, step):
    frozen, corrector = Frozen(), Corrector()
    policy = TemporalMemoryPolicy(frozen, corrector)
    act(policy, 0, 0)
    with pytest.raises(ValueError):
        act(policy, index, step)
    assert len(frozen.calls) == 1 and len(policy.bank) == 1


def test_missing_proprio_and_new_episode_nonzero_start_rejected():
    policy = TemporalMemoryPolicy(Frozen(), Corrector())
    with pytest.raises(ValueError, match="proprio"):
        policy.act(episode_id="a", decision_index=0, observation_step=0)
    act(policy, 0, 0)
    with pytest.raises(ValueError, match="start"):
        act(policy, 1, 2, episode="b")
    assert len(policy.bank) == 0


def test_unbatched_proprio_is_normalized_once_and_owned():
    policy = TemporalMemoryPolicy(Frozen(), Corrector())
    source = torch.ones(2)
    policy.act(episode_id="a", decision_index=0, observation_step=0, proprio=source)
    source.fill_(9)
    assert policy.bank.records[0].state.tolist() == [1, 1]


def test_contract_mismatch_is_rejected():
    from dataclasses import replace

    corrector = Corrector()
    corrector.config = replace(corrector.config, width=24)
    with pytest.raises(ValueError, match="contracts"):
        TemporalMemoryPolicy(Frozen(), corrector)
