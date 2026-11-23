from dataclasses import replace

import pytest
import torch

from reread.config import ReReadConfig
from reread.memory.bank import MemoryBank, MemoryRecord, pack_records


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


def features(value, batch=1, requires_grad=False):
    c = config()
    k = torch.full(
        (batch, len(c.layers), c.video_tokens + c.action_tokens, c.kv_dim),
        float(value),
        requires_grad=requires_grad,
    )
    return {"k": k, "v": k + 100}


def state(value):
    return torch.full((1, 2), float(value))


def test_first_observation_is_current_only_and_owned():
    bank = MemoryBank(config())
    raw, observed = features(3, requires_grad=True), state(4).requires_grad_()
    snapshot = bank.snapshot(raw, observed, 0, 0)
    assert snapshot["valid"].tolist() == [[False, False, True]]
    assert snapshot["executed_steps"].tolist() == [[0, 0, 0]]
    assert not snapshot["k"].requires_grad
    assert not snapshot["states"].requires_grad
    assert len(bank) == 0
    bank.commit(raw, observed, 0, 0)
    record = bank.records[0]
    with torch.no_grad():
        raw["k"].fill_(99)
        observed.fill_(99)
        snapshot["k"].fill_(99)
    assert record.k.unique().tolist() == [3]
    assert record.state.tolist() == [4, 4]
    assert (
        not record.k.requires_grad
        and not record.v.requires_grad
        and not record.state.requires_grad
    )


def test_three_times_right_aligned_and_bounded_actual_prefixes():
    bank = MemoryBank(config())
    times = [0, 2, 3, 7, 9]
    for index, timestamp in enumerate(times):
        raw = features(index)
        snapshot = bank.snapshot(raw, state(index + 10), timestamp, index)
        if index == 3:
            assert snapshot["valid"].all()
            assert snapshot["timestamps"].tolist() == [[2, 3, 7]]
            assert snapshot["executed_steps"].tolist() == [[1, 4, 0]]
            assert snapshot["states"][0, :, 0].tolist() == [11, 12, 13]
        bank.commit(raw, state(index + 10), timestamp, index)
        assert len(bank) <= 2
    assert [record.decision_index for record in bank.records] == [3, 4]
    assert [record.executed_steps for record in bank.records] == [2, 0]
    bank.reset()
    assert len(bank) == 0


def test_failed_snapshot_or_commit_does_not_modify_previous_records():
    bank = MemoryBank(config())
    bank.commit(features(1), state(1), 3, 0)
    original = bank.records
    for timestamp, decision in [(3, 1), (4, 3), (8, 1), (2, 1)]:
        with pytest.raises(ValueError):
            bank.snapshot(features(2), state(2), timestamp, decision)
        with pytest.raises(ValueError):
            bank.commit(features(2), state(2), timestamp, decision)
        assert bank.records is original
    assert original[0].executed_steps == 0


def test_batched_rows_have_independent_histories_and_history_dropout():
    c = config()
    first = MemoryRecord.from_features(features(1), state(1)[0], 0, 0, c)
    second = MemoryRecord.from_features(features(2), state(2)[0], 2, 1, c)
    raw = features(8, batch=2)
    raw["k"][1].fill_(9)
    packed = pack_records(
        [[first, second], []],
        raw,
        torch.tensor([[8.0, 8.0], [9.0, 9.0]]),
        [3, 0],
        c,
        decision_indices=[2, 0],
    )
    assert packed["valid"].tolist() == [[True, True, True], [False, False, True]]
    assert packed["k"][0, :, 0, 0, 0].tolist() == [1, 2, 8]
    assert packed["k"][1, :, 0, 0, 0].tolist() == [0, 0, 9]
    masked = pack_records(
        [[first, second], []],
        raw,
        torch.zeros(2, 2),
        [3, 0],
        c,
        valid=torch.zeros(2, 2, dtype=torch.bool),
    )
    assert masked["valid"].tolist() == [[False, False, True]] * 2
    assert masked["k"][:, :-1].count_nonzero() == 0


def test_record_slice_clone_does_not_retain_batch_storage():
    c = config()
    raw = features(1, batch=8, requires_grad=True)
    record = MemoryRecord.from_features(raw, state(1)[0], 0, 0, c, batch_index=4)
    assert (
        record.k.untyped_storage().nbytes()
        == record.k.numel() * record.k.element_size()
    )
    assert (
        record.v.untyped_storage().nbytes()
        == record.v.numel() * record.v.element_size()
    )
    assert (
        record.k.untyped_storage().data_ptr() != raw["k"].untyped_storage().data_ptr()
    )


def test_plan_tokens_never_shift_to_mimic_executed_actions():
    c = config()
    raw = features(0)
    raw["k"][:, :, c.video_tokens :, 0] = torch.arange(c.horizon)
    bank = MemoryBank(c)
    bank.commit(raw, state(0), 0, 0)
    packed = bank.snapshot(features(1), state(1), 2, 1)
    assert packed["executed_steps"][0, 1] == 2
    assert packed["k"][0, 1, 0, c.video_tokens :, 0].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "value", [True, -1, 0.5, torch.tensor(1.0), torch.tensor(True)]
)
def test_integer_metadata_is_not_coerced(value):
    with pytest.raises(ValueError):
        MemoryBank(config()).snapshot(features(0), state(0), value, 0)


def test_invalid_records_and_unknown_execution_are_rejected():
    c = config()
    record = MemoryRecord.from_features(features(0), state(0)[0], 0, 0, c)
    with pytest.raises(ValueError, match="differs"):
        pack_records(
            [[replace(record, executed_steps=3)]], features(1), state(1), [2], c
        )
    with pytest.raises(ValueError, match="detached"):
        pack_records(
            [[replace(record, k=record.k.requires_grad_())]],
            features(1),
            state(1),
            [2],
            c,
        )
    with pytest.raises(ValueError):
        pack_records(
            [[]],
            features(1),
            state(1),
            [0],
            c,
            valid=torch.zeros(1, 4, dtype=torch.bool),
        )


def test_nan_and_cross_contract_features_fail_closed():
    raw = features(0)
    raw["k"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        MemoryBank(config()).snapshot(raw, state(0), 0, 0)
    with pytest.raises(ValueError):
        MemoryBank(config()).snapshot(
            {"k": torch.zeros(1, 2), "v": torch.zeros(1, 2)}, state(0), 0, 0
        )
