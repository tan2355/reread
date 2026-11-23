from dataclasses import dataclass, replace
from numbers import Integral
from typing import Mapping, Sequence

import torch

from reread.config import ReReadConfig


def _integer(value, name):
    if isinstance(value, torch.Tensor):
        if (
            value.ndim != 0
            or value.dtype == torch.bool
            or value.dtype.is_floating_point
        ):
            raise ValueError(f"{name} must be an integer")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _features(features, config):
    if not isinstance(features, Mapping) or not all(
        name in features for name in ("k", "v")
    ):
        raise ValueError("features must contain raw k and v")
    k, v = features["k"], features["v"]
    shape = (
        len(config.layers),
        config.video_tokens + config.action_tokens,
        config.kv_dim,
    )
    if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise ValueError("k and v must be tensors")
    if k.ndim != 4 or k.shape[1:] != shape or v.shape != k.shape or k.shape[0] < 1:
        raise ValueError(f"raw k/v must be [B,{shape}]")
    if (
        not k.dtype.is_floating_point
        or v.dtype != k.dtype
        or v.device != k.device
        or not torch.isfinite(k).all()
        or not torch.isfinite(v).all()
    ):
        raise ValueError(
            "raw k/v must be finite floating tensors with matching dtype/device"
        )
    return k, v


@dataclass(frozen=True)
class MemoryRecord:

    k: torch.Tensor
    v: torch.Tensor
    state: torch.Tensor
    timestamp: int
    decision_index: int
    executed_steps: int = 0

    @classmethod
    def from_features(
        cls, features, state, timestamp, decision_index, config, *, batch_index=0
    ):
        k, v = _features(features, config)
        batch_index = _integer(batch_index, "batch_index")
        if batch_index >= k.shape[0]:
            raise ValueError("batch_index is outside feature batch")
        if not isinstance(state, torch.Tensor) or state.shape != (config.proprio_dim,):
            raise ValueError("record state must be [proprio_dim]")
        if not state.dtype.is_floating_point or not torch.isfinite(state).all():
            raise ValueError("record state must be finite normalized floating proprio")

        return cls(
            k[batch_index].detach().clone(),
            v[batch_index].detach().clone(),
            state.detach().to(device=k.device).clone(),
            _integer(timestamp, "timestamp"),
            _integer(decision_index, "decision_index"),
        )


def _record(record, config):
    if not isinstance(record, MemoryRecord):
        raise ValueError("history entries must be MemoryRecord instances")
    expected = (
        len(config.layers),
        config.video_tokens + config.action_tokens,
        config.kv_dim,
    )
    if record.k.shape != expected or record.v.shape != expected:
        raise ValueError("history record feature contract differs")
    if (
        not record.k.dtype.is_floating_point
        or record.v.dtype != record.k.dtype
        or record.v.device != record.k.device
        or not torch.isfinite(record.k).all()
        or not torch.isfinite(record.v).all()
    ):
        raise ValueError("history k/v are invalid")
    if (
        record.state.shape != (config.proprio_dim,)
        or not record.state.dtype.is_floating_point
        or not torch.isfinite(record.state).all()
    ):
        raise ValueError("history observed state is invalid")
    for name in ("timestamp", "decision_index", "executed_steps"):
        _integer(getattr(record, name), name)
    if record.executed_steps > config.horizon:
        raise ValueError("executed_steps cannot exceed the original plan horizon")
    if record.k.requires_grad or record.v.requires_grad or record.state.requires_grad:
        raise ValueError("history must contain detached frozen raw features")


def _chronology(history, timestamp, decision_index, config):
    if len(history) > config.memory_frames - 1:
        raise ValueError("history exceeds the bounded two-decision bank")
    for entry in history:
        _record(entry, config)
    if not history:
        if decision_index != 0:
            raise ValueError("An episode must start at decision_index=0")
        return []
    previous = None
    for entry in history:
        if previous is not None:
            if (
                entry.decision_index != previous.decision_index + 1
                or entry.timestamp <= previous.timestamp
            ):
                raise ValueError(
                    "Historical decisions must be consecutive and chronological"
                )
        previous = entry
    if (
        decision_index != history[-1].decision_index + 1
        or timestamp <= history[-1].timestamp
    ):
        raise ValueError(
            "Current decision must be consecutive with a strictly newer observation"
        )
    times = [entry.timestamp for entry in history] + [timestamp]
    durations = [right - left for left, right in zip(times[:-1], times[1:])]
    for entry, duration in zip(history, durations):
        if duration > config.horizon:
            raise ValueError(
                "Observed executed interval exceeds the original plan horizon"
            )
        if entry.executed_steps not in (0, duration):
            raise ValueError(
                "Stored execution interval differs from observed timestamps"
            )
    return durations


def pack_records(
    histories: Sequence[Sequence[MemoryRecord]],
    current,
    states,
    timestamps,
    config=ReReadConfig(),
    *,
    decision_indices=None,
    valid=None,
):
    k, v = _features(current, config)
    batch = k.shape[0]
    if len(histories) != batch:
        raise ValueError("histories must contain one independent list per batch row")
    if not isinstance(states, torch.Tensor) or states.shape != (
        batch,
        config.proprio_dim,
    ):
        raise ValueError("states must be [B,proprio_dim]")
    if not states.dtype.is_floating_point or not torch.isfinite(states).all():
        raise ValueError("states must be finite normalized floating proprio")
    if len(timestamps) != batch:
        raise ValueError("timestamps must contain one observed step per row")
    times = [_integer(value, "timestamp") for value in timestamps]
    if decision_indices is None:
        indices = [row[-1].decision_index + 1 if row else 0 for row in histories]
    else:
        if len(decision_indices) != batch:
            raise ValueError("decision_indices must contain one index per row")
        indices = [_integer(value, "decision_index") for value in decision_indices]
    if valid is not None and (
        not isinstance(valid, torch.Tensor)
        or valid.shape != (batch, config.memory_frames - 1)
        or valid.dtype != torch.bool
    ):
        raise ValueError("valid must be bool [B,2] for historical slots only")
    durations = [
        _chronology(row, time, index, config)
        for row, time, index in zip(histories, times, indices)
    ]
    packed_k = k.new_zeros((batch, config.memory_frames, *k.shape[1:]))
    packed_v = v.new_zeros(packed_k.shape)
    mask = torch.zeros((batch, config.memory_frames), dtype=torch.bool, device=k.device)
    packed_times = torch.zeros(mask.shape, dtype=torch.long, device=k.device)
    executed = torch.zeros_like(packed_times)
    packed_states = states.new_zeros(
        (batch, config.memory_frames, config.proprio_dim), device=k.device
    )
    for row_index, history in enumerate(histories):
        offset = config.memory_frames - 1 - len(history)
        for column, (entry, duration) in enumerate(
            zip(history, durations[row_index]), offset
        ):
            is_valid = valid is None or bool(valid[row_index, column])
            if is_valid:
                packed_k[row_index, column] = entry.k.to(k).detach()
                packed_v[row_index, column] = entry.v.to(v).detach()
                packed_states[row_index, column] = entry.state.to(
                    packed_states
                ).detach()
                packed_times[row_index, column] = entry.timestamp
                executed[row_index, column] = duration
                mask[row_index, column] = True
        packed_k[row_index, -1] = k[row_index].detach()
        packed_v[row_index, -1] = v[row_index].detach()
        packed_states[row_index, -1] = states[row_index].to(packed_states).detach()
        packed_times[row_index, -1] = times[row_index]
        mask[row_index, -1] = True
    return {
        "k": packed_k,
        "v": packed_v,
        "valid": mask,
        "timestamps": packed_times,
        "states": packed_states,
        "executed_steps": executed,
    }


class MemoryBank:

    def __init__(self, config=ReReadConfig()):
        self.config = config
        self.reset()

    def reset(self):
        self._records = ()

    @property
    def records(self):
        return self._records

    def __len__(self):
        return len(self._records)

    def snapshot(self, current, state, observation_step, decision_index):
        k, _ = _features(current, self.config)
        if k.shape[0] != 1:
            raise ValueError(
                "MemoryBank is one rollout lane; use pack_records for batches"
            )
        if state.shape == (self.config.proprio_dim,):
            state = state.unsqueeze(0)
        return pack_records(
            [self._records],
            current,
            state,
            [observation_step],
            self.config,
            decision_indices=[decision_index],
        )

    def commit(self, current, state, observation_step, decision_index):
        k, _ = _features(current, self.config)
        if k.shape[0] != 1:
            raise ValueError(
                "MemoryBank is one rollout lane; use pack_records for batches"
            )
        if state.shape == (1, self.config.proprio_dim):
            state = state[0]
        record = MemoryRecord.from_features(
            current, state, observation_step, decision_index, self.config
        )
        durations = _chronology(
            self._records, record.timestamp, record.decision_index, self.config
        )
        updated = [
            replace(entry, executed_steps=duration)
            for entry, duration in zip(self._records, durations)
        ]
        self._records = tuple((updated + [record])[-(self.config.memory_frames - 1) :])
