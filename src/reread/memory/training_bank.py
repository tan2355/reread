from dataclasses import dataclass, replace
from numbers import Integral
from typing import Mapping

import torch

from reread.config import ReReadConfig
from reread.memory.bank import MemoryRecord


def _integer(value, name):
    if isinstance(value, torch.Tensor):
        if (
            value.device.type != "cpu"
            or value.ndim
            or value.dtype == torch.bool
            or value.dtype.is_floating_point
        ):
            raise ValueError(f"{name} must be a CPU nonnegative integer")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _signature(tensor):
    try:
        version = tensor._version
    except RuntimeError as error:
        raise ValueError(
            "Private training history requires versioned tensors"
        ) from error
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.device,
        tensor.dtype,
        tensor.requires_grad,
        version,
    )


@dataclass(frozen=True)
class _OwnedRecord:
    record: MemoryRecord
    signatures: tuple

    @classmethod
    def create(cls, k, v, state, timestamp, decision_index):
        record = MemoryRecord(
            k.detach().clone(),
            v.detach().clone(),
            state.detach().clone(),
            timestamp,
            decision_index,
        )
        return cls(
            record,
            tuple(_signature(tensor) for tensor in (record.k, record.v, record.state)),
        )

    def check_ownership(self):
        current = tuple(
            _signature(tensor)
            for tensor in (self.record.k, self.record.v, self.record.state)
        )
        if current != self.signatures:
            raise ValueError("Private immutable history was mutated")

    def with_duration(self, duration):
        return replace(self, record=replace(self.record, executed_steps=duration))


@dataclass(frozen=True)
class _Lane:
    episode_id: str
    records: tuple
    trajectory_pass_id: int | None = None


class TrainingMemoryBanks:

    def __init__(self, config=ReReadConfig()):
        self.config = config
        self._lanes = {}

    def reset(self):
        self._lanes.clear()

    def records_for_lane(self, lane):
        stored = self._lanes.get(_integer(lane, "lane"))
        if stored is None:
            return ()
        copies = []
        for owned in stored.records:
            owned.check_ownership()
            record = owned.record
            copies.append(
                replace(
                    record,
                    k=record.k.detach().clone(),
                    v=record.v.detach().clone(),
                    state=record.state.detach().clone(),
                )
            )
        return tuple(copies)

    def _prepare(self, current, samples):
        c = self.config
        if not isinstance(current, Mapping) or not all(
            name in current for name in ("k", "v")
        ):
            raise ValueError("Current features must contain raw k and v")
        k, v = current["k"], current["v"]
        expected = (len(c.layers), c.video_tokens + c.action_tokens, c.kv_dim)
        if (
            not isinstance(k, torch.Tensor)
            or not isinstance(v, torch.Tensor)
            or k.ndim != 4
            or k.shape[1:] != expected
            or v.shape != k.shape
            or k.shape[0] < 1
            or not k.dtype.is_floating_point
            or v.dtype != k.dtype
            or v.device != k.device
        ):
            raise ValueError("Current raw K/V shape, dtype or device differs")
        if not isinstance(samples, (list, tuple)) or len(samples) != k.shape[0]:
            raise ValueError("Samples must contain one row per current feature batch")
        rows, states, histories, intervals, seen = [], [], [], [], set()
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise ValueError("Each trajectory sample must be a mapping")
            if not all(
                name in sample
                for name in ("lane", "episode_id", "frame", "decision_index", "proprio")
            ):
                raise ValueError("Trajectory sample metadata is incomplete")
            lane = _integer(sample["lane"], "lane")
            if lane in seen:
                raise ValueError("A batch cannot contain duplicate trajectory lanes")
            seen.add(lane)
            episode_id = sample["episode_id"]
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError("episode_id must be a nonempty string")
            trajectory_pass_id = sample.get("trajectory_pass_id")
            if trajectory_pass_id is not None:
                trajectory_pass_id = _integer(trajectory_pass_id, "trajectory_pass_id")
            timestamp = _integer(sample["frame"], "frame")
            decision_index = _integer(sample["decision_index"], "decision_index")
            state = sample["proprio"]
            if (
                not isinstance(state, torch.Tensor)
                or state.device.type != "cpu"
                or state.shape != (c.proprio_dim,)
                or not state.is_floating_point()
            ):
                raise ValueError("Current proprio must be CPU floating [proprio_dim]")
            previous = self._lanes.get(lane)
            same_pass = (
                previous is not None
                and previous.episode_id == episode_id
                and previous.trajectory_pass_id == trajectory_pass_id
            )
            history = previous.records if same_pass else ()
            durations = self._chronology(history, timestamp, decision_index)
            rows.append(
                (lane, episode_id, timestamp, decision_index, trajectory_pass_id)
            )
            states.append(state.detach())
            histories.append(history)
            intervals.append(durations)
        states = torch.stack(states)
        if not bool(torch.isfinite(states).all()):
            raise ValueError("Current proprio must be finite")

        if not bool(torch.isfinite(k).all() & torch.isfinite(v).all()):
            raise ValueError("Current raw K/V must be finite")
        return k.detach(), v.detach(), rows, states, histories, intervals

    def _chronology(self, history, timestamp, decision_index):
        c = self.config
        if len(history) > c.memory_frames - 1:
            raise ValueError("History exceeds the bounded two-decision bank")
        if not history:
            if decision_index != 0:
                raise ValueError("An episode must start at decision_index=0")
            return ()
        previous = None
        for owned in history:
            owned.check_ownership()
            record = owned.record
            if previous is not None and (
                record.decision_index != previous.decision_index + 1
                or record.timestamp <= previous.timestamp
            ):
                raise ValueError(
                    "Historical decisions must be consecutive and chronological"
                )
            previous = record
        if (
            decision_index != previous.decision_index + 1
            or timestamp <= previous.timestamp
        ):
            raise ValueError(
                "Current decision must be consecutive with a strictly newer observation"
            )
        times = [owned.record.timestamp for owned in history] + [timestamp]
        durations = tuple(right - left for left, right in zip(times[:-1], times[1:]))
        for owned, duration in zip(history, durations):
            if duration > c.horizon:
                raise ValueError(
                    "Observed executed interval exceeds the original plan horizon"
                )
            if owned.record.executed_steps not in (0, duration):
                raise ValueError(
                    "Stored execution interval differs from observed timestamps"
                )
        return durations

    def snapshot(self, current, samples):
        k, v, rows, states_cpu, histories, intervals = self._prepare(current, samples)
        c, batch = self.config, k.shape[0]
        states = states_cpu.to(device=k.device)
        zero_k = k.new_zeros(k.shape[1:])
        zero_v = v.new_zeros(v.shape[1:])
        zero_state = states.new_zeros((c.proprio_dim,))
        flat_k, flat_v, flat_states = [], [], []
        masks, times, executed = [], [], []
        for index, (row, history, durations) in enumerate(
            zip(rows, histories, intervals)
        ):
            padding = c.memory_frames - 1 - len(history)
            flat_k.extend([zero_k] * padding)
            flat_v.extend([zero_v] * padding)
            flat_states.extend([zero_state] * padding)
            for owned in history:
                record = owned.record
                flat_k.append(record.k.to(k))
                flat_v.append(record.v.to(v))
                flat_states.append(record.state.to(states))
            flat_k.append(k[index])
            flat_v.append(v[index])
            flat_states.append(states[index])
            masks.append([0] * padding + [1] * (len(history) + 1))
            times.append(
                [0] * padding + [owned.record.timestamp for owned in history] + [row[2]]
            )
            executed.append([0] * padding + list(durations) + [0])
        metadata = torch.tensor(
            [masks, times, executed], dtype=torch.long, device=k.device
        )
        return {
            "k": torch.stack(flat_k).reshape(batch, c.memory_frames, *k.shape[1:]),
            "v": torch.stack(flat_v).reshape(batch, c.memory_frames, *v.shape[1:]),
            "states": torch.stack(flat_states).reshape(
                batch, c.memory_frames, c.proprio_dim
            ),
            "valid": metadata[0].bool(),
            "timestamps": metadata[1],
            "executed_steps": metadata[2],
        }

    def commit(self, current, samples):
        k, v, rows, states_cpu, histories, intervals = self._prepare(current, samples)
        states = states_cpu.to(device=k.device)
        changes = {}
        for index, (row, history, durations) in enumerate(
            zip(rows, histories, intervals)
        ):
            lane, episode_id, timestamp, decision_index, trajectory_pass_id = row
            record = _OwnedRecord.create(
                k[index], v[index], states[index], timestamp, decision_index
            )
            updated = [
                owned.with_duration(duration)
                for owned, duration in zip(history, durations)
            ]
            changes[lane] = _Lane(
                episode_id,
                tuple((updated + [record])[-(self.config.memory_frames - 1) :]),
                trajectory_pass_id,
            )
        self._lanes.update(changes)
