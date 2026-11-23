import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from reread.data.cache_contract import CACHE_SCHEMA, ENCODER_IDENTITY_SCHEMA
from reread.data.demonstrations import PHASES

LATENT_SHAPE = (48, 1, 24, 20)
TEXT_SHAPE = (128, 4096)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid {name} SHA256")
    return value


def _relative_file(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Cache file path must be a nonempty relative string")
    path = Path(relative)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError("Cache file path escapes its root")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Cache files and directories must not be symbolic links")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Cache file path escapes its root")
    return resolved


def _json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Cache metadata must be a mapping: {path}")
    return value


def _validate_encoder(identity):
    expected = {
        "dtype": "bfloat16",
        "vae": "native",
        "rgb_shape": [3, 384, 320],
        "latent_shape": list(LATENT_SHAPE),
        "rgb_normalization": "uint8_to_bfloat16_then_multiply_2_over_255_subtract_1",
        "token_length": TEXT_SHAPE[0],
        "text_dim": TEXT_SHAPE[1],
        "tokenizer_clean": "whitespace",
        "add_special_tokens": True,
        "text_padding": "zero_context_after_tokenizer_length_then_all_true_mask",
    }
    if (
        not isinstance(identity, dict)
        or identity.get("schema") != ENCODER_IDENTITY_SCHEMA
    ):
        raise ValueError("Unsupported cache encoder identity")
    execution = identity.get("execution", {})
    batch = execution.get("encode_batch")
    if (
        isinstance(batch, bool)
        or batch not in (32, 64, 128, 256)
        or execution.get("tail_padding") != f"duplicate_last_to_{batch}_then_slice"
    ):
        raise ValueError("Invalid recorded offline VAE batch or tail-padding contract")
    text_batch = execution.get("text_encode_batch", 32)
    if (
        isinstance(text_batch, bool)
        or text_batch not in (32, 64, 128, 256)
        or execution.get("text_tail_padding", "duplicate_last_to_32_then_slice")
        != f"duplicate_last_to_{text_batch}_then_slice"
    ):
        raise ValueError("Invalid recorded offline T5 batch or tail-padding contract")
    if any(execution.get(name) != value for name, value in expected.items()):
        raise ValueError(
            "Cached encoder execution differs from native FastWAM conditioning"
        )
    from reread.data.demonstrations import DEFAULT_PROMPT

    if execution.get("prompt_template") != DEFAULT_PROMPT:
        raise ValueError("Cached text prompt template differs from native FastWAM")
    for group in ("weights", "tokenizer", "source"):
        entries = identity.get(group)
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"Cache encoder identity missing {group}")
        for entry in entries.values():
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("path"), str)
                or _integer(entry.get("bytes"), "encoder bytes") < 1
                or _integer(entry.get("mtime_ns"), "encoder mtime_ns") < 1
            ):
                raise ValueError("Invalid encoder source identity")

            _digest(entry.get("sha256"), "encoder file")


@dataclass(frozen=True)
class CachedEpisode:
    id: str
    path: str
    task: str
    phase: str
    raw_length: int
    instructions: Sequence[str]
    retained_count: int
    split: str

    @property
    def observation_count(self):
        return self.retained_count * 20


def _pass_pair(original_pass, segment):
    total = original_pass + segment
    return total * (total + 1) // 2 + segment


class CachedTrajectoryDataset(torch.utils.data.Dataset):

    def __new__(cls, root=None, *args, **kwargs):
        if cls is CachedTrajectoryDataset and root is not None:
            complete = _json(
                _relative_file(Path(root).resolve(strict=True), "complete.json")
            )
            if complete.get("schema") == CACHE_SCHEMA:
                from reread.data.sharded_dataset import ShardedCachedTrajectoryDataset

                return object.__new__(ShardedCachedTrajectoryDataset)
            raise ValueError("Expected a sealed sharded observation cache (v2)")
        return object.__new__(cls)

    def _configure_sampling(self, instruction_roles_by_phase, augmentation):
        if not isinstance(augmentation, bool):
            raise ValueError("augmentation must be boolean")
        policy = {phase: ("seen",) for phase in PHASES}
        if instruction_roles_by_phase is not None:
            if not isinstance(instruction_roles_by_phase, Mapping) or set(
                instruction_roles_by_phase
            ) != set(PHASES):
                raise ValueError(
                    "instruction_roles_by_phase must specify clean and randomized"
                )
            for phase, roles in instruction_roles_by_phase.items():
                if (
                    not isinstance(roles, (tuple, list))
                    or not roles
                    or any(role not in ("seen", "unseen") for role in roles)
                    or len(set(roles)) != len(roles)
                ):
                    raise ValueError(
                        "Instruction roles must be a nonempty unique seen/unseen sequence"
                    )
                policy[phase] = tuple(roles)
        self.instruction_roles_by_phase = (
            policy if self.split == "train" else {phase: ("seen",) for phase in PHASES}
        )
        self.augmentation = augmentation and self.split == "train"
        self._instruction_role_counts = {}
        self._instruction_inventory = Counter()

    def _register_instruction_counts(self, row, role_counts):
        if any(count < 1 for _, count in role_counts):
            raise ValueError(
                "Requested instruction role is absent from cached episode: " + row.id
            )
        self._instruction_role_counts[row.id] = tuple(role_counts)
        for role, count in role_counts:
            self._instruction_inventory[(row.phase, role)] += count

    def _sampling_metadata(self):
        return {
            "instruction_roles_by_phase": {
                phase: list(roles)
                for phase, roles in self.instruction_roles_by_phase.items()
            },
            "instruction_sampling": "uniform_cached_role_entries_once_per_trajectory_pass",
            "instruction_role_overlap": "same_prompt_in_both_roles_remains_two_entries_role_is_selected_entry",
            "instruction_entries_by_phase_role": {
                phase: {
                    role: self._instruction_inventory[(phase, role)]
                    for role in ("seen", "unseen")
                }
                for phase in PHASES
            },
            "unseen_training_phases": [
                phase
                for phase in PHASES
                if self.split == "train"
                and "unseen" in self.instruction_roles_by_phase[phase]
            ],
            "unseen_is_heldout": (
                not any(
                    "unseen" in roles
                    for roles in self.instruction_roles_by_phase.values()
                )
                if self.split == "train"
                else None
            ),
            "augmentation_enabled": self.augmentation,
            "variant_sampling": (
                "episode_seeded_permutation_cycled_by_trajectory_pass_id"
                if self.augmentation
                else "original_only"
            ),
            "validation_policy": "heldout_episodes_seen_only_original_images",
        }

    def instruction_role_for(self, episode_index, instruction_index):
        episode_index = _integer(episode_index, "episode index")
        instruction_index = _integer(instruction_index, "instruction index")
        row = self.rows[episode_index]
        offset = instruction_index % len(row.instructions)
        for role, count in self._instruction_role_counts[row.id]:
            if offset < count:
                return role
            offset -= count
        raise RuntimeError("Instruction role counts differ from selected entries")

    @staticmethod
    def _entry_metadata(entry):
        if _integer(entry.get("size"), "cache file size") < 1:
            raise ValueError("Cache files must not be empty")
        _digest(entry.get("sha256"), "cache file")

    @property
    def cache_metadata(self):
        return deepcopy(self._metadata)

    def validate_encoder_identity(self, expected):
        _validate_encoder(self._metadata["encoder"])
        planned = deepcopy(self._metadata["encoder"])
        current = deepcopy(expected)
        if not isinstance(current, dict):
            raise ValueError("Expected native encoder identity mapping")
        for name, entry in planned["weights"].items():
            if "sha256" not in current.get("weights", {}).get(name, {}):
                entry.pop("sha256", None)
        if current != planned:
            raise ValueError("Runtime encoder identity differs from sealed cache")

    def __len__(self):
        return len(self.rows)

    def variant_for(self, episode_id, trajectory_pass_id):
        original_pass = _integer(trajectory_pass_id, "trajectory pass")
        if not self.augmentation:
            return 0

        digest = hashlib.sha256(f"{self.seed}:{episode_id}".encode()).digest()
        order = np.random.default_rng(int.from_bytes(digest[:16], "big")).permutation(
            self._variants
        )
        return int(order[original_pass % self._variants])

    def __getitem__(self, request):
        if not isinstance(request, (tuple, list)) or len(request) not in (4, 5):
            raise ValueError(
                "Cached requests require four fields and an optional trajectory pass"
            )
        lane, index, logical_frame, instruction_index = [
            _integer(value, name)
            for value, name in zip(
                request[:4],
                ("lane", "episode index", "logical frame", "instruction index"),
            )
        ]
        original_pass = (
            _integer(request[4], "trajectory pass") if len(request) == 5 else 0
        )
        if (
            index >= len(self.rows)
            or logical_frame % 20
            or logical_frame >= self.rows[index].observation_count
        ):
            raise ValueError(
                "Cached request is not a retained logical-grid observation"
            )
        row, frame_index = self.rows[index], logical_frame // 20
        cached = self._file(index)
        raw_frame, segment = (
            int(cached["frames"][frame_index]),
            int(cached["segments"][frame_index]),
        )
        variant = self.variant_for(row.id, original_pass)
        latents = self._rgb(cached, variant, frame_index)
        if not torch.isfinite(latents).all():
            raise ValueError("Cached RGB latent contains nonfinite values")

        identity = row.instructions[instruction_index % len(row.instructions)]
        instruction_role = self.instruction_role_for(index, instruction_index)
        context, context_mask = self._text(identity)
        state_key = self.processor.shape_meta["state"][0]["key"]
        action_key = self.processor.shape_meta["action"][0]["key"]
        batch = {
            "state": {
                state_key: torch.from_numpy(
                    cached["proprio"][frame_index : frame_index + 1].copy()
                )
            },
            "action": {
                action_key: torch.from_numpy(cached["action"][frame_index].copy())
            },
        }
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        state, action = batch["state"][state_key][0], batch["action"][action_key]
        if (
            state.shape != (14,)
            or action.shape != (32, 14)
            or not torch.isfinite(state).all()
            or not torch.isfinite(action).all()
        ):
            raise ValueError(
                "Official processor changed cached joint dimensions or produced nonfinite values"
            )
        return {
            "lane": lane,
            "episode_id": row.id,
            "task": row.task,
            "phase": row.phase,
            "frame": raw_frame,
            "raw_frame": raw_frame,
            "logical_frame": logical_frame,
            "decision_index": int(cached["decisions"][frame_index]),
            "trajectory_pass_id": _pass_pair(original_pass, segment),
            "original_trajectory_pass_id": original_pass,
            "cache_segment": segment,
            "cache_variant": variant,
            "image_augmented": variant != 0,
            "instruction_role": instruction_role,
            "proprio": state.float(),
            "action": action.float(),
            "valid": torch.from_numpy(cached["valid"][frame_index].copy()),
            "prompt": self._prompt(identity),
            "prompt_id": identity,
            "cached_encoded": {
                "rgb_latents": latents,
                "context": context,
                "context_mask": context_mask,
            },
        }
