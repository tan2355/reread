import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from reread.data.dataset import LATENT_SHAPE
from reread.data.preprocessing import (
    CacheEpisode,
    build_prompt_table,
    history_coverage,
    make_action_window,
)

SCHEMA = "fixture-observation-arrays"

from reread.data.demonstrations import DEFAULT_PROMPT


class FakeNormalizer:
    def forward(self, batch):
        return {
            kind: {key: value / 10 for key, value in values.items()}
            for kind, values in batch.items()
        }


class FakeProcessor:
    shape_meta = {
        "state": [{"key": "default", "shape": 14}],
        "action": [{"key": "default", "shape": 14}],
    }
    normalizer = FakeNormalizer()

    def action_state_transform(self, batch):
        return batch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def encoder_identity():
    entry = {
        "path": "/encoders/native.pth",
        "bytes": 100,
        "mtime_ns": 123,
        "sha256": "a" * 64,
    }
    return {
        "schema": "newm-memory4-offline-encoders-v1",
        "upstream": "/fastwam",
        "wan": "/wan",
        "weights": {"native.pth": deepcopy(entry)},
        "tokenizer": {"tokenizer": deepcopy(entry)},
        "source": {"source.py": deepcopy(entry)},
        "execution": {
            "dtype": "bfloat16",
            "vae": "native",
            "encode_batch": 32,
            "tail_padding": "duplicate_last_to_32_then_slice",
            "rgb_shape": [3, 384, 320],
            "latent_shape": list(LATENT_SHAPE),
            "rgb_normalization": "uint8_to_bfloat16_then_multiply_2_over_255_subtract_1",
            "token_length": 128,
            "text_dim": 4096,
            "tokenizer_clean": "whitespace",
            "add_special_tokens": True,
            "text_padding": "zero_context_after_tokenizer_length_then_all_true_mask",
            "prompt_template": DEFAULT_PROMPT,
        },
    }


def write_fixture(root):
    h5py = pytest.importorskip("h5py")
    root.mkdir()
    definitions = [
        (
            "clean",
            "train",
            201,
            np.array([0, 20, 100, 120, 140, 160, 180], dtype=np.int64),
        ),
        ("clean", "val", 61, np.array([0, 20, 40], dtype=np.int64)),
        ("randomized", "train", 61, np.array([0, 20, 40], dtype=np.int64)),
        ("randomized", "val", 61, np.array([0, 20, 40], dtype=np.int64)),
    ]
    rows = [
        CacheEpisode(
            f"task_one/{phase}/episode{i}",
            f"/raw/episode{i}.hdf5",
            f"/raw/episode{i}.json",
            "task_one",
            phase,
            length,
            ("Move bottle", "Move bottle again"),
            ("Please unseen bottle",),
            split,
        )
        for i, (phase, split, length, _) in enumerate(definitions)
    ]
    table = build_prompt_table(rows)
    write_json(root / "prompts.json", table)
    serialized = [
        dict(
            row.to_dict(),
            source_size=100,
            source_mtime_ns=123,
            instruction_sha256="b" * 64,
        )
        for row in rows
    ]
    plan = {
        "schema": SCHEMA,
        "rows": serialized,
        "stride": 20,
        "horizon": 32,
        "augmented_variants": 2,
        "prompt_roles": "seen_training_only_unseen_eval_only",
        "encoder": encoder_identity(),
        "prompts_sha256": sha(root / "prompts.json"),
        "cleaning": {"original_timeline": True},
        "augmentation": {"finite_bank": True},
        "raw_root": "/raw",
        "source_binding": "stat_and_instruction_hash",
    }
    write_json(root / "plan.json", plan)
    plan_sha = sha(root / "plan.json")
    text_entries = {}
    for number, (identity, prompt) in enumerate(table["prompts"].items()):
        path = root / "text" / (identity + ".pt")
        path.parent.mkdir(exist_ok=True)
        torch.save(
            {
                "schema": SCHEMA,
                "prompt_id": identity,
                "prompt": prompt,
                "plan_sha256": plan_sha,
                "context": torch.full(
                    (128, 4096), float(number + 1), dtype=torch.bfloat16
                ),
                "context_mask": torch.ones(128, dtype=torch.bool),
            },
            path,
        )
        text_entries[identity] = {
            "id": identity,
            "path": "text/" + path.name,
            "size": path.stat().st_size,
            "sha256": sha(path),
        }
    episodes = {}
    for row, (_, _, length, frames) in zip(rows, definitions):
        path = root / "episodes" / (row.id + ".h5")
        path.parent.mkdir(parents=True, exist_ok=True)
        qpos = np.broadcast_to(
            np.arange(length, dtype=np.float32)[:, None], (length, 14)
        ).copy()
        windows = [make_action_window(int(frame), length) for frame in frames]
        bits = torch.empty((3, len(frames), *LATENT_SHAPE), dtype=torch.bfloat16)
        for variant in range(3):
            for i in range(len(frames)):
                bits[variant, i] = float(100 * variant + i)
        with h5py.File(path, "w") as handle:
            handle.attrs["schema"] = SCHEMA
            handle.attrs["episode_id"] = row.id
            handle.attrs["plan_sha256"] = plan_sha
            handle.attrs["rgb_dtype"] = "bfloat16_bits_uint16"
            handle.create_dataset("rgb_latents", data=bits.view(torch.uint16).numpy())
            handle.create_dataset("frames", data=frames)
            handle.create_dataset(
                "segments", data=np.cumsum(np.r_[True, np.diff(frames) != 20]) - 1
            )
            handle.create_dataset("proprio", data=qpos[frames])
            handle.create_dataset(
                "action", data=qpos[np.stack([window[0] for window in windows])]
            )
            handle.create_dataset(
                "valid", data=np.stack([window[1] for window in windows])
            )
        episodes[row.id] = {
            "id": row.id,
            "path": "episodes/" + row.id + ".h5",
            "size": path.stat().st_size,
            "sha256": sha(path),
            "frames": len(frames),
            "variants": 3,
            "split": row.split,
            "task": row.task,
            "phase": row.phase,
            "history": history_coverage(frames),
        }
    manifest = {
        "schema": SCHEMA,
        "plan_sha256": plan_sha,
        "encoder": plan["encoder"],
        "stride": 20,
        "variants": 3,
        "prompt_roles": plan["prompt_roles"],
        "episodes": episodes,
        "text": text_entries,
    }
    write_json(root / "manifest.json", manifest)
    complete = {
        "schema": SCHEMA,
        "complete": True,
        "plan_sha256": plan_sha,
        "manifest_sha256": sha(root / "manifest.json"),
        "variants": 3,
        "episodes": len(episodes),
        "prompts": len(text_entries),
        "train_instruction_type": "seen",
        "eval_instruction_type": "unseen",
    }
    write_json(root / "complete.json", complete)
    return rows, manifest
