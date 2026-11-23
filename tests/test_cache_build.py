import io
import json
import sqlite3
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

import reread.data.build_cache as build
from reread.backbone.observations import encode_current_batch
from reread.data.preprocessing import PHASES


def raw_fixture(root):
    pixels = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), 100, dtype=np.uint8)).save(pixels, format="JPEG")
    blob = np.frombuffer(pixels.getvalue(), dtype=np.uint8)
    for phase in PHASES.values():
        directory = root / "adjust_bottle" / phase
        directory.mkdir(parents=True)
        for index in range(2):
            path = directory / f"episode{index}.hdf5"
            with h5py.File(path, "w") as handle:
                q = np.zeros((66, 14), dtype=np.float32)
                q[:, 0] = np.arange(66) * 0.01
                handle.create_dataset("joint_action/vector", data=q)
                for camera in build.CAMERAS:
                    values = handle.create_dataset(
                        f"observation/{camera}/rgb",
                        (66,),
                        dtype=h5py.vlen_dtype(np.dtype("uint8")),
                    )
                    for frame in range(66):
                        values[frame] = blob
            path.with_suffix(".json").write_text(
                json.dumps(
                    {"seen": ["adjust bottle"], "unseen": ["rotate the bottle upright"]}
                )
            )


def test_greedy_sharding_covers_once_and_balances():
    rows = [dict(id=str(i), raw_length=100 + i * 20) for i in range(17)]
    shards, loads = build.assign_episodes(rows, 16)
    assert sorted(x for shard in shards for x in shard) == sorted(
        row["id"] for row in rows
    )
    assert max(loads) - min(loads) <= 22
    assert build.assign_episodes(rows, 16) == (shards, loads)


def test_cache_atomic_artifacts_never_overwrite(tmp_path):
    target = tmp_path / "test.json"
    build.atomic_json(target, {"first": True})
    with pytest.raises(FileExistsError):
        build.atomic_json(target, {"second": True})
    assert json.loads(target.read_text()) == {"first": True}
    for path in ("../escape", "/absolute"):
        with pytest.raises(ValueError):
            build.safe_file(tmp_path, path)


def test_cpu_mocked_cache_prepare_encode_seal_and_corruption(tmp_path, monkeypatch):
    import reread.data.encoders as cache_encoder

    raw_fixture(tmp_path / "raw")
    root = tmp_path / "cache"
    args = build.parser().parse_args(
        [
            "--stage",
            "prepare",
            "--output-root",
            str(root),
            "--raw-root",
            str(tmp_path / "raw"),
            "--world-size",
            "2",
            "--cpu-workers",
            "0",
        ]
    )
    identity_calls = []

    def identity(**kwargs):
        identity_calls.append(kwargs)
        return {"mock": "encoder_identity"}

    monkeypatch.setattr(cache_encoder, "build_encoder_identity", identity)
    build.prepare(args)
    plan = json.loads((root / "plan.json").read_text())
    assert len(plan["rows"]) == 4
    assert plan["batch_size"] == 128 and plan["text_batch_size"] == 128
    assert (
        identity_calls[0]["vae_batch_size"] == 128
        and identity_calls[0]["text_batch_size"] == 128
    )
    assert len(json.loads((root / "prompts.json").read_text())["prompts"]) == 2
    calls = []

    class Encoder:
        def encode_text(self, prompts):
            calls.append(("text", len(prompts)))
            return torch.zeros(
                (len(prompts), 128, 4096), dtype=torch.bfloat16
            ), torch.ones((len(prompts), 128), dtype=torch.bool)

        def release_text(self):
            calls.append(("release", 0))

        def encode_rgb(self, images):
            assert images.dtype == torch.uint8 and images.shape[1:] == (3, 384, 320)
            calls.append(("rgb", len(images)))
            return torch.zeros((len(images), 48, 1, 24, 20), dtype=torch.bfloat16)

    encoder_loads = []

    def load_encoder(**kwargs):
        encoder_loads.append(kwargs)
        return Encoder()

    monkeypatch.setattr(cache_encoder.OfflineEncoders, "load", load_encoder)
    for rank in range(2):
        args.rank = rank
        args.device = f"cuda:{rank + 2}"
        args.expected_gpu_uuid = f"GPU-physical-{rank}"
        build.encode(args)
    assert [call["device"] for call in encoder_loads] == ["cuda:2", "cuda:3"]
    assert [call["expected_gpu_uuid"] for call in encoder_loads] == [
        "GPU-physical-0",
        "GPU-physical-1",
    ]
    build.seal(args)
    complete = json.loads((root / "complete.json").read_text())
    assert (
        complete["complete"] and complete["episodes"] == 4 and complete["variants"] == 3
    )
    assert complete["prompt_roles"] == plan["prompt_roles"] == "cached_seen_and_unseen"
    assert "train_instruction_type" not in complete
    assert "eval_instruction_type" not in complete
    assert all(size <= args.batch_size for name, size in calls if name == "rgb")

    assert ("rgb", 24) in calls
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["prompt_roles"] == complete["prompt_roles"]
    assert manifest["schema"] == "newm-memory4-observation-cache-v2"
    assert "episodes" not in manifest or isinstance(manifest["episodes"], int)
    with sqlite3.connect(root / manifest["index"]["path"]) as connection:
        entries = [
            json.loads(row[0])
            for row in connection.execute("SELECT entry_json FROM episodes")
        ]
    for entry in entries:
        with h5py.File(root / entry["path"], "r") as shard:
            handle = shard[entry["group"]]
            assert handle["frames"][:].tolist() == [0, 20, 40, 60]
            assert handle["action"][0, 0, 0] == pytest.approx(0.01)
            assert handle["action"][-1, -1, 0] == pytest.approx(0.65)
            assert handle["valid"][-1].sum() == 5
            assert handle["rgb_latents"].shape == (3, 4, 48, 1, 24, 20)
    with pytest.raises(FileExistsError):
        build.encode(args)
    with pytest.raises(FileExistsError):
        build.prepare(args)


def test_encoded_training_bypasses_vae_and_t5():
    frozen = SimpleNamespace(backbone=SimpleNamespace(device="cpu"))
    encoded = {
        "rgb_latents": torch.ones((48, 1, 24, 20), dtype=torch.bfloat16),
        "context": torch.ones((128, 4096), dtype=torch.bfloat16),
        "context_mask": torch.ones(128, dtype=torch.bool),
    }
    samples = [dict(cached_encoded=encoded, proprio=torch.zeros(14))]
    result = encode_current_batch(frozen, samples)
    assert result["rgb_latents"].shape == (1, 48, 1, 24, 20)
    assert set(result) == {"rgb_latents", "context", "context_mask", "proprio"}
    with pytest.raises(ValueError):
        encode_current_batch(frozen, [samples[0], {"proprio": torch.zeros(14)}])


def test_fixed_batch_and_stride_guard():
    with pytest.raises(ValueError):
        build.main(["--stage", "prepare", "--stride", "10"])
    with pytest.raises(SystemExit):
        build.main(["--stage", "prepare", "--batch-size", "16"])
    with pytest.raises(SystemExit):
        build.main(["--stage", "prepare", "--text-batch-size", "16"])


def test_cache_encoder_device_is_explicit_and_independent_of_shard_rank():
    args = build.parser().parse_args(
        [
            "--stage",
            "encode",
            "--rank",
            "12",
            "--world-size",
            "16",
            "--device",
            "cuda:5",
            "--expected-gpu-uuid",
            "GPU-physical-five",
        ]
    )
    assert args.rank == 12 and args.device == "cuda:5"
    assert args.expected_gpu_uuid == "GPU-physical-five"
    defaults = build.parser().parse_args(["--stage", "encode"])
    assert defaults.device == "cuda:0" and defaults.expected_gpu_uuid is None
