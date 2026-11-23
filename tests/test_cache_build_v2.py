import json
import sqlite3

import h5py
import pytest
import torch
from cache_fixtures import FakeProcessor, encoder_identity
from test_cache_build import raw_fixture

import reread.data.build_cache as build


@pytest.fixture
def fake_build(tmp_path, monkeypatch):
    import reread.data.encoders as cache_encoder

    raw_fixture(tmp_path / "raw")
    args = build.parser().parse_args(
        [
            "--stage",
            "prepare",
            "--output-root",
            str(tmp_path / "cache"),
            "--raw-root",
            str(tmp_path / "raw"),
            "--world-size",
            "2",
            "--cpu-workers",
            "0",
            "--text-shard-mib",
            "1",
            "--rgb-shard-mib",
            "1",
        ]
    )

    def identity(**kwargs):
        result = encoder_identity()
        result["execution"].update(
            encode_batch=128,
            tail_padding="duplicate_last_to_128_then_slice",
            text_encode_batch=128,
            text_tail_padding="duplicate_last_to_128_then_slice",
        )
        return result

    monkeypatch.setattr(cache_encoder, "build_encoder_identity", identity)

    class Encoder:
        def encode_text(self, prompts):
            return torch.zeros(
                (len(prompts), 128, 4096), dtype=torch.bfloat16
            ), torch.ones((len(prompts), 128), dtype=torch.bool)

        def release_text(self):
            pass

        def encode_rgb(self, images):
            return torch.ones((len(images), 48, 1, 24, 20), dtype=torch.bfloat16)

    monkeypatch.setattr(
        cache_encoder.OfflineEncoders, "load", lambda **kwargs: Encoder()
    )
    build.prepare(args)
    return args


@pytest.mark.parametrize(
    "older_plan", [False, True], ids=["current-plan", "earlier-plan"]
)
def test_sharded_build_seal_sqlite_covers_roles_and_exact_labels(fake_build, older_plan):
    from pathlib import Path

    from reread.data.dataset import CachedTrajectoryDataset

    args = fake_build
    root = Path(args.output_root)
    plan_path = root / "plan.json"
    if older_plan:
        plan = json.loads(plan_path.read_text())
        plan["prompt_roles"] = "seen_training_only_unseen_eval_only"
        plan_path.write_text(json.dumps(plan))
    plan_sha = build.sha256_file(plan_path)
    for rank in range(2):
        args.rank = rank
        build.encode(args)
    build.seal(args)
    complete = json.loads((root / "complete.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert build.sha256_file(plan_path) == plan_sha
    assert complete["plan_sha256"] == manifest["plan_sha256"] == plan_sha
    assert complete["prompt_roles"] == manifest["prompt_roles"] == "cached_seen_and_unseen"
    dataset = CachedTrajectoryDataset(root, FakeProcessor())
    try:
        assert dataset.cache_metadata["prompt_roles"] == complete["prompt_roles"]
    finally:
        dataset.close()
    assert complete["schema"] == build.SCHEMA and complete["complete"]
    assert complete["episodes"] == 4 and complete["prompts"] == 2
    assert complete["text_shards"] == 2 and complete["rgb_shards"] == 4
    assert "text" not in manifest and manifest["episodes"] == 4
    assert manifest["index"]["sha256"] == build.sha256_file(root / "index.sqlite")
    with sqlite3.connect(root / "index.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM texts").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM roles").fetchone()[0] == 8
        assert connection.execute(
            "SELECT DISTINCT role FROM roles ORDER BY role"
        ).fetchall() == [("seen",), ("unseen",)]
        for _, row_json, entry_json in connection.execute("SELECT * FROM episodes"):
            row, entry = json.loads(row_json), json.loads(entry_json)
            assert row["id"] == entry["id"]
            with h5py.File(root / entry["path"], "r") as handle:
                group = handle[entry["group"]]
                assert group["frames"][:].tolist() == [0, 20, 40, 60]
                assert group["action"][0, 0, 0] == pytest.approx(0.01)
                assert group["action"][-1, -1, 0] == pytest.approx(0.65)
                assert group["valid"][-1].sum() == 5
    with pytest.raises(FileExistsError):
        build.seal(args)
    with pytest.raises(FileExistsError):
        build.encode(args)


def test_seal_checks_whole_shards_once_not_once_per_record(fake_build, monkeypatch):
    args = fake_build
    for rank in range(2):
        args.rank = rank
        build.encode(args)
    observed = []
    original = build.sha256_file

    def recording(path):
        if str(path).endswith(".h5"):
            observed.append(str(path))
        return original(path)

    monkeypatch.setattr(build, "sha256_file", recording)
    build.seal(args)
    assert len(observed) == 6 and len(set(observed)) == 6


def test_writer_failure_prevents_rank_and_global_completion(fake_build, monkeypatch):
    from pathlib import Path

    from reread.data.shards import ShardWriter

    def fail(self, *args):
        raise OSError("simulated CFS failure")

    monkeypatch.setattr(ShardWriter, "append_text", fail)
    with pytest.raises(RuntimeError, match="writer failed"):
        build.encode(fake_build)
    root = Path(fake_build.output_root)
    assert not (root / "ranks" / "rank_00000.json").exists()
    assert not (root / "complete.json").exists()


def test_plan_rejects_shard_or_writer_contract_change(fake_build):
    fake_build.text_shard_mib += 1
    with pytest.raises(ValueError, match="contract differ"):
        build.load_plan(fake_build)


def test_seal_rejects_episode_split_drift_before_index(fake_build):
    from pathlib import Path

    args = fake_build
    for rank in range(2):
        args.rank = rank
        build.encode(args)
    root = Path(args.output_root)
    report_path = root / "ranks" / "rank_00000.json"
    report = json.loads(report_path.read_text())
    entry = report["episodes"][0]
    entry["split"] = "train" if entry["split"] == "val" else "val"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="source/split/variant"):
        build.seal(args)
    assert (
        not (root / "index.sqlite").exists() and not (root / "complete.json").exists()
    )


@pytest.mark.parametrize("workers", [0, 1])
def test_production_builder_index_reader_and_training_batch_roundtrip(
    fake_build, workers
):
    from types import SimpleNamespace

    from reread.backbone.observations import encode_current_batch
    from reread.data.dataset import CachedTrajectoryDataset
    from reread.data.demonstrations import iter_trajectory_batches

    args = fake_build
    for rank in range(2):
        args.rank = rank
        build.encode(args)
    build.seal(args)
    dataset = CachedTrajectoryDataset(args.output_root, FakeProcessor())
    batches = list(
        iter_trajectory_batches(
            dataset,
            seed=0,
            batch_size=2,
            decision_stride=20,
            workers=workers,
            max_batches=3,
        )
    )
    assert len(batches) == 3
    frozen = SimpleNamespace(backbone=SimpleNamespace(device="cpu"))
    for batch in batches:
        assert len(batch) == 2
        encoded = encode_current_batch(frozen, batch)
        assert encoded["rgb_latents"].shape == (2, 48, 1, 24, 20)
        assert encoded["rgb_latents"].eq(1).all() and encoded["context"].eq(0).all()
        assert encoded["context_mask"].all()
        assert all("adjust bottle" in sample["prompt"].lower() for sample in batch)
    dataset.close()
