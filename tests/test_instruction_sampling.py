import json
import pickle
import sqlite3
from collections import Counter
from pathlib import Path

import pytest
import torch
from cache_fixtures import FakeProcessor, sha, write_fixture
from test_cache_dataset_v2 import make_v2, reseal_index

from reread.data.dataset import CachedTrajectoryDataset
from reread.data.distributed import continuous_requests

POLICY = {"clean": ("seen",), "randomized": ("seen", "unseen")}


@pytest.fixture
def cache(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    source = tmp_path / "v1"
    rows, manifest = write_fixture(source)
    root = tmp_path / "v2"
    make_v2(root, source, rows, manifest)
    yield root
    torch.set_num_threads(old_threads)


def opened(root, **kwargs):
    return CachedTrajectoryDataset(
        root,
        FakeProcessor(),
        instruction_roles_by_phase=POLICY,
        augmentation=True,
        **kwargs,
    )


def test_explicit_randomized_seen_and_unseen_uses_existing_encoded_records(cache):
    before = {path: sha(path) for path in cache.rglob("*") if path.is_file()}
    dataset = opened(cache)
    try:
        clean, randomized = (
            next(i for i, r in enumerate(dataset.rows) if r.phase == phase)
            for phase in ("clean", "randomized")
        )
        assert len(dataset.rows[clean].instructions) == 2
        assert len(dataset.rows[randomized].instructions) == 3
        clean_samples = [dataset[(0, clean, 0, i, 4)] for i in range(6)]
        random_samples = [dataset[(0, randomized, 0, i, 4)] for i in range(3)]
        assert all(
            s["instruction_role"] == "seen" and "unseen" not in s["prompt"]
            for s in clean_samples
        )
        assert [s["instruction_role"] for s in random_samples] == [
            "seen",
            "seen",
            "unseen",
        ]
        assert "Please unseen bottle" in random_samples[2]["prompt"]

        source = cache.parent / "v1"
        original_text = torch.load(
            source / "text" / (random_samples[2]["prompt_id"] + ".pt"),
            weights_only=True,
        )
        assert torch.equal(
            random_samples[2]["cached_encoded"]["context"], original_text["context"]
        )
        assert random_samples[2]["prompt_id"] != random_samples[0]["prompt_id"]
        assert dataset.instruction_role_for(randomized, 5) == "unseen"
        metadata = dataset.cache_metadata
        assert metadata["prompt_roles"] == "cached_seen_and_unseen"
        assert metadata["instruction_roles_by_phase"] == {
            "clean": ["seen"],
            "randomized": ["seen", "unseen"],
        }
        assert (
            metadata["unseen_training_phases"] == ["randomized"]
            and metadata["unseen_is_heldout"] is False
        )
        assert metadata["instruction_entries_by_phase_role"] == {
            "clean": {"seen": 2, "unseen": 0},
            "randomized": {"seen": 2, "unseen": 1},
        }
    finally:
        dataset.close()
    assert {path: sha(path) for path in cache.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("batch_size,batch_count", [(2, 120), (64, 16)])
def test_actual_training_requests_cover_roles_and_augmented_variants_with_fixed_history(
    cache, batch_size, batch_count
):
    dataset = opened(cache, seed=19)
    role_counts, variant_counts, passes = Counter(), Counter(), {}
    try:
        batches = continuous_requests(
            dataset.rows,
            list(range(len(dataset))),
            11,
            batch_size,
            20,
            batches=batch_count,
            task_probabilities={"task_one": 1.0},
            phase_probabilities={"clean": 1 / 11, "randomized": 10 / 11},
        )
        for requests in batches:
            for request in requests:
                sample = dataset[request]
                key = (sample["episode_id"], sample["original_trajectory_pass_id"])
                value = (
                    sample["prompt_id"],
                    sample["instruction_role"],
                    sample["cache_variant"],
                )
                assert passes.setdefault(key, value) == value
                assert sample["image_augmented"] == (sample["cache_variant"] != 0)
                assert (
                    sample["cached_encoded"]["rgb_latents"]
                    .eq(100 * sample["cache_variant"] + sample["logical_frame"] // 20)
                    .all()
                )
                role_counts[(sample["phase"], sample["instruction_role"])] += 1
                variant_counts[sample["cache_variant"]] += 1
                assert sample["valid"].any()
        assert (
            role_counts[("clean", "seen")] > 0 and not role_counts[("clean", "unseen")]
        )
        assert (
            role_counts[("randomized", "seen")] > 0
            and role_counts[("randomized", "unseen")] > 0
        )
        assert set(variant_counts) == {0, 1, 2}
        assert (
            sum(role_counts.values())
            == sum(variant_counts.values())
            == batch_size * batch_count
        )
        assert len(passes) < batch_size * batch_count
    finally:
        dataset.close()


def test_each_trajectory_pass_uses_one_variant_and_cycle_is_uniform(cache):
    dataset = opened(cache, seed=3)
    try:
        for index, row in enumerate(dataset.rows):
            counts = Counter()
            for pass_id in range(9):
                samples = [
                    dataset[(0, index, frame, 0, pass_id)]
                    for frame in range(0, row.observation_count, 20)
                ]
                assert len({s["cache_variant"] for s in samples}) == 1
                assert len({s["instruction_role"] for s in samples}) == 1
                counts[samples[0]["cache_variant"]] += 1
            assert counts == {0: 3, 1: 3, 2: 3}
    finally:
        dataset.close()


def test_validation_episodes_text_and_images_remain_isolated(cache):
    train = opened(cache)
    val = opened(cache, split="val")
    try:
        assert {r.id for r in train.rows}.isdisjoint({r.id for r in val.rows})
        assert all(r.split == "train" for r in train.rows)
        assert all(r.split == "val" for r in val.rows)
        for index, row in enumerate(val.rows):
            assert len(row.instructions) == 2
            for choice in range(6):
                sample = val[(0, index, 0, choice, choice)]
                assert sample["instruction_role"] == "seen"
                assert "unseen" not in sample["prompt"]
                assert (
                    sample["cache_variant"] == 0 and sample["image_augmented"] is False
                )
        assert val.cache_metadata["augmentation_enabled"] is False
        assert val.cache_metadata["unseen_training_phases"] == []
        assert val.cache_metadata["unseen_is_heldout"] is None
    finally:
        train.close()
        val.close()


def test_default_consumers_stay_seen_only_and_augmentation_can_be_disabled(cache):
    default = CachedTrajectoryDataset(cache, FakeProcessor())
    original = CachedTrajectoryDataset(cache, FakeProcessor(), augmentation=False)
    try:
        for index, row in enumerate(default.rows):
            assert len(row.instructions) == 2
            assert all(
                default[(0, index, 0, i, i)]["instruction_role"] == "seen"
                for i in range(5)
            )
            assert all(
                original[(0, index, 0, 0, p)]["cache_variant"] == 0 for p in range(5)
            )
    finally:
        default.close()
        original.close()


def test_unseen_role_lookup_is_worker_safe_and_warm_hit_has_no_index_or_filesystem_io(
    cache, monkeypatch
):
    dataset = opened(cache)
    random_index = next(
        i for i, row in enumerate(dataset.rows) if row.phase == "randomized"
    )
    dataset[(0, random_index, 0, 2, 7)]
    restored = pickle.loads(pickle.dumps(dataset))
    dataset.close()
    sample = restored[(0, random_index, 0, 2, 7)]
    assert sample["instruction_role"] == "unseen"
    if restored.cache_metadata["schema"].endswith("-v2"):

        def forbidden(*args, **kwargs):
            raise AssertionError("Warm unseen lookup performed filesystem/index IO")

        monkeypatch.setattr(restored, "_connection", forbidden)
        monkeypatch.setattr(Path, "stat", forbidden)
        monkeypatch.setattr(Path, "resolve", forbidden)
    following = restored[(0, random_index, 20, 2, 7)]
    assert (
        following["instruction_role"] == "unseen"
        and following["prompt_id"] == sample["prompt_id"]
    )
    restored.close()


@pytest.mark.parametrize(
    "policy",
    [
        True,
        {},
        {"randomized": ["seen", "unseen"]},
        {"clean": ["seen"], "randomized": []},
        {"clean": ["seen"], "randomized": ["other"]},
        {"clean": ["seen"], "randomized": ["seen", "seen"]},
    ],
)
def test_instruction_policy_fails_closed(cache, policy):
    with pytest.raises(ValueError, match="instruction|Instruction"):
        CachedTrajectoryDataset(
            cache, FakeProcessor(), instruction_roles_by_phase=policy
        )


@pytest.mark.parametrize("mutation", ["swap_identity", "bad_ordinal", "missing_role"])
def test_selected_unseen_sqlite_role_is_authenticated(tmp_path, mutation):
    source, root = tmp_path / "source", tmp_path / "v2"
    rows, manifest = write_fixture(source)
    make_v2(root, source, rows, manifest)
    episode = next(
        row.id for row in rows if row.split == "train" and row.phase == "randomized"
    )
    with sqlite3.connect(root / "index.sqlite") as db:
        if mutation == "swap_identity":
            seen = db.execute(
                "SELECT prompt_id FROM roles WHERE episode_id=? AND role='seen' AND ordinal=0",
                (episode,),
            ).fetchone()[0]
            db.execute(
                "UPDATE roles SET prompt_id=? WHERE episode_id=? AND role='unseen'",
                (seen, episode),
            )
        elif mutation == "bad_ordinal":
            db.execute(
                "UPDATE roles SET ordinal=9 WHERE episode_id=? AND role='unseen'",
                (episode,),
            )
        else:
            raw = json.loads(
                db.execute(
                    "SELECT row_json FROM episodes WHERE id=?", (episode,)
                ).fetchone()[0]
            )
            raw["unseen"] = []
            db.execute(
                "UPDATE episodes SET row_json=? WHERE id=?", (json.dumps(raw), episode)
            )
            db.execute(
                "DELETE FROM roles WHERE episode_id=? AND role='unseen'", (episode,)
            )
    reseal_index(root)
    dataset = None
    try:
        with pytest.raises(ValueError, match="Unseen|role is absent"):
            dataset = opened(root)
            random_index = next(
                i for i, row in enumerate(dataset.rows) if row.phase == "randomized"
            )
            dataset[(0, random_index, 0, 2, 0)]
    finally:
        if dataset is not None:
            dataset.close()


def test_same_prompt_in_both_roles_preserves_selected_role(tmp_path):
    source, root = tmp_path / "source", tmp_path / "v2"
    rows, manifest = write_fixture(source)
    make_v2(root, source, rows, manifest)
    episode = next(
        row.id for row in rows if row.split == "train" and row.phase == "randomized"
    )

    with sqlite3.connect(root / "index.sqlite") as db:
        raw = json.loads(
            db.execute(
                "SELECT row_json FROM episodes WHERE id=?", (episode,)
            ).fetchone()[0]
        )
        raw["unseen"] = [raw["seen"][0]]
        identity = db.execute(
            "SELECT prompt_id FROM roles WHERE episode_id=? AND role='seen' AND ordinal=0",
            (episode,),
        ).fetchone()[0]
        db.execute(
            "UPDATE episodes SET row_json=? WHERE id=?", (json.dumps(raw), episode)
        )
        db.execute(
            "UPDATE roles SET prompt_id=? WHERE episode_id=? AND role='unseen'",
            (identity, episode),
        )
    reseal_index(root)
    dataset = opened(root)
    try:
        index = next(i for i, row in enumerate(dataset.rows) if row.id == episode)
        seen, unseen = dataset[(0, index, 0, 0, 0)], dataset[(0, index, 0, 2, 0)]
        assert seen["prompt_id"] == unseen["prompt_id"]
        assert (
            seen["instruction_role"] == "seen"
            and unseen["instruction_role"] == "unseen"
        )
        assert torch.equal(
            seen["cached_encoded"]["context"], unseen["cached_encoded"]["context"]
        )
        assert len(dataset.rows[index].instructions) == 3
    finally:
        dataset.close()
