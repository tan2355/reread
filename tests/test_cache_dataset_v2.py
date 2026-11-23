import json
import pickle
import sqlite3
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from cache_fixtures import FakeProcessor, sha, write_fixture, write_json

from reread.data.dataset import CachedTrajectoryDataset
from reread.data.demonstrations import iter_trajectory_batches
from reread.data.preprocessing import history_coverage
from reread.data.shards import LABEL_NAMES, SCHEMA_V2, ShardWriter


def make_v2(
    root, source, source_rows, source_manifest, *, text_shard_mib=2, rgb_shard_mib=2
):
    import h5py

    root.mkdir()
    plan = json.loads((source / "plan.json").read_text())
    plan["schema"] = SCHEMA_V2
    write_json(root / "plan.json", plan)
    plan_sha = sha(root / "plan.json")
    prompts = json.loads((source / "prompts.json").read_text())
    writer = ShardWriter(
        root,
        rank=0,
        plan_sha256=plan_sha,
        text_shard_mib=text_shard_mib,
        rgb_shard_mib=rgb_shard_mib,
    )
    for identity, entry in source_manifest["text"].items():
        payload = torch.load(source / entry["path"], weights_only=True)
        writer.append_text(
            [identity], payload["context"][None], payload["context_mask"][None]
        )
    for row in source_rows:
        old = source_manifest["episodes"][row.id]
        with h5py.File(source / old["path"], "r") as handle:
            payload = {name: np.asarray(handle[name][:]) for name in LABEL_NAMES}
            latents = torch.from_numpy(np.asarray(handle["rgb_latents"][:])).view(
                torch.bfloat16
            )
        payload.update(
            row=row.to_dict(),
            source={},
            augmentation_parameters=[{}, {}, {}],
            history=history_coverage(payload["frames"]),
            raw_history=history_coverage(payload["frames"]),
        )
        writer.append_episode(payload, latents)
    written = writer.finish()
    connection = sqlite3.connect(root / "index.sqlite")
    connection.executescript("""
      CREATE TABLE texts(id TEXT PRIMARY KEY,prompt TEXT,shard TEXT,row INTEGER,sha256 TEXT);
      CREATE TABLE episodes(id TEXT PRIMARY KEY,row_json TEXT,entry_json TEXT);
      CREATE TABLE roles(episode_id TEXT,role TEXT,ordinal INTEGER,prompt_id TEXT,
                         PRIMARY KEY(episode_id,role,ordinal));
    """)
    for record in written["text"]:
        connection.execute(
            "INSERT INTO texts VALUES(?,?,?,?,?)",
            (
                record["id"],
                prompts["prompts"][record["id"]],
                record["path"],
                record["row"],
                record["sha256"],
            ),
        )
    for row, entry in zip(source_rows, written["episodes"]):
        connection.execute(
            "INSERT INTO episodes VALUES(?,?,?)",
            (row.id, json.dumps(row.to_dict()), json.dumps(entry)),
        )
        for role, ids in prompts["episodes"][row.id].items():
            connection.executemany(
                "INSERT INTO roles VALUES(?,?,?,?)",
                [(row.id, role, i, key) for i, key in enumerate(ids)],
            )
    connection.commit()
    connection.close()
    manifest = {
        key: deepcopy(plan.get(key))
        for key in (
            "schema",
            "encoder",
            "stride",
            "horizon",
            "prompt_roles",
            "cleaning",
            "augmentation",
            "raw_root",
            "source_binding",
        )
    }
    manifest.update(
        plan_sha256=plan_sha,
        variants=3,
        shards=written["shards"],
        index={
            "path": "index.sqlite",
            "size": (root / "index.sqlite").stat().st_size,
            "sha256": sha(root / "index.sqlite"),
        },
    )
    write_json(root / "manifest.json", manifest)
    write_json(
        root / "complete.json",
        {
            "schema": SCHEMA_V2,
            "complete": True,
            "plan_sha256": plan_sha,
            "manifest_sha256": sha(root / "manifest.json"),
            "variants": 3,
            "episodes": len(source_rows),
            "prompts": len(written["text"]),
            "train_instruction_type": "seen",
            "eval_instruction_type": "unseen",
        },
    )
    return manifest, written


@pytest.fixture
def pair(tmp_path):
    old, new = tmp_path / "v1", tmp_path / "v2"
    rows, manifest = write_fixture(old)
    new_manifest, written = make_v2(new, old, rows, manifest)
    return old, new, new_manifest, written


def assert_samples_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        if name == "cached_encoded":
            assert left[name].keys() == right[name].keys()
            assert all(
                torch.equal(left[name][key], right[name][key]) for key in left[name]
            )
        elif isinstance(left[name], torch.Tensor):
            assert torch.equal(left[name], right[name])
        else:
            assert left[name] == right[name]


def test_warm_hit_avoids_filesystem_probes_and_sql(pair, monkeypatch):
    _, root, _, _ = pair
    dataset = CachedTrajectoryDataset(root, FakeProcessor())
    expected = dataset[(0, 0, 0, 0)]
    import reread.data.sharded_dataset as module

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected cache-hit filesystem/index probe")

    monkeypatch.setattr(module, "_relative_file", forbidden)
    monkeypatch.setattr(module, "_sha256", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(dataset, "_connection", forbidden)
    monkeypatch.setattr(dataset, "_shard", forbidden)
    assert_samples_equal(expected, dataset[(0, 0, 0, 0)])

    following = dataset[(0, 0, 20, 0)]
    assert following["frame"] == 20
    assert (
        following["cached_encoded"]["rgb_latents"]
        .eq(100 * following["cache_variant"] + 1)
        .all()
    )
    dataset.close()


def test_worker_pickle_is_bounded_and_has_no_live_connections_or_text_bank(pair):
    _, root, _, _ = pair
    dataset = CachedTrajectoryDataset(
        root, FakeProcessor(), max_open_files=1, text_cache_entries=1
    )
    dataset[(0, 0, 0, 0)]
    encoded = pickle.dumps(dataset)
    restored = pickle.loads(encoded)
    assert len(encoded) < 30_000
    assert (
        restored._db is None
        and not restored._handles
        and not restored._files
        and not restored._texts
    )
    assert not restored._rgbs and restored._rgb_bytes == 0
    assert not hasattr(restored, "_prompts") and not hasattr(restored, "_text_entries")
    assert restored.rows[0].instructions.dataset is restored
    for index in range(2):
        restored[(0, index, 0, index)]
        assert (
            len(restored._handles) <= 1
            and len(restored._files) <= 1
            and len(restored._texts) <= 1
        )
    restored.close()
    dataset.close()


def test_rgb_prefetch_budget_eviction_and_oversize_fallback(pair):
    _, root, _, _ = pair
    dataset = CachedTrajectoryDataset(root, FakeProcessor(), rgb_cache_mib=1)
    reference = CachedTrajectoryDataset(root, FakeProcessor(), rgb_cache_mib=0)
    for episode in range(2):
        for replay in range(5):
            query = (0, episode, 0, 0, replay)
            assert_samples_equal(dataset[query], reference[query])
            assert dataset._rgb_bytes == sum(
                value.nbytes for value in dataset._rgbs.values()
            )
            assert dataset._rgb_bytes <= 1024**2
            assert not reference._rgbs and reference._rgb_bytes == 0
    assert len(dataset._rgbs) < 6
    dataset.close()
    reference.close()


@pytest.mark.parametrize("target", ["rgb", "text", "labels", "checksums"])
def test_record_corruption_detected_without_whole_shard_hash(pair, target):
    import h5py

    _, root, _, written = pair
    entry = written["episodes"][0]
    if target == "text":
        with sqlite3.connect(root / "index.sqlite") as db:
            identity = db.execute(
                "SELECT prompt_id FROM roles WHERE episode_id=? AND role='seen' AND ordinal=0",
                (entry["id"],),
            ).fetchone()[0]
            path, ordinal = db.execute(
                'SELECT shard,"row" FROM texts WHERE id=?', (identity,)
            ).fetchone()
        with h5py.File(root / path, "a") as handle:
            handle["contexts"][ordinal, 0, 0] ^= np.uint16(1)
    else:
        with h5py.File(root / entry["path"], "a") as handle:
            group = handle[entry["group"]]
            if target == "labels":
                group["action"][0, 0, 0] += 1
            elif target == "checksums":
                group["rgb_sha256"][0, 0] = b"a" * 64
            else:
                for variant in range(3):
                    group["rgb_latents"][variant, 0, 0, 0, 0, 0] ^= np.uint16(1)
    dataset = CachedTrajectoryDataset(root, FakeProcessor(), verify_files=False)
    with pytest.raises(ValueError, match="SHA256"):
        dataset[(0, 0, 0, 0)]
    dataset.close()


def reseal_index(root):
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["index"].update(
        size=(root / "index.sqlite").stat().st_size, sha256=sha(root / "index.sqlite")
    )
    write_json(root / "manifest.json", manifest)
    complete = json.loads((root / "complete.json").read_text())
    complete["manifest_sha256"] = sha(root / "manifest.json")
    write_json(root / "complete.json", complete)


def test_seen_role_cannot_be_replaced_by_unseen_even_if_index_resealed(pair):
    _, root, _, written = pair
    episode = written["episodes"][0]["id"]
    with sqlite3.connect(root / "index.sqlite") as db:
        unseen = db.execute(
            "SELECT prompt_id FROM roles WHERE episode_id=? AND role='unseen'",
            (episode,),
        ).fetchone()[0]
        db.execute(
            "UPDATE roles SET prompt_id=? WHERE episode_id=? AND role='seen' AND ordinal=0",
            (unseen, episode),
        )
    reseal_index(root)
    dataset = CachedTrajectoryDataset(root, FakeProcessor())
    with pytest.raises(ValueError, match="Seen instruction roles"):
        dataset[(0, 0, 0, 0)]
    dataset.close()


def test_index_tamper_rejected_before_worker_loading(pair):
    _, root, _, _ = pair
    with sqlite3.connect(root / "index.sqlite") as db:
        db.execute("UPDATE texts SET prompt='changed'")
    with pytest.raises(ValueError, match="index SHA256"):
        CachedTrajectoryDataset(root, FakeProcessor())


@pytest.mark.parametrize(
    "change",
    [
        "negative_text_row",
        "oversize_text_row",
        "text_escape",
        "wrong_prompt",
        "wrong_episode_group",
        "role_ordinal",
    ],
)
def test_malformed_resealed_index_is_rejected(pair, change):
    _, root, _, written = pair
    episode = written["episodes"][0]["id"]
    with sqlite3.connect(root / "index.sqlite") as db:
        identity = db.execute(
            "SELECT prompt_id FROM roles WHERE episode_id=? AND role='seen' AND ordinal=0",
            (episode,),
        ).fetchone()[0]
        if change == "negative_text_row":
            db.execute('UPDATE texts SET "row"=-1 WHERE id=?', (identity,))
        elif change == "oversize_text_row":
            db.execute('UPDATE texts SET "row"=999999 WHERE id=?', (identity,))
        elif change == "text_escape":
            db.execute("UPDATE texts SET shard='../outside.h5' WHERE id=?", (identity,))
        elif change == "wrong_prompt":
            db.execute(
                "UPDATE texts SET prompt='not the hashed prompt' WHERE id=?",
                (identity,),
            )
        elif change == "wrong_episode_group":
            entry = json.loads(
                db.execute(
                    "SELECT entry_json FROM episodes WHERE id=?", (episode,)
                ).fetchone()[0]
            )
            entry["group"] = "episodes/" + "0" * 64
            db.execute(
                "UPDATE episodes SET entry_json=? WHERE id=?",
                (json.dumps(entry), episode),
            )
        else:
            db.execute(
                "UPDATE roles SET ordinal=9 WHERE episode_id=? AND role='seen' AND ordinal=0",
                (episode,),
            )
    reseal_index(root)
    dataset = None
    try:
        with pytest.raises(ValueError):
            dataset = CachedTrajectoryDataset(root, FakeProcessor())
            dataset[(0, 0, 0, 0)]
    finally:
        if dataset is not None:
            dataset.close()


def test_v2_encoder_identity_still_enforced(pair):
    from cache_fixtures import encoder_identity

    _, root, _, _ = pair
    dataset = CachedTrajectoryDataset(root, FakeProcessor())
    expected = encoder_identity()
    dataset.validate_encoder_identity(expected)
    expected["weights"]["native.pth"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="Runtime encoder"):
        dataset.validate_encoder_identity(expected)
    dataset.close()


@pytest.mark.parametrize("target", ["rgb_shard", "rgb_group", "text_shard"])
def test_wrong_shard_or_group_header_is_rejected(pair, target):
    import h5py

    _, root, _, written = pair
    if target == "text_shard":
        paths = [entry["path"] for entry in written["text"]]
    else:
        paths = [written["episodes"][0]["path"]]
    for path in set(paths):
        with h5py.File(root / path, "a") as handle:
            owner = (
                handle[written["episodes"][0]["group"]]
                if target == "rgb_group"
                else handle
            )
            owner.attrs.modify("plan_sha256", "f" * 64)
    dataset = CachedTrajectoryDataset(root, FakeProcessor())
    with pytest.raises(ValueError, match="header|size"):
        dataset[(0, 0, 0, 0)]
    dataset.close()


@pytest.mark.parametrize("workers", [1, 2])
def test_current_only_private_outputs_and_worker_loader(pair, workers):
    _, root, _, _ = pair
    dataset = CachedTrajectoryDataset(root, FakeProcessor(), phases=("clean",))
    expected = dataset[(0, 0, 0, 0)]
    modified = dataset[(0, 0, 0, 0)]
    for tensor in modified["cached_encoded"].values():
        tensor.zero_()
    modified["action"].zero_()
    assert_samples_equal(expected, dataset[(0, 0, 0, 0)])
    samples = [
        batch[0]
        for batch in iter_trajectory_batches(
            dataset,
            seed=0,
            batch_size=1,
            decision_stride=20,
            workers=workers,
            max_batches=5,
        )
    ]
    assert [sample["frame"] for sample in samples] == [0, 20, 100, 120, 140]
    assert [sample["decision_index"] for sample in samples] == [0, 1, 0, 1, 2]
    assert all(
        set(s["cached_encoded"]) == {"rgb_latents", "context", "context_mask"}
        for s in samples
    )
    dataset.close()
