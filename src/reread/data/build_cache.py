import argparse
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from reread.data.cache_contract import CACHED_PROMPT_ROLES, LEGACY_PROMPT_ROLES
from reread.data.demonstrations import (
    CAMERAS,
    DEFAULT_RAW_ROOT,
    action_window,
    decode_rgb,
    rgb_montage,
)
from reread.data.shards import SCHEMA_V2

SCHEMA = SCHEMA_V2
DEFAULT_OUTPUT = "artifacts/cache"
DEFAULT_UPSTREAM = "third_party/fastwam"
DEFAULT_WAN = "artifacts/ckpt/Wan2.2-TI2V-5B"


def sha256_file(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".pending.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())

    os.link(temporary, path)
    temporary.unlink()


def safe_file(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Cache path escapes its root")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("Cache artifact path contains a symlink")
    path = candidate.resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Cache path escapes its root")
    return path


def episode_relative(episode_id):
    parts = episode_id.split("/")
    import re

    if len(parts) != 3 or any(not re.fullmatch(r"[a-z0-9_]+", p) for p in parts):
        raise ValueError("Unsafe episode identity")
    return "episodes/" + "/".join(parts) + ".h5"


def assign_episodes(rows, world_size):
    if world_size < 1:
        raise ValueError("world_size must be positive")
    loads, assignments = [0] * world_size, [[] for _ in range(world_size)]
    for row in sorted(rows, key=lambda row: (-row["raw_length"], row["id"])):
        rank = min(range(world_size), key=lambda index: (loads[index], index))
        assignments[rank].append(row["id"])
        loads[rank] += (row["raw_length"] + 19) // 20
    return assignments, loads


def prepare(args):
    from reread.data.encoders import build_encoder_identity
    from reread.data.preprocessing import build_prompt_table, discover_cache_episodes

    root = Path(args.output_root).resolve()
    raw_root = Path(args.raw_root).resolve()
    if (
        root == raw_root
        or root.is_relative_to(raw_root)
        or raw_root.is_relative_to(root)
    ):
        raise ValueError("Cache and raw data roots must be separate")
    if root.exists():
        raise FileExistsError(
            "Use a NEW cache root; existing caches are never overwritten"
        )
    rows = discover_cache_episodes(
        raw_root,
        val_episodes=args.val_episodes,
        split_seed=args.split_seed,
        workers=max(1, args.cpu_workers),
    )
    serialized = []
    for row in rows:
        entry = row.to_dict()
        stat = Path(row.path).stat()
        entry["source_size"] = stat.st_size
        entry["source_mtime_ns"] = stat.st_mtime_ns
        entry["instruction_sha256"] = sha256_file(row.instruction_path)
        serialized.append(entry)
    table = build_prompt_table(rows)

    identity = build_encoder_identity(
        upstream=args.upstream,
        wan=args.wan,
        sha256_weights=True,
        vae_batch_size=args.batch_size,
        text_batch_size=args.text_batch_size,
    )
    assignments, loads = assign_episodes(serialized, args.world_size)
    prompts = sorted(table["prompts"])
    text_assignments = [
        prompts[rank :: args.world_size] for rank in range(args.world_size)
    ]
    plan = dict(
        schema=SCHEMA,
        observation_color_contract="native_robotwin_rgb_v1",
        raw_root=str(raw_root),
        rows=serialized,
        world_size=args.world_size,
        assignments=assignments,
        estimated_rank_windows=loads,
        text_assignments=text_assignments,
        encoder=identity,
        stride=args.stride,
        horizon=32,
        augmented_variants=args.augmented_variants,
        seed=args.seed,
        batch_size=args.batch_size,
        text_batch_size=args.text_batch_size,
        text_shard_mib=getattr(args, "text_shard_mib", 8192),
        rgb_shard_mib=getattr(args, "rgb_shard_mib", 2048),
        writer_queue_depth=getattr(args, "writer_queue_depth", 2),
        val_episodes=args.val_episodes,
        split_seed=args.split_seed,
        prompt_roles=CACHED_PROMPT_ROLES,
        cleaning=dict(
            policy="original_grid_full_horizon_joint_gripper_measured_pose_v1",
            min_static_run=6,
            joint_threshold=1e-6,
            gripper_threshold=1e-6,
            position_threshold_m=5e-4,
            rotation_threshold_rad=2e-3,
            retain_pre_motion_context=3,
        ),
        augmentation=dict(
            kind="photometric_color_jitter",
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0,
            geometric=False,
            temporal_consistency="one_transform_per_episode_variant",
            variant_zero="unaltered",
            finite_bank=True,
        ),
        source_binding="raw_size_mtime_and_joint_sha256_instruction_sha256_not_full_rgb_hash",
        training_started=False,
    )
    root.mkdir(parents=True, exist_ok=False)
    for name in ("shards", "ranks"):
        (root / name).mkdir()
    atomic_json(root / "prompts.json", table)
    plan["prompts_sha256"] = sha256_file(root / "prompts.json")
    atomic_json(root / "plan.json", plan)
    print(
        json.dumps(
            dict(
                event="cache_prepared",
                episodes=len(rows),
                prompts=len(prompts),
                output_root=str(root),
                plan_sha256=sha256_file(root / "plan.json"),
                estimated_latent_gib=sum(loads)
                * (1 + args.augmented_variants)
                * 46080
                / 1024**3,
            )
        ),
        flush=True,
    )


def load_plan(args):
    root = Path(args.output_root).resolve()
    plan = json.loads((root / "plan.json").read_text())
    if (
        plan.get("schema") != SCHEMA
        or plan.get("prompt_roles") not in (CACHED_PROMPT_ROLES, LEGACY_PROMPT_ROLES)
        or plan["world_size"] != args.world_size
        or plan["stride"] != args.stride
        or plan["batch_size"] != args.batch_size
        or plan["text_batch_size"] != args.text_batch_size
        or plan["augmented_variants"] != args.augmented_variants
        or plan["seed"] != args.seed
        or plan.get("text_shard_mib") != getattr(args, "text_shard_mib", 8192)
        or plan.get("rgb_shard_mib") != getattr(args, "rgb_shard_mib", 2048)
        or plan.get("writer_queue_depth") != getattr(args, "writer_queue_depth", 2)
        or Path(plan["raw_root"]).resolve() != Path(args.raw_root).resolve()
        or plan["prompts_sha256"] != sha256_file(root / "prompts.json")
    ):
        raise ValueError("Cache launcher and sealed preparation contract differ")
    return root, plan


def _decode_episode(job):
    import h5py

    from reread.data.preprocessing import (
        CacheEpisode,
        augment_montage,
        augmentation_params,
        history_coverage,
        read_endposes,
        selected_frames,
    )

    row, seed, variants = job
    episode = CacheEpisode.from_dict(
        {
            key: value
            for key, value in row.items()
            if key not in ("source_size", "source_mtime_ns", "instruction_sha256")
        }
    )
    source = Path(episode.path)
    before = source.stat()
    if (
        row.get("source_size", before.st_size) != before.st_size
        or row.get("source_mtime_ns", before.st_mtime_ns) != before.st_mtime_ns
        or row.get("instruction_sha256", sha256_file(episode.instruction_path))
        != sha256_file(episode.instruction_path)
    ):
        raise RuntimeError("Raw source changed after cache planning: " + episode.id)
    with h5py.File(source, "r") as handle:
        qpos = np.asarray(handle["joint_action/vector"][:], dtype=np.float32)
        if len(qpos) != episode.raw_length:
            raise ValueError("Raw trajectory length changed: " + episode.id)
        poses = read_endposes(handle)
        frames = selected_frames(
            qpos,
            poses,
            stride=20,
            horizon=32,
            joint_threshold=1e-6,
            gripper_threshold=1e-6,
            position_threshold_m=5e-4,
            rotation_threshold_rad=2e-3,
        )
        raw_images = [
            rgb_montage(
                {
                    camera: decode_rgb(handle[f"observation/{camera}/rgb"][int(frame)])
                    for camera in CAMERAS
                }
            )
            for frame in frames
        ]
    if (before.st_size, before.st_mtime_ns) != (
        source.stat().st_size,
        source.stat().st_mtime_ns,
    ):
        raise RuntimeError("Raw trajectory changed while caching: " + episode.id)

    parameters = [
        augmentation_params(seed, episode.id, variant)
        for variant in range(variants + 1)
    ]

    images = np.stack(
        [
            np.stack([augment_montage(image, params).numpy() for image in raw_images])
            for params in parameters
        ]
    )
    indices, valid = zip(
        *(action_window(int(frame), len(qpos), 32) for frame in frames)
    )
    segments = np.cumsum(np.r_[True, np.diff(frames) != 20]).astype(np.int64) - 1
    return dict(
        row=row,
        frames=frames,
        segments=segments,
        images=images,
        proprio=qpos[frames],
        action=qpos[np.stack(indices)],
        valid=np.stack(valid),
        augmentation_parameters=parameters,
        history=history_coverage(frames),
        raw_history=history_coverage(np.arange(0, len(qpos) - 1, 20)),
        source=dict(
            path=str(source.resolve()),
            size=before.st_size,
            mtime_ns=before.st_mtime_ns,
            joint_sha256=hashlib.sha256(qpos.tobytes()).hexdigest(),
            instruction_sha256=sha256_file(episode.instruction_path),
        ),
    )


def _bounded_decode(jobs, workers):
    if workers <= 0:
        yield from map(_decode_episode, jobs)
        return
    from collections import deque

    jobs = iter(jobs)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        pending = deque()
        for _ in range(workers * 2):
            job = next(jobs, None)
            if job is not None:
                pending.append(pool.submit(_decode_episode, job))
        while pending:
            yield pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(pool.submit(_decode_episode, job))


def encode(args):
    import torch

    from reread.data.encoders import OfflineEncoders
    from reread.data.shards import AsyncShardWriter

    root, plan = load_plan(args)
    result_path = root / "ranks" / f"rank_{args.rank:05d}.json"
    if result_path.exists() or (root / "complete.json").exists():
        raise FileExistsError(
            "Rank/cache already completed; never re-encode automatically"
        )
    plan_sha = sha256_file(root / "plan.json")
    rows = {row["id"]: row for row in plan["rows"]}
    encoder = OfflineEncoders.load(
        upstream=args.upstream,
        wan=args.wan,
        device=args.device,
        cpu_threads=args.cpu_threads,
        identity=plan["encoder"],
        expected_gpu_uuid=args.expected_gpu_uuid,
    )
    writer = AsyncShardWriter(
        root,
        rank=args.rank,
        plan_sha256=plan_sha,
        text_shard_mib=plan["text_shard_mib"],
        rgb_shard_mib=plan["rgb_shard_mib"],
        depth=plan["writer_queue_depth"],
    )
    started = time.monotonic()
    table = json.loads((root / "prompts.json").read_text())
    text_ids = plan["text_assignments"][args.rank]
    timings = dict(text_encode_seconds=0.0, rgb_encode_seconds=0.0)
    try:
        for offset in range(0, len(text_ids), args.text_batch_size):
            ids = text_ids[offset : offset + args.text_batch_size]
            stamp = time.monotonic()
            context, mask = encoder.encode_text([table["prompts"][key] for key in ids])
            timings["text_encode_seconds"] += time.monotonic() - stamp
            writer.submit_text(ids, context, mask)
            print(
                json.dumps(
                    dict(
                        event="text_cache_progress",
                        rank=args.rank,
                        encoded=offset + len(ids),
                        total=len(text_ids),
                        writer=writer.progress(),
                        **timings,
                    )
                ),
                flush=True,
            )
        if text_ids:
            del context, mask
        encoder.release_text()

        del table
        jobs = (
            (rows[key], args.seed, args.augmented_variants)
            for key in plan["assignments"][args.rank]
        )
        active, batch_images, batch_targets = {}, [], []
        episodes_encoded = 0

        def flush():
            nonlocal batch_images, batch_targets, episodes_encoded
            if not batch_images:
                return
            stamp = time.monotonic()
            output = encoder.encode_rgb(torch.stack(batch_images))
            timings["rgb_encode_seconds"] += time.monotonic() - stamp
            for target, latent in zip(batch_targets, output):
                episode_id, variant, frame_index = target
                item = active[episode_id]
                item["latents"][variant, frame_index].copy_(latent)
                item["remaining"] -= 1
                if not item["remaining"]:
                    metadata = {
                        key: value
                        for key, value in item["payload"].items()
                        if key != "images"
                    }
                    writer.submit_episode(metadata, item["latents"])
                    del active[episode_id]
                    episodes_encoded += 1
                    if episodes_encoded % 20 == 0:
                        print(
                            json.dumps(
                                dict(
                                    event="rgb_cache_progress",
                                    rank=args.rank,
                                    episodes=episodes_encoded,
                                    total=len(plan["assignments"][args.rank]),
                                    elapsed_seconds=time.monotonic() - started,
                                    writer=writer.progress(),
                                    **timings,
                                )
                            ),
                            flush=True,
                        )
            batch_images, batch_targets = [], []

        for payload in _bounded_decode(jobs, args.cpu_workers):
            episode_id = payload["row"]["id"]
            images = payload["images"]
            active[episode_id] = dict(
                payload=payload,
                remaining=images.shape[0] * images.shape[1],
                latents=torch.empty(
                    (*images.shape[:2], 48, 1, 24, 20), dtype=torch.bfloat16
                ),
            )
            for variant in range(images.shape[0]):
                for frame_index in range(images.shape[1]):
                    batch_images.append(torch.from_numpy(images[variant, frame_index]))
                    batch_targets.append((episode_id, variant, frame_index))
                    if len(batch_images) == args.batch_size:
                        flush()
        flush()
        if active:
            raise RuntimeError("Some episode RGB results were not submitted")
        result = writer.finish()
    except BaseException:
        writer.abort()
        raise

    atomic_json(
        result_path,
        dict(
            schema=SCHEMA,
            complete=True,
            rank=args.rank,
            plan_sha256=plan_sha,
            **result,
            elapsed_seconds=time.monotonic() - started,
            timings=dict(**timings, writer=writer.progress()),
        ),
    )
    print(
        json.dumps(
            dict(
                event="rank_complete",
                rank=args.rank,
                episodes=len(result["episodes"]),
                prompts=len(result["text"]),
                shards=len(result["shards"]),
            )
        ),
        flush=True,
    )


def _verify_shard(root, relative, entry, plan_sha, text_records, episode_records):
    import h5py

    from reread.data.preprocessing import history_coverage
    from reread.data.shards import (
        LABEL_NAMES,
        LATENT_SHAPE,
        TEXT_SHAPE,
        array_sha256,
        labels_sha256,
    )

    path = safe_file(root, relative)
    before = path.stat()
    if before.st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
        raise ValueError("Cache shard checksum/size mismatch: " + relative)
    with h5py.File(path, "r") as handle:
        if (
            handle.attrs.get("schema") != SCHEMA
            or handle.attrs.get("plan_sha256") != plan_sha
            or handle.attrs.get("kind") != entry["kind"]
            or handle.attrs.get("records") != entry["records"]
        ):
            raise ValueError("Cache shard header mismatch: " + relative)
        if entry["kind"] == "text":
            records = text_records
            if (
                len(records) != entry["records"]
                or set(handle) != {"contexts", "prompt_id", "sha256"}
                or handle["contexts"].shape != (len(records), *TEXT_SHAPE)
                or handle["contexts"].dtype != np.uint16
                or handle["prompt_id"].shape != (len(records),)
                or handle["sha256"].shape != (len(records),)
                or handle.attrs.get("text_padding")
                != "zero_context_after_tokenizer_length_then_all_true_mask"
            ):
                raise ValueError("Text shard record count/shape mismatch")
            identities, digests = handle["prompt_id"][:], handle["sha256"][:]
            if {record["row"] for record in records} != set(range(len(records))):
                raise ValueError("Text shard row inventory mismatch")
            for record in records:
                index = record["row"]
                if (
                    identities[index].decode() != record["id"]
                    or digests[index].decode() != record["sha256"]
                ):
                    raise ValueError("Text shard row identity/checksum mismatch")
        elif entry["kind"] == "rgb":
            records = episode_records
            if (
                len(records) != entry["records"]
                or set(handle) != {"episodes"}
                or set(handle["episodes"])
                != {record["group"].split("/")[1] for record in records}
            ):
                raise ValueError("RGB shard episode inventory mismatch")
            for record in records:
                group = handle[record["group"]]
                if (
                    group.attrs.get("schema") != SCHEMA
                    or group.attrs.get("episode_id") != record["id"]
                    or group.attrs.get("plan_sha256") != plan_sha
                    or group.attrs.get("rgb_dtype") != "bfloat16_bits_uint16"
                    or group.attrs.get("labels_sha256") != record["labels_sha256"]
                    or group.attrs.get("rgb_checksums_sha256")
                    != record["rgb_checksums_sha256"]
                    or json.loads(group.attrs["source"]) != record["source"]
                    or json.loads(group.attrs["augmentation_parameters"])
                    != record["augmentation_parameters"]
                    or group["rgb_latents"].shape
                    != (record["variants"], record["frames"], *LATENT_SHAPE)
                    or group["rgb_latents"].dtype != np.uint16
                    or group["rgb_sha256"].shape
                    != (record["variants"], record["frames"])
                    or set(group) != {*LABEL_NAMES, "rgb_latents", "rgb_sha256"}
                ):
                    raise ValueError("RGB shard episode header mismatch")
                labels = {name: group[name][:] for name in LABEL_NAMES}
                if (
                    array_sha256(group["rgb_sha256"][:])
                    != record["rgb_checksums_sha256"]
                    or labels_sha256(labels) != record["labels_sha256"]
                    or history_coverage(labels["frames"]) != record["history"]
                ):
                    raise ValueError("RGB shard labels/checksum table mismatch")
        else:
            raise ValueError("Unknown shard kind")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Cache shard changed during sealing: " + relative)


def _write_index(root, rows, episodes, texts, table):
    import sqlite3
    from contextlib import closing

    path = root / "index.sqlite"
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".pending.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    with closing(sqlite3.connect(temporary)) as connection:
        connection.executescript("""
            PRAGMA journal_mode=DELETE;
            CREATE TABLE texts(id TEXT PRIMARY KEY, prompt TEXT NOT NULL, shard TEXT NOT NULL,
                               row INTEGER NOT NULL, sha256 TEXT NOT NULL);
            CREATE TABLE episodes(id TEXT PRIMARY KEY, row_json TEXT NOT NULL, entry_json TEXT NOT NULL);
            CREATE TABLE roles(episode_id TEXT NOT NULL, role TEXT NOT NULL, ordinal INTEGER NOT NULL,
                               prompt_id TEXT NOT NULL, PRIMARY KEY(episode_id, role, ordinal));
        """)
        connection.executemany(
            "INSERT INTO texts VALUES (?,?,?,?,?)",
            (
                (
                    identity,
                    table["prompts"][identity],
                    item["path"],
                    item["row"],
                    item["sha256"],
                )
                for identity, item in texts.items()
            ),
        )
        connection.executemany(
            "INSERT INTO episodes VALUES (?,?,?)",
            (
                (
                    row["id"],
                    json.dumps(row, sort_keys=True),
                    json.dumps(episodes[row["id"]], sort_keys=True),
                )
                for row in rows
            ),
        )
        connection.executemany(
            "INSERT INTO roles VALUES (?,?,?,?)",
            (
                (identity, role, ordinal, prompt_id)
                for identity, roles in table["episodes"].items()
                for role in ("seen", "unseen")
                for ordinal, prompt_id in enumerate(roles[role])
            ),
        )
        connection.commit()
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("Cache index integrity check failed")
    os.link(temporary, path)
    temporary.unlink()
    return dict(path="index.sqlite", size=path.stat().st_size, sha256=sha256_file(path))


def seal(args):
    import re
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    from reread.data.preprocessing import (
        CacheEpisode,
        augmentation_params,
        build_prompt_table,
        history_coverage,
    )

    root, plan = load_plan(args)
    if (root / "complete.json").exists():
        raise FileExistsError("Cache already sealed")
    plan_sha = sha256_file(root / "plan.json")
    episodes, texts, shards, ranks = {}, {}, {}, []
    text_by_shard, episodes_by_shard = defaultdict(list), defaultdict(list)
    for rank in range(args.world_size):
        report = json.loads((root / "ranks" / f"rank_{rank:05d}.json").read_text())
        if (
            report.get("schema") != SCHEMA
            or report.get("complete") is not True
            or report.get("rank") != rank
            or report.get("plan_sha256") != plan_sha
        ):
            raise ValueError("Incomplete or mismatched encoder rank")
        for relative, entry in report["shards"].items():
            if (
                relative in shards
                or entry.get("kind") not in ("text", "rgb")
                or not re.fullmatch(
                    f"shards/{entry['kind']}/rank_{rank:05d}_shard_[0-9]{{5,}}\\.h5",
                    relative,
                )
                or not isinstance(entry.get("records"), int)
                or entry["records"] < 1
                or not isinstance(entry.get("size"), int)
                or entry["size"] < 1
                or not re.fullmatch("[0-9a-f]{64}", entry.get("sha256", ""))
            ):
                raise ValueError("Duplicate or invalid shard path")
            shards[relative] = entry
        for kind, target, expected, grouped in (
            ("episodes", episodes, set(plan["assignments"][rank]), episodes_by_shard),
            ("text", texts, set(plan["text_assignments"][rank]), text_by_shard),
        ):
            entries = report[kind]
            if (
                len(entries) != len(expected)
                or {entry["id"] for entry in entries} != expected
            ):
                raise ValueError("Rank is missing or duplicating source data")
            for entry in entries:
                if entry["id"] in target or entry["path"] not in report["shards"]:
                    raise ValueError("Duplicate global identity or unknown shard")
                if report["shards"][entry["path"]]["kind"] != (
                    "rgb" if kind == "episodes" else "text"
                ):
                    raise ValueError("Record points to wrong shard kind")
                if (
                    kind == "episodes"
                    and entry["group"]
                    != "episodes/" + hashlib.sha256(entry["id"].encode()).hexdigest()
                ):
                    raise ValueError("Episode group identity mismatch")
                target[entry["id"]] = entry
                grouped[entry["path"]].append(entry)
        ranks.append(
            dict(
                rank=rank,
                elapsed_seconds=report["elapsed_seconds"],
                timings=report.get("timings", {}),
            )
        )
    if set(episodes) != {row["id"] for row in plan["rows"]}:
        raise ValueError("Cache coverage differs from original inventory")
    for row in plan["rows"]:
        entry = episodes[row["id"]]
        source = entry.get("source", {})
        raw_frames = np.arange(0, row["raw_length"] - 1, 20)
        if (
            any(entry.get(key) != row[key] for key in ("task", "phase", "split"))
            or entry.get("variants") != 1 + plan["augmented_variants"]
            or entry.get("raw_grid_frames") != len(raw_frames)
            or not isinstance(entry.get("frames"), int)
            or not 1 <= entry["frames"] <= len(raw_frames)
            or entry.get("raw_history") != history_coverage(raw_frames)
            or Path(source.get("path", "")) != Path(row["path"])
            or source.get("size") != row["source_size"]
            or source.get("mtime_ns") != row["source_mtime_ns"]
            or source.get("instruction_sha256") != row["instruction_sha256"]
            or not re.fullmatch("[0-9a-f]{64}", source.get("joint_sha256", ""))
            or entry.get("augmentation_parameters")
            != [
                augmentation_params(plan["seed"], row["id"], variant)
                for variant in range(1 + plan["augmented_variants"])
            ]
        ):
            raise ValueError(
                "Episode record differs from sealed source/split/variant plan: "
                + row["id"]
            )
    table = json.loads((root / "prompts.json").read_text())
    clean_rows = [
        CacheEpisode.from_dict(
            {
                key: value
                for key, value in row.items()
                if key not in ("source_size", "source_mtime_ns", "instruction_sha256")
            }
        )
        for row in plan["rows"]
    ]
    if table != build_prompt_table(clean_rows) or set(texts) != set(table["prompts"]):
        raise ValueError("Seen/unseen prompt inventory or role binding is incomplete")

    def verify(item):
        relative, entry = item
        _verify_shard(
            root,
            relative,
            entry,
            plan_sha,
            text_by_shard[relative],
            episodes_by_shard[relative],
        )

    with ThreadPoolExecutor(max_workers=max(1, min(8, args.cpu_workers or 4))) as pool:
        for _ in pool.map(verify, shards.items()):
            pass
    index = _write_index(root, plan["rows"], episodes, texts, table)
    history = {str(k): 0 for k in range(1, 5)}
    raw_history = {str(k): 0 for k in range(1, 5)}
    for entry in episodes.values():
        for key in history:
            history[key] += entry["history"]["available_counts"][key]
            raw_history[key] += entry["raw_history"]["available_counts"][key]
    coverage = {}
    for phase in ("clean", "randomized"):
        selected = [
            e
            for e in episodes.values()
            if e["phase"] == phase and e["split"] == "train"
        ]
        coverage[phase] = dict(
            episodes=len(selected),
            frames=sum(e["frames"] for e in selected),
            fewer_than_four_anchors=sum(e["frames"] < 4 for e in selected),
            full_memory_decisions=sum(
                e["history"]["available_counts"]["4"] for e in selected
            ),
        )
    manifest = dict(
        schema=SCHEMA,
        observation_color_contract=plan["observation_color_contract"],
        plan_sha256=plan_sha,
        shards=shards,
        index=index,
        encoder=plan["encoder"],
        stride=20,
        horizon=32,
        variants=1 + args.augmented_variants,
        prompt_roles=CACHED_PROMPT_ROLES,
        cleaning=plan["cleaning"],
        augmentation=plan["augmentation"],
        raw_root=plan["raw_root"],
        source_binding=plan["source_binding"],
        ranks=ranks,
        prompts_sha256=plan["prompts_sha256"],
        history_available_counts=history,
        raw_history_available_counts=raw_history,
        train_history_by_phase=coverage,
        episodes=len(episodes),
        prompts=len(texts),
        text_mask="implicit_all_true",
        record_integrity="sha256_context_latent_and_labels_on_first_access",
        text_shard_mib=plan["text_shard_mib"],
        rgb_shard_mib=plan["rgb_shard_mib"],
        writer_queue_depth=plan["writer_queue_depth"],
        training_started=False,
    )
    atomic_json(root / "manifest.json", manifest)
    summary = dict(
        schema=SCHEMA,
        complete=True,
        observation_color_contract=plan["observation_color_contract"],
        plan_sha256=plan_sha,
        manifest_sha256=sha256_file(root / "manifest.json"),
        episodes=len(episodes),
        prompts=len(texts),
        frames=sum(e["frames"] for e in episodes.values()),
        raw_grid_frames=sum(e["raw_grid_frames"] for e in episodes.values()),
        variants=manifest["variants"],
        history_available_counts=history,
        raw_history_available_counts=raw_history,
        train_history_by_phase=coverage,
        shard_count=len(shards),
        text_shards=sum(e["kind"] == "text" for e in shards.values()),
        rgb_shards=sum(e["kind"] == "rgb" for e in shards.values()),
        bytes=sum(e["size"] for e in shards.values()) + index["size"],
        prompt_roles=CACHED_PROMPT_ROLES,
        training_started=False,
    )
    atomic_json(root / "complete.json", summary)
    print(json.dumps(dict(event="cache_sealed", **summary)), flush=True)


def parser():
    p = argparse.ArgumentParser(
        description=(
            "Build sharded RGB and text caches with original action timestamps.\n\n"
            "Independent GPU ranks encode observations; instruction splits remain separate."
        )
    )
    p.add_argument("--stage", choices=("prepare", "encode", "seal"), required=True)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT)
    p.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    p.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    p.add_argument("--wan", default=DEFAULT_WAN)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA runtime logical device, as verified by launcher preflight",
    )
    p.add_argument(
        "--expected-gpu-uuid",
        help="Verify physical GPU identity before loading encoder weights",
    )
    p.add_argument("--batch-size", type=int, choices=(32, 64, 128, 256), default=128)
    p.add_argument(
        "--text-batch-size", type=int, choices=(32, 64, 128, 256), default=128
    )
    p.add_argument("--text-shard-mib", type=int, default=8192)
    p.add_argument("--rgb-shard-mib", type=int, default=2048)
    p.add_argument("--writer-queue-depth", type=int, default=2)
    p.add_argument("--cpu-workers", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--augmented-variants", type=int, default=2)
    p.add_argument("--val-episodes", type=int, default=1)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if (
        args.world_size < 1
        or not 0 <= args.rank < args.world_size
        or args.batch_size not in (32, 64, 128, 256)
        or args.text_batch_size not in (32, 64, 128, 256)
        or args.cpu_workers < 0
        or args.cpu_threads < 1
        or args.text_shard_mib < 1
        or args.rgb_shard_mib < 1
        or args.writer_queue_depth < 1
        or args.augmented_variants < 1
        or args.stride != 20
        or args.val_episodes < 1
    ):
        raise ValueError("Invalid VAE/T5 batch/stride20 cache or CPU resource contract")
    if not args.device.startswith("cuda:") or not args.device[5:].isdigit():
        raise ValueError(
            "Offline cache device must be an explicit CUDA logical ordinal, for example cuda:0"
        )
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    return {"prepare": prepare, "encode": encode, "seal": seal}[args.stage](args)


if __name__ == "__main__":
    main()
