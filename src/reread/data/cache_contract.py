import hashlib
import json
import re
import stat
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath

CACHE_SCHEMA = "newm-memory4-observation-cache-v2"
ENCODER_IDENTITY_SCHEMA = "newm-memory4-offline-encoders-v1"
PROMPT_TABLE_SCHEMA = "newm-memory4-prompts-v1"
TEXT_REUSE_SCHEMA = "newm-memory4-text-reuse-v1"

COLOR_CONTRACT = "native_robotwin_rgb_v1"
CACHED_PROMPT_ROLES = "cached_seen_and_unseen"
LEGACY_PROMPT_ROLES = "seen_training_only_unseen_eval_only"
TEXT_REUSE_VERIFICATION = (
    "sealed_source_manifest_same_inode_size_header_and_per_record_sha256_on_read"
)
TEXT_REUSE_FIELDS = {
    "schema",
    "mode",
    "source_cache",
    "source_manifest_sha256",
    "source_plan_sha256",
    "source_index_sha256",
    "source_index_size",
    "source_prompts_sha256",
    "verification",
}
_DIGEST = re.compile(r"[0-9a-f]{64}")
_ENCODER_SOURCE_REVISIONS = (
    (
        (49604, "cb2b1b53f35a1a201c640ec52d12a7c4cb4ba1df322155addd3dc409e04dc07c"),
        (47473, "3c5d9dd236bc14368363cf5f175e5d0aed4b04af891f2b1d2b568e941ed3a2f1"),
    ),
    (
        (50979, "5ea4c2db67b06ee5ad194872ebe97db69914fde9f2ff08109a5228b226f91c78"),
        (48771, "ed92ff7a8c31f38198ced1edde888bf2652f7399aab1a287df84a5ada58b2718"),
    ),
    (
        (11459, "aedf7f131556a39b99a273deec3a85d67f41b5367cef41f310f33a807ae60a87"),
        (10576, "389f0495b484ceaefe8b1f2c9e0243dbc970a7028fe01468cdeb37bc3725ede2"),
    ),
    (
        (11801, "422f9e78f23a36890fa3c20478553ec44d6b9d44bf41336d8dfcb2ce915dc20b"),
        (10892, "d745f7c1d23f1fcb02ef56e35633f65b0529aa77e046e2050a07c972bb5293b3"),
    ),
    (
        (8279, "792f66bd0609c1365256aa24d43301340f81260c1651d89b04d2ae8224a08726"),
        (7931, "e22f17f361e94682549751ad0dfa01a4526c8d7e2871692d51df839fb7e00338"),
    ),
    (
        (8500, "108e834b78394ce223b464830b8b4bbbedb17b5a705f7d5d1ac1256a04dc8ea6"),
        (8149, "5b7613f17d1972e2b6f1d3218bec028b864e34d156ecdaf04c066de9c956b9bb"),
    ),
)


def matches_encoder_source(recorded, size, digest):
    previous = (recorded.get("bytes"), recorded.get("sha256"))
    current = (size, digest)
    return previous == current or any(
        previous in versions and current in versions
        for versions in _ENCODER_SOURCE_REVISIONS
    )


def validate_prompt_roles(plan, manifest, complete):
    roles = manifest.get("prompt_roles")
    if roles == LEGACY_PROMPT_ROLES:
        valid = (
            plan.get("prompt_roles", roles) == roles
            and complete.get("prompt_roles", roles) == roles
            and complete.get("train_instruction_type") == "seen"
            and complete.get("eval_instruction_type") == "unseen"
        )
    else:
        valid = (
            roles == CACHED_PROMPT_ROLES
            and plan.get("prompt_roles") in (CACHED_PROMPT_ROLES, LEGACY_PROMPT_ROLES)
            and complete.get("prompt_roles") == roles
            and "train_instruction_type" not in complete
            and "eval_instruction_type" not in complete
        )
    if not valid:
        raise ValueError("Inconsistent cached instruction pool metadata")


def _digest(value, name):
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"RGB cache has an invalid {name} SHA256")
    return value


def _absolute(value):
    return isinstance(value, str) and (
        Path(value).is_absolute() or PurePosixPath(value).is_absolute()
    )


def require_rgb_cache(metadata):
    if not isinstance(metadata, Mapping):
        raise ValueError("Require sealed native-RGB cache metadata")
    if (
        metadata.get("sealed_complete") is not True
        or metadata.get("schema") != CACHE_SCHEMA
    ):
        raise ValueError(
            "Require complete sealed observation cache v2; encoding/prepared caches cannot train"
        )
    if metadata.get("observation_color_contract") != COLOR_CONTRACT:
        raise ValueError(
            "Require native_robotwin_rgb_v1 cache; old/missing color contracts are forbidden"
        )
    variants = metadata.get("variants")
    if isinstance(variants, bool) or not isinstance(variants, int) or variants != 3:
        raise ValueError(
            "Native-RGB training requires exactly 3 variants: original plus 2 augmentations"
        )
    for key in ("manifest_sha256", "plan_sha256", "index_sha256"):
        _digest(metadata.get(key), key)
    reuse = metadata.get("text_reuse")
    binding = metadata.get("source_cache_identity")
    if reuse is None and binding is None:
        pass
    else:
        if (
            not isinstance(reuse, Mapping)
            or set(reuse) != TEXT_REUSE_FIELDS
            or not isinstance(binding, Mapping)
            or dict(reuse) != dict(binding)
        ):
            raise ValueError(
                "Require exact text_reuse provenance bound to authenticated plan.source_cache_identity"
            )
        if (
            reuse.get("schema") != TEXT_REUSE_SCHEMA
            or reuse.get("mode") != "hardlink_unchanged_shards"
            or reuse.get("verification") != TEXT_REUSE_VERIFICATION
        ):
            raise ValueError("Unsupported immutable T5 hard-link reuse contract")
        if not _absolute(reuse.get("source_cache")):
            raise ValueError("T5 source-cache provenance must be an absolute path")
        for key in (
            "source_manifest_sha256",
            "source_plan_sha256",
            "source_index_sha256",
            "source_prompts_sha256",
        ):
            _digest(reuse.get(key), key)
        size = reuse.get("source_index_size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("T5 source-index size must be a positive integer")
    result = {
        key: deepcopy(metadata[key])
        for key in (
            "schema",
            "sealed_complete",
            "observation_color_contract",
            "variants",
            "manifest_sha256",
            "plan_sha256",
            "index_sha256",
        )
    }
    if reuse is not None:
        result.update(
            text_reuse=deepcopy(reuse), source_cache_identity=deepcopy(binding)
        )
    if "root" in metadata:
        if not _absolute(metadata["root"]):
            raise ValueError("Cache root must be an absolute path")
        result["root"] = metadata["root"]
    return result


def _json_with_sha(root, name):
    path = root / name
    if path.is_symlink():
        raise ValueError(f"Cache metadata cannot be a symlink: {name}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(
            f"Native-RGB cache is not sealed: missing {name}; finish encoding/sealing first"
        ) from error
    if not resolved.is_relative_to(root):
        raise ValueError(f"Cache metadata escapes root: {name}")
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Cache metadata must be a regular file: {name}")
    with resolved.open("rb") as stream:
        raw = stream.read()
    after = resolved.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"Cache metadata changed during admission: {name}")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid sealed cache metadata JSON: {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Cache metadata must be an object: {name}")
    return value, hashlib.sha256(raw).hexdigest()


def sealed_rgb_cache(root):
    try:
        root = Path(root).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(
            "Native-RGB cache root does not exist; finish caching before training"
        ) from error
    if not root.is_dir() or (root / "failed.json").exists():
        raise ValueError("Native-RGB cache is not a successful sealed cache directory")
    complete, _ = _json_with_sha(root, "complete.json")
    if complete.get("complete") is not True or complete.get("schema") != CACHE_SCHEMA:
        raise ValueError(
            "Native-RGB cache complete.json is not complete observation cache v2"
        )
    manifest, manifest_sha = _json_with_sha(root, "manifest.json")
    if (
        manifest.get("schema") != CACHE_SCHEMA
        or complete.get("manifest_sha256") != manifest_sha
    ):
        raise ValueError(
            "Native-RGB cache manifest SHA/schema differs from completion seal"
        )
    if (
        manifest.get("observation_color_contract") != COLOR_CONTRACT
        or complete.get("observation_color_contract") != COLOR_CONTRACT
    ):
        raise ValueError(
            "Require native_robotwin_rgb_v1 in manifest and seal; old/missing color is forbidden"
        )
    if (
        manifest.get("variants") != 3
        or isinstance(manifest.get("variants"), bool)
        or complete.get("variants") != 3
    ):
        raise ValueError("Native-RGB cache must seal exactly 3 image variants")
    plan, plan_sha = _json_with_sha(root, "plan.json")
    if (
        plan.get("schema") != CACHE_SCHEMA
        or plan.get("observation_color_contract") != COLOR_CONTRACT
        or manifest.get("plan_sha256") != plan_sha
        or complete.get("plan_sha256") != plan_sha
    ):
        raise ValueError(
            "Native-RGB cache plan SHA/schema/color differs from sealed metadata"
        )
    if plan.get("text_reuse") != manifest.get("text_reuse") or complete.get(
        "text_reuse"
    ) != manifest.get("text_reuse"):
        raise ValueError(
            "T5 text reuse differs between authenticated plan, manifest and seal"
        )
    index = manifest.get("index")
    if not isinstance(index, dict) or index.get("path") != "index.sqlite":
        raise ValueError("Sealed native-RGB cache must declare index.sqlite")
    size = index.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("Sealed native-RGB index size must be positive")
    metadata = dict(
        root=str(root),
        schema=manifest["schema"],
        sealed_complete=True,
        variants=manifest["variants"],
        observation_color_contract=manifest["observation_color_contract"],
        manifest_sha256=manifest_sha,
        plan_sha256=plan_sha,
        index_sha256=index.get("sha256"),
        text_reuse=manifest.get("text_reuse"),
        source_cache_identity=plan.get("source_cache_identity"),
    )
    return require_rgb_cache(metadata)
