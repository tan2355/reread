import hashlib
import json
from pathlib import Path

from reread.data.cache_contract import matches_encoder_source
from reread.data.dataset import _validate_encoder

_SOURCE = (
    "wan_video_vae.py",
    "wan_video_text_encoder.py",
    "helpers/loader.py",
    "helpers/io.py",
    "helpers/state_dict_converters.py",
)
_WEIGHTS = ("Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth")
_TOKENIZER = (
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


def validate_cached_encoder_source(identity, upstream):
    _validate_encoder(identity)
    for group, names in (
        ("source", _SOURCE),
        ("weights", _WEIGHTS),
        ("tokenizer", _TOKENIZER),
    ):
        if set(identity[group]) != set(names):
            raise ValueError(f"Unexpected native cache encoder {group} inventory")
    root = Path(upstream).resolve(strict=True)
    source = root / "src/fastwam/models/wan22"
    verified = {}
    for name in _SOURCE:
        path = (source / name).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Encoder source escapes selected upstream checkout")
        before = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        after = path.stat()
        expected = identity["source"][name]
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Encoder source changed during validation")
        if not matches_encoder_source(expected, before.st_size, digest):
            raise ValueError(
                f"Cached encoder source differs from selected upstream: {name}"
            )
        verified[name] = digest
    return {
        "schema": "newm-memory3-cached-input-validation-v1",
        "encoder_identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_sha256": verified,
        "encoder_assets": "recorded_hashes_from_sealed_manifest_not_reopened",
    }
