import hashlib
from copy import deepcopy
from pathlib import Path

import pytest
from cache_fixtures import encoder_identity

from reread.data.provenance import (
    _SOURCE,
    _TOKENIZER,
    _WEIGHTS,
    validate_cached_encoder_source,
)


def identity_and_source(tmp_path):
    identity = encoder_identity()
    missing = {
        "path": "/deliberately/absent/encoder",
        "bytes": 12,
        "mtime_ns": 123,
        "sha256": "b" * 64,
    }
    identity["weights"] = {name: dict(missing) for name in _WEIGHTS}
    identity["tokenizer"] = {name: dict(missing) for name in _TOKENIZER}
    source = tmp_path / "src/fastwam/models/wan22"
    identity["source"] = {}
    for name in _SOURCE:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        identity["source"][name] = {
            "path": "/old/location/" + name,
            "bytes": path.stat().st_size,
            "mtime_ns": 123,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return identity, source


def test_cached_training_validation_never_reads_encoder_assets(tmp_path, monkeypatch):
    identity, _ = identity_and_source(tmp_path)
    original = deepcopy(identity)
    real_stat = Path.stat

    def guarded_stat(path, *args, **kwargs):
        assert "deliberately" not in str(path), "Encoder asset was accessed"
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    result = validate_cached_encoder_source(identity, tmp_path)
    assert len(result["source_sha256"]) == len(_SOURCE)
    assert (
        result["encoder_assets"] == "recorded_hashes_from_sealed_manifest_not_reopened"
    )
    assert identity == original


def test_changed_source_is_rejected(tmp_path):
    identity, source = identity_and_source(tmp_path)
    (source / _SOURCE[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source differs"):
        validate_cached_encoder_source(identity, tmp_path)


@pytest.mark.parametrize("group", ["weights", "tokenizer", "source"])
def test_incomplete_native_inventory_is_rejected(tmp_path, group):
    identity, _ = identity_and_source(tmp_path)
    identity[group].pop(next(iter(identity[group])))
    with pytest.raises(ValueError, match="inventory"):
        validate_cached_encoder_source(identity, tmp_path)


def test_different_conditioning_contract_is_rejected(tmp_path):
    identity, _ = identity_and_source(tmp_path)
    identity["execution"]["token_length"] = 512
    with pytest.raises(ValueError, match="execution differs"):
        validate_cached_encoder_source(identity, tmp_path)
