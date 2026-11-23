import re
import stat
from collections.abc import Mapping
from pathlib import Path

CERTIFICATE_SCHEMA = "newm-endpoint-local-file-certificate-v1"
CORRECTOR_CERTIFICATE_SCHEMA = "newm-endpoint-local-corrector-certificate-v1"
_FIELDS = {"path", "size", "mtime_ns", "ctime_ns", "dev", "ino"}
_PROVENANCE = {"base": "base_sha256", "stats": "stats_sha256"}


def file_signature(path):
    resolved = Path(path).resolve(strict=True)
    identity = resolved.stat()
    if not stat.S_ISREG(identity.st_mode):
        raise ValueError("Preverified model inputs must be regular files")
    return dict(
        path=str(resolved),
        size=identity.st_size,
        mtime_ns=identity.st_mtime_ns,
        ctime_ns=identity.st_ctime_ns,
        dev=identity.st_dev,
        ino=identity.st_ino,
    )


def _signature(value):
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError("Invalid local file signature fields")
    if (
        type(value["path"]) is not str
        or not Path(value["path"]).is_absolute()
        or any(type(value[name]) is not int for name in _FIELDS - {"path"})
        or value["size"] < 0
    ):
        raise ValueError("Invalid local file signature values")
    return dict(value)


def _provenance_sha(provenance, key):
    value = provenance.get(key) if isinstance(provenance, Mapping) else None
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("Missing certified checkpoint file SHA: " + key)
    return value


def issue_file_certificate(before, after, provenance):
    for value in (before, after):
        if not isinstance(value, Mapping) or set(value) != set(_PROVENANCE):
            raise ValueError("Both base and stats signatures are required")
    result = dict(schema=CERTIFICATE_SCHEMA)
    for name, key in _PROVENANCE.items():
        first, last = _signature(before[name]), _signature(after[name])
        if first != last:
            raise ValueError(
                "Model input changed during coordinator verification: " + name
            )
        result[name] = dict(last, sha256=_provenance_sha(provenance, key))
    return result


def validate_file_certificate(certificate, *, base_path, stats_path, provenance):
    if (
        not isinstance(certificate, Mapping)
        or set(certificate) != {"schema", "base", "stats"}
        or certificate["schema"] != CERTIFICATE_SCHEMA
    ):
        raise ValueError("Invalid node-local preverified file certificate")
    for name, path in (("base", base_path), ("stats", stats_path)):
        record = certificate[name]
        if not isinstance(record, Mapping) or set(record) != _FIELDS | {"sha256"}:
            raise ValueError("Invalid certified file fields: " + name)
        certified = _signature({key: record[key] for key in _FIELDS})
        if record["sha256"] != _provenance_sha(provenance, _PROVENANCE[name]):
            raise ValueError(
                "Certified SHA differs from loaded checkpoint provenance: " + name
            )
        if file_signature(path) != certified:
            raise ValueError(
                "Preverified model input changed or has a different path: " + name
            )


def issue_corrector_certificate(before, after, sha256):
    first, last = _signature(before), _signature(after)
    if first != last:
        raise ValueError("Corrector changed during coordinator verification")
    if type(sha256) is not str or re.fullmatch("[0-9a-f]{64}", sha256) is None:
        raise ValueError("Invalid expected corrector SHA256")
    return dict(schema=CORRECTOR_CERTIFICATE_SCHEMA, signature=last, sha256=sha256)


def validate_corrector_certificate(certificate, *, path, expected_sha256):
    if (
        not isinstance(certificate, Mapping)
        or set(certificate) != {"schema", "signature", "sha256"}
        or certificate["schema"] != CORRECTOR_CERTIFICATE_SCHEMA
    ):
        raise ValueError("Invalid local corrector certificate")
    if (
        type(expected_sha256) is not str
        or re.fullmatch("[0-9a-f]{64}", expected_sha256) is None
        or certificate["sha256"] != expected_sha256
    ):
        raise ValueError("Corrector certificate differs from shared checkpoint SHA256")
    if file_signature(path) != _signature(certificate["signature"]):
        raise ValueError(
            "Corrector changed or has a different path after SHA256 verification"
        )
