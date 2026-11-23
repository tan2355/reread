import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATTERNS = (
    re.compile(r"/(?:mnt/cfs|root/workspace|home/[^/\s]+)(?:/|\b)", re.I),
    re.compile(r"\b[a-z][a-z0-9]*-dev-\d+\b|[A-Z]:[\\/]Users[\\/]", re.I),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{25,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)


def files(root=ROOT):
    excluded = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "build",
        "dist",
    }
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and not excluded.intersection(path.relative_to(root).parts):
            if not any(
                part.endswith(".egg-info") for part in path.relative_to(root).parts
            ):
                yield path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root=ROOT):
    root = Path(root)
    failures = []
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = {entry["path"]: entry for entry in manifest["files"]}
    actual = {
        p.relative_to(root).as_posix(): p
        for p in files(root)
        if p.name != "manifest.json"
    }
    if set(expected) != set(actual):
        failures.append("File inventory differs from manifest.json")
    for name, path in actual.items():
        entry = expected.get(name)
        if entry and (
            path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]
        ):
            failures.append(name + ": SHA256/size mismatch")
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=name)
            except (SyntaxError, UnicodeError) as error:
                failures.append(name + ": " + str(error))
    return {"files_checked": len(actual), "failures": failures}


if __name__ == "__main__":
    result = check()
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["failures"]))
