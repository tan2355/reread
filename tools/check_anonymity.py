import argparse
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import tomllib
from check import PRIVATE_PATTERNS, ROOT, files

EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@(?:[\w-]+\.)+[A-Za-z]{2,}")
URL = re.compile(r"https?://[^\s<>\"')]+")
IDENTITY_KEYS = {
    "host",
    "hostname",
    "username",
    "user_name",
    "email",
    "gpu_uuid",
    "device_uuid",
}
ANONYMOUS_VALUES = {"", "anonymous", "compute-node", "redacted"}
NONVENDOR_HOSTS = {
    "arxiv.org",
    "www.apache.org",
    "cvpr.thecvf.com",
    "archive.ubuntu.com",
    "security.ubuntu.com",
    "deb.debian.org",
    "security.debian.org",
}
UPSTREAM_REPOSITORIES = ("/yuantianyuan01/FastWAM", "/RoboTwin-Platform/RoboTwin")


def is_example_host(host: str) -> bool:
    return host in {"localhost", "127.0.0.1"} or host.endswith((".invalid", ".test"))


def metadata_issues(value: object, location: str) -> list[str]:
    issues = []
    if isinstance(value, dict):
        for key, item in value.items():
            field = location + "/" + key
            if (
                key.lower() in IDENTITY_KEYS
                and isinstance(item, str)
                and item.lower() not in ANONYMOUS_VALUES
            ):
                issues.append(field + ": identifying metadata needs review")
            issues.extend(metadata_issues(item, field))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(metadata_issues(item, location + "/" + str(index)))
    return issues


def scan(root: Path) -> dict:
    failures = []
    external_hosts = set()
    checked = 0
    for path in files(root):
        relative = path.relative_to(root)
        name = relative.as_posix()
        vendor = "third_party" in relative.parts
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        checked += 1
        if any(pattern.search(text) for pattern in PRIVATE_PATTERNS):
            failures.append(name + ": private path, host or credential pattern")
        if vendor:
            continue
        for match in EMAIL.finditer(text):
            if not is_example_host(match.group().split("@", 1)[1]):
                failures.append(name + ": non-placeholder email address")
        for match in URL.finditer(text):
            try:
                parsed = urlsplit(match.group())
                host = parsed.hostname or ""
            except ValueError:
                failures.append(name + ": unclassified URL")
                continue
            if "tests" in relative.parts:
                continue
            external_hosts.add(host)
            upstream = host == "github.com" and any(
                parsed.path == prefix or parsed.path.startswith(prefix + "/")
                for prefix in UPSTREAM_REPOSITORIES
            )
            if not (host in NONVENDOR_HOSTS or upstream or is_example_host(host)):
                failures.append(name + ": unapproved external link")
        if path.suffix == ".json":
            failures.extend(metadata_issues(json.loads(text), name))
        elif path.suffix == ".xml":
            for element in ET.fromstring(text).iter():
                if element.get("hostname", "").lower() not in ANONYMOUS_VALUES:
                    failures.append(name + ": original host metadata")
        elif path.name == "pyproject.toml":
            project = tomllib.loads(text).get("project", {})
            if (
                project.get("authors")
                or project.get("maintainers")
                or project.get("urls")
            ):
                failures.append(name + ": author, maintainer or project URL metadata")
    return {
        "text_files_checked": checked,
        "failures": sorted(set(failures)),
        "nonvendor_external_hosts": sorted(external_hosts),
        "third_party_attributions_retained": True,
    }


def check_archive(path: Path) -> dict:
    failures = []
    with zipfile.ZipFile(path) as archive:
        if archive.comment:
            failures.append("Archive comment is not empty")
        for entry in archive.infolist():
            if entry.date_time != (1980, 1, 1, 0, 0, 0) or entry.comment or entry.extra:
                failures.append(entry.filename + ": non-neutral archive metadata")
            parts = Path(entry.filename).parts
            if any(
                part in {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}
                for part in parts
            ):
                failures.append(
                    entry.filename + ": local environment or history included"
                )
            if entry.filename.startswith(("/", "\\")) or ".." in parts:
                failures.append(entry.filename + ": non-relative archive path")
    return {"archive_metadata_failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the anonymous source snapshot and optional archive metadata."
    )
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    result = scan(ROOT)
    if args.archive:
        result.update(check_archive(args.archive))
    print(json.dumps(result, indent=2))
    raise SystemExit(
        int(bool(result["failures"] or result.get("archive_metadata_failures")))
    )


if __name__ == "__main__":
    main()
