import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_source_inventory_prunes_installed_environments(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "public_source_check", ROOT / "tools/check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (".venv", "venv", ".pytest_cache", "package.egg-info"):
        directory = tmp_path / name / "nested"
        directory.mkdir(parents=True)
        (directory / "private.key").write_text("not a project source file")
        (directory / "invalid.py").write_text("invalid Python syntax !")
    (tmp_path / "source.py").write_text("value = 1\n")
    (tmp_path / "README.md").write_text("Source files\n")
    assert [
        path.relative_to(tmp_path).as_posix() for path in module.files(tmp_path)
    ] == ["README.md", "source.py"]
    entries = [
        {
            "path": p.relative_to(tmp_path).as_posix(),
            "size": p.stat().st_size,
            "sha256": module.sha256(p),
        }
        for p in module.files(tmp_path)
    ]
    (tmp_path / "manifest.json").write_text(json.dumps({"files": entries}))
    assert module.check(tmp_path) == {"files_checked": 2, "failures": []}
