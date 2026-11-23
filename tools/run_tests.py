import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
env["PYTHONPATH"] = os.pathsep.join(
    str(ROOT / p)
    for p in ("src", "scripts", "third_party/fastwam/src", "experiments/robotwin")
)
failed = []
for path in sorted((ROOT / "tests").glob("test_*.py")):
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(path),
        ],
        cwd=ROOT,
        env=env,
    )
    if result.returncode:
        failed.append(path.name)
if failed:
    print("Failed files:", ", ".join(failed))
raise SystemExit(bool(failed))
