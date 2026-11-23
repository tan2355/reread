import json

from check import ROOT, files, sha256

if __name__ == "__main__":
    entries = [
        {
            "path": p.relative_to(ROOT).as_posix(),
            "size": p.stat().st_size,
            "sha256": sha256(p),
        }
        for p in files()
        if p.name != "manifest.json"
    ]
    (ROOT / "manifest.json").write_text(
        json.dumps({"schema": "reread-source-v1", "files": entries}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Recorded {len(entries)} source files.")
