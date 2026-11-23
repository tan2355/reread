import argparse
import zipfile
from pathlib import Path

from check import ROOT, check, files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = check()
    if result["failures"]:
        raise SystemExit("\n".join(result["failures"]))
    with zipfile.ZipFile(
        args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files():
            info = zipfile.ZipInfo(
                "reread/" + path.relative_to(ROOT).as_posix(),
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    print(args.output)
