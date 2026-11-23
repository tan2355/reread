import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts", ROOT / "third_party/fastwam/src"):
    sys.path.insert(0, str(directory))


@pytest.fixture(scope="session")
def gloo_device():
    import torch.distributed as dist

    try:
        dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")
    except RuntimeError as error:
        if sys.platform == "win32" and "unsupported gloo device" in str(error):
            pytest.skip("This Windows PyTorch build cannot create a Gloo device")
        raise
