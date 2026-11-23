from __future__ import annotations

import re
import sys
import uuid


def canonical_cuda_uuid(value) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text[:4].upper() == "GPU-":
            text = text[4:]
        if not re.fullmatch(
            r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", text
        ):
            raise ValueError("CUDA device identity must be a full canonical GPU UUID")
        parsed = uuid.UUID(text)
    else:
        raw = getattr(value, "bytes", None)
        try:
            if raw is None or isinstance(raw, int):
                raise ValueError("Missing sixteen-byte CUDA device UUID")
            raw = bytes(raw)
            if len(raw) != 16:
                raise ValueError("CUDA device UUID must contain sixteen bytes")
            parsed = uuid.UUID(bytes=raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Invalid sixteen-byte CUDA device UUID") from error
    return "GPU-" + str(parsed)


def probe_runtime() -> dict:
    import torch

    torch.cuda.init()
    devices = []
    for ordinal in range(torch.cuda.device_count()):
        torch.cuda.set_device(ordinal)
        properties = torch.cuda.get_device_properties(ordinal)
        free, total = torch.cuda.mem_get_info(ordinal)
        devices.append(
            dict(
                ordinal=ordinal,
                uuid=canonical_cuda_uuid(properties.uuid),
                name=properties.name,
                total_memory=int(properties.total_memory),
                memory_free=int(free),
                memory_total=int(total),
            )
        )
    return dict(
        event="cache_cuda_preflight",
        initialized=True,
        devices=devices,
        python=sys.executable,
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
    )
