from contextlib import contextmanager

import torch
from torch.nn import functional as F


def linear_patch_projection(module, x):
    if (
        not isinstance(module, torch.nn.Conv3d)
        or module.groups != 1
        or module.padding != (0, 0, 0)
        or module.dilation != (1, 1, 1)
        or module.stride != module.kernel_size
        or module.padding_mode != "zeros"
    ):
        raise ValueError(
            "Linear patch projection requires unpadded, non-overlapping Conv3d"
        )
    if x.ndim != 5 or x.shape[1] != module.in_channels:
        raise ValueError("Invalid patch projection input")
    b, c, t, h, w = x.shape
    kt, kh, kw = module.kernel_size
    if t % kt or h % kh or w % kw:
        raise ValueError("Patch projection input must be divisible by kernel size")

    patches = x.reshape(b, c, t // kt, kt, h // kh, kh, w // kw, kw)
    patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(
        b, t // kt, h // kh, w // kw, -1
    )
    result = F.linear(patches, module.weight.flatten(1), module.bias)
    return result.permute(0, 4, 1, 2, 3)


@contextmanager
def projection_backend(backbone, backend):
    if backend not in ("linear", "conv3d"):
        raise ValueError(f"Unknown projection backend: {backend}")
    expert = backbone.video_expert
    modules = [expert.patch_embedding]
    restored = []
    try:
        for module in modules:
            if any(p.requires_grad for p in module.parameters()):
                raise ValueError(
                    "Projection replacement is limited to the frozen backbone"
                )
            restored.append(
                (module, "forward" in module.__dict__, module.__dict__.get("forward"))
            )
            if backend == "linear":
                module.forward = lambda x, m=module: linear_patch_projection(m, x)
            else:
                module.forward = lambda x, m=module: m._conv_forward(
                    x, m.weight, m.bias
                )
        yield
    finally:
        for module, had_override, original in reversed(restored):
            if had_override:
                module.forward = original
            else:
                delattr(module, "forward")
