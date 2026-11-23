from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

MODES = ("native", "channels_last_3d", "single_frame_2d")
VAE_CONTROLLER_ATTR = "_reread_current_vae_controller"


def _binding(parameter):
    if parameter is None:
        return None
    return (
        id(parameter),
        parameter.data_ptr(),
        parameter._version,
        parameter.device,
        parameter.dtype,
        tuple(parameter.shape),
        tuple(parameter.stride()),
    )


def _causal_spec(layer):
    if not isinstance(layer, nn.Conv3d) or type(layer).__name__ != "CausalConv3d":
        raise ValueError("single_frame_2d requires native CausalConv3d modules")
    padding = getattr(layer, "_padding", None)
    if (
        not isinstance(padding, (tuple, list))
        or len(padding) != 6
        or tuple(layer.padding) != (0, 0, 0)
        or layer.stride[0] != 1
        or layer.dilation[0] != 1
        or padding[4] != layer.kernel_size[0] - 1
        or padding[5] != 0
        or padding[0] != padding[1]
        or padding[2] != padding[3]
        or layer.padding_mode != "zeros"
    ):
        raise ValueError("Unsupported causal Conv3D temporal/spatial padding contract")
    return (padding[2], padding[0])


class CurrentFrameVAEExecution:

    def __init__(self, model, mode):
        self.model = model
        self.mode = mode
        self._forward_overrides = []
        self._weight_storage = []
        self._encode_override = None
        self._restored = False
        if mode == "native":
            return
        roots = (model.encoder, model.conv1)
        if any(child.training for root in roots for child in root.modules()) or any(
            parameter.requires_grad for root in roots for parameter in root.parameters()
        ):
            raise ValueError(
                "VAE execution optimization requires frozen eval encoder roots"
            )
        convolutions = [
            child
            for root in roots
            for child in root.modules()
            if isinstance(child, nn.Conv3d)
        ]

        specs = []
        if mode == "single_frame_2d":
            for child in convolutions:
                try:
                    padding = _causal_spec(child)
                except ValueError:
                    continue
                specs.append((child, padding))
        self.optimized_convolution_count = len(specs)
        self.native_convolution_count = len(convolutions) - len(specs)
        if mode == "single_frame_2d" and any(
            not bool(torch.isfinite(parameter).all())
            for layer, _ in specs
            for parameter in (layer.weight, layer.bias)
            if parameter is not None
        ):
            raise ValueError(
                "Single-frame specialization requires finite frozen convolution weights"
            )
        if not callable(getattr(model, "encode", None)):
            raise ValueError("Expected the native VAE tensor model with encode()")
        original_encode = model.encode
        had_encode = "encode" in model.__dict__
        self._encode_override = (had_encode, model.__dict__.get("encode"))

        def current_encode(_model, *args, **kwargs):
            pixels = args[0] if args else kwargs.get("x")
            if (
                not isinstance(pixels, torch.Tensor)
                or pixels.ndim != 5
                or pixels.shape[2] != 1
            ):
                raise ValueError(
                    "Current-frame VAE optimization accepts only [B,C,1,H,W]"
                )
            if any(child.training for root in roots for child in root.modules()) or any(
                parameter.requires_grad
                for root in roots
                for parameter in root.parameters()
            ):
                raise RuntimeError(
                    "The optimized VAE encoder must remain frozen and in eval mode"
                )

            prepared = self.prepare_input(pixels)
            if args:
                args = (prepared, *args[1:])
            else:
                kwargs = dict(kwargs, x=prepared)
            result = original_encode(*args, **kwargs)

            return result.contiguous()

        model.encode = MethodType(current_encode, model)
        if mode == "channels_last_3d":
            for layer in convolutions:
                self._weight_storage.append((layer, layer.weight.detach()))
                with torch.no_grad():
                    layer.weight.data = layer.weight.detach().clone(
                        memory_format=torch.channels_last_3d
                    )
        else:
            for layer, padding in specs:
                had_forward = "forward" in layer.__dict__
                self._forward_overrides.append(
                    (layer, had_forward, layer.__dict__.get("forward"))
                )

                packed_weight = layer.weight.detach()[:, :, -1].contiguous()
                weight_binding, bias_binding = (
                    _binding(layer.weight),
                    _binding(layer.bias),
                )

                def current_conv(
                    _layer,
                    x,
                    cache_x=None,
                    *,
                    _weight=packed_weight,
                    _weight_binding=weight_binding,
                    _bias_binding=bias_binding,
                    _padding=padding,
                ):
                    if x.ndim != 5 or x.shape[2] != 1 or cache_x is not None:
                        raise ValueError(
                            "Single-frame Conv2D cannot consume temporal history or clips"
                        )
                    if (
                        _binding(_layer.weight) != _weight_binding
                        or _binding(_layer.bias) != _bias_binding
                    ):
                        raise RuntimeError(
                            "VAE weights/storage changed; restore and reprepare the encoder"
                        )
                    output = F.conv2d(
                        x[:, :, 0],
                        _weight,
                        _layer.bias,
                        stride=_layer.stride[1:],
                        padding=_padding,
                        dilation=_layer.dilation[1:],
                        groups=_layer.groups,
                    )
                    return output.unsqueeze(2)

                layer.forward = MethodType(current_conv, layer)

    def prepare_input(self, pixels):
        if self.mode != "native" and (pixels.ndim != 5 or pixels.shape[2] != 1):
            raise ValueError(
                "Current-frame encoder input must have temporal length one"
            )
        if self.mode == "channels_last_3d":
            return pixels.contiguous(memory_format=torch.channels_last_3d)
        return pixels

    def restore(self):
        if self._restored:
            return
        for layer, existed, previous in self._forward_overrides:
            if existed:
                layer.forward = previous
            else:
                delattr(layer, "forward")
        for layer, original_storage in self._weight_storage:
            with torch.no_grad():
                layer.weight.data = original_storage
        if self._encode_override is not None:
            existed, previous = self._encode_override
            if existed:
                self.model.encode = previous
            else:
                delattr(self.model, "encode")
        if getattr(self.model, VAE_CONTROLLER_ATTR, None) is self:
            delattr(self.model, VAE_CONTROLLER_ATTR)
        self._restored = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.restore()


def prepare_current_vae(model, *, mode="native"):
    if mode not in MODES:
        raise ValueError(f"Unknown current VAE mode: {mode!r}; expected one of {MODES}")
    existing = getattr(model, VAE_CONTROLLER_ATTR, None)
    if existing is not None:
        if existing.mode == mode and not existing._restored:
            return existing
        existing.restore()
    controller = CurrentFrameVAEExecution(model, mode)
    setattr(model, VAE_CONTROLLER_ATTR, controller)
    return controller
