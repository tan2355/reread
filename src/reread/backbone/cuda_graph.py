import threading
from contextlib import contextmanager

import torch

from reread.backbone.projection_backend import projection_backend


def _method_identity(method):
    return getattr(method, "__func__", method)


def _tensor_metadata(value):
    return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)


_UNSUPPORTED_CONFIG = object()
_DIAGNOSTIC_CONFIG_NAMES = frozenset(
    {
        "training",
        "calls",
        "call_count",
        "call_counts",
        "forward_calls",
        "prepare_calls",
        "encode_calls",
        "replay_calls",
        "hits",
        "misses",
    }
)


def _primitive_config(value):
    if type(value) in (bool, int, float, str, type(None)):
        return type(value).__name__, value
    if isinstance(value, tuple):
        pieces = tuple(_primitive_config(item) for item in value)
        if all(piece is not _UNSUPPORTED_CONFIG for piece in pieces):
            return "tuple", pieces
    return _UNSUPPORTED_CONFIG


def _module_configuration(module):
    values = []
    for name, value in vars(module).items():
        if name.startswith("_") or name in _DIAGNOSTIC_CONFIG_NAMES:
            continue
        captured = _primitive_config(value)
        if captured is not _UNSUPPORTED_CONFIG:
            values.append((name, captured))
    return tuple(sorted(values))


def _model_signature(frozen):
    base = frozen.backbone
    if any(module.training for module in base.modules()):
        raise ValueError(
            "CUDA graph requires the entire frozen backbone in evaluation mode"
        )
    result = []
    for name, value in (*base.named_parameters(), *base.named_buffers()):
        if value.requires_grad:
            raise ValueError(
                "CUDA graph requires frozen backbone parameters and buffers"
            )
        if value.is_inference():
            raise ValueError("CUDA graph requires versioned frozen tensor storage")
        result.append(
            (
                name,
                id(value),
                value.data_ptr(),
                value._version,
                *_tensor_metadata(value),
            )
        )

    for expert_name in ("video_expert", "action_expert"):
        expert = getattr(base, expert_name)
        frequencies = getattr(expert, "freqs", ())
        if isinstance(frequencies, torch.Tensor):
            frequencies = (frequencies,)
        for index, value in enumerate(frequencies):
            if isinstance(value, torch.Tensor):
                if value.is_inference():
                    raise ValueError("CUDA graph requires versioned frequency storage")
                result.append(
                    (
                        expert_name,
                        index,
                        id(value),
                        value.data_ptr(),
                        value._version,
                        *_tensor_metadata(value),
                    )
                )
    methods = (
        base._append_proprio_to_context,
        base._build_mot_attention_mask,
        base._denoise_action_with_video_cache,
        base.video_expert.prepare,
        base.mot._build_expert_attention_io,
        base.mot.prefill_video_cache_tensor,
        base.infer_action_scheduler.build_inference_schedule,
        base.infer_action_scheduler.step,
    )

    optional = (
        (base.video_expert, "get_freqs"),
        (base.action_expert, "prepare"),
        (base.action_expert, "post"),
        (base.action_expert, "get_freqs"),
        (base.mot, "forward_action_with_video_cache_tensor"),
        (base.mot, "_split_modulation"),
        (base.mot, "_apply_expert_post_block_tensor"),
    )
    methods += tuple(
        getattr(owner, name)
        for owner, name in optional
        if callable(getattr(owner, name, None))
    )
    overrides = tuple(
        (name, id(module.__dict__["forward"]))
        for name, module in base.named_modules()
        if "forward" in module.__dict__
    )

    module_forwards = tuple(
        (name, _method_identity(module.forward))
        for name, module in base.named_modules()
    )

    convolution_forwards = tuple(
        (name, _method_identity(module._conv_forward))
        for name, module in base.named_modules()
        if isinstance(module, torch.nn.Conv3d)
    )

    module_configuration = tuple(
        (name, _module_configuration(module)) for name, module in base.named_modules()
    )
    scheduler = base.infer_action_scheduler

    scheduler_signature = (
        type(scheduler),
        id(scheduler),
        tuple(
            sorted(
                (name, value)
                for name, value in vars(scheduler).items()
                if not name.startswith("_")
                and isinstance(value, (int, float, bool, str, type(None)))
            )
        ),
        _method_identity(getattr(scheduler, "_phi", None)),
    )
    if any(
        module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
        for module in base.modules()
    ):
        raise ValueError("CUDA graph cannot replay external Python module hooks")
    if getattr(frozen, "unmasked_attention", False) or getattr(
        frozen, "timestep_cache", False
    ):
        raise ValueError(
            "CUDA graph prototype uses native attention/timestep kernels only"
        )
    execution_flags = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.get_float32_matmul_precision(),
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.is_autocast_enabled("cuda"),
        torch.get_autocast_dtype("cuda"),
    )
    return (
        tuple(result),
        tuple(_method_identity(m) for m in methods),
        overrides,
        module_forwards,
        convolution_forwards,
        module_configuration,
        scheduler_signature,
        execution_flags,
        frozen.config,
        frozen.projection,
        frozen.cache_conditioning,
        base.device,
        base.torch_dtype,
    )


def _validate_inputs(
    frozen, rgb_latents, proprio, context, context_mask
):
    base, config = frozen.backbone, frozen.config
    if (
        not isinstance(rgb_latents, torch.Tensor)
        or rgb_latents.ndim != 5
        or rgb_latents.shape[2] != 1
    ):
        raise ValueError("Only current RGB latent [B,C,1,H,W] is allowed")
    batch = rgb_latents.shape[0]
    patch = tuple(base.video_expert.patch_size)
    if (
        batch < 1
        or rgb_latents.shape[1] != base.video_expert.in_dim
        or len(patch) != 3
        or tuple(rgb_latents.shape[-2:])
        != tuple(config.video_grid[i] * patch[i + 1] for i in range(2))
    ):
        raise ValueError("Current RGB latent channels/grid violate the memory contract")
    if (
        not isinstance(proprio, torch.Tensor)
        or proprio.shape != (batch, base.proprio_dim)
        or not isinstance(context, torch.Tensor)
        or context.ndim != 3
        or context.shape[0] != batch
        or not isinstance(context_mask, torch.Tensor)
        or context_mask.shape != context.shape[:2]
        or context_mask.dtype != torch.bool
    ):
        raise ValueError(
            "Batched current proprio/context/mask shapes or dtypes do not match"
        )
    device = torch.device(base.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError(
            "Frozen inference CUDA graph requires an available CUDA device"
        )
    values = {
        "rgb_latents": rgb_latents,
        "proprio": proprio,
        "context": context,
        "context_mask": context_mask,
    }
    for name, value in values.items():
        if value.requires_grad:
            raise ValueError("Frozen inference graph inputs must be detached")

        if not value.is_contiguous():
            raise ValueError(
                f"Frozen inference graph requires canonical contiguous current inputs: {name}"
            )
        if name != "context_mask" and not value.is_floating_point():
            raise ValueError("Frozen inference graph inputs must be floating tensors")
        if name != "context_mask" and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite current input: {name}")
    return values


@torch.no_grad()
def _tensor_core(
    frozen, static, noise, timestep_video, attention_mask, timesteps, deltas
):
    base, config = frozen.backbone, frozen.config
    batch = static["rgb_latents"].shape[0]
    context, context_mask = base._append_proprio_to_context(
        static["context"], static["context_mask"], static["proprio"]
    )
    latents_action = noise.clone()
    with frozen._verified_grid():
        with projection_backend(base, frozen.projection):
            (
                video_tokens,
                _t_video,
                video_t_mod,
                video_context,
                video_context_mask,
                video_freqs,
                _f_video,
                _h_video,
                _w_video,
                tokens_per_frame,
            ) = base.video_expert.prepare(
                x=static["rgb_latents"],
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=bool(
                    getattr(base.video_expert, "fuse_vae_embedding_in_latents", False)
                ),
            )
        seq_len = video_tokens.shape[1]
        if (
            seq_len != config.video_grid[0] * config.video_grid[1]
            or tokens_per_frame != seq_len
        ):
            raise RuntimeError("Current-frame graph video grid changed")
        with frozen._conditioning_cache(), frozen._capture() as (saved, counts):
            video_cache_k, video_cache_v = base.mot.prefill_video_cache_tensor(
                video_tokens=video_tokens,
                video_freqs=video_freqs,
                video_t_mod=video_t_mod,
                video_context=video_context,
                video_context_mask=video_context_mask,
                video_attention_mask=attention_mask[:seq_len, :seq_len],
            )
            for step_t, step_delta in zip(timesteps, deltas):
                prediction = base._denoise_action_with_video_cache(
                    latents_action=latents_action,
                    timestep_action=step_t.expand(batch).to(
                        device=base.device, dtype=latents_action.dtype
                    ),
                    context=context,
                    context_mask=context_mask,
                    video_cache_k=video_cache_k,
                    video_cache_v=video_cache_v,
                    action_attention_mask=attention_mask[seq_len:, :],
                )
                latents_action = base.infer_action_scheduler.step(
                    prediction, step_delta, latents_action
                )
        for modality in ("video", "action"):
            expected = 1 if modality == "video" else config.inference_steps
            if any(
                counts.get((modality, layer)) != expected for layer in config.layers
            ):
                raise RuntimeError("Upstream graph inference call graph changed")
        features = {
            name: torch.stack(
                [
                    torch.cat(
                        [
                            saved[(modality, layer)][name]
                            for modality in ("video", "action")
                        ],
                        dim=1,
                    )
                    for layer in config.layers
                ],
                dim=1,
            ).detach()
            for name in ("k", "v")
        }
    return latents_action.float().detach(), features


class GraphedSpatialFastWAM:

    def __init__(self, frozen, *, warmup_steps=3):
        if (
            isinstance(warmup_steps, bool)
            or not isinstance(warmup_steps, int)
            or not 1 <= warmup_steps <= 5
        ):
            raise ValueError("Graph warmup must be a bounded 1..5 complete extractions")
        if not all(
            callable(getattr(frozen, name, None))
            for name in ("_verified_grid", "_capture", "_conditioning_cache")
        ):
            raise TypeError("Graph requires the current spatial frozen extractor")
        self.frozen = frozen
        self.warmup_steps = warmup_steps
        self._guard = threading.Lock()
        self._graph = None
        self._signature = None
        self._shape = None
        self._seed = None
        self._static = None
        self._outputs = None
        self._constants = None
        self._last_replay = None
        self._replays = self._builds = 0

    @property
    def backbone(self):
        return self.frozen.backbone

    @property
    def config(self):
        return self.frozen.config

    @property
    def graph_ready(self):
        return self._graph is not None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.frozen, name)

    @contextmanager
    def _locked(self):
        if not self._guard.acquire(blocking=False):
            raise RuntimeError(
                "One frozen CUDA graph cannot serve concurrent extraction"
            )
        if not self.frozen._lock.acquire(blocking=False):
            self._guard.release()
            raise RuntimeError(
                "One frozen extractor cannot serve concurrent graph/eager extraction"
            )
        try:
            yield
        finally:
            self.frozen._lock.release()
            self._guard.release()

    def _bind_checked_signature(self, values, seed, signature):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("A fixed integer inference seed is required")
        if self._seed is not None and self._seed != seed:
            raise ValueError(
                "Frozen CUDA graph inference seed is fixed for its lifetime"
            )
        shape = tuple((name, tuple(value.shape)) for name, value in values.items())
        self._seed = seed
        changed = (
            self._graph is None or signature != self._signature or shape != self._shape
        )
        return changed, signature, shape

    def _bind(self, values, seed):
        return self._bind_checked_signature(values, seed, _model_signature(self.frozen))

    def _copy(self, values):
        for name, target in self._static.items():
            target.copy_(values[name], non_blocking=True)

    def _clear_unlocked(self):
        if self._last_replay is not None:
            self._last_replay.synchronize()
        self._last_replay = None
        self._graph = self._static = self._outputs = self._constants = None
        self._signature = self._shape = None

    def _build(self, values, seed, signature, shape):
        self._clear_unlocked()
        base, config = self.backbone, self.config
        device, dtype = torch.device(base.device), base.torch_dtype
        batch = values["rgb_latents"].shape[0]
        caller_stream = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(caller_stream)
        try:
            with torch.cuda.device(device), torch.cuda.stream(stream):
                static = {
                    name: value.detach()
                    .to(
                        device=device,
                        dtype=torch.bool if name == "context_mask" else dtype,
                    )
                    .contiguous()
                    .clone()
                    for name, value in values.items()
                }
                generator = torch.Generator(device="cpu").manual_seed(seed)
                noise = (
                    torch.randn(
                        (1, config.horizon, config.action_dim),
                        generator=generator,
                        dtype=torch.float32,
                    )
                    .expand(batch, -1, -1)
                    .clone()
                    .to(device=device, dtype=dtype)
                )
                timestep_video = torch.zeros((batch,), dtype=dtype, device=device)
                seq_len = config.video_grid[0] * config.video_grid[1]
                mask = base._build_mot_attention_mask(
                    video_seq_len=seq_len,
                    action_seq_len=config.horizon,
                    video_tokens_per_frame=seq_len,
                    device=device,
                )
                timesteps, deltas = (
                    base.infer_action_scheduler.build_inference_schedule(
                        num_inference_steps=config.inference_steps,
                        device=device,
                        dtype=dtype,
                        shift_override=config.sigma_shift,
                    )
                )
                constants = (noise, timestep_video, mask, timesteps, deltas)
                for _ in range(self.warmup_steps):
                    _tensor_core(self.frozen, static, *constants)

            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, capture_error_mode="global"):
                outputs = _tensor_core(self.frozen, static, *constants)
            caller_stream.wait_stream(stream)
            self._static, self._constants, self._outputs = static, constants, outputs
            self._graph, self._signature, self._shape = graph, signature, shape
            self._builds += 1
        except BaseException:
            self._clear_unlocked()
            raise

    @torch.no_grad()
    def prepare_graph(
        self, *, rgb_latents, proprio, context, context_mask, seed=0
    ):
        values = _validate_inputs(
            self.frozen, rgb_latents, proprio, context, context_mask
        )
        with self._locked():
            changed, signature, shape = self._bind(values, seed)
            if changed:
                self._build(values, seed, signature, shape)
        return self

    @torch.no_grad()
    def extract_batch(
        self, *, rgb_latents, proprio, context, context_mask, seed=0
    ):
        values = _validate_inputs(
            self.frozen, rgb_latents, proprio, context, context_mask
        )
        with self._locked():
            return self._replay_prepared_locked(
                values, seed, _model_signature(self.frozen)
            )

    @torch.no_grad()
    def _extract_batch_checked_signature(
        self,
        current_signature,
        *,
        rgb_latents,
        proprio,
        context,
        context_mask,
        seed=0,
    ):
        values = _validate_inputs(
            self.frozen, rgb_latents, proprio, context, context_mask
        )
        with self._locked():
            return self._replay_prepared_locked(values, seed, current_signature)

    def _replay_prepared_locked(self, values, seed, current_signature):
        changed, _, _ = self._bind_checked_signature(values, seed, current_signature)
        if changed:
            raise RuntimeError(
                "Frozen CUDA graph is not prepared for the current shape/storage/runtime; "
                "call prepare_graph explicitly while GPU producers are stopped"
            )
        stream = torch.cuda.current_stream(torch.device(self.backbone.device))
        if self._last_replay is not None:
            stream.wait_event(self._last_replay)
        self._copy(values)
        self._graph.replay()
        self._replays += 1
        actions, features = self._outputs
        owned = (
            actions.detach().clone(),
            {name: value.detach().clone() for name, value in features.items()},
        )
        self._last_replay = torch.cuda.Event()
        self._last_replay.record(stream)
        return owned

    def clear(self):
        with self._locked():
            self._clear_unlocked()

    def stats(self):
        return {
            "builds": self._builds,
            "replays": self._replays,
            "built": self._graph is not None,
            "seed": self._seed,
        }
