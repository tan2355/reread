import threading
import time
from collections import OrderedDict

import torch


def text_cache_bytes(mib):
    if type(mib) is not int or mib < 0:
        raise ValueError("endpoint_text_cache_mib must be a nonnegative integer")
    return mib * 1024**2


def _gpu_tensors(tensors):
    return all(value.is_cuda for value in tensors)


def _function_identity(call):
    return id(getattr(call, "__func__", call))


def _tensor_identity(value):
    try:
        version = value._version
    except RuntimeError as error:
        raise ValueError(
            "Frozen T5 weights must support mutation versioning"
        ) from error
    return (id(value), str(value.device), str(value.dtype), tuple(value.shape), version)


def _frozen_identity(model):
    encoder = getattr(model, "text_encoder", None)
    tokenizer = getattr(model, "tokenizer", None)
    if (
        not isinstance(encoder, torch.nn.Module)
        or tokenizer is None
        or not callable(tokenizer)
    ):
        raise TypeError(
            "Prompt cache requires a frozen torch T5 module and callable tokenizer"
        )
    children = tuple(encoder.named_modules())
    if any(child.training for _, child in children):
        raise ValueError("Prompt cache requires T5 in eval mode")
    params = tuple(encoder.named_parameters())
    if any(value.requires_grad for _, value in params):
        raise ValueError("Prompt cache requires frozen T5 parameters")
    modules = tuple(
        (name, id(child), _function_identity(child.forward)) for name, child in children
    )
    tensors = tuple(
        (name, _tensor_identity(value))
        for name, value in (*params, *encoder.named_buffers())
    )

    tokenizer_metadata = tuple(
        (name, repr(getattr(tokenizer, name, None)))
        for name in ("name", "seq_len", "clean", "vocab_size")
    )
    return (
        modules,
        tensors,
        id(tokenizer),
        id(getattr(tokenizer, "tokenizer", None)),
        _function_identity(type(tokenizer).__call__),
        tokenizer_metadata,
    )


def _prompt_key(prompt):
    if type(prompt) is str:
        return ("str", (), (prompt,))
    if (
        type(prompt) not in (list, tuple)
        or not prompt
        or any(type(value) is not str for value in prompt)
    ):
        raise TypeError("Prompt cache accepts only a str or nonempty list/tuple of str")
    return (type(prompt).__name__, (len(prompt),), tuple(prompt))


def _execution_key(model):
    device = torch.device(model.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = model.torch_dtype
    if not isinstance(dtype, torch.dtype):
        raise TypeError("Frozen prompt model torch_dtype must be a torch.dtype")
    enabled = torch.is_autocast_enabled(device.type)
    autocast_dtype = torch.get_autocast_dtype(device.type) if enabled else None
    return str(device), dtype, enabled, autocast_dtype


class FrozenPromptCache:

    def __init__(self, model, *, max_bytes=256 * 1024**2):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("Prompt cache max_bytes must be a nonnegative integer")
        if not callable(getattr(model, "encode_prompt", None)):
            raise TypeError("Frozen model must expose encode_prompt")
        self.model = model
        self._original = model.encode_prompt
        self.max_bytes = max_bytes
        self._signature = _frozen_identity(model)
        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self.bytes = self.calls = self.hits = self.misses = self.evictions = (
            self.invalidations
        ) = 0
        self.encode_host_s = self.lookup_host_s = 0.0
        self._rollout_start = self._counters()

    def _counters(self):
        return dict(
            calls=self.calls,
            hits=self.hits,
            misses=self.misses,
            evictions=self.evictions,
            invalidations=self.invalidations,
            encode_host_s=self.encode_host_s,
            lookup_host_s=self.lookup_host_s,
        )

    @property
    def original_encode(self):
        return self._original

    def reset_rollout_stats(self):
        with self._lock:
            self._rollout_start = self._counters()

    def rollout_stats(self):
        with self._lock:
            return {
                "prompt_cache_" + name: value - self._rollout_start[name]
                for name, value in self._counters().items()
            }

    def stats(self):
        with self._lock:
            return dict(
                self._counters(),
                entries=len(self._entries),
                bytes=self.bytes,
                limit_bytes=self.max_bytes,
                gpu_only=True,
                persistent=False,
                clone_on_read=True,
                timing="host_elapsed_no_cuda_synchronization",
            )

    def clear(self):
        with self._lock:
            self._entries.clear()
            self.bytes = 0
            self._signature = _frozen_identity(self.model)

    def encode(self, prompt):
        prompt_key = _prompt_key(prompt)
        started = time.perf_counter()
        with self._lock:
            signature = _frozen_identity(self.model)
            if signature != self._signature:
                self._entries.clear()
                self.bytes = 0
                self._signature = signature
                self.invalidations += 1
            key = (prompt_key, _execution_key(self.model))
            self.calls += 1
            entry = self._entries.get(key)
            if entry is not None:
                self.hits += 1
                self._entries.move_to_end(key)
                tensors, size, ready = entry
                if ready is not None:
                    stream = torch.cuda.current_stream(tensors[0].device)
                    stream.wait_event(ready)
                    for value in tensors:
                        value.record_stream(stream)

                result = tuple(value.clone() for value in tensors)
                self.lookup_host_s += time.perf_counter() - started
                return result
            self.misses += 1
            self.lookup_host_s += time.perf_counter() - started
            encoded_started = time.perf_counter()
            result = self._original(prompt)
            self.encode_host_s += time.perf_counter() - encoded_started
            if (
                type(result) is not tuple
                or len(result) != 2
                or any(not isinstance(value, torch.Tensor) for value in result)
            ):
                raise TypeError(
                    "Frozen T5 must return the exact (context, context_mask) tensor pair"
                )
            context, mask = result
            batch = len(prompt_key[2])
            if (
                context.ndim != 3
                or context.shape[0] != batch
                or not context.is_floating_point()
                or mask.ndim != 2
                or mask.shape != context.shape[:2]
                or mask.dtype != torch.bool
                or context.device != mask.device
                or any(value.requires_grad for value in result)
            ):
                raise ValueError("Frozen T5 prompt encoding contract changed")
            size = sum(value.numel() * value.element_size() for value in result)
            if self.max_bytes and size <= self.max_bytes and _gpu_tensors(result):
                while self._entries and self.bytes + size > self.max_bytes:
                    _, (_, old_size, _) = self._entries.popitem(last=False)
                    self.bytes -= old_size
                    self.evictions += 1
                owned = tuple(value.detach().clone() for value in result)
                ready = None
                if owned[0].is_cuda:
                    ready = torch.cuda.Event()
                    ready.record(torch.cuda.current_stream(owned[0].device))
                self._entries[key] = owned, size, ready
                self.bytes += size
            return result
