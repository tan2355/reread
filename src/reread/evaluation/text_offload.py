import threading
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def gpu_text_lease(path, *, timeout=1800):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with path.open("a+b") as stream:
        acquired = False
        try:
            while not acquired:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    if time.monotonic() - started >= timeout:
                        raise TimeoutError(
                            "Timed out waiting for GPU-local temporary T5 residency"
                        )
                    time.sleep(0.02)
            yield
        finally:
            if acquired:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def text_lease_path(config):
    return (
        Path(config["output_root"])
        / "runtime_cache"
        / f"node_{config.get('node_rank', 0)}_gpu_{config['gpu_index']}_text_encoder.lock"
    )


class FrozenTextOffload:
    def __init__(
        self,
        model,
        original_encode,
        lock_path,
        *,
        torch_module=None,
        lease_factory=gpu_text_lease,
        allow_cpu_fixture=False,
    ):
        if torch_module is None:
            import torch as torch_module
        self.torch, self.model = torch_module, model
        self.encoder, self.original = model.text_encoder, original_encode
        self.device = torch_module.device(model.device)
        if self.device.type != "cuda" and not allow_cpu_fixture:
            raise ValueError(
                "T5 offload preserves the original CUDA encoder, not CPU encoding"
            )
        if not isinstance(self.encoder, torch_module.nn.Module):
            raise TypeError("Expected the frozen native torch T5 module")
        if any(m.training for m in self.encoder.modules()) or any(
            p.requires_grad for p in self.encoder.parameters()
        ):
            raise ValueError("Only an eval-mode, frozen text encoder may be offloaded")
        self.lock_path, self.lease_factory = Path(lock_path), lease_factory
        self._lock = threading.RLock()
        self.calls = self.wait_seconds = self.encode_seconds = self.transfer_seconds = 0
        self.cleanup_errors = []
        self._synchronize()

        self.encoder.to(device="cpu")
        self._masters, self._bindings = {}, []
        for owner in self.encoder.modules():
            for kind, values in (
                ("parameter", owner._parameters),
                ("buffer", owner._buffers),
            ):
                for name, tensor in values.items():
                    if tensor is None:
                        continue
                    if tensor.device.type != "cpu":
                        raise RuntimeError(
                            "Native T5 CPU residency was not established"
                        )
                    key = id(tensor)
                    self._masters.setdefault(key, (tensor, tensor.detach()))
                    self._bindings.append((owner, kind, name, key))
        self.weight_bytes = sum(
            master.numel() * master.element_size()
            for _, master in self._masters.values()
        )
        self._identity = self._cpu_identity()
        self._empty_cache()

    def _synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _empty_cache(self):
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

    def _cpu_identity(self):
        rows = []
        for owner, kind, name, key in self._bindings:
            values = owner._parameters if kind == "parameter" else owner._buffers
            value, master = values[name], self._masters[key][1]
            rows.append(
                (
                    id(value),
                    value._version,
                    str(value.device),
                    str(value.dtype),
                    tuple(value.shape),
                    value.data_ptr(),
                    master.data_ptr(),
                )
            )
        return tuple(rows)

    def _assert_cpu_identity(self):
        if self._cpu_identity() != self._identity:
            raise RuntimeError(
                "Frozen T5 CPU storage/precision/version changed between calls"
            )
        if any(m.training for m in self.encoder.modules()) or any(
            p.requires_grad for p in self.encoder.parameters()
        ):
            raise RuntimeError("Frozen T5 training/gradient state changed")

    def _stage(self):
        staged = {
            key: master.to(device=self.device, non_blocking=False)
            for key, (_, master) in self._masters.items()
        }
        for owner, kind, name, key in self._bindings:
            if kind == "parameter":
                owner._parameters[name].data = staged[key]
            else:
                owner._buffers[name] = staged[key]

    def _restore(self):
        for owner, kind, name, key in self._bindings:
            original, master = self._masters[key]
            if kind == "parameter":
                original.data = master
                owner._parameters[name] = original
            else:
                owner._buffers[name] = original
        self._empty_cache()

    def encode(self, prompt):
        with self._lock:
            self._assert_cpu_identity()
            started = time.perf_counter()
            with self.lease_factory(self.lock_path):
                self.wait_seconds += time.perf_counter() - started
                primary_error = None
                try:
                    self._synchronize()
                    started = time.perf_counter()
                    self._stage()
                    self.transfer_seconds += time.perf_counter() - started
                    started = time.perf_counter()
                    with self.torch.no_grad():
                        result = self.original(prompt)
                    self._synchronize()
                    self.encode_seconds += time.perf_counter() - started
                    self.calls += 1
                    return result
                except BaseException as error:
                    primary_error = error
                    raise
                finally:
                    failures = []
                    for name, cleanup in (
                        ("synchronize", self._synchronize),
                        ("restore_cpu", self._restore),
                        ("verify_cpu", self._assert_cpu_identity),
                    ):
                        try:
                            cleanup()
                        except Exception as error:
                            failures.append(
                                dict(
                                    stage=name,
                                    error_type=type(error).__name__,
                                    error=str(error),
                                )
                            )
                    self.cleanup_errors.extend(failures)
                    self.cleanup_errors = self.cleanup_errors[-16:]
                    if failures and primary_error is None:
                        raise RuntimeError(
                            "T5 residency cleanup failed: " + str(failures)
                        )
                    if (
                        failures
                        and primary_error is not None
                        and hasattr(primary_error, "add_note")
                    ):
                        primary_error.add_note(
                            "T5 cleanup diagnostics: " + str(failures)
                        )

    def stats(self):
        self._assert_cpu_identity()
        return dict(
            schema="newm-frozen-text-offload-v1",
            resident_device="cpu",
            compute_device=str(self.device),
            compute_precision_unchanged=True,
            cpu_weight_bytes=self.weight_bytes,
            cache_miss_calls=self.calls,
            lease_wait_seconds=self.wait_seconds,
            transfer_seconds=self.transfer_seconds,
            encode_seconds=self.encode_seconds,
            stable_cpu_storage=True,
            cleanup_errors=list(self.cleanup_errors),
        )


def install_text_encoder_offload(policy, config):
    if getattr(policy, "_text_encoder_offload", None) is not None:
        raise ValueError("Text encoder offload already installed")
    cache, model = policy.prompt_cache, policy.base.model
    lock = text_lease_path(config)
    offload = FrozenTextOffload(model, cache.original_encode, lock)
    cache._original = offload.encode
    cache.clear()
    policy._text_encoder_offload = offload
    return offload
