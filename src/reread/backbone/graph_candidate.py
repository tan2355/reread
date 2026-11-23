import time

import torch

from reread.backbone.cuda_graph import (
    GraphedSpatialFastWAM,
    _model_signature,
    _validate_inputs,
)


class _CurrentModuleInventory:

    def __init__(self, backbone):
        self._backbone = backbone
        self._named_modules = tuple(backbone.named_modules())

    def __getattr__(self, name):
        return getattr(self._backbone, name)

    def modules(self):
        return (module for _, module in self._named_modules)

    def named_modules(self):
        return iter(self._named_modules)


class _SignatureView:
    def __init__(self, frozen):
        self._frozen = frozen
        self.backbone = _CurrentModuleInventory(frozen.backbone)

    def __getattr__(self, name):
        return getattr(self._frozen, name)


def one_pass_model_signature(frozen):
    return _model_signature(_SignatureView(frozen))


class OnePassSignatureGraph(GraphedSpatialFastWAM):

    def __init__(self, frozen, *, warmup_steps=3):
        super().__init__(frozen, warmup_steps=warmup_steps)
        self._signature_calls = 0
        self._signature_seconds = 0.0

    def _current_signature(self):
        started = time.perf_counter()
        try:
            return one_pass_model_signature(self.frozen)
        finally:
            self._signature_calls += 1
            self._signature_seconds += time.perf_counter() - started

    def _bind(self, values, seed):
        return self._bind_checked_signature(values, seed, self._current_signature())

    @torch.no_grad()
    def extract_batch(
        self, *, rgb_latents, proprio, context, context_mask, seed=0
    ):
        values = _validate_inputs(
            self.frozen, rgb_latents, proprio, context, context_mask
        )
        with self._locked():
            return self._replay_prepared_locked(values, seed, self._current_signature())

    def stats(self):
        return super().stats() | {
            "signature_scanner": "fresh_one_pass_module_inventory",
            "signature_calls": self._signature_calls,
            "signature_cpu_seconds": self._signature_seconds,
        }


def _compare_outputs(reference, actual):
    if (
        not isinstance(reference, tuple)
        or not isinstance(actual, tuple)
        or len(reference) != 2
        or len(actual) != 2
        or not isinstance(reference[1], dict)
        or not isinstance(actual[1], dict)
        or set(reference[1]) != {"k", "v"}
        or set(actual[1]) != {"k", "v"}
    ):
        raise RuntimeError(
            "Graph numerical check requires actions and complete spatial K/V"
        )
    result = {}
    for name, expected, observed in (
        ("actions", reference[0], actual[0]),
        ("k", reference[1]["k"], actual[1]["k"]),
        ("v", reference[1]["v"], actual[1]["v"]),
    ):
        if (
            not isinstance(expected, torch.Tensor)
            or not isinstance(observed, torch.Tensor)
            or expected.shape != observed.shape
            or expected.dtype != observed.dtype
            or expected.device != observed.device
            or expected.requires_grad
            or observed.requires_grad
        ):
            raise RuntimeError("Graph numerical tensor contract changed: " + name)
        if not bool(torch.isfinite(expected).all()) or not bool(
            torch.isfinite(observed).all()
        ):
            raise RuntimeError("Graph numerical check found nonfinite output: " + name)

        if not torch.equal(
            expected.contiguous().view(torch.uint8),
            observed.contiguous().view(torch.uint8),
        ):
            raise RuntimeError("Graph output is not bitwise equal to eager: " + name)
        result[name] = {
            "bitwise_equal": True,
            "shape": list(expected.shape),
            "dtype": str(expected.dtype),
        }
    return result


@torch.no_grad()
def verify_graph_output(frozen, graph, encoded, *, seed=0):
    if graph.frozen is not frozen:
        raise ValueError("Numerical comparison requires the same frozen instance")
    reference = frozen.extract_batch(**encoded, seed=seed)
    actual = graph.extract_batch(**encoded, seed=seed)
    return _compare_outputs(reference, actual)


def prepare_verified_graph(
    frozen, encoded, *, seed=0, warmup_steps=3, one_pass_signature=True
):
    if type(one_pass_signature) is not bool:
        raise TypeError("one_pass_signature must be an explicit bool")
    wrapper = OnePassSignatureGraph if one_pass_signature else GraphedSpatialFastWAM
    graph = wrapper(frozen, warmup_steps=warmup_steps)
    started = time.perf_counter()
    try:
        graph.prepare_graph(**encoded, seed=seed)
        numerical = verify_graph_output(frozen, graph, encoded, seed=seed)
        report = {
            "scope": "single_input_exact_comparison_not_full_dataset_proof",
            "batch_size": int(encoded["rgb_latents"].shape[0]),
            "inference_steps": frozen.config.inference_steps,
            "preparation_and_validation_wall_seconds": time.perf_counter() - started,
            "numerical": numerical,
            "graph": graph.stats(),
        }
        return graph, report
    except BaseException:
        graph.clear()
        raise
