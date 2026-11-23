from contextlib import contextmanager

import torch

import reread.backbone.observations as observation_train
from reread.data.prefetch import AsyncEncodedBatches


class _OwnedSource:

    def __init__(self, source):
        self.source = source
        self.iterator = iter(source)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        return next(self.iterator)

    def close(self):
        if self.closed:
            return
        self.closed = True
        first_error = None
        closed = set()
        for value in (self.iterator, self.source):
            if id(value) in closed:
                continue
            closed.add(id(value))
            close = getattr(value, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        if first_error is not None:
            raise first_error


def _cached_only(samples):
    if (
        not isinstance(samples, (list, tuple))
        or not samples
        or not all(isinstance(row, dict) and "cached_encoded" in row for row in samples)
    ):
        raise ValueError(
            "Frozen-compute lookahead requires cached-only observation batches"
        )


@contextmanager
def feature_batches(
    source,
    frozen,
    *,
    inference_seed=0,
    prefetch=True,
    prefetch_frozen=False,
    extract=observation_train.extract_current_batch,
):
    owned = _OwnedSource(source)
    first = first_encoded = None
    try:
        if not prefetch_frozen:
            with observation_train._feature_batches(
                owned,
                frozen,
                extract=extract,
                inference_seed=inference_seed,
                preencode_next=prefetch,
            ) as batches:
                yield batches
            return
        if not prefetch:
            raise ValueError(
                "Frozen-compute lookahead requires observation prefetch enabled"
            )
        if extract is not observation_train.extract_current_batch:
            raise ValueError(
                "Frozen-compute lookahead requires the canonical frozen extractor"
            )
        device = torch.device(frozen.backbone.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError(
                "Frozen-compute lookahead supports CPU or explicit rank-local CUDA"
            )
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        pending = False
        prepare = getattr(frozen, "prepare_graph", None)
        if callable(prepare) and not frozen.graph_ready:
            try:
                first = next(owned)
            except StopIteration:
                yield iter(())
                return
            _cached_only(first)

            with torch.no_grad(), torch.autocast(device.type, enabled=False):
                first_encoded = observation_train.encode_current_batch(frozen, first)
                prepare(**first_encoded, seed=inference_seed)
            pending = True

        class PreparedSource:
            def __iter__(self):
                return self

            def __next__(self):
                nonlocal pending
                if pending:
                    pending = False
                    return first
                return next(owned)

            def close(self):
                owned.close()

        first_encoding_pending = first_encoded is not None

        def encode_and_extract(samples):
            nonlocal first_encoding_pending, first, first_encoded
            _cached_only(samples)
            with torch.no_grad(), torch.autocast(device.type, enabled=False):
                if first_encoding_pending and samples is first:
                    encoded = first_encoded
                    first_encoding_pending = False
                    first, first_encoded = None, None
                else:
                    encoded = observation_train.encode_current_batch(frozen, samples)
                actions, features = observation_train.extract_encoded_batch(
                    frozen, encoded, seed=inference_seed
                )
                return dict(actions=actions, k=features["k"], v=features["v"])

        with AsyncEncodedBatches(
            PreparedSource(), encode_and_extract, device=device
        ) as batches:
            yield (
                (samples, (value["actions"], {"k": value["k"], "v": value["v"]}))
                for samples, value in batches
            )
    except BaseException:
        try:
            owned.close()
        except BaseException:
            pass
        raise
    finally:
        first = first_encoded = None
        owned.close()
