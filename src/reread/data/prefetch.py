import queue
import threading
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable, Iterable, Mapping

import torch


@dataclass
class _Batch:
    samples: Any
    encoded: Mapping[str, Any]
    ready: Any = None


_END = object()


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)


def _validate(encoded):
    if not isinstance(encoded, Mapping):
        raise TypeError("Preprocessing must return an encoded mapping")
    if any(tensor.requires_grad for tensor in _tensors(encoded)):
        raise ValueError("Preprocessing outputs must be detached frozen features")
    return encoded


class AsyncEncodedBatches:

    def __init__(
        self, source_batches: Iterable, encode_callable: Callable, device, capacity=1
    ):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity != 1:
            raise ValueError("Exactly one encoded GPU lookahead batch is supported")
        if not callable(encode_callable):
            raise TypeError("encode_callable must be callable")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Preprocessing supports CPU or CUDA devices only")
        self._cuda = self.device.type == "cuda"
        if self._cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA preprocessing requested but CUDA is unavailable")
        if self._cuda and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._source = source_batches
        self._encode = encode_callable
        self._iterator = None
        self._queue = queue.Queue(maxsize=capacity)
        self._permit = threading.Semaphore(capacity)
        self._stop = threading.Event()
        self._thread = None
        self._stream = None
        self._closed = False
        self._failure: tuple[BaseException, TracebackType | None] | None = None
        self._failure_reported = False

        self._autocast_enabled = torch.is_autocast_enabled(self.device.type)
        self._autocast_dtype = torch.get_autocast_dtype(self.device.type)

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(raise_errors=exc_type is None)
        return False

    def _remember_failure(self, error):
        if self._failure is None:
            self._failure = (error, error.__traceback__)

    def _raise_failure(self):
        if self._failure is not None and not self._failure_reported:
            error, traceback = self._failure
            self._failure_reported = True
            raise error.with_traceback(traceback)

    def _close_source(self):
        closed_ids = set()
        for item in (self._iterator, self._source):
            if item is None or id(item) in closed_ids:
                continue
            closed_ids.add(id(item))
            close = getattr(item, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException as error:
                    self._remember_failure(error)

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _encode_batch(self, samples):
        with (
            torch.no_grad(),
            torch.autocast(
                self.device.type,
                enabled=self._autocast_enabled,
                dtype=self._autocast_dtype,
            ),
        ):
            try:
                return _validate(self._encode(samples))
            except StopIteration as error:
                raise RuntimeError(
                    "Preprocessing raised StopIteration; this is not source exhaustion"
                ) from error

    def _produce(self):
        permit_owned = False
        try:
            with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
                self._iterator = iter(self._source)
                while not self._stop.is_set():
                    self._permit.acquire()
                    permit_owned = True
                    if self._stop.is_set():
                        break
                    try:
                        samples = next(self._iterator)
                    except StopIteration:
                        break
                    if self._stop.is_set():
                        break
                    encoded = self._encode_batch(samples)
                    ready = torch.cuda.Event()
                    ready.record(self._stream)
                    if not self._put(_Batch(samples, encoded, ready)):
                        break
                    permit_owned = False
        except BaseException as error:
            self._remember_failure(error)
        finally:
            if permit_owned:
                self._permit.release()
            self._close_source()

            try:
                self._stream.synchronize()
            except BaseException as error:
                self._remember_failure(error)
            self._put(_END)

    def _start(self):
        if self._thread is None:
            self._stream = torch.cuda.Stream(device=self.device)

            self._stream.wait_stream(torch.cuda.current_stream(self.device))
            self._thread = threading.Thread(
                target=self._produce,
                name="newm-current-observation-encode",
                daemon=True,
            )
            self._thread.start()

    def __next__(self):
        if self._closed:
            raise StopIteration
        if not self._cuda:
            try:
                if self._iterator is None:
                    self._iterator = iter(self._source)
                samples = next(self._iterator)
                return samples, self._encode_batch(samples)
            except StopIteration:
                self.close()
                raise
            except BaseException as error:
                self._remember_failure(error)
                self._failure_reported = True
                self.close(raise_errors=False)
                raise
        self._start()
        item = self._queue.get()
        if item is _END:
            self.close()
            raise StopIteration
        self._permit.release()
        try:
            consumer_stream = torch.cuda.current_stream(self.device)
            consumer_stream.wait_event(item.ready)
            for tensor in _tensors(item.encoded):
                if tensor.is_cuda:
                    tensor.record_stream(consumer_stream)
            return item.samples, item.encoded
        except BaseException as error:
            self._remember_failure(error)
            self._failure_reported = True
            self.close(raise_errors=False)
            raise

    def close(self, *, raise_errors=True):
        if not self._closed:
            self._closed = True
            self._stop.set()
            self._permit.release()
            if self._thread is not None:
                self._thread.join()
            else:
                self._close_source()
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        if raise_errors:
            self._raise_failure()
