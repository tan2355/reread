import errno
import hashlib
import json
import os
import socket
import time
import uuid
from collections import deque
from pathlib import Path

from reread.evaluation.scheduler import (
    _validate_job_templates,
    aggregate_results,
    validate_result,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


_READ_RETRY_DELAYS = (0.05, 0.1, 0.2)


def _transient_read_error(error):
    return isinstance(error, FileNotFoundError) or error.errno == getattr(
        errno, "ESTALE", 116
    )


def read_json(path):
    for attempt in range(len(_READ_RETRY_DELAYS) + 1):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except OSError as error:
            if not _transient_read_error(error) or attempt == len(_READ_RETRY_DELAYS):
                raise
            time.sleep(_READ_RETRY_DELAYS[attempt])


def visible_json(path):
    try:
        return read_json(path)
    except OSError as error:
        if not _transient_read_error(error):
            raise
        return None


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    pending = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    try:
        with pending.open("x", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(pending, path)
        else:
            os.replace(pending, path)
    finally:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass


class SharedEpisodeQueue:
    def __init__(self, output, jobs, node_rank, *, local_workers=16):
        self.root = Path(output)
        self.jobs = list(jobs)
        self.lookup = _validate_job_templates(self.jobs)
        if node_rank not in (0, 1) or local_workers < 1:
            raise ValueError("Two node ranks and positive worker count required")
        self.node_rank, self.local_workers = node_rank, local_workers
        self.claims = self.root / "claims"
        self.claims.mkdir(exist_ok=True)
        self.pending = deque(
            sorted(
                jobs,
                key=lambda j: (
                    -j.estimated_cost,
                    j.official_success_percent,
                    j.task,
                    j.phase,
                    j.episode_index,
                ),
            )
        )
        self.inflight = {}
        self._results = {}

    @staticmethod
    def key(job):
        return hashlib.sha256(job.job_id.encode()).hexdigest()

    def claim(self, worker_id):
        if (
            isinstance(worker_id, bool)
            or not isinstance(worker_id, int)
            or not self.node_rank * self.local_workers
            <= worker_id
            < (self.node_rank + 1) * self.local_workers
        ):
            raise ValueError("Worker belongs to another node")
        if worker_id in self.inflight:
            raise ValueError("Worker already has an unfinished episode")
        if (self.root / "failed.json").exists():
            return None
        while self.pending:
            job = self.pending.popleft()
            directory = self.claims / self.key(job)
            try:
                directory.mkdir()
            except FileExistsError:
                continue
            atomic_json(
                directory / "owner.json",
                dict(node_rank=self.node_rank, worker_id=worker_id, job_id=job.job_id),
                exclusive=True,
            )
            self.inflight[worker_id] = job
            return job
        return None

    def finish(self, worker_id, result):
        if worker_id not in self.inflight:
            raise ValueError("No in-flight episode for this worker")
        job = self.inflight[worker_id]
        validate_result(job, result)
        directory = self.claims / self.key(job)
        owner = read_json(directory / "owner.json")
        if owner != dict(
            node_rank=self.node_rank, worker_id=worker_id, job_id=job.job_id
        ):
            raise ValueError("Cannot finish a different owner claim")
        record = dict(
            result, worker_id=worker_id, node_rank=self.node_rank, seed_start=job.seed
        )
        atomic_json(directory / "result.json", record, exclusive=True)
        del self.inflight[worker_id]
        self._results[job.job_id] = record
        return record

    def abandon(self, worker_id, reason):
        job = self.inflight[worker_id]
        return self.finish(
            worker_id,
            dict(
                job_id=job.job_id,
                task=job.task,
                phase=job.phase,
                episode_index=job.episode_index,
                status="error",
                success=None,
                error=str(reason),
            ),
        )

    def results(self):

        for job in self.jobs:
            if job.job_id in self._results:
                continue
            directory = self.claims / self.key(job)
            path = directory / "result.json"
            if not path.is_file():
                continue
            record, owner = visible_json(path), visible_json(directory / "owner.json")
            if record is None or owner is None:
                continue
            validate_result(job, record)
            rank, worker = owner.get("node_rank"), owner.get("worker_id")
            if (
                type(rank) is not int
                or rank not in (0, 1)
                or type(worker) is not int
                or not rank * self.local_workers
                <= worker
                < (rank + 1) * self.local_workers
                or record.get("worker_id") != worker
                or record.get("node_rank") != rank
                or owner.get("job_id") != job.job_id
                or record.get("seed_start") != job.seed
            ):
                raise ValueError("Published result ownership mismatch")
            self._results[job.job_id] = record
        return list(self._results.values())

    def summary(self):
        return aggregate_results(self.jobs, self.results())


class SingleNodeSession:

    def __init__(
        self,
        output,
        contract,
        *,
        hostname=None,
        join_timeout=1800,
        heartbeat_timeout=600,
    ):
        self.root = Path(output)
        self.host = hostname or socket.gethostname()
        self.contract = contract
        self.started = time.monotonic()
        self.heartbeat_timeout = heartbeat_timeout
        self.rank = 0
        self.heartbeat_sequence = 0
        self.root.mkdir(parents=False, exist_ok=False)
        (self.root / "nodes").mkdir()
        self.node_dir = (
            self.root / "nodes" / hashlib.sha256(self.host.encode()).hexdigest()[:20]
        )
        self.node_dir.mkdir()
        atomic_json(
            self.root / "session.json",
            dict(contract=contract, created_at=time.time()),
            exclusive=True,
        )
        atomic_json(
            self.node_dir / "registration.json",
            dict(hostname=self.host, contract=contract),
            exclusive=True,
        )
        self.heartbeat("joining")

    def heartbeat(self, stage, **fields):
        self.heartbeat_sequence += 1
        atomic_json(
            self.node_dir / "heartbeat.json",
            dict(
                **fields,
                stage=stage,
                nonce=uuid.uuid4().hex,
                sequence=self.heartbeat_sequence,
            ),
        )

    def check(self):
        if (self.root / "failed.json").is_file() or (
            self.node_dir / "error.json"
        ).is_file():
            raise RuntimeError(
                "Evaluation aborted; inspect failed.json and nodes/*/error.json"
            )
        nodes = list((self.root / "nodes").iterdir())
        if nodes != [self.node_dir]:
            raise RuntimeError("Single-node run must have exactly one coordinator")
        return nodes

    def join(self):
        self.check()
        return self.rank

    def ready(self, info):
        self.check()
        atomic_json(
            self.node_dir / "ready.json", dict(node_rank=0, **info), exclusive=True
        )
        atomic_json(self.root / "go.json", dict(ready_nodes=1), exclusive=True)

    def fail(self, error):
        value = dict(
            complete=False,
            hostname=self.host,
            node_rank=0,
            error=f"{type(error).__name__}: {error}",
        )
        for path in (self.node_dir / "error.json", self.root / "failed.json"):
            try:
                atomic_json(path, value, exclusive=True)
            except FileExistsError:
                pass


class TwoNodeSession:

    def __init__(
        self,
        output,
        contract,
        *,
        hostname=None,
        join_timeout=1800,
        heartbeat_timeout=600,
    ):
        self.root = Path(output)
        self.host = hostname or socket.gethostname()
        self.contract = contract
        self.join_timeout = join_timeout
        self.heartbeat_timeout = heartbeat_timeout
        self.started = time.monotonic()
        self.peer_observed = {}
        self.peer_sequences = {}
        self.heartbeat_sequence = 0
        self.rank = None
        self.node_dir = None
        created = False
        try:
            self.root.mkdir(parents=False)
            created = True
        except FileExistsError:
            if self.root.is_symlink() or not self.root.is_dir():
                raise ValueError("Unsafe output directory")
        if created:
            (self.root / "nodes").mkdir()
            atomic_json(
                self.root / "session.json",
                dict(contract=contract, created_at=time.time()),
                exclusive=True,
            )
        while (session := visible_json(self.root / "session.json")) is None:
            if time.monotonic() - self.started > min(join_timeout, 60):
                raise TimeoutError(
                    "Output exists without a completed fresh session; choose a new run ID"
                )
            time.sleep(0.1)
        if canonical(session["contract"]) != canonical(contract):
            raise ValueError(
                "Nodes disagree on immutable evaluation/checkpoint contract"
            )
        if time.time() - session["created_at"] > join_timeout + 60 or any(
            (self.root / name).exists()
            for name in ("go.json", "complete.json", "failed.json")
        ):
            raise FileExistsError(
                "Run already started/ended or stale; use a fresh run ID, no resume"
            )
        node = self.root / "nodes" / hashlib.sha256(self.host.encode()).hexdigest()[:20]
        node.mkdir()
        self.node_dir = node
        atomic_json(
            node / "registration.json",
            dict(hostname=self.host, contract=contract),
            exclusive=True,
        )
        self.heartbeat("joining")

    def heartbeat(self, stage, **fields):
        self.heartbeat_sequence += 1
        atomic_json(
            self.node_dir / "heartbeat.json",
            dict(
                **fields,
                stage=stage,
                nonce=uuid.uuid4().hex,
                sequence=self.heartbeat_sequence,
            ),
        )

    def check(self):
        if (self.root / "failed.json").is_file():
            raise RuntimeError(
                "Evaluation aborted; inspect failed.json and nodes/*/error.json"
            )
        nodes = list((self.root / "nodes").iterdir())
        if len(nodes) > 2:
            raise RuntimeError("More than two coordinators joined this run")
        now = time.monotonic()
        for node in nodes:
            if (node / "error.json").exists():
                raise RuntimeError("A node failed; inspect " + str(node / "error.json"))

            if (node / "done.json").exists():
                continue
            heartbeat = visible_json(node / "heartbeat.json")
            previous, changed = self.peer_observed.get(node.name, (None, now))
            if heartbeat is not None:
                if (
                    not isinstance(heartbeat, dict)
                    or not isinstance(heartbeat.get("nonce"), str)
                    or not heartbeat["nonce"]
                    or not isinstance(heartbeat.get("stage"), str)
                    or not heartbeat["stage"]
                    or type(heartbeat.get("sequence")) is not int
                    or heartbeat["sequence"] < 1
                ):
                    raise ValueError(
                        "Malformed node heartbeat: " + str(node / "heartbeat.json")
                    )
                sequence = heartbeat["sequence"]
                last_sequence = self.peer_sequences.get(node.name, 0)
                if sequence > last_sequence:
                    previous, changed = heartbeat["nonce"], now
                    self.peer_sequences[node.name] = sequence
                elif sequence == last_sequence and heartbeat["nonce"] != previous:
                    raise ValueError(
                        "Heartbeat identity changed without advancing sequence: "
                        + node.name
                    )

            self.peer_observed[node.name] = previous, changed
            if now - changed > self.heartbeat_timeout:
                raise TimeoutError(
                    "Node heartbeat stalled; no claims will be reassigned: " + node.name
                )
        return nodes

    def join(self):
        while True:
            nodes = self.check()
            registrations = [
                value
                for n in nodes
                if (value := visible_json(n / "registration.json")) is not None
            ]
            if len(registrations) == 2:
                if any(
                    canonical(r["contract"]) != canonical(self.contract)
                    for r in registrations
                ):
                    raise ValueError("Node contract mismatch")
                hosts = sorted(r["hostname"] for r in registrations)
                if len(set(hosts)) != 2:
                    raise ValueError("Two distinct hosts required")
                self.rank = hosts.index(self.host)
                return self.rank
            if time.monotonic() - self.started > self.join_timeout:
                raise TimeoutError("Second node did not join")
            self.heartbeat("joining")
            time.sleep(1)

    def ready(self, info):
        atomic_json(
            self.node_dir / "ready.json",
            dict(node_rank=self.rank, **info),
            exclusive=True,
        )
        started = time.monotonic()
        while True:
            nodes = self.check()
            if len(nodes) == 2 and all((n / "ready.json").is_file() for n in nodes):
                if self.rank == 0:
                    atomic_json(
                        self.root / "go.json", dict(ready_nodes=2), exclusive=True
                    )
                break
            if time.monotonic() - started > self.join_timeout:
                raise TimeoutError(
                    "Peer preflight/private simulator did not become ready"
                )
            self.heartbeat("waiting_ready")
            time.sleep(1)
        while not (self.root / "go.json").is_file():
            self.check()
            if time.monotonic() - started > self.join_timeout:
                raise TimeoutError("Coordinator start gate missing")
            self.heartbeat("waiting_start_gate")
            time.sleep(0.2)

    def fail(self, error):
        value = dict(
            complete=False,
            hostname=self.host,
            node_rank=self.rank,
            error=f"{type(error).__name__}: {error}",
        )
        for path in (self.node_dir / "error.json", self.root / "failed.json"):
            try:
                atomic_json(path, value, exclusive=True)
            except FileExistsError:
                pass
