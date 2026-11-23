from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from reread.data.sampling import load_task_list

EVAL_ROOT = "artifacts/eval"
CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
PHASES = ("demo_clean", "demo_randomized")
EPISODES_PER_PHASE = {"hard11": 25, "all50": 100}
SELECTED_TASKS = (
    "open_microwave",
    "hanging_mug",
    "turn_switch",
    "place_can_basket",
    "move_stapler_pad",
    "stack_bowls_three",
    "pick_diverse_bottles",
    "place_mouse_pad",
    "handover_block",
    "place_object_basket",
    "move_can_pot",
)
RESULT_STATUSES = {"success", "failure", "error"}
SEED_PROTOCOL = "official_expert_admission_strided_subset"


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def load_difficulty(
    path: str | Path | None = None, *, task_set: str = "hard11"
) -> dict[str, dict[str, float]]:
    if task_set not in EPISODES_PER_PHASE:
        raise ValueError(f"Unknown evaluation task set: {task_set}")
    if path is None:
        path = CONFIG_DIR / f"{task_set}_reference.json"
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    selected: dict[str, dict[str, float]] = {}
    for task, row in document["tasks"].items():
        clean = float(row["official_clean_percent"])
        random = float(row["official_randomized_percent"])
        if not all(math.isfinite(x) and 0 <= x <= 100 for x in (clean, random)):
            raise ValueError(f"Invalid official success rate for {task}")
        mean = (clean + random) / 2
        if not math.isclose(float(row["official_mean_percent"]), mean, abs_tol=1e-9):
            raise ValueError(f"Incorrect mean success rate for {task}")
        if task_set == "all50" or mean < 90:
            selected[task] = {
                "demo_clean": clean,
                "demo_randomized": random,
                "mean": mean,
            }
    if task_set == "all50":
        tasks = load_task_list(CONFIG_DIR / "robotwin_tasks.json")
        if set(selected) != set(tasks):
            raise ValueError("Expected reference rates for all 50 RoboTwin tasks")
        return {task: selected[task] for task in tasks}
    if set(selected) != set(SELECTED_TASKS):
        raise ValueError(
            "Difficulty table no longer selects exactly the requested 11 tasks"
        )
    return {task: selected[task] for task in SELECTED_TASKS}


@dataclass(frozen=True)
class EpisodeJob:
    job_id: str
    task: str
    phase: str
    episode_index: int
    seed: int
    candidate_stride: int
    max_seed_attempts: int
    max_steps: int
    estimated_cost: float
    official_success_percent: float
    rng_state: dict[str, Any] | None = None

    @property
    def chain_id(self) -> tuple[str, str]:
        return self.task, self.phase

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeJob":
        return cls(**dict(value))


def build_jobs(
    task_limits: Mapping[str, int],
    episodes_per_phase: int | None = None,
    seed: int = 0,
    *,
    task_set: str = "hard11",
    max_seed_attempts: int = 1000,
    difficulty_path: str | Path | None = None,
) -> list[EpisodeJob]:
    difficulty = load_difficulty(difficulty_path, task_set=task_set)
    if episodes_per_phase is None:
        episodes_per_phase = EPISODES_PER_PHASE[task_set]
    _integer(episodes_per_phase, "episodes_per_phase", minimum=1)
    _integer(seed, "seed")
    _integer(max_seed_attempts, "max_seed_attempts", minimum=1)
    initial_seed = 100000 * (seed + 1)
    if initial_seed + episodes_per_phase * max_seed_attempts >= 2**32:
        raise ValueError("Scene seed lanes exceed the NumPy uint32 seed range")
    jobs = []
    for task in difficulty:
        limit = _integer(task_limits[task], f"step limit for {task}", minimum=1)
        for phase in PHASES:
            success = difficulty[task][phase]
            cost = limit * (1 - 0.5 * success / 100)
            for episode_index in range(episodes_per_phase):
                jobs.append(
                    EpisodeJob(
                        job_id=f"{task}/{phase}/{episode_index:03d}",
                        task=task,
                        phase=phase,
                        episode_index=episode_index,
                        seed=initial_seed + episode_index,
                        candidate_stride=episodes_per_phase,
                        max_seed_attempts=max_seed_attempts,
                        max_steps=limit,
                        estimated_cost=cost,
                        official_success_percent=success,
                    )
                )
    return jobs


def _validate_job_templates(jobs: Sequence[EpisodeJob]) -> dict[str, EpisodeJob]:
    allowed_tasks = set(load_task_list(CONFIG_DIR / "robotwin_tasks.json"))
    lookup: dict[str, EpisodeJob] = {}
    for job in jobs:
        if job.job_id in lookup:
            raise ValueError(f"Duplicate planned job: {job.job_id}")
        if job.task not in allowed_tasks or job.phase not in PHASES:
            raise ValueError(f"Unexpected task/phase: {job.job_id}")
        _integer(job.episode_index, "episode_index")
        _integer(job.seed, "seed")
        _integer(job.candidate_stride, "candidate_stride", minimum=1)
        if not math.isfinite(job.estimated_cost) or job.estimated_cost <= 0:
            raise ValueError("estimated_cost must be finite and positive")
        lookup[job.job_id] = job
    if not lookup:
        raise ValueError("No evaluation jobs")
    return lookup


def validate_result(job: EpisodeJob, result: Mapping[str, Any]) -> None:
    for key in ("job_id", "task", "phase", "episode_index"):
        if result.get(key) != getattr(job, key):
            raise ValueError(f"Result {key} does not match dispatched job {job.job_id}")
    status = result.get("status")
    if status not in RESULT_STATUSES:
        raise ValueError(f"Unknown result status: {status}")
    if status == "error":
        if result.get("success") is not None:
            raise ValueError(
                "An infrastructure error must not be a policy failure/success"
            )
        if not result.get("error"):
            raise ValueError("Infrastructure error requires an error description")
        return
    success = result.get("success")
    if not isinstance(success, bool) or success != (status == "success"):
        raise ValueError("Result success must agree with status")
    accepted_seed = _integer(result.get("seed"), "accepted seed")
    if accepted_seed < job.seed:
        raise ValueError("Accepted scene seed precedes the job's next seed")
    attempts = _integer(result.get("seed_attempts"), "seed_attempts", minimum=1)
    if (
        accepted_seed != job.seed + (attempts - 1) * job.candidate_stride
        or attempts > job.max_seed_attempts
        or result.get("candidate_stride") != job.candidate_stride
    ):
        raise ValueError(
            "Seed admission must stay within its bounded deterministic seed lane"
        )
    if result.get("next_seed") != accepted_seed + job.candidate_stride:
        raise ValueError(
            "next_seed must follow the accepted scene seed within its lane"
        )
    elapsed = result.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError("elapsed_seconds must be finite and nonnegative")
    if success and result.get("video"):
        raise ValueError("Successful episode must not retain a video")
    if not success and not result.get("video"):
        raise ValueError(
            "A completed failed episode must have a retained failure video"
        )


class DynamicEpisodeScheduler:

    def __init__(self, jobs: Sequence[EpisodeJob], *, worker_count: int = 16):
        self.worker_count = _integer(worker_count, "worker_count", minimum=1)
        self.jobs = list(jobs)
        self._lookup = _validate_job_templates(self.jobs)
        self._pending = deque(
            sorted(
                jobs,
                key=lambda item: (
                    -item.estimated_cost,
                    item.official_success_percent,
                    item.task,
                    item.phase,
                    item.episode_index,
                ),
            )
        )
        self._inflight: dict[int, EpisodeJob] = {}
        self._results: dict[str, dict[str, Any]] = {}
        self._infrastructure_error = False

    @property
    def results(self) -> list[dict[str, Any]]:
        return list(self._results.values())

    @property
    def inflight(self) -> dict[int, EpisodeJob]:
        return dict(self._inflight)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    @property
    def completed_count(self) -> int:
        return sum(row["status"] != "error" for row in self._results.values())

    @property
    def has_failures(self) -> bool:
        return self._infrastructure_error

    @property
    def done(self) -> bool:
        return self.pending_count == 0 and self.inflight_count == 0

    def reserve(self, worker_id: int) -> EpisodeJob | None:
        _integer(worker_id, "worker_id")
        if worker_id >= self.worker_count:
            raise ValueError("worker_id exceeds worker_count")
        if worker_id in self._inflight:
            raise ValueError(f"Worker {worker_id} already owns an unfinished episode")
        if self.has_failures:
            return None
        if not self._pending:
            return None

        job = self._pending.popleft()
        self._inflight[worker_id] = job
        return job

    def finish(self, worker_id: int, result: Mapping[str, Any]) -> None:
        if worker_id not in self._inflight:
            raise ValueError(f"Worker {worker_id} has no dispatched episode")
        job = self._inflight[worker_id]
        validate_result(job, result)
        if job.job_id in self._results:
            raise ValueError(f"Duplicate result: {job.job_id}")
        record = dict(result)
        record["worker_id"] = worker_id
        record["seed_start"] = job.seed
        self._results[job.job_id] = record
        del self._inflight[worker_id]
        if record["status"] == "error":
            self._infrastructure_error = True

    def abandon(self, worker_id: int, error: str) -> dict[str, Any]:
        if not error:
            raise ValueError("Worker failure requires a description")
        job = self._inflight[worker_id]
        result = {
            key: getattr(job, key)
            for key in ("job_id", "task", "phase", "episode_index")
        }
        result.update(status="error", success=None, error=str(error))
        self.finish(worker_id, result)
        return self._results[job.job_id]


def aggregate_results(
    jobs: Sequence[EpisodeJob], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    lookup = _validate_job_templates(jobs)
    indexed: dict[str, Mapping[str, Any]] = {}
    for result in results:
        job_id = result.get("job_id")
        if job_id not in lookup:
            raise ValueError(f"Unplanned result: {job_id}")
        if job_id in indexed:
            raise ValueError(f"Duplicate result: {job_id}")
        job = lookup[job_id]
        if result.get("seed_start", job.seed) != job.seed:
            raise ValueError(f"Result changed its planned initial seed: {job_id}")
        validate_result(job, result)
        indexed[job_id] = result
    by_cell: dict[tuple[str, str], list[EpisodeJob]] = {}
    for job in jobs:
        by_cell.setdefault(job.chain_id, []).append(job)
    tasks = list(dict.fromkeys(job.task for job in jobs))
    cells = []
    for task in tasks:
        for phase in PHASES:
            plan = sorted(
                by_cell.get((task, phase), ()),
                key=lambda j: j.episode_index,
            )
            if not plan:
                continue
            records = []
            for job in plan:
                row = indexed.get(job.job_id)
                if row is None:
                    continue
                records.append(row)
            completed = [r for r in records if r["status"] != "error"]
            seeds = [r["seed"] for r in completed]
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"Repeated accepted scene seed for {task}/{phase}")
            successes = sum(r["status"] == "success" for r in completed)
            cells.append(
                {
                    "task": task,
                    "phase": phase,
                    "expected": len(plan),
                    "completed": len(completed),
                    "successes": successes,
                    "policy_failures": len(completed) - successes,
                    "infrastructure_errors": sum(
                        r["status"] == "error" for r in records
                    ),
                    "missing": len(plan) - len(records),
                    "success_rate_percent": 100 * successes / len(completed)
                    if completed
                    else None,
                    "complete": len(completed) == len(plan),
                }
            )
    errors = sum(c["infrastructure_errors"] for c in cells)
    completed = sum(c["completed"] for c in cells)
    complete = completed == len(jobs) and errors == 0
    summary: dict[str, Any] = {
        "schema": "newm-memory3-evaluation-summary-v1",
        "complete": complete,
        "seed_protocol": SEED_PROTOCOL,
        "expected_episodes": len(jobs),
        "completed_episodes": completed,
        "successes": sum(c["successes"] for c in cells),
        "policy_failures": sum(c["policy_failures"] for c in cells),
        "infrastructure_errors": errors,
        "missing_episodes": len(jobs) - len(indexed),
        "cells": cells,
        "tasks": [],
        "macro_success_rate_percent": None,
        "phase_success_rate_percent": {phase: None for phase in PHASES},
    }
    for task in tasks:
        phases = {cell["phase"]: cell for cell in cells if cell["task"] == task}
        if not phases:
            continue
        task_complete = len(phases) == 2 and all(
            cell["complete"] for cell in phases.values()
        )
        summary["tasks"].append(
            dict(
                task=task,
                complete=task_complete,
                clean_success_rate_percent=phases.get("demo_clean", {}).get(
                    "success_rate_percent"
                ),
                randomized_success_rate_percent=phases.get("demo_randomized", {}).get(
                    "success_rate_percent"
                ),
                mean_success_rate_percent=(
                    sum(cell["success_rate_percent"] for cell in phases.values()) / 2
                    if task_complete
                    else None
                ),
            )
        )
    if complete:
        summary["macro_success_rate_percent"] = sum(
            c["success_rate_percent"] for c in cells
        ) / len(cells)
        for phase in PHASES:
            rates = [c["success_rate_percent"] for c in cells if c["phase"] == phase]
            summary["phase_success_rate_percent"][phase] = (
                sum(rates) / len(rates) if rates else None
            )
    return summary


def validate_output_root(output: str | Path, *, allowed_root: str = EVAL_ROOT) -> str:
    candidate, root = Path(output), Path(allowed_root).resolve()
    resolved = candidate.resolve()
    if ".." in candidate.parts or resolved == root or not resolved.is_relative_to(root):
        raise ValueError(
            f"Evaluation output must be a unique subdirectory beneath {allowed_root}"
        )
    return str(resolved)


def create_run_directory(output: str | Path, *, allowed_root: str = EVAL_ROOT) -> Path:
    guarded = validate_output_root(output, allowed_root=allowed_root)
    root = Path(allowed_root).resolve(strict=True)
    destination = Path(guarded)
    resolved = destination.resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        raise ValueError("Evaluation output resolves outside the allowed root")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Evaluation output already exists; choose a new run: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=False)
    return destination
