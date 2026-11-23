import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from reread.data.sampling import normalized_probabilities


def partition_episodes(
    rows, rank, world_size, *, seed=0, coverage_plan=None, decision_stride=20
):
    if world_size < 1 or not 0 <= rank < world_size or not rows:
        raise ValueError("Invalid rank/world size or empty dataset")
    if decision_stride < 1:
        raise ValueError("Decision stride must be positive")
    if coverage_plan is not None and coverage_plan["world_size"] != world_size:
        raise ValueError("Coverage plan world size differs from episode partition")
    groups = defaultdict(list)
    identities = set()
    for index, row in enumerate(rows):
        if row.id in identities:
            raise ValueError("Duplicate source episode identity")
        identities.add(row.id)
        groups[(row.task, row.phase)].append(index)
    tasks, phases = {row.task for row in rows}, {row.phase for row in rows}
    expected = {(task, phase) for task in tasks for phase in phases}
    if set(groups) != expected:
        raise ValueError("Each selected task must contain every selected phase")
    if (
        coverage_plan is not None
        and set(coverage_plan["target_task_probabilities"]) != tasks
    ):
        raise ValueError("Coverage plan tasks differ from episode partition")
    result = []
    for group in sorted(groups):
        indices = sorted(groups[group], key=lambda index: rows[index].id)
        if len(indices) < world_size:
            raise ValueError(f"Every rank needs an episode in task/phase group {group}")
        digest = hashlib.sha256(f"{seed}:{group[0]}:{group[1]}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:16], "big"))
        order = rng.permutation(indices).tolist()
        if coverage_plan is None:
            result.extend(order[rank::world_size])
            continue
        task = group[0]
        residual = coverage_plan["flexible_task_probabilities"][task]
        capacities = [
            int(task in coverage_plan["rank_anchor_tasks"][r])
            + coverage_plan["rank_flexible_lanes"][r] * residual
            for r in range(world_size)
        ]
        if min(capacities) <= 0:
            raise ValueError(
                "Disjoint nonempty pools on every rank require positive per-rank task capacity"
            )
        costs = {
            index: math.ceil(rows[index].observation_count / decision_stride)
            for index in order
        }
        if min(costs.values()) < 1:
            raise ValueError("Episode partition requires positive decision costs")

        order.sort(key=lambda index: -costs[index])
        target = [
            sum(costs.values()) * capacity / sum(capacities) for capacity in capacities
        ]
        rank_order = rng.permutation(world_size).tolist()
        rank_order.sort(key=lambda r: -capacities[r])
        assigned, pools = [0] * world_size, [[] for _ in range(world_size)]
        for owner, index in zip(rank_order, order[-world_size:]):
            pools[owner].append(index)
            assigned[owner] += costs[index]
        for index in order[:-world_size]:
            owner = max(rank_order, key=lambda r: target[r] - assigned[r])
            pools[owner].append(index)
            assigned[owner] += costs[index]
        result.extend(pools[rank])
    return tuple(sorted(result))


def global_coverage_plan(task_probabilities, *, world_size, batch_size, seed=0):
    if (
        isinstance(world_size, bool)
        or isinstance(batch_size, bool)
        or not isinstance(world_size, int)
        or not isinstance(batch_size, int)
        or min(world_size, batch_size) < 1
        or not task_probabilities
    ):
        raise ValueError(
            "Positive integer world/batch size and nonempty task probabilities required"
        )
    probabilities = normalized_probabilities(
        task_probabilities, set(task_probabilities), name="Global task probabilities"
    )
    tasks, global_batch = sorted(probabilities), world_size * batch_size
    if len(tasks) > global_batch:
        raise ValueError("Global batch is too small to cover every task")
    if any(value < 1 / global_batch for value in probabilities.values()):
        raise ValueError(
            "Task probability below 1/global_batch is incompatible with every-batch coverage"
        )
    flexible = global_batch - len(tasks)
    residual = (
        {task: (global_batch * probabilities[task] - 1) / flexible for task in tasks}
        if flexible
        else {task: 0.0 for task in tasks}
    )
    if flexible and not math.isclose(
        sum(residual.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Invalid residual task probability budget")
    order = np.random.default_rng(seed).permutation(tasks).tolist()
    anchors = [order[rank::world_size] for rank in range(world_size)]
    return dict(
        schema="newm-memory3-global-task-coverage-v1",
        seed=seed,
        world_size=world_size,
        batch_per_rank=batch_size,
        global_batch=global_batch,
        task_count=len(tasks),
        anchor_lanes=len(tasks),
        flexible_lanes=flexible,
        rank_anchor_tasks=anchors,
        rank_flexible_lanes=[batch_size - len(value) for value in anchors],
        target_task_probabilities=probabilities,
        flexible_task_probabilities=residual,
        scope="every_global_optimizer_batch",
        allocation="fixed_task_anchors_plus_difficulty_residual_trajectory_credit",
        probability_semantics="long_run_consumed_decisions_not_per_step_proportional_quota",
    )


def assert_local_task_coverage(samples, coverage_plan, rank):
    if not 0 <= rank < coverage_plan["world_size"]:
        raise ValueError("Invalid coverage rank")
    batch = coverage_plan["batch_per_rank"]
    lanes = {sample["lane"]: sample for sample in samples}
    if len(samples) != batch or len(lanes) != batch or set(lanes) != set(range(batch)):
        raise RuntimeError(
            "Coverage requires each private lane exactly once per consumed batch"
        )
    for lane, task in enumerate(coverage_plan["rank_anchor_tasks"][rank]):
        sample = lanes[lane]
        actual = sample["episode_id"].split("/")[0]
        if actual != task or sample.get("task", task) != task:
            raise RuntimeError(
                f"Coverage anchor lane {lane} on rank {rank} did not consume task {task}"
            )


def continuous_requests(
    rows,
    indices,
    seed,
    batch_size,
    decision_stride,
    *,
    batches,
    task_probabilities,
    phase_probabilities,
    coverage_plan=None,
    rank=None,
):
    if min(batch_size, decision_stride, batches) < 1 or not indices:
        raise ValueError(
            "Positive batch/stride/update budget and nonempty rank pool required"
        )
    if len(set(indices)) != len(indices) or any(
        i < 0 or i >= len(rows) for i in indices
    ):
        raise ValueError("Invalid or duplicate rank episode index")
    tasks, phases = {rows[i].task for i in indices}, {rows[i].phase for i in indices}
    task_p = normalized_probabilities(
        task_probabilities, tasks, name="Global task probabilities"
    )
    phase_p = normalized_probabilities(
        phase_probabilities, phases, name="Global phase probabilities"
    )
    groups, costs = defaultdict(list), {}
    for index in indices:
        row = rows[index]
        if row.observation_count < 1 or not len(row.instructions):
            raise ValueError(
                "Each trajectory requires observations and seen instructions"
            )
        groups[(row.task, row.phase)].append(index)
        costs[index] = math.ceil(row.observation_count / decision_stride)
    if set(groups) != {(task, phase) for task in tasks for phase in phases}:
        raise ValueError("Each rank requires every task/phase group")
    anchor_tasks = []
    flexible_p = task_p
    if coverage_plan is not None:
        expected = global_coverage_plan(
            task_probabilities,
            world_size=coverage_plan["world_size"],
            batch_size=batch_size,
            seed=coverage_plan["seed"],
        )
        if (
            coverage_plan != expected
            or rank is None
            or not 0 <= rank < expected["world_size"]
        ):
            raise ValueError(
                "Coverage plan differs from task probabilities, batch geometry, or rank"
            )
        anchor_tasks = coverage_plan["rank_anchor_tasks"][rank]
        flexible_p = coverage_plan["flexible_task_probabilities"]
    probabilities = {
        g: flexible_p[g[0]] * phase_p[g[1]] for g in groups if flexible_p[g[0]] > 0
    }
    rng = np.random.default_rng(seed)
    group_order = [tuple(g) for g in rng.permutation(sorted(groups)).tolist()]
    orders = {g: rng.permutation(groups[g]).tolist() for g in group_order}
    positions = dict.fromkeys(group_order, 0)
    flexible_order = [g for g in group_order if g in probabilities]
    flexible_reserved = dict.fromkeys(flexible_order, 0)
    anchor_orders = {
        task: [g for g in group_order if g[0] == task] for task in anchor_tasks
    }
    anchor_reserved = {g: 0 for task in anchor_tasks for g in anchor_orders[task]}
    lanes, next_pass = {}, 0
    for _ in range(batches):
        requests = []
        for lane in range(batch_size):
            entry = lanes.get(lane)
            if entry is None or entry[1] >= rows[entry[0]].observation_count:
                if lane < len(anchor_tasks):
                    group = min(
                        anchor_orders[anchor_tasks[lane]],
                        key=lambda g: anchor_reserved[g] / phase_p[g[1]],
                    )
                    reserved = anchor_reserved
                else:
                    group = min(
                        flexible_order,
                        key=lambda g: flexible_reserved[g] / probabilities[g],
                    )
                    reserved = flexible_reserved
                if positions[group] == len(orders[group]):
                    orders[group] = rng.permutation(groups[group]).tolist()
                    positions[group] = 0
                index = orders[group][positions[group]]
                positions[group] += 1
                reserved[group] += costs[index]
                entry = (
                    index,
                    0,
                    int(rng.integers(len(rows[index].instructions))),
                    next_pass,
                )
                next_pass += 1
            index, frame, instruction, pass_id = entry
            requests.append((lane, index, frame, instruction, pass_id))
            lanes[lane] = index, frame + decision_stride, instruction, pass_id
        yield requests


def _identity(value):
    return value


def _worker_init(_worker_id):
    torch.set_num_threads(1)


def iter_rank_batches(
    dataset,
    indices,
    *,
    rank,
    seed,
    batch_size,
    decision_stride,
    batches,
    workers,
    task_probabilities,
    phase_probabilities,
    coverage_plan=None,
    pin_memory=None,
):
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    if pin_memory is not None and not isinstance(pin_memory, bool):
        raise ValueError("pin_memory must be a boolean or None")
    requests = continuous_requests(
        dataset.rows,
        indices,
        seed + 100003 * rank,
        batch_size,
        decision_stride,
        batches=batches,
        task_probabilities=task_probabilities,
        phase_probabilities=phase_probabilities,
        coverage_plan=coverage_plan,
        rank=rank,
    )
    stopping = False

    def sampler():
        for batch in requests:
            if stopping:
                return
            yield batch

    kwargs = dict(
        dataset=dataset,
        batch_sampler=sampler(),
        collate_fn=_identity,
        num_workers=workers,
        pin_memory=bool(workers) if pin_memory is None else pin_memory,
        worker_init_fn=_worker_init,
        generator=torch.Generator().manual_seed(seed + rank),
    )
    if workers:
        kwargs.update(multiprocessing_context="spawn", prefetch_factor=2, timeout=120)
    iterator = iter(torch.utils.data.DataLoader(**kwargs))
    try:
        for samples in iterator:
            if len(samples) != batch_size:
                raise RuntimeError("Distributed trajectory batch was not full")
            if coverage_plan is not None:
                assert_local_task_coverage(samples, coverage_plan, rank)
            for sample in samples:
                if sample.get("decision_index") is None:
                    sample["decision_index"] = sample["frame"] // decision_stride
            yield samples
    finally:
        stopping = True

        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()


def global_weighted_loss(prediction, target, valid, *, group=None):
    if prediction.shape != target.shape or valid.shape != prediction.shape[:2]:
        raise ValueError("Prediction, target, and valid mask shapes differ")
    valid = valid.to(device=prediction.device, dtype=torch.bool)
    target = target.detach().to(device=prediction.device, dtype=torch.float32)
    local_sum = (prediction.float()[valid] - target[valid]).square().sum()
    local_weight = valid.sum(dtype=torch.int64) * prediction.shape[-1]
    stats = torch.stack(
        (
            local_sum.detach().to(torch.float64),
            local_weight.to(torch.float64),
            (~torch.isfinite(local_sum.detach())).to(torch.float64),
        )
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
    global_sum, global_weight, invalid = stats.tolist()
    if invalid or not math.isfinite(global_sum):
        raise FloatingPointError(
            "Nonfinite loss on at least one rank; no backward or history commit"
        )
    if global_weight < 1:
        raise ValueError("No valid labels across all ranks")
    world = dist.get_world_size(group)
    return local_sum * (world / global_weight), {
        "mse": global_sum / global_weight,
        "label_elements": int(global_weight),
    }


def effective_cpus(
    *,
    affinity=None,
    cgroup_root=Path("/sys/fs/cgroup"),
    membership=Path("/proc/self/cgroup"),
):
    if affinity is None:
        affinity = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )
    count = int(affinity)
    root = Path(cgroup_root)
    directories = {root, root / "cpu"}
    try:
        for line in Path(membership).read_text().splitlines():
            _, controllers, path = line.split(":", 2)
            if not controllers or "cpu" in controllers.split(","):
                prefix = root if not controllers else root / "cpu"
                candidate = prefix / path.lstrip("/")
                if ".." not in candidate.parts and candidate.is_relative_to(prefix):
                    directories.update(
                        [
                            candidate,
                            *[p for p in candidate.parents if p.is_relative_to(prefix)],
                        ]
                    )
    except (OSError, ValueError):
        pass
    for directory in directories:
        try:
            quota, period = (directory / "cpu.max").read_text().split()
            if quota != "max" and int(period) > 0:
                count = min(count, max(1, int(quota) // int(period)))
        except (OSError, ValueError, ZeroDivisionError):
            pass
        try:
            quota = int((directory / "cpu.cfs_quota_us").read_text())
            period = int((directory / "cpu.cfs_period_us").read_text())
            if quota > 0 and period > 0:
                count = min(count, max(1, quota // period))
        except (OSError, ValueError):
            pass
    return count


def cpu_budget(*, world_size, threads, workers, available=None):
    available = effective_cpus() if available is None else available
    assigned = world_size * (threads + workers + 2)
    if min(world_size, threads, available) < 1 or workers < 0 or assigned > available:
        raise ValueError(
            f"CPU budget exceeded: {assigned} assigned, {available} available"
        )
    return dict(
        effective_cpus=available,
        ranks=world_size,
        threads_per_rank=threads,
        workers_per_rank=workers,
        service_slots_per_rank=2,
        assigned_cpus=assigned,
        spare_cpus=available - assigned,
    )


class DistributedContext:
    def __init__(self, *, device):
        self.device = torch.device(device)
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()

    @classmethod
    def initialize_h800(cls, *, timeout_seconds=600):
        required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
        if any(name not in os.environ for name in required):
            raise ValueError("Use the single-node eight-GPU torchrun launcher")
        rank, world, local_rank, local_world = (
            int(os.environ[name]) for name in required
        )
        if (world, local_world) != (8, 8) or rank != local_rank or not 0 <= rank < 8:
            raise ValueError("This entry requires exactly one node and eight ranks")
        if timeout_seconds < 30:
            raise ValueError("Distributed timeout must be at least 30 seconds")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
            raise RuntimeError("Exactly eight runtime-visible CUDA GPUs are required")
        torch.cuda.set_device(local_rank)
        name = torch.cuda.get_device_name(local_rank)
        if "H800" not in name.upper():
            raise RuntimeError(f"This entry is H800-only; rank {rank} sees {name}")
        free_bytes, total_bytes = torch.cuda.mem_get_info(local_rank)
        if total_bytes < 75 * 1024**3 or free_bytes < 60 * 1024**3:
            raise RuntimeError(
                f"Rank {rank} requires an 80GB H800 with at least 60GiB free; existing jobs were not touched"
            )
        dist.init_process_group(
            "nccl", init_method="env://", timeout=timedelta(seconds=timeout_seconds)
        )
        return cls(device=f"cuda:{local_rank}")

    def wrap(self, model):
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("Corrector parameters must remain FP32")
        kwargs = dict(
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=25,
        )
        if self.device.type == "cuda":
            kwargs.update(
                device_ids=[self.device.index], output_device=self.device.index
            )
        return DistributedDataParallel(model, **kwargs)

    def loss(self, prediction, target, valid):
        return global_weighted_loss(prediction, target, valid)

    def sum_counts(self, values):
        tensor = torch.tensor(values, dtype=torch.int64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.cpu().tolist()

    def require_all(self, condition, message):
        flag = torch.tensor(int(bool(condition)), dtype=torch.int32, device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if not flag.item():
            raise RuntimeError(message)

    def agree(self, value):
        digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).digest()
        local = torch.tensor(list(digest), dtype=torch.uint8, device=self.device)
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local)
        if any(not torch.equal(local, other) for other in gathered):
            raise RuntimeError("Ranks disagree on cache/config/sampling identity")

    def rank_zero(self, function):
        result = [None]
        if self.rank == 0:
            try:
                result[0] = {"value": function(), "error": None}
            except Exception as error:
                result[0] = {"error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(result, src=0, device=self.device)
        if result[0]["error"]:
            raise RuntimeError(result[0]["error"])
        return result[0]["value"]

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()
