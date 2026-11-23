import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_task_list(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = document.get("tasks")
    if (
        document.get("schema") != "reread-uniform-tasks-v1"
        or not isinstance(tasks, list)
        or len(tasks) != 50
        or any(not isinstance(task, str) or not task for task in tasks)
        or len(set(tasks)) != 50
    ):
        raise ValueError("Expected 50 unique RoboTwin 2.0 task names")
    return tuple(sorted(tasks))


def uniform_task_probabilities(tasks):
    tasks = tuple(tasks)
    if (
        not tasks
        or any(not isinstance(task, str) or not task for task in tasks)
        or len(set(tasks)) != len(tasks)
    ):
        raise ValueError("Task names must be nonempty and unique")
    return {task: 1.0 / len(tasks) for task in sorted(tasks)}


def normalized_probabilities(values, expected, *, name):
    if not isinstance(values, dict) or set(values) != set(expected):
        raise ValueError(f"{name} must match the selected raw data exactly")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in values.values()
    ):
        raise ValueError(f"{name} must be finite and strictly positive")
    total = sum(values.values())
    if not math.isfinite(total):
        raise ValueError(f"{name} total must be finite")
    return {key: values[key] / total for key in sorted(values)}


def weighted_trajectory_requests(
    rows,
    seed,
    batch_size,
    decision_stride,
    *,
    task_probabilities=None,
    phase_probabilities=None,
    max_batches=None,
):
    if batch_size < 1 or decision_stride < 1 or not rows:
        raise ValueError("Positive batch/stride and nonempty raw rows required")
    if max_batches is not None and (isinstance(max_batches, bool) or max_batches < 0):
        raise ValueError("max_batches must be nonnegative")
    if max_batches == 0:
        return
    tasks, phases = {row.task for row in rows}, {row.phase for row in rows}
    task_probabilities = normalized_probabilities(
        {task: 1.0 for task in tasks}
        if task_probabilities is None
        else task_probabilities,
        tasks,
        name="Task probabilities",
    )
    phase_probabilities = normalized_probabilities(
        {phase: 1.0 for phase in phases}
        if phase_probabilities is None
        else phase_probabilities,
        phases,
        name="Phase probabilities",
    )
    groups = defaultdict(list)
    costs = []
    for index, row in enumerate(rows):
        if row.observation_count < 1 or not row.instructions:
            raise ValueError(
                "Every raw trajectory requires observations and seen instructions"
            )
        groups[(row.task, row.phase)].append(index)
        costs.append((row.observation_count + decision_stride - 1) // decision_stride)
    expected = {(task, phase) for task in tasks for phase in phases}
    if set(groups) != expected:
        raise ValueError("Every selected task must contain every selected phase")
    probabilities = {
        group: task_probabilities[group[0]] * phase_probabilities[group[1]]
        for group in sorted(groups)
    }
    rng = np.random.default_rng(seed)

    group_order = [tuple(group) for group in rng.permutation(sorted(groups)).tolist()]
    orders = {group: rng.permutation(groups[group]).tolist() for group in group_order}
    positions = {group: 0 for group in group_order}
    reserved = {group: 0 for group in group_order}
    lanes, pass_id, emitted, batches = {}, 0, 0, 0
    budget = sum(costs)
    while emitted < budget:
        requests = []
        for lane in range(min(batch_size, budget - emitted)):
            entry = lanes.get(lane)
            if entry is None or entry[1] >= rows[entry[0]].observation_count:
                group = min(
                    group_order, key=lambda key: reserved[key] / probabilities[key]
                )
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
                    pass_id,
                )
                pass_id += 1
            index, frame, instruction, current_pass = entry
            requests.append((lane, index, frame, instruction, current_pass))
            lanes[lane] = (index, frame + decision_stride, instruction, current_pass)
        yield requests
        emitted += len(requests)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            return
