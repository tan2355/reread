import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from reread.data.cache_contract import require_rgb_cache, sealed_rgb_cache
from reread.data.sampling import (
    load_task_list,
    uniform_task_probabilities,
    weighted_trajectory_requests,
)
from reread.train import parser, validate_args

ROOT = Path(__file__).parents[1]


@dataclass
class Row:
    task: str
    phase: str
    observation_count: int
    instructions: tuple = ("instruction",)


def test_all_50_tasks_are_uniform_and_training_parser_uses_that_inventory():
    tasks = load_task_list(ROOT / "configs/robotwin_tasks.json")
    probabilities = uniform_task_probabilities(tasks)
    assert len(probabilities) == 50
    assert set(probabilities.values()) == {0.02}
    args = parser().parse_args(
        ["--launch-id", "a" * 32, "--cache-root", "cache", "--output", "weights"]
    )
    validate_args(args)
    assert load_task_list(args.task_list) == tasks
    assert not hasattr(args, "task_policy")
    assert args.phase_weights == [1.0, 10.0]
    assert args.steps == 20000


@pytest.mark.parametrize("tasks", [[], ["a", "a"], [""], ["a", 1]])
def test_invalid_task_inventory_is_rejected(tasks):
    with pytest.raises(ValueError):
        uniform_task_probabilities(tasks)


def test_difficulty_score_dictionary_is_not_a_task_inventory(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps(
            {"schema": "reread-uniform-tasks-v1", "tasks": {"task": {"weight": 5}}}
        )
    )
    with pytest.raises(ValueError):
        load_task_list(path)


def test_uniform_decision_exposure_with_unequal_trajectory_lengths():
    tasks = [f"task_{i:02}" for i in range(50)]
    rows = [
        Row(task, phase, 100 + 20 * (index % 17))
        for i, task in enumerate(tasks)
        for phase in ("clean", "randomized")
        for index in range(i, i + 30)
    ]
    probabilities = uniform_task_probabilities(tasks)
    phase_p = {"clean": 1 / 11, "randomized": 10 / 11}
    counts, phases = Counter(), Counter()
    previous = {}
    total = 0
    for batch in weighted_trajectory_requests(
        rows, 7, 64, 20, task_probabilities=probabilities, phase_probabilities=phase_p
    ):
        for lane, index, frame, instruction, pass_id in batch:
            if lane in previous and previous[lane][3] == pass_id:
                old_index, old_frame, old_instruction, _ = previous[lane]
                assert index == old_index
                assert frame == old_frame + 20
                assert instruction == old_instruction
            else:
                assert frame == 0
            previous[lane] = (index, frame, instruction, pass_id)
            counts[rows[index].task] += 1
            phases[rows[index].phase] += 1
            total += 1
    assert set(counts) == set(tasks)
    assert max(abs(counts[task] / total - 0.02) for task in tasks) < 0.002
    assert abs(phases["clean"] / total - 1 / 11) < 0.01


def test_fresh_native_rgb_cache_requires_no_legacy_text_cache():
    metadata = dict(
        schema="newm-memory4-observation-cache-v2",
        sealed_complete=True,
        observation_color_contract="native_robotwin_rgb_v1",
        variants=3,
        manifest_sha256="a" * 64,
        plan_sha256="b" * 64,
        index_sha256="c" * 64,
    )
    result = require_rgb_cache(metadata)
    assert result["manifest_sha256"] == "a" * 64
    assert "text_reuse" not in result
    with pytest.raises(ValueError):
        require_rgb_cache({**metadata, "observation_color_contract": "bgr"})


def test_fresh_native_rgb_seal_binds_plan_and_manifest(tmp_path):
    import hashlib

    def save(name, value):
        raw = json.dumps(value, sort_keys=True).encode()
        (tmp_path / name).write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    schema = "newm-memory4-observation-cache-v2"
    color = "native_robotwin_rgb_v1"
    plan_sha = save("plan.json", dict(schema=schema, observation_color_contract=color))
    manifest_sha = save(
        "manifest.json",
        dict(
            schema=schema,
            observation_color_contract=color,
            variants=3,
            plan_sha256=plan_sha,
            index=dict(path="index.sqlite", size=1, sha256="c" * 64),
        ),
    )
    save(
        "complete.json",
        dict(
            schema=schema,
            complete=True,
            observation_color_contract=color,
            variants=3,
            plan_sha256=plan_sha,
            manifest_sha256=manifest_sha,
        ),
    )
    assert sealed_rgb_cache(tmp_path)["manifest_sha256"] == manifest_sha
    save("plan.json", dict(schema=schema, observation_color_contract="bgr"))
    with pytest.raises(ValueError):
        sealed_rgb_cache(tmp_path)


@pytest.mark.parametrize(
    "selection,task_set,task_count,episodes_per_phase",
    [([], "hard11", 11, 25), (["--task-set", "all50"], "all50", 50, 100)],
)
def test_evaluation_plan_has_equal_episode_counts_and_no_legacy_modes(
    tmp_path, monkeypatch, selection, task_set, task_count, episodes_per_phase
):
    import evaluate as launcher

    monkeypatch.setattr(launcher, "verify_release", lambda *args: "a" * 64)
    monkeypatch.setattr(launcher, "configure_environment", lambda *args: None)
    plan = launcher.main(
        [
            "--run-id",
            "plan_check",
            "--corrector",
            "checkpoint.pt",
            "--output-root",
            str(tmp_path),
            "--print-plan",
            *selection,
        ]
    )
    assert plan["task_set"] == task_set
    assert len(plan["tasks"]) == task_count
    assert plan["episodes_per_task_phase"] == episodes_per_phase
    assert plan["total_episodes"] == task_count * 2 * episodes_per_phase
    assert plan["denoising_steps"] == plan["replan_steps"] == 20
    options = {
        name for action in launcher.parser()._actions for name in action.option_strings
    }
    assert (
        not {"--r10-history20", "--recover-from", "--wait-training-complete"} & options
    )


@pytest.mark.parametrize("task_set", ["hard11", "all50"])
def test_archived_summaries_record_the_selected_task_set(tmp_path, task_set):
    import evaluate as launcher
    from reread.evaluation.shared_queue import SingleNodeSession

    session = SingleNodeSession(tmp_path / "run", {"task_set": task_set})
    rows = [{"job_id": "example", "status": "success"}]
    queue = SimpleNamespace(
        results=lambda: rows,
        summary=lambda: {"complete": True, "completed_episodes": len(rows)},
    )
    summary = launcher.finish_run(session, queue, node_count=1, local_workers=1)
    assert summary["task_set"] == task_set
    for name in ("summary.json", "progress.json", "speed_summary.json"):
        assert json.loads((session.root / name).read_text())["task_set"] == task_set
