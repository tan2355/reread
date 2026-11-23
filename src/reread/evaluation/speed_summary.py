import math
import statistics


def build_speed_summary(results, elapsed_seconds):
    rows = list(results)
    scored = [r for r in rows if r.get("status") in ("success", "failure")]
    usable = []
    for row in scored:
        decisions = row.get("memory_cadence_audit", {}).get("decisions")
        seconds = row.get("infer_s")
        if (
            isinstance(decisions, list)
            and decisions
            and type(seconds) in (int, float)
            and math.isfinite(seconds)
            and seconds > 0
        ):
            usable.append((row, len(decisions), float(seconds)))
    total_decisions = sum(n for _, n, _ in usable)
    infer_seconds = sum(s for _, _, s in usable)
    actions = sum(row.get("actions", 0) for row in scored)
    by_cell = {}
    for row, count, seconds in usable:
        key = row["task"] + "/" + row["phase"]
        cell = by_cell.setdefault(
            key, dict(episodes=0, decisions=0, infer_seconds_sum=0.0)
        )
        cell["episodes"] += 1
        cell["decisions"] += count
        cell["infer_seconds_sum"] += seconds
    for cell in by_cell.values():
        cell["seconds_per_decision"] = cell["infer_seconds_sum"] / cell["decisions"]
    return dict(
        schema="newm-live-evaluation-speed-v1",
        scored_episodes=len(scored),
        timing_episodes=len(usable),
        timing_complete=len(usable) == len(scored),
        decisions=total_decisions,
        infer_seconds_sum_across_workers=infer_seconds,
        mean_policy_seconds_per_decision=infer_seconds / total_decisions
        if total_decisions
        else None,
        median_episode_mean_seconds_per_decision=statistics.median(
            s / n for _, n, s in usable
        )
        if usable
        else None,
        evaluation_elapsed_seconds=elapsed_seconds,
        evaluation_decisions_per_wall_second=total_decisions / elapsed_seconds
        if elapsed_seconds > 0
        else None,
        evaluation_actions_per_wall_second=actions / elapsed_seconds
        if elapsed_seconds > 0
        else None,
        evaluation_episodes_per_hour=len(scored) * 3600 / elapsed_seconds
        if elapsed_seconds > 0
        else None,
        by_task_phase=by_cell,
        timing_scope="existing live infer_s host wall timing with32 GPU workers; includes observed cold/graph setup costs",
        throughput_scope="whole evaluation after checkpoint acceptance: includes load/admission/simulation/video/cleanup",
        training_wait_excluded=True,
        timing_includes_co_resident_worker_contention=True,
        isolated_gpu_or_kernel_benchmark=False,
    )
