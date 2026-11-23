#!/usr/bin/env python3
import sys as _site_sys
from pathlib import Path as _SitePath

_site_sys.path.insert(0, str(_SitePath(__file__).resolve().parents[1] / "src"))
import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from multiprocessing.connection import wait
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
from eval_utils import (
    INFERENCE_SETTINGS,
    ROOT,
    checkpoint_preflight,
    configure_environment,
    cpu_plan,
    policy_settings,
    runtime_preflight,
    sha256,
    verify_release,
)


def worker_entry(connection, config):
    generation = config.get("worker_generation", 0)
    suffix = "" if generation == 0 else f"_generation_{generation:03d}"
    path = (
        Path(config["output_root"])
        / "logs"
        / f"worker_{config['worker_id']:02d}{suffix}.log"
    )
    with path.open("x", encoding="utf-8", buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        try:
            spec = importlib.util.spec_from_file_location(
                "worker_binding", ROOT / "src/reread/evaluation/runtime_setup.py"
            )
            binding = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(binding)
            binding.bind_worker_visibility(config)
            from reread.evaluation.worker import worker_process

            worker_process(connection, config)
        except BaseException as error:
            traceback.print_exc()
            try:
                connection.send(
                    dict(type="error", error=f"{type(error).__name__}: {error}")
                )
            except (OSError, EOFError):
                pass
            raise


def log_event(path, name, **fields):
    value = dict(event=name, **fields)
    line = json.dumps(value, allow_nan=False)
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def run_summary(session, queue):
    return dict(queue.summary(), task_set=session.contract["task_set"])


def run_local_workers(
    args,
    session,
    queue,
    config_template,
    gpus,
    cpu,
    *,
    entry=worker_entry,
    local_workers=16,
):
    from reread.evaluation.runtime_setup import worker_cuda_token
    from reread.evaluation.shared_queue import atomic_json

    context = mp.get_context("spawn")
    processes, pipes, states, activity, idle = {}, {}, {}, {}, set()
    generations, episode_counts = {}, {}
    max_episodes = getattr(args, "worker_max_episodes", 0)
    last_pulse = last_progress = 0.0
    event_path = session.node_dir / "events.jsonl"

    def start_worker(local_id, generation):
        worker_id = session.rank * local_workers + local_id
        gpu, slot = local_id % len(gpus), local_id // len(gpus)
        config = dict(
            config_template,
            worker_id=worker_id,
            node_rank=session.rank,
            worker_generation=generation,
            gpu_index=gpu,
            slot=slot,
            render_pci=gpus[gpu]["pci"],
            cpu_affinity=cpu["worker_affinity"][local_id],
            cpu_threads=cpu.get(
                "worker_threads", [cpu["cpu_threads_per_worker"]] * local_workers
            )[local_id],
            ffmpeg_threads=1,
            cuda_visible_devices=worker_cuda_token(
                gpu, os.environ.get("CUDA_VISIBLE_DEVICES")
            ),
            worker_cuda_index=0,
        )
        parent, child = context.Pipe()
        process = context.Process(
            target=entry,
            args=(child, config),
            name=f"endpoint-eval-{worker_id}-g{generation}",
        )
        try:
            process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        processes[worker_id], pipes[worker_id] = process, parent
        states[worker_id], activity[worker_id] = "starting", time.monotonic()
        generations[worker_id], episode_counts[worker_id] = generation, 0
        suffix = "" if generation == 0 else f"_generation_{generation:03d}"
        atomic_json(
            session.root / "workers" / f"{worker_id:02d}{suffix}.json",
            dict(pid=process.pid, **config),
            exclusive=True,
        )

    try:
        for local_id in range(local_workers):
            start_worker(local_id, 0)
        while any(state != "stopped" for state in states.values()):
            now = time.monotonic()

            for worker in list(states):
                if states[worker] != "recycle_join":
                    continue
                process = processes[worker]
                process.join(timeout=0)
                if process.exitcode is None:
                    if time.monotonic() - activity[worker] > 60:
                        raise TimeoutError(
                            f"Worker {worker} did not exit for bounded resource recycling"
                        )
                    continue
                if process.exitcode != 0:
                    raise RuntimeError(
                        f"Worker {worker} recycling exit={process.exitcode}"
                    )
                pipes[worker].close()
                log_event(
                    event_path,
                    "worker_recycled",
                    worker_id=worker,
                    generation=generations[worker],
                    episodes=episode_counts[worker],
                )
                start_worker(worker % local_workers, generations[worker] + 1)
            if now - last_pulse >= 5:
                session.heartbeat(
                    "evaluating",
                    workers=states,
                    active={str(w): j.job_id for w, j in queue.inflight.items()},
                )
                session.check()
                last_pulse = now
            busy = {
                g: sum((w % local_workers) % len(gpus) == g for w in queue.inflight)
                for g in range(len(gpus))
            }
            for worker in sorted(
                idle, key=lambda w: (busy[(w % local_workers) % len(gpus)], w)
            ):
                job = queue.claim(worker)
                if job is None:
                    pipes[worker].send(dict(type="stop"))
                    states[worker] = "stopping"
                else:
                    pipes[worker].send(dict(type="job", job=job.to_dict()))
                    states[worker] = "episode"
                    busy[(worker % local_workers) % len(gpus)] += 1
                activity[worker] = time.monotonic()
            idle.clear()
            active = {
                p: w
                for w, p in pipes.items()
                if states[w] not in ("stopped", "recycle_join")
            }
            for pipe in wait(list(active), timeout=0.5) if active else []:
                worker = active[pipe]
                try:
                    message = pipe.recv()
                except (EOFError, OSError) as error:
                    raise RuntimeError(
                        f"Worker {worker} exited without clean acknowledgement"
                    ) from error
                kind = message.get("type")
                activity[worker] = time.monotonic()
                if kind == "ready" and states[worker] == "starting":
                    states[worker] = "idle"
                    idle.add(worker)
                    log_event(
                        event_path,
                        "worker_ready",
                        node_rank=session.rank,
                        worker_id=worker,
                    )
                elif kind == "result" and states[worker] == "episode":
                    result = queue.finish(worker, message["result"])
                    if result["status"] == "error":
                        raise RuntimeError(f"Worker {worker}: {result['error']}")
                    episode_counts[worker] += 1
                    if max_episodes and episode_counts[worker] >= max_episodes:
                        pipes[worker].send(dict(type="stop"))
                        states[worker] = "recycling"
                    else:
                        states[worker] = "idle"
                        idle.add(worker)
                    log_event(
                        event_path,
                        "episode_complete",
                        worker_id=worker,
                        job_id=result["job_id"],
                        status=result["status"],
                    )
                elif kind == "progress" and states[worker] == "episode":
                    if message.get("job_id") != queue.inflight[worker].job_id:
                        raise ValueError(
                            "Worker progress refers to a different episode"
                        )
                elif kind == "stopped" and states[worker] == "stopping":
                    states[worker] = "stopped"
                elif kind == "stopped" and states[worker] == "recycling":
                    states[worker] = "recycle_join"
                elif kind == "error":
                    raise RuntimeError(f"Worker {worker}: {message.get('error')}")
                else:
                    raise ValueError(
                        f"Out-of-order worker {worker} message {kind!r} in {states[worker]}"
                    )
            for worker, process in processes.items():
                if states[worker] in ("stopped", "recycle_join"):
                    continue
                limit = (
                    args.startup_timeout
                    if states[worker] == "starting"
                    else args.episode_timeout
                )
                if states[worker] in ("stopping", "recycling"):
                    limit = 60

                undrained_exit = process.exitcode is not None and pipes[worker].poll()
                if (
                    process.exitcode is not None and not undrained_exit
                ) or time.monotonic() - activity[worker] > limit:
                    raise RuntimeError(
                        f"Worker {worker} exit={process.exitcode}, state={states[worker]}, timeout={limit}s"
                    )
            if session.rank == 0 and now - last_progress >= 10:
                atomic_json(session.root / "progress.json", run_summary(session, queue))
                last_progress = now
        for worker, process in processes.items():
            process.join(timeout=10)
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker {worker} did not exit cleanly: {process.exitcode}"
                )
    except BaseException as error:
        session.fail(error)
        for worker in list(queue.inflight):
            queue.abandon(worker, f"Node aborted: {type(error).__name__}: {error}")
        raise
    finally:
        for process in processes.values():
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for pipe in pipes.values():
            pipe.close()


def visible_final_summary(session, queue):
    started = time.monotonic()
    while True:
        session.check()
        summary = run_summary(session, queue)
        if summary["complete"]:
            return summary
        if summary["infrastructure_errors"]:
            raise RuntimeError("Nodes ended with infrastructure errors; no final score")
        if time.monotonic() - started >= min(60, session.heartbeat_timeout):
            raise TimeoutError(
                "Nodes ended but final results remained incomplete/invisible; no retries or final score"
            )
        session.heartbeat(
            "waiting_result_visibility", missing=summary["missing_episodes"]
        )
        time.sleep(0.5)


def finish_run(session, queue, *, node_count=2, local_workers=16):
    from reread.evaluation.shared_queue import atomic_json, read_json

    atomic_json(session.node_dir / "done.json", dict(clean_exit=True), exclusive=True)
    last_progress = None
    while True:
        nodes = session.check()
        session.heartbeat(
            "waiting_for_peer_completion"
            if node_count == 2
            else "finalizing_local_results"
        )
        if len(nodes) == node_count and all((n / "done.json").is_file() for n in nodes):
            break

        if session.rank == 0 and (
            last_progress is None or time.monotonic() - last_progress >= 10
        ):
            atomic_json(session.root / "progress.json", run_summary(session, queue))
            last_progress = time.monotonic()
        time.sleep(1)
    if session.rank == 0:
        summary = visible_final_summary(session, queue)
        summary.update(
            node_count=node_count,
            gpu_count=8 * node_count,
            worker_count=local_workers * node_count,
            elapsed_seconds=time.monotonic() - session.started,
        )
        atomic_json(session.root / "summary.json", summary, exclusive=True)
        atomic_json(session.root / "progress.json", summary)
        from reread.evaluation.speed_summary import build_speed_summary

        speed = build_speed_summary(queue.results(), summary["elapsed_seconds"])
        speed["task_set"] = summary["task_set"]
        atomic_json(session.root / "speed_summary.json", speed, exclusive=True)

        with (session.root / "episodes.jsonl").open("x", encoding="utf-8") as stream:
            for row in sorted(queue.results(), key=lambda r: r["job_id"]):
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        atomic_json(
            session.root / "complete.json",
            dict(
                complete=True,
                completed=summary["completed_episodes"],
                summary_sha256=sha256(session.root / "summary.json"),
                speed_sha256=sha256(session.root / "speed_summary.json"),
            ),
            exclusive=True,
        )
    started = time.monotonic()
    while not (session.root / "complete.json").exists():
        session.check()
        if time.monotonic() - started > session.heartbeat_timeout:
            raise TimeoutError("Final coordinator summary did not arrive")
        session.heartbeat("waiting_summary")
        time.sleep(1)
    return read_json(session.root / "summary.json")


def parser():
    p = argparse.ArgumentParser(
        description="Evaluate ReRead on RoboTwin 2.0 using the native D20/S20 policy."
    )
    p.add_argument(
        "--task-set", choices=("hard11", "all50"), default="hard11",
        help="Evaluate Hard11 (default) or all 50 RoboTwin tasks",
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="Fresh run ID; explicit two-node mode requires the same ID on both nodes",
    )
    p.add_argument("--node-count", type=int, choices=(1, 2), default=2)
    p.add_argument(
        "--processes-per-gpu",
        type=int,
        choices=(2,),
        default=2,
        help="Two GPU workers, with CPU-resident T5 and serialized GPU text staging",
    )
    p.add_argument(
        "--worker-max-episodes",
        type=int,
        default=8,
        help="Cleanly exit/reload after this many completed episodes; no job retries",
    )
    p.add_argument("--corrector", required=True)
    p.add_argument("--output-root", default="artifacts/eval")
    p.add_argument(
        "--checkpoint",
        default="artifacts/ckpt/fastwam_v0_3/robotwin_uncond_3cam_384.pt",
    )
    p.add_argument(
        "--stats",
        default="artifacts/ckpt/fastwam_v0_3/robotwin_uncond_3cam_384_dataset_stats.json",
    )
    p.add_argument("--wan", default="artifacts/ckpt/Wan2.2-TI2V-5B")
    p.add_argument("--upstream", default=str(ROOT / "third_party/fastwam"))
    p.add_argument("--simulator-source", default=str(ROOT / "third_party/RoboTwin"))
    p.add_argument("--assets", default="artifacts/robotwin2_eval_assets")
    p.add_argument("--runtime", default="artifacts/fswm_robotwin_eval_runtime_v1")
    p.add_argument("--extra-deps", default="artifacts/eval_dependencies")
    p.add_argument(
        "--cpu-budget",
        type=int,
        default=123,
        help="PER NODE, capped by actual affinity/cgroup quota",
    )
    p.add_argument("--reserve-cpus", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--text-cache-mib", type=int, default=256)
    p.add_argument(
        "--frozen-graph",
        choices=("auto", "native"),
        default="auto",
        help="Reuse frozen D20 CUDA graph only after two full live byte-equality audits",
    )
    p.add_argument("--minimum-free-mib", type=int, default=70000)
    p.add_argument("--join-timeout", type=int, default=1800)
    p.add_argument("--heartbeat-timeout", type=int, default=600)
    p.add_argument("--startup-timeout", type=int, default=1800)
    p.add_argument("--episode-timeout", type=int, default=3600)
    p.add_argument("--expected-release-sha")
    p.add_argument(
        "--expected-checkpoint-sha",
        help="Optional additional pin; completion SHA always verified on every selected node",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--print-plan", action="store_true")
    group.add_argument("--checkpoint-preflight-only", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if (args.node_count, args.processes_per_gpu) not in ((2, 2), (1, 2)):
        raise ValueError(
            "Supported memory-optimized topologies: one or two nodes x8 GPUs x2 workers"
        )
    local_workers = 8 * args.processes_per_gpu
    if "LOCAL_RANK" in os.environ or "TORCHELASTIC_RUN_ID" in os.environ:
        raise ValueError("Execute once per node outside torchrun/DeepSpeed")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.run_id):
        raise ValueError("Use a short safe common run ID")
    if (
        args.seed < 0
        or not 0 <= args.text_cache_mib <= 4096
        or args.minimum_free_mib < 60000
        or not 1 <= args.worker_max_episodes <= 8
        or min(
            args.join_timeout,
            args.heartbeat_timeout,
            args.startup_timeout,
            args.episode_timeout,
        )
        < 60
    ):
        raise ValueError("Invalid seed/cache/memory/timeouts")
    for digest in (args.expected_release_sha, args.expected_checkpoint_sha):
        if digest is not None and not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Invalid SHA256")
    release_sha = verify_release(ROOT, args.expected_release_sha)
    configure_environment(args, 2)
    import yaml

    from reread.evaluation.scheduler import (
        EPISODES_PER_PHASE,
        SEED_PROTOCOL,
        build_jobs,
        validate_output_root,
    )

    limits = yaml.safe_load(
        (Path(args.simulator_source) / "task_config/_eval_step_limit.yml").read_text()
    )
    jobs = build_jobs(limits, seed=args.seed, task_set=args.task_set)
    output_root = Path(args.output_root).resolve()
    output = output_root / args.run_id
    validate_output_root(output, allowed_root=str(output_root))
    plan = dict(
        schema="newm-endpoint-eval-topology-v2",
        source=str(ROOT),
        release_sha256=release_sha,
        corrector=args.corrector,
        checkpoint_step=20000,
        model_schema="newm-endpoint-memory3-action-residual-v1",
        output=str(output),
        run_id=args.run_id,
        nodes=args.node_count,
        gpus_per_node=8,
        processes_per_gpu=args.processes_per_gpu,
        workers=local_workers * args.node_count,
        collectives=False,
        task_set=args.task_set,
        tasks=list(dict.fromkeys(j.task for j in jobs)),
        episodes_per_task_phase=EPISODES_PER_PHASE[args.task_set],
        total_episodes=len(jobs),
        warmup_decisions=0,
        correction_start="first_decision",
        correction_start_decision=0,
        denoising_steps=20,
        replan_steps=20,
        training_history_dropout=0.05,
        inference_history_dropout=0,
        corrector_autocast=True,
        precision_runtime_source="explicit_checkpoint_metadata",
        policy_rgb_mode="native_rgb",
        video="failures_only",
        scheduling="global_atomic_claims_difficulty_times_action_limit_longest_first",
        seed_protocol=SEED_PROTOCOL,
        seed=args.seed,
        instruction_type={"demo_clean": "seen", "demo_randomized": "unseen"},
        randomized_unseen_note="included in training; not held-out language",
        endpoint_runtime=INFERENCE_SETTINGS,
        text_cache_mib=args.text_cache_mib,
        frozen_graph=args.frozen_graph,
        graph_live_audit_decisions=2,
        memory_safety=dict(
            one_worker_per_gpu=False,
            explicit_episode_cleanup=True,
            text_encoder_cpu_resident=True,
            text_compute="native_GPU_BF16",
            gpu_sibling_text_staging_mutex=True,
            worker_max_episodes=args.worker_max_episodes,
        ),
        cpu_budget_per_node=args.cpu_budget,
        reserve_cpus_per_node=args.reserve_cpus,
        output_must_be_fresh=True,
        automatic_retries=False,
        training_loss="masked_action_mse",
        corrector_identity="node_local_signature_around_SHA_load_and_worker_construction",
    )
    if args.print_plan:
        print(json.dumps(plan, indent=2), flush=True)
        return plan
    if os.name != "posix" and not args.checkpoint_preflight_only:
        raise RuntimeError("Evaluation execution requires Linux CUDA nodes")
    if output.parent.resolve() != output_root or output.is_symlink():
        raise ValueError(
            "Evaluation output parent must be the real eval directory, without symlink redirection"
        )
    identity = checkpoint_preflight(args)
    plan["model_source_sha256"] = identity["provenance"]["source_manifest_sha256"]
    if args.checkpoint_preflight_only:
        result = dict(
            plan,
            checkpoint_identity=identity,
            cuda_initialized=False,
            evaluation_started=False,
        )
        print(json.dumps(result, indent=2), flush=True)
        return result
    runtime = runtime_preflight(args)
    contract = dict(
        plan,
        checkpoint_sha256=identity["checkpoint_sha256"],
        policy=policy_settings(args, identity),
        jobs=[j.to_dict() for j in jobs],
        simulator_source=args.simulator_source,
        assets=args.assets,
        simulator_limits_sha256=sha256(
            Path(args.simulator_source) / "task_config/_eval_step_limit.yml"
        ),
        runtime=runtime,
        join_timeout=args.join_timeout,
        heartbeat_timeout=args.heartbeat_timeout,
        startup_timeout=args.startup_timeout,
        episode_timeout=args.episode_timeout,
    )
    cpu = cpu_plan(args.cpu_budget, args.reserve_cpus, local_workers=local_workers)
    configure_environment(args, cpu["cpu_threads_per_worker"])
    from reread.evaluation.runtime_setup import prepare_simulator, probe_gpus
    from reread.evaluation.shared_queue import (
        SharedEpisodeQueue,
        SingleNodeSession,
        TwoNodeSession,
        atomic_json,
    )

    gpus = probe_gpus(args.minimum_free_mib)
    session_type = TwoNodeSession if args.node_count == 2 else SingleNodeSession
    session = session_type(
        output,
        contract,
        join_timeout=args.join_timeout,
        heartbeat_timeout=args.heartbeat_timeout,
    )
    queue = None
    try:
        session.join()
        for name in ("logs", "workers", "runtime_cache"):
            (output / name).mkdir(exist_ok=True)
        session.heartbeat("preparing_private_simulator")
        sim = prepare_simulator(
            session.node_dir,
            args.simulator_source,
            args.assets,
            args.upstream,
            ROOT / "experiments/robotwin/reread_policy",
        )
        if session.rank == 0:
            atomic_json(
                output / "manifest.json",
                dict(plan, checkpoint_identity=identity, contract=contract),
                exclusive=True,
            )
            atomic_json(output / "jobs.json", contract["jobs"], exclusive=True)
        queue = SharedEpisodeQueue(
            output, jobs, session.rank, local_workers=local_workers
        )
        session.ready(
            dict(cpu=cpu, gpu_inventory=gpus, runtime=runtime, simulator_root=str(sim))
        )
        os.sched_setaffinity(0, cpu["coordinator_affinity"])

        local_policy = dict(
            contract["policy"],
            _endpoint_preverified_files=identity["local_preverified_files"],
            _endpoint_preverified_corrector=identity["local_preverified_corrector"],
        )
        template = dict(
            output_root=str(output),
            sim_root=str(sim),
            policy_args=local_policy,
            seed=args.seed,
            text_encoder_offload=True,
        )
        run_local_workers(
            args, session, queue, template, gpus, cpu, local_workers=local_workers
        )
        summary = finish_run(
            session, queue, node_count=args.node_count, local_workers=local_workers
        )
        print(
            json.dumps(dict(event="complete", output=str(output), summary=summary)),
            flush=True,
        )
        return summary
    except BaseException as error:
        session.fail(error)
        if queue is not None:
            diagnostic = dict(
                run_summary(session, queue), complete=False, coordinator_error=str(error)
            )
            atomic_json(session.node_dir / "partial_summary.json", diagnostic)
            if session.rank == 0:
                atomic_json(output / "summary.json", diagnostic)
        raise


if __name__ == "__main__":
    main()
