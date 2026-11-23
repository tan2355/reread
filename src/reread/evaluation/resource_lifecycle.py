import gc
import os
import random
import sys
import time
from contextlib import contextmanager


def acknowledge_worker_exit(
    connection, message, *, torch_module=None, exit_process=None
):
    module = _loaded_torch(torch_module)
    if module is not None and module.cuda.is_initialized():
        module.cuda.synchronize()
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()
    if connection is not None:
        connection.send(message)
        connection.close()
    (os._exit if exit_process is None else exit_process)(0)


def _loaded_torch(module=None):
    return (
        sys.modules.get("torch")
        if module is None
        else (None if module is False else module)
    )


def _error(stage, error):
    return dict(stage=stage, error_type=type(error).__name__, error=str(error))


def memory_snapshot(torch_module=None):
    module = _loaded_torch(torch_module)
    result = dict(
        cuda_initialized=False,
        device=None,
        allocated_bytes=None,
        reserved_bytes=None,
        free_bytes=None,
        total_bytes=None,
    )
    if module is None:
        result["unavailable_reason"] = "torch_not_loaded"
        return result
    try:
        cuda = module.cuda
        result["cuda_initialized"] = bool(cuda.is_initialized())
        if not result["cuda_initialized"]:
            result["unavailable_reason"] = "cuda_not_initialized"
            return result
        device = cuda.current_device()
        free, total = cuda.mem_get_info(device)
        result.update(
            device=int(device),
            allocated_bytes=int(cuda.memory_allocated(device)),
            reserved_bytes=int(cuda.memory_reserved(device)),
            free_bytes=int(free),
            total_bytes=int(total),
        )
    except Exception as error:
        result["snapshot_error"] = _error("memory_snapshot", error)
    return result


@contextmanager
def _preserve_rng(torch_module, errors):
    saved = [("python", random, random.getstate())]
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        try:
            saved.append(("numpy", numpy.random, numpy.random.get_state()))
        except Exception as error:
            errors.append(_error("capture_numpy_rng", error))
    if torch_module is not None:
        try:
            saved.append(("torch_cpu", torch_module, torch_module.get_rng_state()))
        except Exception as error:
            errors.append(_error("capture_torch_cpu_rng", error))
        try:
            if torch_module.cuda.is_initialized():
                saved.append(
                    (
                        "torch_cuda",
                        torch_module.cuda,
                        torch_module.cuda.get_rng_state_all(),
                    )
                )
        except Exception as error:
            errors.append(_error("capture_torch_cuda_rng", error))
    try:
        yield
    finally:
        for name, owner, state in reversed(saved):
            try:
                if name == "python":
                    owner.setstate(state)
                elif name == "numpy":
                    owner.set_state(state)
                elif name == "torch_cuda":
                    owner.set_rng_state_all(state)
                else:
                    owner.set_rng_state(state)
            except Exception as error:
                errors.append(_error("restore_" + name + "_rng", error))


def _detach_environment(env, report):
    attributes = getattr(env, "__dict__", {})
    robot = attributes.get("robot")
    robot_attributes = getattr(robot, "__dict__", {})
    seen = set()
    planners = [
        value
        for name, value in robot_attributes.items()
        if name.endswith("_planner") and value is not None
    ]
    for planner in planners:
        if id(planner) in seen:
            continue
        seen.add(id(planner))
        method_name = next(
            (
                name
                for name in ("close", "dispose")
                if callable(getattr(planner, name, None))
            ),
            None,
        )
        report["planners"].append(dict(type=type(planner).__name__, method=method_name))
        if method_name is not None:
            try:
                getattr(planner, method_name)()
            except Exception as error:
                report["errors"].append(_error("planner_" + method_name, error))

    report["robot_attributes_released"] = len(robot_attributes)
    robot_attributes.clear()
    roots = (
        "eval_video_ffmpeg",
        "sink",
        "now_obs",
        "observation",
        "obs",
        "robot",
        "cameras",
        "camera",
        "viewer",
        "scene",
        "renderer",
        "engine",
    )
    for name in roots:
        if name in attributes:
            del attributes[name]
            report["environment_roots_released"].append(name)

    planners.clear()
    planner = robot = None


def finalize_environment(
    env, *, drop_local_references=None, torch_module=None, collector=None
):
    started = time.perf_counter()
    module = _loaded_torch(torch_module)
    report = dict(
        schema="newm-episode-resource-lifecycle-v1",
        before=memory_snapshot(module if module is not None else False),
        after=None,
        errors=[],
        planners=[],
        environment_roots_released=[],
        robot_attributes_released=0,
        cuda_synchronized=False,
        allocator_cache_released=False,
        garbage_collected=None,
        env_close_called=False,
        model_or_history_modified=False,
    )
    with _preserve_rng(module, report["errors"]):
        if module is not None:
            try:
                if module.cuda.is_initialized():
                    module.cuda.synchronize()
                    report["cuda_synchronized"] = True
            except Exception as error:
                report["errors"].append(_error("cuda_synchronize", error))
        try:
            if env is not None:
                _detach_environment(env, report)
        except Exception as error:
            report["errors"].append(_error("detach_environment", error))
        finally:
            env = None
            if drop_local_references is not None:
                try:
                    drop_local_references()
                except Exception as error:
                    report["errors"].append(_error("drop_local_references", error))
        try:
            report["garbage_collected"] = int(
                (gc.collect if collector is None else collector)()
            )
        except Exception as error:
            report["errors"].append(_error("gc_collect", error))
        if module is not None:
            try:
                if module.cuda.is_initialized():
                    module.cuda.empty_cache()
                    report["allocator_cache_released"] = True
            except Exception as error:
                report["errors"].append(_error("cuda_empty_cache", error))
        report["after"] = memory_snapshot(module if module is not None else False)
    report["elapsed_seconds"] = time.perf_counter() - started
    return report
