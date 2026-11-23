import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

import reread.training as training
from reread.data.distributed import DistributedContext, cpu_budget

TRAIN_STEPS = 20_000
WORLD_SIZE = 16
LOCAL_WORLD_SIZE = 8


def topology(environment):
    keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    if any(key not in environment for key in keys):
        raise ValueError("Use the two-node 16-GPU launcher")
    rank, world, local_rank, local_world = (int(environment[key]) for key in keys)
    if (world, local_world) != (16, 8) or not 0 <= rank < 16 or local_rank != rank % 8:
        raise ValueError(
            "Require exactly two nodes, eight local ranks each, sixteen global ranks"
        )
    node_rank = rank // 8
    if int(environment.get("GROUP_RANK", node_rank)) != node_rank:
        raise ValueError("Node/global/local rank mapping differs")
    if int(environment.get("GROUP_WORLD_SIZE", 2)) != 2:
        raise ValueError("Require exactly two nodes")
    return dict(
        rank=rank,
        world_size=world,
        local_rank=local_rank,
        local_world_size=local_world,
        node_rank=node_rank,
        node_count=2,
    )


class TrainingContext(DistributedContext):

    @classmethod
    def initialize(cls, *, timeout_seconds=600, nccl_high_priority=False):
        layout = topology(os.environ)
        if type(timeout_seconds) is not int or timeout_seconds < 30:
            raise ValueError("Distributed timeout must be at least 30 seconds")
        if type(nccl_high_priority) is not bool:
            raise ValueError("NCCL priority must be boolean")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
            raise RuntimeError("Each node must expose exactly eight CUDA GPUs")
        local_rank = layout["local_rank"]
        torch.cuda.set_device(local_rank)
        name = torch.cuda.get_device_name(local_rank)
        free, total = torch.cuda.mem_get_info(local_rank)
        if total < 75 * 1024**3 or free < 60 * 1024**3:
            raise RuntimeError("Require 80GB GPUs and at least 60GiB free on every GPU")
        options = {}
        if nccl_high_priority:
            pg_options = dist.ProcessGroupNCCL.Options()
            pg_options.is_high_priority_stream = True
            options["pg_options"] = pg_options
        dist.init_process_group(
            "nccl",
            init_method="env://",
            timeout=timedelta(seconds=timeout_seconds),
            **options,
        )
        context = cls(device=f"cuda:{local_rank}")
        context.gpu_family, context.gpu_name = "CUDA_80GB", name
        context.layout = layout
        return context


def communication_probe(context):
    value = torch.full(
        (2 * 1024 * 1024,), float(context.rank + 1), device=context.device
    )
    dist.all_reduce(value)
    context.require_all(
        bool(torch.all(value == 136).item()), "16-rank all-reduce correctness failed"
    )
    value.zero_()
    for _ in range(3):
        dist.all_reduce(value)
    torch.cuda.synchronize(context.device)
    started = time.monotonic()
    for _ in range(10):
        dist.all_reduce(value)
    torch.cuda.synchronize(context.device)
    seconds = (time.monotonic() - started) / 10
    print(
        json.dumps(
            dict(
                event="communication_probe",
                rank=context.rank,
                world_size=16,
                payload_bytes=value.numel() * value.element_size(),
                seconds_per_all_reduce=seconds,
                dtype="float32",
                scope="startup_8MiB_collective_not_full_gradient_or_training_throughput",
            )
        ),
        flush=True,
    )


def parser():
    p = training.parser()
    p.description = "Train ReRead for 20K updates on two nodes with eight 80GB GPUs each."

    p.set_defaults(steps=TRAIN_STEPS)
    return p


def validate_args(args):
    if args.steps != TRAIN_STEPS:
        raise ValueError("The mainline recipe uses 20K updates")
    training.validate_args(args, expected_steps=args.steps)


@record
def main(argv=None):
    args = parser().parse_args(argv)
    validate_args(args)

    budget = cpu_budget(
        world_size=LOCAL_WORLD_SIZE, threads=args.cpu_threads, workers=args.data_workers
    )
    budget.update(node_count=2, global_ranks=16, scope="per_node")
    torch.set_num_threads(args.cpu_threads)
    context = TrainingContext.initialize(
        timeout_seconds=args.distributed_timeout_seconds
    )
    try:
        context.agree(
            dict(
                schema="newm-endpoint-baseline-mse-d20-16gpu-v1",
                launch_id=args.launch_id,
                output=str(Path(args.output).resolve()),
                source_manifest_sha256=args.source_manifest_sha256,
                steps=args.steps,
                batch_per_rank=args.batch_size,
                world_size=16,
                node_count=2,
                local_world_size=8,
                lr=args.lr,
                min_lr=args.min_lr,
                warmup_steps=args.warmup_steps,
            )
        )
        communication_probe(context)
        return training.run(args, context, budget)
    finally:
        context.close()


if __name__ == "__main__":
    main()
