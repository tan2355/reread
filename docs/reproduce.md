# Reproduction

Use Python 3.12. From the source root, install the dependencies with:

~~~bash
python -m pip install -r requirements.txt
~~~
On Windows, a short extraction path such as C:/src/reread avoids dependency
installation path-length limits.

The reference GPU runtime uses Linux, PyTorch 2.9.0+cu126, CUDA 12.6 and
NCCL 2.27.5. The original version inventory is in
constraints/gpu-reference.txt. To apply those version constraints:

~~~bash
python -m pip install -r requirements.txt -c constraints/gpu-reference.txt
~~~

Training uses two nodes with eight 80GB GPUs each.

Add third_party/fastwam/src to PYTHONPATH when running training.

Evaluation requires RoboTwin assets, SAPIEN 3.0.0b1, NumPy 1.x, mplib, toppra,
trimesh, cuRobo and ffmpeg with libx264. Its --runtime directory contains
cuda/, curobo/src/, sim/, geometry/, flash_attn/ and bin/. Additional Python
packages can be supplied with --extra-deps. These external binaries, assets
and model weights are not part of the source archive.


## Assets

- Official Fast-WAM RoboTwin weights: robotwin_uncond_3cam_384.pt.
- Matching statistics: robotwin_uncond_3cam_384_dataset_stats.json.
- Wan2.2-TI2V-5B encoder/tokenizer assets.
- RoboTwin demonstrations with three-camera RGB, proprioception,
  instruction files and expert actions.
- RoboTwin simulator assets for closed-loop evaluation.

Checkpoint metadata identifies the frozen backbone, normalization statistics,
source manifest and observation cache by SHA-256.

## 1. Build the observation cache

Prepare a fresh directory:

~~~bash
python -m reread.data.build_cache --stage prepare \
  --raw-root /data/robotwin --output-root /data/reread_cache \
  --upstream third_party/fastwam --wan /weights/Wan2.2-TI2V-5B \
  --world-size 8
~~~

Encode once per rank 0–7 on its assigned GPU, keeping preparation arguments
identical. For rank 0:

~~~bash
python -m reread.data.build_cache --stage encode \
  --raw-root /data/robotwin --output-root /data/reread_cache \
  --upstream third_party/fastwam --wan /weights/Wan2.2-TI2V-5B \
  --world-size 8 --rank 0 --device cuda:0
~~~

After all ranks finish:

~~~bash
python -m reread.data.build_cache --stage seal \
  --raw-root /data/robotwin --output-root /data/reread_cache \
  --upstream third_party/fastwam --wan /weights/Wan2.2-TI2V-5B \
  --world-size 8
~~~

Fresh caches encode RGB and text. The seal
verifies shard/index hashes and binds them to the preparation plan.
Two photometric variants supplement the original observations. Action labels retain their original timestamps.

## 2. Train

The recipe uses two nodes, eight 80GB GPUs per node, batch size 64 per GPU,
and 20K updates. Both nodes must see the same source, cache, weights and output.
Choose one fresh 32-hex LAUNCH_ID and reuse it on both nodes. NODE_RANK is 0
or 1; MASTER_ADDR identifies node 0.

~~~bash
export PYTHONPATH="$PWD/src:$PWD/third_party/fastwam/src"
torchrun --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port=29500 \
  -m reread.train --launch-id "$LAUNCH_ID" \
  --cache-root /data/reread_cache --output /weights/reread_run \
  --upstream third_party/fastwam \
  --checkpoint /weights/robotwin_uncond_3cam_384.pt \
  --stats /weights/robotwin_uncond_3cam_384_dataset_stats.json \
  --wan /weights/Wan2.2-TI2V-5B
~~~

The task inventory defines all 50 tasks, each with target probability 0.02. Task/phase allocation balances decision counts, accounting for
trajectory length. Anchors advance chronologically with private history.

Training writes run.json, endpoint_step_20000.pt and complete.json.
The final file binds the checkpoint name, size and hash.
After intentional source edits, run python tools/refresh_manifest.py.

## 3. Infer and evaluate

The deployment adapter loads the frozen backbone and corrector. It corrects
the full candidate chunk, executes 20 steps and resets history between episodes.

The default task set, `hard11`, runs 25 clean and 25 randomized episodes per
task (550 total). Add `--task-set all50` to the command below to run 100 clean
and 100 randomized episodes on each of the 50 tasks (10,000 total).
Run the command once per node with the same task set, run ID and output root.
The default is two nodes. Use `--node-count 1` for one eight-GPU node.

~~~bash
python scripts/evaluate.py \
  --run-id evaluation_01 --output-root /data/eval \
  --corrector /weights/reread_run/endpoint_step_20000.pt \
  --checkpoint /weights/robotwin_uncond_3cam_384.pt \
  --stats /weights/robotwin_uncond_3cam_384_dataset_stats.json \
  --wan /weights/Wan2.2-TI2V-5B --assets /data/robotwin_assets \
  --runtime /opt/robotwin_runtime --extra-deps /opt/robotwin_dependencies
~~~

Use `--print-plan` to inspect the schedule or `--checkpoint-preflight-only` to
validate weights on CPU. The plan, run manifest and summaries record the
selected `task_set`. Each run uses a fresh output directory. Every task and
condition receives the same number of episodes and the same weight in the score.

Within each task and condition, episode `i` starts at seed `base + i`. Each
time the expert rejects a scene, the seed advances by 25 for `hard11` or 100
for `all50`. Policy failures count as episodes. Infrastructure failures stop
the run.
