# Implementation

## Model

ReRead.forward(candidate, memory) returns a corrected action chunk.
Candidate and output tensors have shape [B, 32, 14] in normalized action space.

The memory dictionary contains:

| Field | Shape | Meaning |
|---|---|---|
| k, v | [B, 3, 4, 96, 3072] | Native K/V, ordered from oldest to current decision |
| states | [B, 3, 14] | Proprioception at each decision |
| valid | [B, 3] | Boolean record validity |
| timestamps | [B, 3] | Raw action-step indices |
| executed_steps | [B, 3] | Executed steps for each stored record |

K/V are captured before RoPE from layers 4, 14, 24, and 29. Within each layer,
head-camera tokens are pooled to 4 x 8 and wrist-camera tokens to 4 x 4 each.
This gives 64 visual tokens; all 32 action tokens are retained.

Five reader layers use width 1536, 24 attention heads, and FFN width 6144.
The zero-initialized output head makes the initial correction zero.
The backbone and stored historical K/V remain detached.

## Training

data.build_cache encodes current RGB and instructions.
data.dataset.CachedTrajectoryDataset reads the sealed shards.
train configures distributed execution, and engine.stream_updates generates
frozen candidates, reads causal memory, computes MSE, and commits current K/V
after a successful update.

All 50 tasks have equal target decision exposure. The sampler accounts for
trajectory length while preserving chronological order. Clean data use seen
instructions; randomized data use seen and unseen instruction pools. Each
trajectory keeps its selected instruction and image variant.

Candidate arm actions receive Gaussian noise with probability 0.2 and
standard deviation 0.05. Grippers, invalid label rows, and native K/V are
unchanged. History dropout masks whole records: both past records are
dropped with probability 0.05, and only the older record with probability
0.0475. All reader layers share the mask; the current record remains visible.

The loss averages squared action error over valid elements. Distributed
training normalizes by the global valid-element count.

## Inference

The RoboTwin adapter calls the frozen backbone for 20 denoising steps,
corrects the full 32-step candidate, and executes 20 steps before replanning.
History is cleared between episodes. Checkpoints store model tensors,
configuration, execution settings, sampling metadata, and asset/source hashes.
