import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real

FEATURE_SCHEMA = "reread-pre-rope-camera-v1"
LEGACY_FEATURE_SCHEMA = "newm-memory4-pre-rope-camera-v1"
MODEL_SCHEMA = "newm-endpoint-memory3-action-residual-v1"
CHECKPOINT_SCHEMA = "newm-endpoint-memory3-weights-v1"
GRIPPER_DIMENSIONS = (6, 13)


@dataclass(frozen=True)
class ReReadConfig:
    action_dim: int = 14
    proprio_dim: int = 14
    horizon: int = 32
    kv_dim: int = 3072
    layers: tuple[int, ...] = (4, 14, 24, 29)
    video_tokens: int = 64
    action_tokens: int = 32
    memory_frames: int = 3
    width: int = 1536
    heads: int = 24
    depth: int = 5
    ff_dim: int = 6144
    inference_steps: int = 20
    sigma_shift: float = 5.0
    decision_stride: int = 20
    history_dropout: float = 0.05
    activation_checkpointing: bool = False
    video_grid: tuple[int, int] = (12, 10)
    head_rows: int = 8
    wrist_split: int = 5
    camera_pools: tuple[tuple[int, int], ...] = ((4, 8), (4, 4), (4, 4))
    feature_schema: str = FEATURE_SCHEMA
    action_perturb_probability: float = 0.2
    action_perturb_std: float = 0.05
    model_schema: str = MODEL_SCHEMA

    def __post_init__(self):
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "video_grid", tuple(self.video_grid))
        object.__setattr__(
            self, "camera_pools", tuple(tuple(p) for p in self.camera_pools)
        )
        for key in (
            "action_dim",
            "proprio_dim",
            "horizon",
            "kv_dim",
            "video_tokens",
            "action_tokens",
            "memory_frames",
            "width",
            "heads",
            "depth",
            "ff_dim",
            "inference_steps",
            "decision_stride",
            "head_rows",
            "wrist_split",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for value in (
            *self.layers,
            *self.video_grid,
            *(v for p in self.camera_pools for v in p),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(
                    "Layer, grid and pooling dimensions must be positive integers"
                )
        if (
            isinstance(self.sigma_shift, bool)
            or not isinstance(self.sigma_shift, Real)
            or not math.isfinite(self.sigma_shift)
            or self.sigma_shift <= 0
        ):
            raise ValueError("sigma_shift must be finite and positive")
        if (
            not self.layers
            or self.layers != tuple(sorted(set(self.layers)))
            or min(self.layers) < 1
        ):
            raise ValueError("layers must be sorted, unique and >= 1")
        if self.memory_frames != 3:
            raise ValueError(
                "Memory3 requires the current observation + two past decisions"
            )
        if not isinstance(self.activation_checkpointing, bool):
            raise ValueError("activation_checkpointing must be boolean")
        if self.action_tokens != self.horizon:
            raise ValueError(
                "Action plan tokens must retain the complete horizon, without pooling"
            )
        if self.width % self.heads or self.decision_stride > self.horizon:
            raise ValueError("Invalid head width or executed action stride")
        if (
            isinstance(self.history_dropout, bool)
            or not isinstance(self.history_dropout, Real)
            or not math.isfinite(self.history_dropout)
            or not 0 <= self.history_dropout <= 1
        ):
            raise ValueError("history_dropout must be in [0,1]")
        if len(self.video_grid) != 2 or min(self.video_grid) <= 0:
            raise ValueError("video_grid must contain two positive dimensions")
        h, w = self.video_grid
        if not 0 < self.head_rows < h or not 0 < self.wrist_split < w:
            raise ValueError("Invalid camera boundaries")
        regions = (
            (self.head_rows, w),
            (h - self.head_rows, self.wrist_split),
            (h - self.head_rows, w - self.wrist_split),
        )
        if len(self.camera_pools) != 3 or any(len(p) != 2 for p in self.camera_pools):
            raise ValueError("Three camera pooling grids are required")
        if any(
            min(p) <= 0 or p[0] > r[0] or p[1] > r[1]
            for p, r in zip(self.camera_pools, regions)
        ):
            raise ValueError("Camera pooling cannot upsample its source grid")
        if sum(h * w for h, w in self.camera_pools) != self.video_tokens:
            raise ValueError("Camera pooling grids must sum to video_tokens")
        if self.feature_schema == LEGACY_FEATURE_SCHEMA:
            object.__setattr__(self, "feature_schema", FEATURE_SCHEMA)
        elif self.feature_schema != FEATURE_SCHEMA:
            raise ValueError(
                "Unsupported feature schema; old pooled/RoPE features cannot be reused"
            )
        for name in (
            "action_perturb_probability",
            "action_perturb_std",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.action_perturb_probability > 1:
            raise ValueError("action_perturb_probability must be in [0,1]")
        if self.model_schema != MODEL_SCHEMA:
            raise ValueError("Unsupported endpoint model schema")

    def to_dict(self):
        return asdict(self)

    @property
    def tokens_per_decision(self):
        return len(self.layers) * (self.video_tokens + self.action_tokens)

    @property
    def total_memory_tokens(self):
        return self.memory_frames * self.tokens_per_decision

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise TypeError("Endpoint configuration must be a dictionary")
        settings = dict(value)
        for key in (
            "plan_perturb_probability", "plan_perturb_std", "velocity_weight",
            "acceleration_weight", "seam_velocity_weight", "seam_acceleration_weight",
        ):
            if key in settings:
                retired = settings.pop(key)
                if isinstance(retired, bool) or not isinstance(retired, Real) or retired != 0:
                    raise ValueError("The checkpoint uses unsupported training settings")
        return cls(**settings)


def joint_dimensions(config):
    return tuple(
        index for index in range(config.action_dim) if index not in GRIPPER_DIMENSIONS
    )
