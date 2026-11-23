import torch
from torch import nn
from torch.nn import functional as F

from reread.config import MODEL_SCHEMA, ReReadConfig
from reread.memory.spatial import video_token_metadata


class ReaderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config
        self.self_attn = nn.MultiheadAttention(
            c.width, c.heads, dropout=0.0, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            c.width, c.heads, dropout=0.0, batch_first=True
        )
        self.linear1 = nn.Linear(c.width, c.ff_dim)
        self.linear2 = nn.Linear(c.ff_dim, c.width)
        self.norm1 = nn.LayerNorm(c.width)
        self.norm2 = nn.LayerNorm(c.width)
        self.norm3 = nn.LayerNorm(c.width)

    def forward(self, query, content, position, padding_mask, key_content=None):
        if key_content is None:
            key_content = content + position
        normalized = self.norm1(query)
        query = (
            query
            + self.self_attn(normalized, normalized, normalized, need_weights=False)[0]
        )
        normalized = self.norm2(query)
        query = (
            query
            + self.cross_attn(
                normalized,
                key_content,
                content,
                key_padding_mask=padding_mask,
                need_weights=False,
            )[0]
        )
        return query + self.linear2(F.relu(self.linear1(self.norm3(query))))


class ReRead(nn.Module):

    model_schema = MODEL_SCHEMA

    def __init__(self, config=None):
        super().__init__()
        self.config = c = ReReadConfig() if config is None else config
        if not isinstance(c, ReReadConfig):
            raise TypeError("ReRead requires ReReadConfig")
        self.key_proj = nn.Linear(c.kv_dim, c.width)
        self.value_proj = nn.Linear(c.kv_dim, c.width)
        self.state_proj = nn.Linear(c.proprio_dim, c.width)
        self.memory_norm = nn.LayerNorm(c.width)
        self.layer_embed = nn.Embedding(len(c.layers), c.width)
        self.camera_embed = nn.Embedding(4, c.width)
        self.modality_embed = nn.Embedding(2, c.width)
        self.spatial_proj = nn.Linear(2, c.width, bias=False)
        self.time_proj = nn.Linear(2, c.width, bias=False)
        self.plan_position = nn.Embedding(c.horizon, c.width)
        self.action_proj = nn.Linear(c.action_dim, c.width)
        self.action_position = nn.Embedding(c.horizon, c.width)
        self.reader = nn.ModuleList([ReaderLayer(c) for _ in range(c.depth)])
        self.output_norm = nn.LayerNorm(c.width)
        self.output = nn.Linear(c.width, c.action_dim)
        cameras, positions = video_token_metadata(c)
        self.register_buffer("video_cameras", cameras, persistent=False)
        self.register_buffer("video_positions", positions, persistent=False)

        for layer in self.reader:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _memory(self, memory, *, return_visibility=False):
        c = self.config
        required = {"k", "v", "valid", "timestamps", "states", "executed_steps"}
        if set(memory) != required:
            raise ValueError(f"Memory fields must be exactly {sorted(required)}")
        k, v = memory["k"], memory["v"]
        shape = (
            c.memory_frames,
            len(c.layers),
            c.video_tokens + c.action_tokens,
            c.kv_dim,
        )
        if k.ndim != 5 or k.shape[1:] != shape or v.shape != k.shape or k.shape[0] < 1:
            raise ValueError(f"Memory KV must have shape [B,{shape}]")
        batch = k.shape[0]
        for name in ("valid", "timestamps", "executed_steps"):
            if memory[name].shape != (batch, c.memory_frames):
                raise ValueError(f"Invalid memory {name} shape")
        if memory["states"].shape != (batch, c.memory_frames, c.proprio_dim):
            raise ValueError("Invalid memory states shape")
        if memory["valid"].dtype != torch.bool or not bool(
            memory["valid"][:, -1].all()
        ):
            raise ValueError("Current observation must be visible in every row")
        for name in ("timestamps", "executed_steps"):
            if memory[name].dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must contain integer environment steps")
        device, dtype = self.key_proj.weight.device, self.key_proj.weight.dtype
        source_valid = memory["valid"].to(device=device)
        valid = source_valid.clone()
        if self.training and c.history_dropout:
            draw = torch.rand((batch,), device=device)
            valid[:, 1] &= draw >= c.history_dropout
            valid[:, 0] &= draw >= c.history_dropout * (2 - c.history_dropout)

        padding = ~valid[:, :, None, None, None]
        k = k.detach().to(device=device, dtype=dtype).masked_fill(padding, 0)
        v = v.detach().to(device=device, dtype=dtype).masked_fill(padding, 0)
        states = memory["states"].detach().to(device=device, dtype=dtype)
        states = states.masked_fill(~valid[:, :, None], 0)
        content = self.key_proj(k) + self.value_proj(v)
        content = self.memory_norm(content + self.state_proj(states)[:, :, None, None])

        spatial = self.spatial_proj(self.video_positions.to(dtype=dtype))
        visual_position = (
            spatial
            + self.camera_embed(self.video_cameras)
            + self.modality_embed.weight[0]
        )
        plan_position = (
            self.plan_position.weight
            + self.camera_embed.weight[3]
            + self.modality_embed.weight[1]
        )
        position = torch.cat((visual_position, plan_position), dim=0)[None, None, None]
        position = position + self.layer_embed.weight[None, None, :, None]
        times = memory["timestamps"].detach().to(device=device)
        executed = memory["executed_steps"].detach().to(device=device)
        age = times[:, -1:] - times
        if (
            bool(((age < 0) & source_valid).any())
            or bool(((executed < 0) & source_valid).any())
            or bool(((times < 0) & source_valid).any())
        ):
            raise ValueError("Future records or negative executed steps are forbidden")
        if bool(((age[:, :-1] <= 0) & source_valid[:, :-1]).any()):
            raise ValueError(
                "Historical observations must precede the current observation"
            )
        if (
            bool((executed[:, -1] != 0).any())
            or bool(((executed > c.horizon) & source_valid).any())
            or bool(((executed[:, :-1] == 0) & source_valid[:, :-1]).any())
        ):
            raise ValueError("Invalid observed execution prefix length")
        timing = torch.stack(
            (
                torch.log1p(age.clamp_min(0).float() / c.decision_stride),
                executed.float() / c.horizon,
            ),
            dim=-1,
        ).to(dtype=dtype)
        timing = timing.masked_fill(~valid[:, :, None], 0)
        position = position + self.time_proj(timing)[:, :, None, None]
        token_mask = (~valid)[:, :, None, None].expand(
            -1, -1, len(c.layers), c.video_tokens + c.action_tokens
        )
        content = content.masked_fill(token_mask[..., None], 0).flatten(1, 3)
        position = position.expand(batch, -1, -1, -1, -1)
        position = position.masked_fill(token_mask[..., None], 0).flatten(1, 3)
        result = (content, position, token_mask.flatten(1))
        return (*result, valid) if return_visibility else result

    def forward(self, action, memory, *, return_aux=False):
        config = self.config
        if action.ndim != 3 or action.shape[1:] != (config.horizon, config.action_dim):
            raise ValueError("Action must be [B,horizon,action_dim]")
        if action.shape[0] != memory["k"].shape[0]:
            raise ValueError("Action and memory batch sizes differ")

        candidate = action.detach().to(
            device=self.action_proj.weight.device, dtype=torch.float32
        )
        content, position, mask, visible = self._memory(memory, return_visibility=True)
        key_content = content + position
        query = self.action_proj(candidate.to(self.action_proj.weight.dtype))
        query = query + self.action_position.weight[None]
        for layer in self.reader:
            if (
                config.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                from torch.utils.checkpoint import checkpoint

                query = checkpoint(
                    layer,
                    query,
                    content,
                    position,
                    mask,
                    key_content,
                    use_reentrant=False,
                )
            else:
                query = layer(query, content, position, mask, key_content)
        actions = candidate + self.output(self.output_norm(query)).float()
        return (
            {"actions": actions, "history_visible": visible} if return_aux else actions
        )

    def loss(self, action, memory, target, valid=None):
        from reread.loss import action_mse

        if valid is None:
            valid = torch.ones(action.shape[:2], device=action.device, dtype=torch.bool)
        return action_mse(self(action, memory), target, valid, self.config)[0]
