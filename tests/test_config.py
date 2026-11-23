import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reread.config import CHECKPOINT_SCHEMA, MODEL_SCHEMA, ReReadConfig
from reread.model import ReRead


def test_config_roundtrip_and_published_defaults():
    config = ReReadConfig.from_dict(
        json.loads((Path(__file__).parents[1] / "configs/reread.json").read_text())
    )
    assert config == ReReadConfig()
    assert ReReadConfig.from_dict(config.to_dict()) == config
    assert config.model_schema == MODEL_SCHEMA
    assert CHECKPOINT_SCHEMA == "newm-endpoint-memory3-weights-v1"
    assert config.action_perturb_probability == 0.2
    assert config.action_perturb_std == 0.05
    assert config.activation_checkpointing is False
    with torch.device("meta"):
        model = ReRead(config)
    assert sum(p.numel() for p in model.parameters()) == 198_523_406
    assert model.config.total_memory_tokens == 1152


@pytest.mark.parametrize(
    "changes",
    [
        dict(model_schema="old"),
        dict(action_perturb_probability=1.1),
        dict(action_perturb_probability=True),
        dict(action_perturb_std=-0.1),
    ],
)
def test_invalid_endpoint_configs_rejected(changes):
    with pytest.raises(ValueError):
        replace(ReReadConfig(), **changes)


def test_unknown_config_and_wrong_model_type_rejected():
    with pytest.raises(TypeError):
        ReReadConfig.from_dict({"old_noise_mode": 1})
    with pytest.raises(TypeError):
        ReReadConfig.from_dict([])
    with pytest.raises(TypeError):
        ReRead({})
