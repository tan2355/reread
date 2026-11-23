from collections.abc import Mapping

RGB_MODES = ("native_rgb",)
COMPATIBLE_CHECKPOINT_SCHEMA = "newm-endpoint-memory3-weights-v1"


def validate_rgb_mode(mode):
    if mode != "native_rgb":
        raise ValueError("endpoint_rgb_mode must be native_rgb")
    return mode


def validate_checkpoint_rgb_mode(mode, payload):
    validate_rgb_mode(mode)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != COMPATIBLE_CHECKPOINT_SCHEMA
        or payload.get("sampling", {}).get("observation_color_contract")
        != "native_robotwin_rgb_v1"
    ):
        raise ValueError(
            "Require a checkpoint trained with native RoboTwin RGB observations"
        )
    return mode


def policy_observation(observation, mode):
    validate_rgb_mode(mode)
    return observation
