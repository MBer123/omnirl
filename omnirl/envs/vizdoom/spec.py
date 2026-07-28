from dataclasses import dataclass

import numpy as np

from .builder import build_env
from .evaluation import MetricTracker


@dataclass(frozen=True)
class EnvSpec:
    """Algorithm-facing description of a single-agent ViZDoom environment."""

    state_dim: tuple[int, int, int]
    state_dtype: np.dtype
    action_dim: int
    is_continuous: bool
    action_low: np.ndarray | None
    action_high: np.ndarray | None
    default_eval_metrics: dict


def get_env_spec(env_config):
    """Describe a ViZDoom environment without exposing native game objects."""
    env = build_env(env_config=env_config, role="specification")
    try:
        return EnvSpec(
            state_dim=env.observation_shape,
            state_dtype=env.observation_dtype,
            action_dim=env.num_actions,
            is_continuous=False,
            action_low=None,
            action_high=None,
            default_eval_metrics=dict(MetricTracker.default_metrics),
        )
    finally:
        env.close()
