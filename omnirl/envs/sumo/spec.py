from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from .builder import build_env
from .evaluation import MetricTracker


@dataclass(frozen=True)
class EnvSpec:
    """Algorithm-facing description of a single-agent SUMO environment."""

    state_dim: int
    state_dtype: np.dtype
    action_dim: int
    is_continuous: bool
    action_low: np.ndarray | None
    action_high: np.ndarray | None
    default_eval_metrics: dict


def get_env_spec(env_config):
    """Describe a SUMO environment without exposing SUMO-RL objects."""
    env = build_env(env_config=env_config, role="specification")
    try:
        observation_space = env.observation_space
        action_space = env.action_space

        if not isinstance(observation_space, spaces.Box):
            raise ValueError("The SUMO backend requires a Box observation space.")
        if len(observation_space.shape) != 1:
            raise ValueError(
                "The SUMO backend supports vector observations; "
                f"received shape {observation_space.shape}."
            )
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError("The SUMO backend requires a discrete action space.")

        return EnvSpec(
            state_dim=int(observation_space.shape[0]),
            state_dtype=observation_space.dtype,
            action_dim=int(action_space.n),
            is_continuous=False,
            action_low=None,
            action_high=None,
            default_eval_metrics=dict(MetricTracker.default_metrics),
        )
    finally:
        env.close()
