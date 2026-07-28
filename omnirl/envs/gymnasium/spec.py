from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from .builder import build_env
from .evaluation import MetricTracker


@dataclass(frozen=True)
class EnvSpec:
    """
    Algorithm-facing description of a Gymnasium environment.

    Args:
        state_dim: Dimension of the environment state space.
        state_dtype: Data type of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        action_low: Lower bounds of the continuous action space.
        action_high: Upper bounds of the continuous action space.
        default_eval_metrics: Initial values of environment evaluation metrics.
    """

    state_dim: int | tuple[int, ...]
    state_dtype: np.dtype
    action_dim: int
    is_continuous: bool
    action_low: np.ndarray | None
    action_high: np.ndarray | None
    default_eval_metrics: dict


def get_env_spec(env_config):
    """Describe an environment without exposing Gymnasium spaces."""
    env = build_env(env_config=env_config, role="specification")
    obs_shape = env.observation_space.shape
    state_dim = obs_shape[0] if len(obs_shape) == 1 else obs_shape
    state_dtype = env.observation_space.dtype

    if isinstance(env.action_space, spaces.Discrete):
        action_dim = env.action_space.n
        is_continuous = False
        action_low = None
        action_high = None
    else:
        action_dim = env.action_space.shape[0]
        is_continuous = True
        action_low = np.asarray(env.action_space.low, dtype=np.float32)
        action_high = np.asarray(env.action_space.high, dtype=np.float32)

    env.close()
    return EnvSpec(
        state_dim=state_dim,
        state_dtype=state_dtype,
        action_dim=action_dim,
        is_continuous=is_continuous,
        action_low=action_low,
        action_high=action_high,
        default_eval_metrics=dict(MetricTracker.default_metrics),
    )
