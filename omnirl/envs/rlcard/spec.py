from dataclasses import dataclass

import numpy as np

from .builder import build_env
from .evaluation import MetricTracker
from .protocol import extract_observation, seed_and_reset


@dataclass(frozen=True)
class EnvSpec:
    """
    Algorithm-facing description of an RLCard environment.

    Args:
        state_dim: Dimension of the environment state space.
        state_dtype: Data type of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        action_low: Lower bounds of the continuous action space.
        action_high: Upper bounds of the continuous action space.
        default_eval_metrics: Initial values of environment evaluation metrics.
    """

    state_dim: int
    state_dtype: np.dtype
    action_dim: int
    is_continuous: bool
    action_low: np.ndarray | None
    action_high: np.ndarray | None
    default_eval_metrics: dict


def get_env_spec(env_config):
    """Describe an environment without exposing RLCard game state."""
    env = build_env(env_config=env_config, role="specification")
    state = seed_and_reset(env)
    observation = extract_observation(state)

    return EnvSpec(
        state_dim=int(observation.shape[0]),
        state_dtype=observation.dtype,
        action_dim=int(env.num_actions),
        is_continuous=False,
        action_low=None,
        action_high=None,
        default_eval_metrics=dict(MetricTracker.default_metrics),
    )
