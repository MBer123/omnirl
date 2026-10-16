import numpy as np
from gymnasium import spaces

from .builder import build_env
from .evaluation import MetricTracker


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

    env_spec = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "is_continuous": is_continuous,
    }
    state_spec = {
        **env_spec,
        "state_dtype": state_dtype,
    }
    action_spec = {
        **env_spec,
        "action_high": action_high,
        "action_low": action_low,
    }
    env.close()

    return {
        "env_spec": env_spec,
        "state_spec": state_spec,
        "action_spec": action_spec,
        "eval_metrics": MetricTracker.default_metrics,
    }
