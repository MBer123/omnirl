import gymnasium as gym
import numpy as np
from gymnasium import spaces

from omnirl.envs.base import BaseEnv, BaseEnvProvider


class EnvAdapter(BaseEnv):
    """
    Build and adapt a Gymnasium environment.

    Args:
        env_id: Registered Gymnasium environment ID.
        render_mode: Render mode passed to the environment.
        **kwargs: Additional options passed to gym.make.
    """

    def __init__(self, env_id, render_mode=None, **kwargs):
        super().__init__(gym.make(env_id, render_mode=render_mode, **kwargs))


class EnvProvider(BaseEnvProvider):
    """Build and describe Gymnasium environments."""

    @staticmethod
    def build(env_config, role=None):
        """Build a Gymnasium environment for the requested role."""
        config = dict(env_config)
        env_id = config.pop("id")
        render_mode = config.pop("render_mode", None)

        if role in {"collection", "specification"}:
            render_mode = None

        return EnvAdapter(env_id, render_mode=render_mode, **config)

    @classmethod
    def get_spec(cls, env_config):
        """Describe an environment without exposing Gymnasium spaces."""
        env = cls.build(env_config=env_config, role="specification")

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
        env.close()

        return {
            "env_spec": env_spec,
            "state_spec": {**env_spec, "state_dtype": state_dtype},
            "action_spec": {
                **env_spec,
                "action_high": action_high,
                "action_low": action_low,
            },
        }
