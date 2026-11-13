from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """
    Forward interaction and spaces to a wrapped environment.

    Args:
        env: The underlying environment instance.
    """

    def __init__(self, env):
        self.env = env

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def render_mode(self):
        return self.env.render_mode

    def reset(self, seed=None, options=None):
        """Reset the environment and return observation, info."""
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        """Return observation, reward, terminated, truncated, info."""
        return self.env.step(action)

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        return self.env.close()


class BaseEnvProvider(ABC):
    """Shared interface for environment construction and specifications."""

    @staticmethod
    @abstractmethod
    def build(env_config: dict, role: str | None = None):
        """Build an environment for the requested role."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_spec(cls, env_config: dict) -> dict:
        """Return env_spec, state_spec and action_spec for an environment."""
        raise NotImplementedError
