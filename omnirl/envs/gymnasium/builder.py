import gymnasium as gym

from .adapter import EnvAdapter


def build_env(env_config, role=None):
    """Build and adapt a Gymnasium environment."""
    config = dict(env_config)
    env_id = config.pop("id")
    render_mode = config.pop("render_mode", None)

    if role in {"collection", "specification"}:
        render_mode = None

    env = gym.make(env_id, render_mode=render_mode)
    return EnvAdapter(env, render_mode)
