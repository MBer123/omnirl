import gymnasium as gym

from .adapter import EnvAdapter


def build_env(env_config, role=None):
    """Build and adapt a Gymnasium environment."""
    if role not in {None, "collection", "evaluation", "specification"}:
        raise ValueError(f"Unsupported environment role: {role!r}.")

    config = dict(env_config)
    try:
        env_id = config.pop("id")
    except KeyError:
        raise ValueError(
            "Gymnasium environment configuration requires an 'id'."
        ) from None
    render_mode = config.pop("render_mode", None)
    if config:
        fields = ", ".join(sorted(config))
        raise ValueError(f"Unsupported Gymnasium environment fields: {fields}.")

    if role in {"collection", "specification"}:
        render_mode = None

    env = gym.make(env_id, render_mode=render_mode)
    return EnvAdapter(env, render_mode)
