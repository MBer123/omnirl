import rlcard as rlcard_lib

from .adapter import EnvAdapter


def build_env(env_config, role=None):
    """Build and adapt a single-player RLCard environment."""
    if role not in {None, "collection", "evaluation", "specification"}:
        raise ValueError(f"Unsupported environment role: {role!r}.")

    config = dict(env_config)
    try:
        env_id = config.pop("id")
    except KeyError:
        raise ValueError(
            "RLCard environment configuration requires an 'id'."
        ) from None
    if config:
        fields = ", ".join(sorted(config))
        raise ValueError(f"Unsupported RLCard environment fields: {fields}.")

    raw_env = rlcard_lib.make(env_id)
    if raw_env.num_players != 1:
        raise ValueError(
            "The RLCard backend currently supports only single-player games; "
            f"'{env_id}' has {raw_env.num_players} players."
        )
    return EnvAdapter(raw_env)
