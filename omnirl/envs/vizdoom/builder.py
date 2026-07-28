from itertools import product
import os
from pathlib import Path
import tempfile
import uuid

from .adapter import EnvAdapter


_ROLES = {None, "collection", "evaluation", "specification"}
_SCREEN_FORMATS = {"GRAY8": 1, "RGB24": 3}


def _doom_config_path():
    """Return a process-local temporary ZDoom configuration path."""
    name = f"omnirl-vizdoom-{os.getpid()}-{uuid.uuid4().hex}.ini"
    return Path(tempfile.gettempdir()) / name


def _load_vizdoom():
    """Load ViZDoom lazily so importing OmniRL keeps it optional."""
    try:
        import vizdoom
    except ImportError as error:
        raise ImportError(
            "The ViZDoom backend requires 'vizdoom'. Install the ViZDoom "
            "environment requirements before starting training."
        ) from error
    return vizdoom


def _scenario_path(vizdoom, env_id):
    """Resolve a bundled scenario name or an explicit configuration path."""
    path = Path(env_id)
    if path.suffix == "":
        path = path.with_suffix(".cfg")

    candidates = (path, Path(vizdoom.scenarios_path) / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(f"ViZDoom scenario configuration not found: {env_id}")


def _enum_value(enum_cls, name, field):
    """Resolve a user-facing enum name with a clear configuration error."""
    try:
        return getattr(enum_cls, name)
    except (AttributeError, TypeError):
        raise ValueError(f"Unsupported ViZDoom {field}: {name!r}.") from None


def _build_action_table(vizdoom, game, max_buttons_pressed):
    """Build a discrete table of no-op and bounded button combinations."""
    buttons = list(game.get_available_buttons())
    if not buttons:
        raise ValueError("The ViZDoom scenario must define available buttons.")

    delta_buttons = [button.name for button in buttons if vizdoom.is_delta_button(button)]
    if delta_buttons:
        names = ", ".join(delta_buttons)
        raise ValueError(
            "The single-agent ViZDoom backend supports only binary buttons; "
            f"the scenario defines delta buttons: {names}."
        )

    if isinstance(max_buttons_pressed, bool) or not isinstance(
        max_buttons_pressed, int
    ):
        raise ValueError("ViZDoom max_buttons_pressed must be an integer.")
    if not 1 <= max_buttons_pressed <= len(buttons):
        raise ValueError(
            "ViZDoom max_buttons_pressed must be between 1 and the number "
            f"of available buttons ({len(buttons)})."
        )

    return [
        action
        for action in product((0, 1), repeat=len(buttons))
        if sum(action) <= max_buttons_pressed
    ]


def build_env(env_config, role=None):
    """Build a native single-player ViZDoom environment."""
    if role not in _ROLES:
        raise ValueError(f"Unsupported environment role: {role!r}.")

    config = dict(env_config)
    try:
        env_id = config.pop("id")
    except KeyError:
        raise ValueError(
            "ViZDoom environment configuration requires an 'id'."
        ) from None

    frame_skip = config.pop("frame_skip", 4)
    max_buttons_pressed = config.pop("max_buttons_pressed", 1)
    screen_resolution = config.pop("screen_resolution", "RES_160X120")
    screen_format = config.pop("screen_format", "RGB24")
    render_mode = config.pop("render_mode", None)
    if config:
        fields = ", ".join(sorted(config))
        raise ValueError(f"Unsupported ViZDoom environment fields: {fields}.")

    if isinstance(frame_skip, bool) or not isinstance(frame_skip, int):
        raise ValueError("ViZDoom frame_skip must be a positive integer.")
    if frame_skip <= 0:
        raise ValueError("ViZDoom frame_skip must be a positive integer.")
    if render_mode not in {None, "human"}:
        raise ValueError("ViZDoom render_mode must be None or 'human'.")
    if role in {"collection", "specification"}:
        render_mode = None
    if screen_format not in _SCREEN_FORMATS:
        choices = ", ".join(sorted(_SCREEN_FORMATS))
        raise ValueError(
            f"ViZDoom screen_format must be one of: {choices}."
        )

    vizdoom = _load_vizdoom()
    scenario_path = _scenario_path(vizdoom, env_id)
    resolution = _enum_value(
        vizdoom.ScreenResolution,
        screen_resolution,
        "screen_resolution",
    )
    pixel_format = _enum_value(
        vizdoom.ScreenFormat,
        screen_format,
        "screen_format",
    )

    config_path = _doom_config_path()
    game = vizdoom.DoomGame()
    try:
        if not game.load_config(str(scenario_path)):
            raise ValueError(
                f"ViZDoom could not load scenario configuration: {scenario_path}"
            )
        game.set_mode(vizdoom.Mode.PLAYER)
        game.set_doom_config_path(str(config_path))
        game.set_screen_resolution(resolution)
        game.set_screen_format(pixel_format)
        game.set_window_visible(render_mode == "human")
        game.set_sound_enabled(False)

        if game.is_multiplayer_game():
            raise ValueError(
                "The ViZDoom backend currently supports only single-player games."
            )

        actions = _build_action_table(
            vizdoom,
            game,
            max_buttons_pressed=max_buttons_pressed,
        )
        observation_shape = (
            int(game.get_screen_height()),
            int(game.get_screen_width()),
            _SCREEN_FORMATS[screen_format],
        )
        return EnvAdapter(
            game=game,
            actions=actions,
            frame_skip=frame_skip,
            render_mode=render_mode,
            observation_shape=observation_shape,
            config_path=config_path,
            ammo_variable=vizdoom.GameVariable.AMMO2,
            killcount_variable=vizdoom.GameVariable.KILLCOUNT,
        )
    except Exception:
        game.close()
        config_path.unlink(missing_ok=True)
        raise
