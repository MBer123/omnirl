import os
from pathlib import Path
import uuid

from .adapter import EnvAdapter


_ROLES = {None, "collection", "evaluation", "specification"}
_RESERVED_FIELDS = {
    "add_per_agent_info",
    "add_system_info",
    "max_depart_delay",
    "out_csv_name",
    "single_agent",
    "time_to_teleport",
    "use_gui",
}


def _tripinfo_output():
    """Return a worker-local output path without whitespace-sensitive arguments."""
    name = f".omnirl-sumo-tripinfo-{os.getpid()}-{uuid.uuid4().hex}.xml"
    return name, Path.cwd() / name


def _configure_sumo_home():
    """Discover SUMO_HOME for installations provided by eclipse-sumo."""
    if os.environ.get("SUMO_HOME"):
        return

    try:
        import sumo
    except ImportError:
        return

    sumo_home = getattr(sumo, "SUMO_HOME", None)
    if sumo_home:
        os.environ["SUMO_HOME"] = str(sumo_home)


def _load_sumo_environment():
    """Load SUMO-RL lazily so importing OmniRL does not start SUMO."""
    _configure_sumo_home()
    try:
        from sumo_rl.environment.env import SumoEnvironment
    except ImportError as error:
        raise ImportError(
            "The SUMO backend requires 'sumo-rl' and a working SUMO "
            "installation. Install the project requirements or set SUMO_HOME."
        ) from error
    return SumoEnvironment


def build_env(env_config, role=None):
    """Build a single-traffic-signal SUMO-RL environment."""
    if role not in _ROLES:
        raise ValueError(f"Unsupported environment role: {role!r}.")

    config = dict(env_config)
    missing = [name for name in ("net_file", "route_file") if name not in config]
    if missing:
        fields = ", ".join(f"'{name}'" for name in missing)
        raise ValueError(f"SUMO environment configuration requires {fields}.")

    reserved = sorted(_RESERVED_FIELDS.intersection(config))
    if reserved:
        fields = ", ".join(reserved)
        raise ValueError(f"SUMO environment fields are backend-owned: {fields}.")

    render_mode = config.pop("render_mode", None)
    if render_mode not in {None, "human"}:
        raise ValueError("SUMO render_mode must be None or 'human'.")
    if role in {"collection", "specification"}:
        render_mode = None

    additional_sumo_cmd = config.pop("additional_sumo_cmd", None)
    if additional_sumo_cmd and "--tripinfo-output" in additional_sumo_cmd:
        raise ValueError("SUMO tripinfo output is owned by the backend.")

    tripinfo_path = None
    if role == "evaluation":
        tripinfo_argument, tripinfo_path = _tripinfo_output()
        metric_command = (
            f"--tripinfo-output {tripinfo_argument} "
            "--tripinfo-output.write-unfinished true "
            "--tripinfo-output.write-undeparted true "
            "--eager-insert"
        )
        additional_sumo_cmd = " ".join(
            part for part in (additional_sumo_cmd, metric_command) if part
        )

    environment_cls = _load_sumo_environment()
    raw_env = environment_cls(
        **config,
        single_agent=True,
        use_gui=render_mode == "human",
        render_mode=render_mode,
        add_system_info=False,
        add_per_agent_info=role == "evaluation",
        out_csv_name=None,
        max_depart_delay=-1,
        time_to_teleport=-1,
        additional_sumo_cmd=additional_sumo_cmd,
    )
    env = EnvAdapter(
        raw_env,
        render_mode=render_mode,
        tripinfo_path=tripinfo_path,
    )

    if len(env.traffic_signal_ids) != 1:
        count = len(env.traffic_signal_ids)
        env.close()
        raise ValueError(
            "The SUMO backend supports exactly one traffic signal; "
            f"the network contains {count}."
        )
    return env
