from collections.abc import Mapping

import numpy as np


def extract_observation(observation):
    """Validate and copy a vector observation returned by SUMO-RL."""
    observation = np.asarray(observation)
    if observation.ndim != 1:
        raise ValueError(
            "The SUMO backend supports vector observations; "
            f"received shape {observation.shape}."
        )
    return observation.copy()


def episode_seed(base_seed, episode):
    """Return a distinct deterministic SUMO seed for each episode."""
    return None if base_seed is None else int(base_seed) + int(episode)


def extract_queue_length(info, traffic_signal_ids):
    """Extract RESCO's per-signal stopped-vehicle queue snapshot."""
    if not isinstance(info, Mapping):
        raise ValueError("SUMO step info must be a mapping.")

    fields = [f"{signal_id}_stopped" for signal_id in traffic_signal_ids]
    missing = [field for field in fields if field not in info]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"SUMO step info is missing queue metrics: {names}.")

    queues = [float(info[field]) for field in fields]
    if not queues or not np.all(np.isfinite(queues)):
        raise ValueError("SUMO step info contains invalid queue metrics.")
    return float(np.mean(queues))
