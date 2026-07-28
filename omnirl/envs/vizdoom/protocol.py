import numpy as np


def episode_seed(base_seed, episode):
    """Return a deterministic ViZDoom seed for one episode."""
    return None if base_seed is None else int(base_seed) + int(episode)


def extract_observation(state, expected_shape):
    """Return a normalized HWC screen observation from a ViZDoom state."""
    if state is None or state.screen_buffer is None:
        raise ValueError("ViZDoom did not provide a screen buffer.")

    observation = np.asarray(state.screen_buffer)
    if observation.ndim == 2:
        observation = observation[..., None]
    if observation.shape != tuple(expected_shape):
        raise ValueError(
            "Unexpected ViZDoom screen shape: "
            f"expected {tuple(expected_shape)}, received {observation.shape}."
        )

    return observation.astype(np.float32, copy=True) / 255.0
