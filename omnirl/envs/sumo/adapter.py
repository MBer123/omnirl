from pathlib import Path

from .tripinfo import parse_tripinfo


class EnvAdapter:
    """
    Adapt a single-agent SUMO-RL environment behind the environment backend.

    Args:
        env: The underlying SUMO-RL environment.
        render_mode: Render mode used by the environment.
    """

    def __init__(self, env, render_mode=None, tripinfo_path=None):
        self.env = env
        self.render_mode = render_mode
        self.tripinfo_path = (
            None if tripinfo_path is None else Path(tripinfo_path)
        )

    @property
    def action_space(self):
        """Return the discrete traffic-signal action space."""
        return self.env.action_space

    @property
    def observation_space(self):
        """Return the traffic observation space."""
        return self.env.observation_space

    @property
    def traffic_signal_ids(self):
        """Return the traffic signals discovered in the SUMO network."""
        return tuple(self.env.ts_ids)

    def reset(self, seed=None, options=None):
        """Reset the simulation."""
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        """Validate and apply one traffic-signal action."""
        if not self.action_space.contains(action):
            raise ValueError(
                f"SUMO action {action!r} is outside {self.action_space}."
            )
        return self.env.step(int(action))

    def render(self):
        """Render the simulation when interactive evaluation is enabled."""
        return self.env.render()

    def finish_episode_metrics(self):
        """Flush and parse one episode's RESCO-compatible trip metrics."""
        if self.tripinfo_path is None:
            raise RuntimeError(
                "Trip metrics are available only for evaluation environments."
            )

        self.env.close()
        try:
            return parse_tripinfo(self.tripinfo_path)
        finally:
            self.tripinfo_path.unlink(missing_ok=True)

    def close(self):
        """Close the TraCI connection and SUMO process."""
        try:
            return self.env.close()
        finally:
            if self.tripinfo_path is not None:
                self.tripinfo_path.unlink(missing_ok=True)
