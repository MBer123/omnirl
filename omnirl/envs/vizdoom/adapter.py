import operator
from pathlib import Path

import numpy as np

from .protocol import extract_observation


class EnvAdapter:
    """
    Adapt one native ViZDoom game behind the environment backend.

    Args:
        game: Configured native ``DoomGame`` instance.
        actions: Discrete action table in available-button order.
        frame_skip: Number of game tics advanced for each agent action.
        render_mode: ``"human"`` for a visible game window, otherwise ``None``.
        observation_shape: Algorithm-facing HWC screen shape.
        config_path: Worker-local ZDoom configuration file removed on close.
        ammo_variable: Native game variable used to count fired shots.
        killcount_variable: Native game variable used to count kills.
    """

    def __init__(
        self,
        game,
        actions,
        frame_skip,
        render_mode,
        observation_shape,
        config_path,
        ammo_variable,
        killcount_variable,
    ):
        self.game = game
        self.actions = tuple(tuple(action) for action in actions)
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.observation_shape = tuple(observation_shape)
        self.observation_dtype = np.dtype(np.float32)
        self.config_path = Path(config_path)
        self.ammo_variable = ammo_variable
        self.killcount_variable = killcount_variable
        self.initial_ammo = None
        self.elapsed_tics = 0

    @property
    def num_actions(self):
        """Return the size of the discrete action table."""
        return len(self.actions)

    def reset(self, seed=None, options=None):
        """Start a new single-player episode and return its first screen."""
        if options:
            fields = ", ".join(sorted(options))
            raise ValueError(f"Unsupported ViZDoom reset options: {fields}.")

        if seed is not None:
            seed = int(seed)
            if not 0 <= seed <= np.iinfo(np.uint32).max:
                raise ValueError("ViZDoom seed must fit in an unsigned 32-bit integer.")
            self.game.set_seed(seed)

        if self.game.is_running():
            self.game.new_episode()
        else:
            self.game.init()

        if self.game.is_multiplayer_game():
            self.close()
            raise ValueError(
                "The ViZDoom backend currently supports only single-player games."
            )

        observation = extract_observation(
            self.game.get_state(), self.observation_shape
        )
        self.elapsed_tics = 0
        self.initial_ammo = float(
            self.game.get_game_variable(self.ammo_variable)
        )
        return observation, {}

    def step(self, action):
        """Validate and execute one discrete ViZDoom action."""
        try:
            action = operator.index(action)
        except TypeError as error:
            raise ValueError(
                f"ViZDoom action must be an integer, received {action!r}."
            ) from error

        if not 0 <= action < self.num_actions:
            raise ValueError(
                f"ViZDoom action {action} is outside [0, {self.num_actions})."
            )
        if not self.game.is_running() or self.game.is_episode_finished():
            raise RuntimeError("Reset ViZDoom before calling step().")

        reward = float(
            self.game.make_action(list(self.actions[action]), self.frame_skip)
        )
        finished = bool(self.game.is_episode_finished())
        truncated = finished and bool(self.game.is_episode_timeout_reached())
        terminated = finished and not truncated
        self._update_elapsed_tics(finished)

        state = self.game.get_state()
        if state is None:
            observation = np.zeros(
                self.observation_shape, dtype=self.observation_dtype
            )
        else:
            observation = extract_observation(state, self.observation_shape)

        info = {
            "episode_tics": self._episode_tics(),
            "total_reward": float(self.game.get_total_reward()),
        }
        return observation, reward, terminated, truncated, info

    def _native_episode_tics(self):
        return max(
            0,
            int(self.game.get_episode_time())
            - int(self.game.get_episode_start_time()),
        )

    def _update_elapsed_tics(self, finished):
        native_tics = self._native_episode_tics()
        if native_tics > self.elapsed_tics:
            self.elapsed_tics = native_tics
        elif finished:
            # Some scenarios reset the native clock when their ACS script
            # finishes the map. Preserve the elapsed time across that edge.
            self.elapsed_tics += self.frame_skip

    def _episode_tics(self):
        return self.elapsed_tics

    def episode_metrics(self):
        """Return shooting and survival metrics for one completed episode."""
        if not self.game.is_episode_finished() or self.initial_ammo is None:
            raise RuntimeError(
                "ViZDoom episode metrics require a completed episode."
            )

        remaining_ammo = float(
            self.game.get_game_variable(self.ammo_variable)
        )
        shots = max(0, int(round(self.initial_ammo - remaining_ammo)))
        kills = max(
            0,
            int(round(self.game.get_game_variable(self.killcount_variable))),
        )
        return {
            "kills": kills,
            "shots": shots,
            "episode_tics": self._episode_tics(),
        }

    def render(self):
        """Return the current frame; human rendering is handled by Doom itself."""
        state = self.game.get_state()
        if state is None:
            return None
        return np.asarray(state.screen_buffer).copy()

    def close(self):
        """Close the native game process."""
        try:
            return self.game.close()
        finally:
            self.config_path.unlink(missing_ok=True)
