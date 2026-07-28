import numpy as np

from .protocol import episode_seed


class CollectionLoop:
    """
    Collect transitions from a native single-player ViZDoom game.

    Args:
        env_builder: Callable that builds a new environment instance.
        seed: Base seed used to derive a distinct seed for every episode.
    """

    def __init__(self, env_builder, seed=None):
        self.env = env_builder()
        self.base_seed = seed
        self.episode = 0
        self.observation = self._reset()

    def _reset(self):
        seed = episode_seed(self.base_seed, self.episode)
        self.episode += 1
        observation, _ = self.env.reset(seed=seed)
        return observation

    def collect(self, sample_action, num_steps):
        """Collect a fixed-length rollout and preserve opaque policy data."""
        states = []
        rewards = []
        next_states = []
        terminals = []
        dones = []
        policy_data = {}

        for _ in range(num_steps):
            decision = sample_action(self.observation)
            action = decision["env_action"]
            next_observation, reward, terminated, truncated, _ = self.env.step(
                action
            )
            done = bool(terminated) or bool(truncated)

            states.append(self.observation)
            rewards.append(reward)
            next_states.append(next_observation)
            # ViZDoom exposes no terminal screen, so timeout transitions must
            # not bootstrap from the zero placeholder returned by the adapter.
            terminals.append(done)
            dones.append(done)

            for name, value in decision.get("data", {}).items():
                policy_data.setdefault(name, []).append(value)

            self.observation = next_observation
            if done:
                self.observation = self._reset()

        return {
            "states": np.asarray(states, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_states": np.asarray(next_states, dtype=np.float32),
            "terminals": np.asarray(terminals, dtype=np.bool_),
            "dones": np.asarray(dones, dtype=np.bool_),
            "policy_data": {
                name: np.asarray(values) for name, values in policy_data.items()
            },
        }
