import numpy as np

from .protocol import (
    extract_observation,
    seed_and_reset,
    terminal_payoff,
    validate_action,
    validate_player,
)


class CollectionLoop:
    """
    Collect transitions using the RLCard interaction protocol.

    Args:
        env_builder: Callable that builds a new environment instance.
        seed: Random seed used for the initial environment reset.
    """

    def __init__(self, env_builder, seed=None):
        self.env = env_builder()
        self.state = seed_and_reset(self.env, seed=seed)
        self.observation = extract_observation(self.state)

    def _reset(self):
        self.state = seed_and_reset(self.env)
        self.observation = extract_observation(self.state)

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
            action = validate_action(self.state, decision["env_action"])

            next_state, player_id = self.env.step(action)
            validate_player(player_id)
            next_observation = extract_observation(next_state)
            terminal = bool(self.env.is_over())
            reward = terminal_payoff(self.env) if terminal else 0.0

            states.append(self.observation)
            rewards.append(reward)
            next_states.append(next_observation)
            terminals.append(terminal)
            dones.append(terminal)

            for name, value in decision.get("data", {}).items():
                policy_data.setdefault(name, []).append(value)

            self.state = next_state
            self.observation = next_observation
            if terminal:
                self._reset()

        return {
            "states": np.asarray(states),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_states": np.asarray(next_states),
            "terminals": np.asarray(terminals, dtype=np.bool_),
            "dones": np.asarray(dones, dtype=np.bool_),
            "policy_data": {
                name: np.asarray(values) for name, values in policy_data.items()
            },
        }
