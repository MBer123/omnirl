class EnvAdapter:
    """
    Adapt an RLCard environment behind the environment backend.

    Args:
        env: The underlying RLCard environment.
    """

    def __init__(self, env):
        self.env = env

    @property
    def num_players(self):
        """Return the number of players in the environment."""
        return self.env.num_players

    @property
    def num_actions(self):
        """Return the action dimension of the environment."""
        return self.env.num_actions

    @property
    def state_shape(self):
        """Return the state shapes of the environment."""
        return self.env.state_shape

    @property
    def action_shape(self):
        """Return the action shapes of the environment."""
        return self.env.action_shape

    def seed(self, seed=None):
        """Seed the environment."""
        return self.env.seed(seed)

    def reset(self):
        """Reset the environment."""
        return self.env.reset()

    def step(self, action):
        """Execute one environment step."""
        return self.env.step(action)

    def is_over(self):
        """Return whether the environment episode has ended."""
        return self.env.is_over()

    def get_payoffs(self):
        """Return the terminal payoffs of the environment."""
        return self.env.get_payoffs()

    def close(self):
        """Close the environment."""
        close = getattr(self.env, "close", None)
        return None if close is None else close()
