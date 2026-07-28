class EnvAdapter:
    """
    Adapt a Gymnasium environment behind the environment backend.

    Args:
        env: The underlying Gymnasium environment.
        render_mode: Render mode passed to the environment.
    """

    def __init__(self, env, render_mode):
        self.env = env
        self.render_mode = render_mode

    @property
    def action_space(self):
        """Return the action space of the environment."""
        return self.env.action_space

    @property
    def observation_space(self):
        """Return the observation space of the environment."""
        return self.env.observation_space

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        """Execute one environment step."""
        return self.env.step(action)

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        return self.env.close()
