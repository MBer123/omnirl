import numpy as np
import torch
from torch.distributions import Normal

from omnirl.networks.actors import StochasticActorNetwork


class Policy:
    """
    TQC policy used for local inference.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        action_low: Lower bounds of the continuous action space.
        action_high: Upper bounds of the continuous action space.
        network_kwargs: Keyword arguments used to build the actor network.
        random_warmup: Whether fresh training uses random warm-up actions.
    """

    def __init__(
        self,
        state_dim: int | tuple[int, int, int],
        action_dim: int,
        is_continuous: bool = False,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        network_kwargs: dict | None = None,
        random_warmup: bool = True,
    ):
        self.device = torch.device("cpu")
        self.random_warmup = random_warmup
        self.version = -1

        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)

        action_low = torch.tensor(action_low, device=self.device)
        action_high = torch.tensor(action_high, device=self.device)
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

        network_kwargs = dict(network_kwargs or {})
        actor_kwargs = dict(network_kwargs.get("actor", network_kwargs))
        layer_type = actor_kwargs.pop("layer_type")
        log_std_min = actor_kwargs.pop("log_std_min", -20.0)
        log_std_max = actor_kwargs.pop("log_std_max", 2.0)
        actor = StochasticActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            is_continuous=True,
            layer_type=layer_type,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            network_kwargs=actor_kwargs,
        )

        self.actor = actor.to(self.device)
        self.actor.eval()

    def set_weights(self, weights, version=None):
        """Load external network weights into the actor network."""
        self.actor.load_state_dict(weights)
        if version is not None:
            self.version = version

    @torch.no_grad()
    def choose_action(self, observation):
        """Select a deterministic action for evaluation."""
        observation = np.asarray(observation, dtype=np.float32)
        state = torch.as_tensor(observation, device=self.device).unsqueeze(0)

        action_mean, _ = self.actor(state)
        action = torch.tanh(action_mean)
        action = action * self.action_scale + self.action_bias

        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def sample_action(self, observations):
        """Sample stochastic normalized actions for a batch of observations."""
        if self.random_warmup and self.version == 0:
            scale = self.action_scale.expand(len(observations), -1)
            action = torch.empty_like(scale).uniform_(-1, 1)
        else:
            observations = np.asarray(observations, dtype=np.float32)
            state = torch.as_tensor(observations, device=self.device)
            action_mean, action_log_std = self.actor(state)
            action = torch.tanh(Normal(action_mean, action_log_std.exp()).sample())

        env_action = action * self.action_scale + self.action_bias
        action, env_action = action.cpu().numpy(), env_action.cpu().numpy()
        return {"env_action": env_action, "data": {"action": action}}
