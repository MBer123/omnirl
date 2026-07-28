import numpy as np
import torch
from torch.distributions import Categorical, Normal

from omnirl.networks.actors import StochasticActorNetwork


class Policy:
    """
    IMPALA policy used for local inference.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        action_low: Lower bounds of the continuous action space.
        action_high: Upper bounds of the continuous action space.
        network_kwargs: Keyword arguments used to build the actor network.
    """

    def __init__(
        self,
        state_dim: int | tuple[int, int, int],
        action_dim: int,
        is_continuous: bool = False,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        network_kwargs: dict | None = None,
    ):
        self.device = torch.device("cpu")
        self.is_continuous = is_continuous
        self.version = -1

        if self.is_continuous:
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
            is_continuous=is_continuous,
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

    def choose_action(self, observation):
        """Select a deterministic action for evaluation."""
        observation = np.asarray(observation, dtype=np.float32)
        state = torch.as_tensor(observation, device=self.device).unsqueeze(0)

        with torch.no_grad():
            if self.is_continuous:
                action_mean, _ = self.actor(state)
                action = torch.tanh(action_mean)
                action = action * self.action_scale + self.action_bias
                return action.squeeze(0).cpu().numpy()
            else:
                action_logits = self.actor(state)
                action = torch.argmax(action_logits, dim=-1)
                return int(action.item())

    def sample_action(self, observation):
        """Select a probabilistic action and its log probability."""
        observation = np.asarray(observation, dtype=np.float32)
        state = torch.as_tensor(observation, device=self.device).unsqueeze(0)

        with torch.no_grad():
            if self.is_continuous:
                action_mean, action_log_std = self.actor(state)
                dist = Normal(action_mean, action_log_std.exp())
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                env_action = torch.tanh(action) * self.action_scale + self.action_bias
                action = action.squeeze(0).cpu().numpy()
                env_action = env_action.squeeze(0).cpu().numpy()
                log_prob = log_prob.squeeze(0).item()
            else:
                action_logits = self.actor(state)
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action).item()
                action = int(action.item())
                env_action = action

        return {
            "env_action": env_action,
            "data": {"action": action, "log_prob": log_prob},
        }
