import numpy as np
import torch

from omnirl.networks.actors import DeterministicActorNetwork


class Policy:
    """
    TD3 policy used for local inference.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        action_low: Lower bounds of the continuous action space.
        action_high: Upper bounds of the continuous action space.
        network_kwargs: Keyword arguments used to build the actor network.
        action_noise: Standard deviation of Gaussian exploration noise.
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
        action_noise: float = 0.1,
        random_warmup: bool = True,
    ):
        self.device = torch.device("cpu")
        self.action_noise = action_noise
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
        actor = DeterministicActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            layer_type=layer_type,
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

        action = self.actor(state)
        action = action * self.action_scale + self.action_bias

        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def sample_action(self, observation, batched=False):
        """Select a noisy normalized action during training."""
        if self.random_warmup and self.version == 0:
            scale = self.action_scale
            scale = scale.expand(len(observation), -1) if batched else scale
            action = torch.empty_like(scale).uniform_(-1, 1)
        else:
            observation = np.asarray(observation, dtype=np.float32)
            state = torch.as_tensor(observation, device=self.device)
            state = state if batched else state.unsqueeze(0)
            action = self.actor(state)
            noise = torch.randn_like(action) * self.action_noise
            action = torch.clamp(action + noise, -1, 1)
            action = action if batched else action[0]

        env_action = action * self.action_scale + self.action_bias
        action, env_action = action.cpu().numpy(), env_action.cpu().numpy()
        return {"env_action": env_action, "data": {"action": action}}
