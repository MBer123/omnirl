from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.distributions import Categorical, Normal

from omnirl.networks.actors import DeterministicActorNetwork, StochasticActorNetwork
from omnirl.networks.critics import DeepQNetwork


class BasePolicy(ABC):
    """Shared policy interface for local collection and evaluation."""

    @abstractmethod
    def set_weights(self, weights: dict, version: int | None = None) -> None:
        """Load learner weights and optionally update the policy version."""
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, observations):
        """Return deterministic environment actions for a batch of observations."""
        raise NotImplementedError

    @abstractmethod
    def sample_action(self, observations) -> dict:
        """Return action, env_action and extra_data for a batch of observations."""
        raise NotImplementedError


class DQNPolicy(BasePolicy):
    """
    Shared epsilon-greedy policy for DQN, DDQN, and PER-DQN.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        action_low: Lower bounds of the continuous action space.
        action_high: Upper bounds of the continuous action space.
        network_kwargs: Keyword arguments used to build the Q-network.
        epsilon_start: Initial epsilon for epsilon-greedy exploration.
        epsilon_end: Final epsilon after decay.
        decay_steps: Number of learner updates over which epsilon decays.
        random_warmup: Whether fresh training uses random warm-up actions.
        device: Torch device to run the network on.
    """

    def __init__(
        self,
        state_dim: int | tuple[int, int, int],
        action_dim: int,
        is_continuous: bool = False,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        network_kwargs: dict | None = None,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        decay_steps: int = 100000,
        random_warmup: bool = True,
        device: str = "cpu",
    ):
        # Initialize device
        self.device = torch.device("cuda:0" if device == "gpu" else "cpu")

        # Initialize parameters
        self.action_dim = action_dim
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_end / epsilon_start) ** (1 / decay_steps)
        self.random_warmup = random_warmup

        # Initialize online network
        self.version = -1
        network_kwargs = dict(network_kwargs or {})
        layer_type = network_kwargs.pop("layer_type")
        q_net = DeepQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            layer_type=layer_type,
            network_kwargs=network_kwargs,
        )

        self.q_eval = q_net.to(self.device)
        self.q_eval.eval()

    def set_weights(self, weights, version=None):
        """Load external network weights into the online Q-network."""
        self.q_eval.load_state_dict(weights)
        if version is not None:
            self.version = version
            new_epsilon = self.epsilon_start * self.epsilon_decay**self.version
            self.epsilon = max(self.epsilon_end, new_epsilon)

    @torch.no_grad()
    def choose_action(self, observations):
        """Select greedy actions for a batch of observations."""
        observations = np.asarray(observations, dtype=np.float32)
        state = torch.as_tensor(observations, device=self.device)

        q_values = self.q_eval(state)

        return torch.argmax(q_values, dim=-1).cpu().numpy()

    @torch.no_grad()
    def sample_action(self, observations):
        """Select epsilon-greedy actions for a batch of observations."""
        observations = np.asarray(observations, dtype=np.float32)
        state = torch.as_tensor(observations, device=self.device)

        action = torch.randint(self.action_dim, (len(state),)).numpy()
        if not (self.random_warmup and self.version == 0):
            random_mask = np.random.random(len(state)) < self.epsilon
            if not random_mask.all():
                q_values = self.q_eval(state)
                greedy_action = torch.argmax(q_values, dim=-1).cpu().numpy()
                action = np.where(random_mask, action, greedy_action)

        return {"action": action, "env_action": action, "extra_data": {}}


class DeterministicPolicy(BasePolicy):
    """
    Deterministic actor policy with Gaussian exploration noise.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        action_low: Lower bounds of the continuous action space.
        action_high: Upper bounds of the continuous action space.
        network_kwargs: Keyword arguments used to build the actor network.
        action_noise: Standard deviation of Gaussian exploration noise.
        random_warmup: Whether fresh training uses random warm-up actions.
        device: Torch device to run the network on.
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
        device: str = "cpu",
    ):
        # Initialize device
        self.device = torch.device("cuda:0" if device == "gpu" else "cpu")

        # Initialize parameters
        self.action_noise = action_noise
        self.random_warmup = random_warmup

        # Initialize action scaling
        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)

        action_low = torch.tensor(action_low, device=self.device)
        action_high = torch.tensor(action_high, device=self.device)
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

        # Initialize online network
        self.version = -1
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
    def choose_action(self, observations):
        """Select deterministic actions for a batch of observations."""
        observations = np.asarray(observations, dtype=np.float32)
        state = torch.as_tensor(observations, device=self.device)

        action = self.actor(state)
        action = action * self.action_scale + self.action_bias

        return action.cpu().numpy()

    @torch.no_grad()
    def sample_action(self, observations):
        """Sample noisy normalized actions for a batch of observations."""
        if self.random_warmup and self.version == 0:
            scale = self.action_scale.expand(len(observations), -1)
            action = torch.empty_like(scale).uniform_(-1, 1)
        else:
            observations = np.asarray(observations, dtype=np.float32)
            state = torch.as_tensor(observations, device=self.device)
            action = self.actor(state)
            noise = torch.randn_like(action) * self.action_noise
            action = torch.clamp(action + noise, -1, 1)

        env_action = action * self.action_scale + self.action_bias
        action, env_action = action.cpu().numpy(), env_action.cpu().numpy()
        return {"action": action, "env_action": env_action, "extra_data": {}}


class StochasticPolicy(BasePolicy):
    """
    Stochastic actor policy that returns log probabilities with sampled actions.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        action_low: Lower bounds of the continuous action space.
        action_high: Upper bounds of the continuous action space.
        network_kwargs: Keyword arguments used to build the actor network.
        device: Torch device to run the network on.
    """

    def __init__(
        self,
        state_dim: int | tuple[int, int, int],
        action_dim: int,
        is_continuous: bool = False,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        network_kwargs: dict | None = None,
        device: str = "cpu",
    ):
        # Initialize device
        self.device = torch.device("cuda:0" if device == "gpu" else "cpu")

        # Initialize parameters
        self.is_continuous = is_continuous

        # Initialize action scaling
        if self.is_continuous:
            action_low = np.asarray(action_low, dtype=np.float32)
            action_high = np.asarray(action_high, dtype=np.float32)

            action_low = torch.tensor(action_low, device=self.device)
            action_high = torch.tensor(action_high, device=self.device)
            self.action_scale = (action_high - action_low) / 2.0
            self.action_bias = (action_high + action_low) / 2.0

        # Initialize actor network
        self.version = -1
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

    @torch.no_grad()
    def choose_action(self, observations):
        """Select deterministic actions for a batch of observations."""
        observations = np.asarray(observations, dtype=np.float32)
        state = torch.as_tensor(observations, device=self.device)

        if self.is_continuous:
            action_mean, _ = self.actor(state)
            action = torch.tanh(action_mean)
            action = action * self.action_scale + self.action_bias
            return action.cpu().numpy()
        else:
            action_logits = self.actor(state)
            action = torch.argmax(action_logits, dim=-1)
            return action.cpu().numpy()

    @torch.no_grad()
    def sample_action(self, observations):
        """Sample actions and log probabilities for a batch of observations."""
        observations = np.asarray(observations, dtype=np.float32)
        state = torch.as_tensor(observations, device=self.device)

        if self.is_continuous:
            action_mean, action_log_std = self.actor(state)
            dist = Normal(action_mean, action_log_std.exp())
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            env_action = torch.tanh(action) * self.action_scale + self.action_bias
            action, env_action = action.cpu().numpy(), env_action.cpu().numpy()
        else:
            action_logits = self.actor(state)
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = action.cpu().numpy()
            env_action = action

        log_prob = log_prob.cpu().numpy()
        extra_data = {"log_probs": log_prob}
        return {"action": action, "env_action": env_action, "extra_data": extra_data}
