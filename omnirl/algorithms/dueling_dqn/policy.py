import numpy as np
import torch

from omnirl.networks.critics import DuelingQNetwork


class Policy:
    """
    Dueling DQN policy used for local inference.

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
    ):
        self.device = torch.device("cpu")
        self.action_dim = action_dim
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_end / epsilon_start) ** (1 / decay_steps)
        self.random_warmup = random_warmup
        self.version = -1

        network_kwargs = dict(network_kwargs or {})
        layer_type = network_kwargs.pop("layer_type")
        q_net = DuelingQNetwork(
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
    def choose_action(self, observation):
        """Greedy action selection during evaluation."""
        observation = np.asarray(observation, dtype=np.float32)
        state = torch.as_tensor(observation, device=self.device).unsqueeze(0)

        q_values = self.q_eval(state)

        return int(torch.argmax(q_values, dim=-1).item())

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
