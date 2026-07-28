import torch
import torch.nn as nn

from omnirl.networks.base import CNN, MLP, ResNet


class StochasticActorNetwork(nn.Module):
    """
    A simple actor network.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        layer_type: Type of feature network to use.
        log_std_min: Minimum value for log standard deviation.
        log_std_max: Maximum value for log standard deviation.
        network_kwargs: Keyword arguments used to build the feature network.
    """

    def __init__(
        self,
        state_dim: int | tuple[int, int, int],
        action_dim: int,
        layer_type: str,
        is_continuous: bool = False,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        network_kwargs: dict | None = None,
    ):
        super().__init__()

        self.is_continuous = is_continuous
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        network_kwargs = dict(network_kwargs or {})
        self.layer_type = layer_type.upper()

        # Select the configured feature network.
        if self.layer_type == "MLP":
            self.layers = MLP(state_dim, **network_kwargs)
            head_dim = self.layers.output_dim
        elif self.layer_type == "CNN":
            self.layers = CNN(state_dim, **network_kwargs)
            self.fc = MLP(self.layers.output_dim, self.layers.conv_hidden_dims)
            head_dim = self.fc.output_dim
        elif self.layer_type == "RESNET":
            self.layers = ResNet(state_dim, **network_kwargs)
            head_dim = self.layers.output_dim
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")

        # Initialize the actor head based on the action type
        if self.is_continuous:
            self.action_mean = nn.Linear(head_dim, action_dim)
            self.action_log_std = nn.Linear(head_dim, action_dim)
        else:
            self.action_logits = nn.Linear(head_dim, action_dim)

    def forward(self, state):
        """Return mean and log_std for continuous action, logits for discrete action."""
        x = self.layers(state)
        if self.layer_type == "CNN":
            x = self.fc(x)

        if self.is_continuous:
            action_mean = self.action_mean(x)
            action_log_std = self.action_log_std(x)
            action_log_std = torch.clamp(
                action_log_std,
                min=self.log_std_min,
                max=self.log_std_max,
            )
            return action_mean, action_log_std

        return self.action_logits(x)


class DeterministicActorNetwork(nn.Module):
    """
    A simple actor network.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        layer_type: Type of feature network to use.
        network_kwargs: Keyword arguments used to build the feature network.
    """

    def __init__(
        self,
        state_dim: int | tuple[int, int, int],
        action_dim: int,
        layer_type: str,
        network_kwargs: dict | None = None,
    ):
        super().__init__()

        network_kwargs = dict(network_kwargs or {})
        self.layer_type = layer_type.upper()

        # Select the configured feature network.
        if self.layer_type == "MLP":
            self.layers = MLP(state_dim, **network_kwargs)
            head_dim = self.layers.output_dim
        elif self.layer_type == "CNN":
            self.layers = CNN(state_dim, **network_kwargs)
            self.fc = MLP(self.layers.output_dim, self.layers.conv_hidden_dims)
            head_dim = self.fc.output_dim
        elif self.layer_type == "RESNET":
            self.layers = ResNet(state_dim, **network_kwargs)
            head_dim = self.layers.output_dim
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")

        # Initialize the actor head
        self.action = nn.Linear(head_dim, action_dim)

    def forward(self, state):
        """Return deterministic action."""
        x = self.layers(state)
        if self.layer_type == "CNN":
            x = self.fc(x)

        return torch.tanh(self.action(x))
