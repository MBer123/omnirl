import torch
import torch.nn as nn

from omnirl.networks.base import CNN, MLP, ResNet


class ValueCriticNetwork(nn.Module):
    """
    A simple critic network.

    Args:
        state_dim: Dimension of the environment state space.
        layer_type: Type of feature network to use.
        network_kwargs: Keyword arguments used to build the feature network.
    """

    def __init__(
        self,
        state_dim: int | tuple[int, int, int],
        layer_type: str,
        network_kwargs: dict | None = None,
    ):
        super().__init__()

        network_kwargs = dict(network_kwargs or {})
        self.layer_type = layer_type.upper()

        # Select the configured feature network.
        if self.layer_type == "MLP":
            self.layers = MLP(state_dim, **network_kwargs)
            self.V = nn.Linear(self.layers.output_dim, 1)
        elif self.layer_type == "CNN":
            self.layers = CNN(state_dim, **network_kwargs)
            self.fc = MLP(self.layers.output_dim, self.layers.conv_hidden_dims)
            self.V = nn.Linear(self.fc.output_dim, 1)
        elif self.layer_type == "RESNET":
            self.layers = ResNet(state_dim, **network_kwargs)
            self.V = nn.Linear(self.layers.output_dim, 1)
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")

    def forward(self, state):
        """Return state value V(s)."""
        x = self.layers(state)
        if self.layer_type == "CNN":
            x = self.fc(x)

        return self.V(x)


class QCriticNetwork(nn.Module):
    """
    A simple critic network.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        layer_type: Type of feature network to use.
        network_kwargs: Keyword arguments used to build the feature network.
    """

    def __init__(
        self,
        state_dim: int | tuple[int, int, int],
        action_dim: int,
        layer_type: str,
        is_continuous: bool = False,
        network_kwargs: dict | None = None,
    ):
        super().__init__()

        self.is_continuous = is_continuous
        network_kwargs = dict(network_kwargs or {})
        self.layer_type = layer_type.upper()

        # Select the configured feature network.
        if self.layer_type == "MLP":
            if is_continuous:
                input_dim = state_dim + action_dim
                output_dim = 1
            else:
                input_dim = state_dim
                output_dim = action_dim

            self.layers = MLP(input_dim, **network_kwargs)
            self.Q = nn.Linear(self.layers.output_dim, output_dim)
        elif self.layer_type == "CNN":
            self.layers = CNN(state_dim, **network_kwargs)

            if is_continuous:
                input_dim = self.layers.output_dim + action_dim
                output_dim = 1
            else:
                input_dim = self.layers.output_dim
                output_dim = action_dim

            self.fc = MLP(input_dim, self.layers.conv_hidden_dims)
            self.Q = nn.Linear(self.fc.output_dim, output_dim)
        elif self.layer_type == "RESNET":
            self.layers = ResNet(state_dim, **network_kwargs)
            input_dim = self.layers.output_dim
            output_dim = action_dim
            if is_continuous:
                input_dim += action_dim
                output_dim = 1
            self.Q = nn.Linear(input_dim, output_dim)
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")

    def forward(self, state, action=None):
        """Return Q(s, a) for continuous actions, or Q(s, ·) for discrete actions."""
        if self.layer_type == "MLP":
            if self.is_continuous:
                state = torch.cat([state, action], dim=-1)
            x = self.layers(state)
        else:
            x = self.layers(state)
            if self.is_continuous:
                x = torch.cat([x, action], dim=-1)
            if self.layer_type == "CNN":
                x = self.fc(x)

        return self.Q(x)


class QuantileCriticNetwork(nn.Module):
    """
    A simple quantile critic network.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        layer_type: Type of feature network to use.
        num_quantiles: Number of quantiles.
        network_kwargs: Keyword arguments used to build the feature network.
    """

    def __init__(
        self,
        state_dim: int | tuple[int, int, int],
        action_dim: int,
        layer_type: str,
        num_quantiles: int = 25,
        network_kwargs: dict | None = None,
    ):
        super().__init__()

        network_kwargs = dict(network_kwargs or {})
        self.layer_type = layer_type.upper()

        # Select the configured feature network.
        if self.layer_type == "MLP":
            self.layers = MLP(state_dim + action_dim, **network_kwargs)
            self.quantiles = nn.Linear(self.layers.output_dim, num_quantiles)
        elif self.layer_type == "CNN":
            self.layers = CNN(state_dim, **network_kwargs)
            self.fc = MLP(
                self.layers.output_dim + action_dim,
                self.layers.conv_hidden_dims,
            )
            self.quantiles = nn.Linear(self.fc.output_dim, num_quantiles)
        elif self.layer_type == "RESNET":
            self.layers = ResNet(state_dim, **network_kwargs)
            self.quantiles = nn.Linear(
                self.layers.output_dim + action_dim,
                num_quantiles,
            )
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")

    def forward(self, state, action=None):
        """Return quantile values for continuous actions."""
        if self.layer_type == "MLP":
            state = torch.cat([state, action], dim=-1)
            x = self.layers(state)
        else:
            x = self.layers(state)
            x = torch.cat([x, action], dim=-1)
            if self.layer_type == "CNN":
                x = self.fc(x)

        return self.quantiles(x)
