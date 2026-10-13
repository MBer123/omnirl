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

        # Initialize network configuration
        network_kwargs = dict(network_kwargs or {})
        layer_type = layer_type.upper()

        # Select the configured feature network.
        if layer_type == "MLP":
            self.layers = MLP(state_dim, **network_kwargs)
        elif layer_type == "CNN":
            self.layers = CNN(state_dim, **network_kwargs)
        elif layer_type == "RESNET":
            self.layers = ResNet(state_dim, **network_kwargs)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        # Initialize the value head
        self.output = nn.Linear(self.layers.output_dim, 1)

    def forward(self, state):
        """Return state value V(s)."""
        x = self.layers(state)
        return self.output(x)


# ==================== Continuous Q Networks ====================


class QCriticNetwork(nn.Module):
    """
    A simple critic network.

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

        # Initialize network configuration
        network_kwargs = dict(network_kwargs or {})
        layer_type = layer_type.upper()

        # Select the configured feature network.
        if layer_type == "MLP":
            self.layers = MLP(state_dim, **network_kwargs)
        elif layer_type == "CNN":
            self.layers = CNN(state_dim, **network_kwargs)
        elif layer_type == "RESNET":
            self.layers = ResNet(state_dim, **network_kwargs)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        # Initialize the Q-network head
        self.head_dim = self.layers.output_dim
        self.output = nn.Sequential(
            nn.Linear(self.head_dim + action_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
        )

    def _encode(self, state, action):
        state_features = self.layers(state)
        joint_feature = torch.cat([state_features, action], dim=-1)
        return joint_feature

    def forward(self, state, action):
        """Return Q(s, a) for continuous actions."""
        x = self._encode(state, action)
        return self.output(x)


class QuantileCriticNetwork(QCriticNetwork):
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
        super().__init__(state_dim, action_dim, layer_type, network_kwargs)

        # Initialize the quantile head
        self.output = nn.Sequential(
            nn.Linear(self.head_dim + action_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, num_quantiles),
        )

    def forward(self, state, action):
        """Return quantile values for continuous actions."""
        x = self._encode(state, action)
        return self.output(x)


# ==================== Discret Q Networks ====================


class DeepQNetwork(nn.Module):
    """
    A simple Q-network.

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

        # Initialize network configuration
        network_kwargs = dict(network_kwargs or {})
        layer_type = layer_type.upper()

        # Select the configured feature network.
        if layer_type == "MLP":
            self.layers = MLP(state_dim, **network_kwargs)
        elif layer_type == "CNN":
            self.layers = CNN(state_dim, **network_kwargs)
        elif layer_type == "RESNET":
            self.layers = ResNet(state_dim, **network_kwargs)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        # Initialize the Q-network head
        self.action_dim = action_dim
        self.head_dim = self.layers.output_dim
        self.output = nn.Linear(self.head_dim, self.action_dim)

    def forward(self, state):
        """Return Q-values for all actions given the input state."""
        x = self.layers(state)
        return self.output(x)


class DuelingQNetwork(DeepQNetwork):
    """
    A dueling Q-network.

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
        super().__init__(state_dim, action_dim, layer_type, network_kwargs)

        # Initialize the value and advantage heads
        self.output = nn.ModuleDict(
            {
                "V": nn.Sequential(
                    nn.Linear(self.head_dim, self.head_dim),
                    nn.ReLU(),
                    nn.Linear(self.head_dim, 1),
                ),
                "A": nn.Sequential(
                    nn.Linear(self.head_dim, self.head_dim),
                    nn.ReLU(),
                    nn.Linear(self.head_dim, self.action_dim),
                ),
            }
        )

    def forward(self, state):
        """Return Q-values for all actions given the input state."""
        x = self.layers(state)
        V = self.output["V"](x)
        A = self.output["A"](x)
        return V + A - A.mean(dim=-1, keepdim=True)


class QuantileQNetwork(DeepQNetwork):
    """
    A simple quantile Q-network.

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
        super().__init__(state_dim, action_dim, layer_type, network_kwargs)

        # Initialize the quantile head
        self.num_quantiles = num_quantiles
        self.output = nn.Linear(self.head_dim, action_dim * num_quantiles)

    def forward(self, state):
        """Return quantile values for all actions given the input state."""
        x = self.layers(state)
        x = self.output(x)
        return x.view(-1, self.action_dim, self.num_quantiles)


class CategoricalQNetwork(DeepQNetwork):
    """
    A simple categorical Q-network.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        layer_type: Type of feature network to use.
        num_atoms: Number of atoms for C51.
        network_kwargs: Keyword arguments used to build the feature network.
    """

    def __init__(
        self,
        state_dim: int | tuple[int, int, int],
        action_dim: int,
        layer_type: str,
        num_atoms: int = 51,
        network_kwargs: dict | None = None,
    ):
        super().__init__(state_dim, action_dim, layer_type, network_kwargs)

        # Initialize the distributional Q-network head
        self.num_atoms = num_atoms
        self.output = nn.Linear(self.head_dim, action_dim * num_atoms)

    def forward(self, state):
        """Return Q-logits for all actions given the input state."""
        x = self.layers(state)
        x = self.output(x)
        return x.view(-1, self.action_dim, self.num_atoms)
