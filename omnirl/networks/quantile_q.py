import torch.nn as nn

from omnirl.networks.base import CNN, MLP, ResNet


class QuantileQNetwork(nn.Module):
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
        super().__init__()

        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
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

        # Initialize the quantile head
        self.quantiles = nn.Linear(head_dim, action_dim * num_quantiles)

    def forward(self, state):
        """Return quantile values for all actions given the input state."""
        x = self.layers(state)
        if self.layer_type == "CNN":
            x = self.fc(x)

        x = self.quantiles(x)
        return x.view(-1, self.action_dim, self.num_quantiles)
