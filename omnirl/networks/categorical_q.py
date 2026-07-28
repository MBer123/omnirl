import torch.nn as nn

from omnirl.networks.base import CNN, MLP, ResNet


class CategoricalQNetwork(nn.Module):
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
        super().__init__()

        self.action_dim = action_dim
        self.num_atoms = num_atoms
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

        # Initialize the distributional Q-network head
        self.Q_logits = nn.Linear(head_dim, action_dim * num_atoms)

    def forward(self, state):
        """Return Q-logits for all actions given the input state."""
        x = self.layers(state)
        if self.layer_type == "CNN":
            x = self.fc(x)

        x = self.Q_logits(x)
        return x.view(-1, self.action_dim, self.num_atoms)
