import torch
import torch.nn as nn

from .mlp import MLP


class CNN(nn.Sequential):
    """
    A simple convolutional neural network.

    Args:
        state_dim: Dimension of the environment state space.
        conv_layers: CNN layer configurations.
        conv_hidden_dims: Hidden sizes of fully-connected layers after CNN trunk.
    """

    def __init__(
        self,
        state_dim: tuple[int, int, int],
        conv_layers: tuple[tuple[int, ...], ...] = ((32, 8, 4, 0),),
        conv_hidden_dims: tuple[int, ...] = (256,),
    ):
        super().__init__()
        H, W, C = state_dim

        layers = []
        in_channels = C
        for out_channels, kernel_size, stride, padding in conv_layers:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.layers = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            x = self.layers(dummy)
            flatten = x.view(1, -1).size(1)

        self.projection = MLP(flatten, conv_hidden_dims)
        self.output_dim = self.projection.output_dim

    def forward(self, state):
        """Return flattened CNN features."""
        state = state.permute(0, 3, 1, 2).contiguous()
        x = self.layers(state).flatten(1)
        return self.projection(x)
