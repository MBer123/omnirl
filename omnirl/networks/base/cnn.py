import torch
import torch.nn as nn


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
        conv_layers: tuple[tuple[int, int, int, int], ...] = (
            (32, 8, 4, 0),
            (64, 4, 2, 0),
        ),
        conv_hidden_dims: tuple[int, ...] = (256,),
    ):
        H, W, C = state_dim
        self.conv_hidden_dims = conv_hidden_dims

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

        super().__init__(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            x = super().forward(dummy)
            self.output_dim = x.view(1, -1).size(1)

    def forward(self, state):
        """Return flattened CNN features."""
        state = state.permute(0, 3, 1, 2).contiguous()
        x = super().forward(state)
        return x.view(x.size(0), -1)
