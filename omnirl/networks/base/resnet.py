import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class ResidualBlock(nn.Module):
    """
    A two-convolution residual block with optional spatial downsampling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the first convolution and shortcut projection.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
            )
        )

    def forward(self, x):
        """Return the residual block output."""
        residual = self.shortcut(x)
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))
        return x + residual


class ResNet(nn.Module):
    """
    A residual image encoder with global average pooling.

    Args:
        state_dim: HWC observation shape.
        stem_channels: Number of channels produced by the input stem.
        stage_channels: Output channels for each residual stage.
        blocks_per_stage: Number of residual blocks in each stage.
        resnet_hidden_dims: Hidden sizes after global average pooling.
    """

    def __init__(
        self,
        state_dim: tuple[int, int, int],
        stem_channels: int = 64,
        stage_channels: tuple[int, ...] = (64, 128, 256, 512),
        blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2),
        resnet_hidden_dims: tuple[int, ...] = (512,),
    ):
        super().__init__()

        if len(stage_channels) != len(blocks_per_stage):
            raise ValueError(
                "stage_channels and blocks_per_stage must have the same length"
            )
        if not stage_channels or any(blocks < 1 for blocks in blocks_per_stage):
            raise ValueError("ResNet requires at least one block in every stage")

        _, _, in_channels = state_dim
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        stages = []
        in_channels = stem_channels
        for stage_index, (out_channels, num_blocks) in enumerate(
            zip(stage_channels, blocks_per_stage)
        ):
            blocks = [
                ResidualBlock(
                    in_channels,
                    out_channels,
                    stride=1 if stage_index == 0 else 2,
                )
            ]
            blocks.extend(
                ResidualBlock(out_channels, out_channels) for _ in range(num_blocks - 1)
            )
            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels

        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = MLP(in_channels, resnet_hidden_dims)
        self.output_dim = self.projection.output_dim

    def forward(self, state):
        """Return projected residual image features."""
        x = state.permute(0, 3, 1, 2).contiguous()
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(F.relu(x)).flatten(1)
        return self.projection(x)
