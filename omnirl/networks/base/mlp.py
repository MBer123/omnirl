import torch.nn as nn


class MLP(nn.Sequential):
    """
    A simple multi-layer perceptron.

    Args:
        input_dim: Dimension of the input features.
        mlp_hidden_dims: Hidden sizes of MLP layers.
    """

    def __init__(
        self,
        input_dim: int,
        mlp_hidden_dims: tuple[int, ...] = (256, 256),
    ):
        layers = []
        in_dim = input_dim
        for out_dim in mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        super().__init__(*layers)
        self.output_dim = mlp_hidden_dims[-1]
