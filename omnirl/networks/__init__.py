"""Network architectures used by OmniRL algorithms."""

from .actors import (
    DeterministicActorNetwork,
    StochasticActorNetwork,
)
from .critics import (
    CategoricalQNetwork,
    DeepQNetwork,
    DuelingQNetwork,
    QCriticNetwork,
    QuantileCriticNetwork,
    QuantileQNetwork,
    ValueCriticNetwork,
)

__all__ = [
    "CategoricalQNetwork",
    "DeepQNetwork",
    "DeterministicActorNetwork",
    "DuelingQNetwork",
    "QCriticNetwork",
    "QuantileCriticNetwork",
    "QuantileQNetwork",
    "StochasticActorNetwork",
    "ValueCriticNetwork",
]
