"""Abstract interfaces for algorithm learners and policies."""

from .learner import BaseLearner
from .policy import BasePolicy

__all__ = ["BaseLearner", "BasePolicy"]
