"""DQN with prioritized experience replay components."""

from omnirl.buffers import PrioritizedReplayBuffer as Memory

from .learner import Learner
from .policy import Policy

__all__ = ["Learner", "Memory", "Policy"]
