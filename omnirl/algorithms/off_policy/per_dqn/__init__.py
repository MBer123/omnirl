"""DQN with prioritized experience replay components."""

from omnirl.algorithms.base.policy import DQNPolicy as Policy
from omnirl.buffers import PrioritizedReplayBuffer as Memory

from .learner import Learner

__all__ = ["Learner", "Memory", "Policy"]
