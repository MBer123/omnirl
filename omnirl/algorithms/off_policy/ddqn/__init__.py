"""Double DQN algorithm components."""

from omnirl.algorithms.base.policy import DQNPolicy as Policy
from omnirl.buffers import ReplayBuffer as Memory

from .learner import Learner

__all__ = ["Learner", "Memory", "Policy"]
