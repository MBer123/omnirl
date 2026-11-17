"""Experience buffers used by OmniRL algorithms."""

import ray

from .replay_buffer import PrioritizedReplayBuffer as _PrioritizedReplayBuffer
from .replay_buffer import ReplayBuffer as _ReplayBuffer
from .rollout_buffer import RolloutBuffer as _RolloutBuffer

PrioritizedReplayBuffer = ray.remote(_PrioritizedReplayBuffer)
ReplayBuffer = ray.remote(_ReplayBuffer)
RolloutBuffer = ray.remote(_RolloutBuffer)

__all__ = ["PrioritizedReplayBuffer", "ReplayBuffer", "RolloutBuffer"]
