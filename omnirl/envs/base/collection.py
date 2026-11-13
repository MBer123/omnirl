from abc import ABC, abstractmethod


class BaseCollectionLoop(ABC):
    """Shared collection interface used by samplers."""

    @abstractmethod
    def run(self, policy, num_steps: int) -> dict:
        """Return a rollout with leading [num_envs, num_steps] dimensions."""
        raise NotImplementedError
