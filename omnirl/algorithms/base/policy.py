from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """Shared policy interface for local collection and evaluation."""

    @abstractmethod
    def set_weights(self, weights: dict, version: int | None = None) -> None:
        """Load learner weights and optionally update the policy version."""
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, observation):
        """Return a deterministic environment action for one observation."""
        raise NotImplementedError

    @abstractmethod
    def sample_action(self, observations) -> dict:
        """Return action, env_action and extra_data for a batch of observations."""
        raise NotImplementedError
