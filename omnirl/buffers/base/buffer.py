from abc import ABC, abstractmethod


class BaseBuffer(ABC):
    """Shared interface for buffers used by samplers and learners."""

    @abstractmethod
    def add(self, rollout: dict) -> None:
        """Store a rollout with leading [num_envs, num_steps] dimensions."""
        raise NotImplementedError

    @abstractmethod
    def get_batch(self, rank: int, world_size: int) -> dict | None:
        """Return a rank-specific training batch, or None when not ready."""
        raise NotImplementedError

    @abstractmethod
    def on_learn(self, train_infos: list[dict]) -> None:
        """Update buffer state using feedback from the learners."""
        raise NotImplementedError

    @abstractmethod
    def ready(self) -> bool:
        """Return whether enough data is available to start training."""
        raise NotImplementedError

    @abstractmethod
    def needs_backpressure(self) -> bool:
        """Return whether collectors must wait for permission to collect."""
        raise NotImplementedError

    @abstractmethod
    def can_sample(self, version: int) -> bool:
        """Return whether a collector with this policy version may collect."""
        raise NotImplementedError

    @abstractmethod
    def supports_prefetch(self) -> bool:
        """Return whether batches can be fetched before an update finishes."""
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> dict[str, int]:
        """Return statistics including available_data and collected_data."""
        raise NotImplementedError
