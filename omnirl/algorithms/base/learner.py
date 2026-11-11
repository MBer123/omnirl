from abc import ABC, abstractmethod


class BaseLearner(ABC):
    """Shared learner interface used by the training runner."""

    @abstractmethod
    def get_weights(self) -> dict:
        """Export policy weights as detached CPU tensors."""
        raise NotImplementedError

    @abstractmethod
    def get_version(self) -> int:
        """Return the current model version."""
        raise NotImplementedError

    @abstractmethod
    def learn(self, batch: dict) -> dict:
        """Update from a batch and return log, memory, version and trained_samples."""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self) -> None:
        """Save training state and policy weights using the learner configuration."""
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self) -> dict:
        """Restore training state and return the restored version and weights."""
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load policy weights and version from a configured checkpoint filename."""
        raise NotImplementedError
