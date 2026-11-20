from abc import ABC, abstractmethod


class BaseLearner(ABC):
    """Shared learner interface used by the training runner."""

    version: int

    @staticmethod
    def _state_dict(model) -> dict:
        """Return the model state as detached CPU tensors."""
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    @abstractmethod
    def get_weights(self) -> dict:
        """Export policy weights as detached CPU tensors."""
        raise NotImplementedError

    def get_version(self) -> int:
        """Return the current model version."""
        return self.version

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
