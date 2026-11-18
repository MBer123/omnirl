from abc import ABC, abstractmethod


class BaseEvaluationLoop(ABC):
    """Shared evaluation interface used by evaluators."""

    default_metrics = {}

    @abstractmethod
    def run(self, policy, num_episodes_per_env: int) -> dict:
        """Evaluate episodes per environment and return results for aggregation."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def aggregate(eval_infos: list[dict]) -> dict:
        """Aggregate evaluation results into episode counts and metrics."""
        raise NotImplementedError
