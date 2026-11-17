import ray
import torch

from omnirl.utils.seed import set_global_seed


@ray.remote
class Evaluator:
    """
    Generic distributed evaluation worker.

    Args:
        policy_cls: Algorithm-specific policy class.
        evaluation_cls: Environment-specific evaluation loop class.
        env_builder: Callable that builds an environment instance.
        action_spec: Environment arguments for policy initialization.
        network_kwargs: Keyword arguments used to build the policy network.
        policy_kwargs: Keyword arguments used to build the policy.
        num_episodes_per_evaluator: Number of episodes to run during each evaluation.
        device: Torch device to run the policy on.
        seed: Random seed used to ensure consistent results across runs.
    """

    def __init__(
        self,
        policy_cls,
        evaluation_cls,
        env_builder,
        action_spec,
        network_kwargs: dict | None = None,
        policy_kwargs: dict | None = None,
        num_episodes_per_evaluator: int = 1,
        device: str = "cpu",
        seed: int | None = None,
    ):
        # Limit PyTorch CPU threads
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        # Initialize random seed
        if seed is not None:
            set_global_seed(seed)

        # Initialize abstract algorithm and environment components
        self.policy = policy_cls(
            network_kwargs=network_kwargs,
            device=device,
            **action_spec,
            **policy_kwargs,
        )
        self.evaluation = evaluation_cls(env_builder=env_builder, seed=seed)
        self.num_episodes_per_evaluator = num_episodes_per_evaluator
        self.version = -1

    def evaluate(self, weights, version):
        """Evaluate the current policy through the injected evaluation loop."""
        self.policy.set_weights(weights, version)
        self.version = version
        result = self.evaluation.run(self.policy, self.num_episodes_per_evaluator)
        return {"version": self.version, **result}
