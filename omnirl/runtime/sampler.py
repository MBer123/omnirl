import ray
import torch

from omnirl.utils.seed import set_global_seed


@ray.remote(num_cpus=1)
class Sampler:
    """
    Generic distributed sampling worker.

    Args:
        policy_cls: Algorithm-specific policy class.
        collection_cls: Environment-specific collection loop class.
        env_builder: Callable that builds an environment instance.
        action_spec: Environment arguments for policy initialization.
        network_kwargs: Keyword arguments used to build the policy network.
        policy_kwargs: Keyword arguments used to build the policy.
        num_envs_per_sampler: Number of environments sharing one inference batch.
        collection_fragment_length: Steps collected per environment per transfer.
        seed: Random seed used to ensure consistent results across runs.
    """

    def __init__(
        self,
        policy_cls,
        collection_cls,
        env_builder,
        action_spec,
        network_kwargs: dict | None = None,
        policy_kwargs: dict | None = None,
        num_envs_per_sampler: int = 1,
        collection_fragment_length: int = 1,
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
            **action_spec,
            **policy_kwargs,
        )
        self.collection = collection_cls(
            env_builder=env_builder,
            seed=seed,
            num_envs=num_envs_per_sampler,
        )
        self.fragment_length = collection_fragment_length
        self.version = -1

    def _sync_weights(self, weight_store):
        """Fetch and load newer policy weights from the weight store."""
        result = ray.get(weight_store.get.remote(self.version))
        weights = result["weights"]
        if weights is not None:
            self.version = result["version"]
            self.policy.set_weights(weights, self.version)

    def rollout(self, memory, weight_store):
        """Continuously collect environment data and send it to the buffer."""
        pending_add = None
        while True:
            self._sync_weights(weight_store)
            rollout = self.collection.collect(self.policy, self.fragment_length)
            rollout["version"] = self.version
            if pending_add is not None:
                ray.get(pending_add)
            pending_add = memory.add.remote(rollout)
