import time
from concurrent.futures import ThreadPoolExecutor

import ray
import torch

from omnirl.utils.seed import set_global_seed


@ray.remote
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
        device: Torch device to run the policy on.
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
        device: str = "cpu",
        seed: int | None = None,
    ):
        # Limit PyTorch CPU threads
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        # Initialize random seed
        if seed is not None:
            set_global_seed(seed)

        # Initialize abstract algorithm
        self.policy = policy_cls(
            network_kwargs=network_kwargs,
            device=device,
            **action_spec,
            **policy_kwargs,
        )

        # Initialize environment components
        self.collection = collection_cls(
            env_builder=env_builder,
            seed=seed,
            num_envs=num_envs_per_sampler,
        )
        self.fragment_length = collection_fragment_length
        self.pending_weights = None
        self.version = -1

    @staticmethod
    def _fetch_weights(weight_store, version):
        """Fetch a newer CPU snapshot for the requested policy version."""
        result = ray.get(weight_store.get.remote(version))
        weights_ref = result["weights_ref"]
        weights = None if weights_ref is None else ray.get(weights_ref)
        return weights, result["version"]

    def _sync_weights(self, weight_store, executor):
        """Load ready weights between fragments and prefetch the next snapshot."""
        if self.pending_weights is None:
            weights, version = self._fetch_weights(weight_store, self.version)
        elif self.pending_weights.done():
            weights, version = self.pending_weights.result()
        else:
            return self.pending_weights

        if weights is not None:
            self.policy.set_weights(weights, version)
            self.version = version
        return executor.submit(self._fetch_weights, weight_store, self.version)

    def _can_collect(self, memory):
        """Check dynamic admission only for buffers that need backpressure."""
        if not self.needs_backpressure:
            return True
        return ray.get(memory.can_sample.remote(self.version))

    def rollout(self, memory, weight_store):
        """Continuously collect environment data and send it to the buffer."""
        pending_add = None
        self.needs_backpressure = ray.get(memory.needs_backpressure.remote())

        with ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                self.pending_weights = self._sync_weights(weight_store, executor)
                if not self._can_collect(memory):
                    time.sleep(0.001)
                    continue

                rollout = self.collection.run(self.policy, self.fragment_length)
                rollout["version"] = self.version
                if pending_add is not None:
                    ray.get(pending_add)
                pending_add = memory.add.remote(rollout)
