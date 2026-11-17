import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


class ResourceAllocator:
    """
    CPU and GPU allocation for runtime actors.

    Args:
        num_learners: Number of learner actors.
        num_samplers: Number of sampler actors.
        num_evaluators: Number of evaluator actors.
        learner_device: Device selector used to run learners.
        num_sampler_gpus: Number of GPUs used by samplers, shared when needed.
        num_samplers_per_gpu: Maximum number of samplers sharing each GPU.
        num_evaluator_gpus: Number of GPUs used by evaluators, shared when needed.
        num_evaluators_per_gpu: Maximum number of evaluators sharing each GPU.
    """

    def __init__(
        self,
        num_learners: int,
        num_samplers: int,
        num_evaluators: int,
        learner_device: str,
        num_sampler_gpus: int,
        num_samplers_per_gpu: int,
        num_evaluator_gpus: int,
        num_evaluators_per_gpu: int,
    ):
        # Check CPU availability
        num_buffer = 1
        num_weight_store = 1
        available_resources = ray.available_resources()
        num_workers = num_learners + num_samplers + num_evaluators
        requested_cpus = num_workers + num_weight_store + num_buffer
        if requested_cpus > available_resources.get("CPU", 0):
            raise ValueError("Not enough CPUs available for the current configuration.")

        # Check GPU availability
        num_gpus = num_learners if learner_device == "gpu" else 0
        shared_gpus = max(num_sampler_gpus, num_evaluator_gpus)
        requested_gpus = max(num_gpus, shared_gpus)
        if num_gpus > 1:
            requested_gpus = num_gpus + shared_gpus
        if requested_gpus > available_resources.get("GPU", 0):
            raise ValueError("Not enough GPUs available for the current configuration.")

        # Initialize actor resource configuration
        self.available_gpus = int(available_resources.get("GPU", 0))
        self.actors = {
            "learner": (num_gpus, num_learners, 1),
            "sampler": (num_sampler_gpus, num_samplers, num_samplers_per_gpu),
            "evaluator": (num_evaluator_gpus, num_evaluators, num_evaluators_per_gpu),
            "memory": (0, 1, 0),
            "weight_store": (0, 1, 0),
        }

        self.group = None
        self.allocations = {}

    def reserve(self, timeout_seconds=30):
        """Reserve CPU and GPU bundles within the specified timeout."""
        bundles = []
        gpu_indices = []
        learner_indices = []
        gpu_totals = {}

        for role, (num_gpus, num_actors, actors_per_gpu) in self.actors.items():
            allocations = []

            # Prefer new GPUs, then non-learner GPUs, then a single learner's GPU.
            allow_learner_sharing = len(learner_indices) < 2
            shared_indices = [
                index
                for index in gpu_indices
                if allow_learner_sharing or index not in learner_indices
            ]
            shared_indices.sort(key=lambda index: index in learner_indices)
            new_gpus = min(num_gpus, self.available_gpus - len(gpu_indices))

            for i in range(num_gpus):
                remaining = num_actors - i * actors_per_gpu
                actor_count = max(0, min(actors_per_gpu, remaining))

                if i < new_gpus:
                    index = len(bundles)
                    bundles.append({"CPU": 0, "GPU": 1})
                    gpu_indices.append(index)
                    gpu_totals[index] = 0
                else:
                    index = shared_indices[i - new_gpus]

                if role == "learner":
                    learner_indices.append(index)

                bundles[index]["CPU"] += actor_count
                gpu_totals[index] += actor_count / actors_per_gpu
                allocation = (index, 1 / actors_per_gpu)
                allocations.extend([allocation] * actor_count)

            for _ in range(num_actors - len(allocations)):
                bundles.append({"CPU": 1})
                allocations.append((len(bundles) - 1, 0))

            self.allocations[role] = allocations

        # Normalize shared requests; Ray uses 1e-4 resource precision.
        for allocations in self.allocations.values():
            for i, (index, share) in enumerate(allocations):
                total = gpu_totals.get(index, 0)
                if total > 1:
                    share = int(share / total * 10_000) / 10_000
                    allocations[i] = (index, share)

        self.group = placement_group(bundles)
        if not self.group.wait(timeout_seconds=timeout_seconds):
            remove_placement_group(self.group)
            self.group = None
            msg = f"Could not reserve resources within {timeout_seconds} seconds."
            raise RuntimeError(msg)

    def get_allocation(self, role, index=0):
        """Return the device and Ray options for a worker's role and index."""
        bundle_index, num_gpus = self.allocations[role][index]
        pgStrategy = PlacementGroupSchedulingStrategy
        strategy = pgStrategy(self.group, bundle_index)
        device = "gpu" if num_gpus else "cpu"
        option = {"num_cpus": 1, "num_gpus": num_gpus, "scheduling_strategy": strategy}
        return device, option
