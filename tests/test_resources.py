import itertools
import unittest
from unittest.mock import patch

from omnirl.runtime.resources import ResourceAllocator


def gpu_indices(allocator, role):
    return {index for index, share in allocator.allocations[role] if share > 0}


class ResourceAllocatorTest(unittest.TestCase):
    def reserve(self, available_gpus, **changes):
        kwargs = {
            "num_learners": 1,
            "num_samplers": 2,
            "num_evaluators": 2,
            "num_buffers": 1,
            "learner_device": "gpu",
            "num_sampler_gpus": 1,
            "num_samplers_per_gpu": 2,
            "num_evaluator_gpus": 1,
            "num_evaluators_per_gpu": 2,
            **changes,
        }
        actor_counts = [
            kwargs[f"num_{role}s"] for role in ("learner", "sampler", "evaluator")
        ]
        num_cpus = sum(actor_counts) + kwargs.get("num_buffers", 1) + 1
        resources = {"CPU": num_cpus, "GPU": available_gpus}
        target = "omnirl.runtime.resources"
        with patch(f"{target}.ray.available_resources", return_value=resources):
            with patch(f"{target}.placement_group") as create_group:
                create_group.return_value.wait.return_value = True
                allocator = ResourceAllocator(**kwargs)
                allocator.reserve()
                bundles = create_group.call_args.args[0]
        self.assertEqual(sum(bundle["CPU"] for bundle in bundles), resources["CPU"])
        reserved_gpus = sum(bundle.get("GPU", 0) for bundle in bundles)
        self.assertLessEqual(reserved_gpus, available_gpus)
        return allocator, bundles

    def test_single_gpu_is_shared_by_all_roles(self):
        allocator, bundles = self.reserve(1)
        for role in ("learner", "sampler", "evaluator"):
            self.assertEqual(gpu_indices(allocator, role), {0})
        self.assertEqual(bundles[0], {"CPU": 5, "GPU": 1})
        for role in ("memory", "weight_store"):
            self.assertEqual(allocator.get_allocation(role)[0], "cpu")

    def test_multiple_buffers_reserve_separate_cpus(self):
        allocator, bundles = self.reserve(1, num_buffers=2)
        self.assertEqual(len(allocator.allocations["memory"]), 2)
        for index, share in allocator.allocations["memory"]:
            self.assertEqual(bundles[index], {"CPU": 1})
            self.assertEqual(share, 0)

    def test_free_cards_precede_sharing(self):
        allocator, _ = self.reserve(3)
        assignments = [
            gpu_indices(allocator, role) for role in ("learner", "sampler", "evaluator")
        ]
        self.assertEqual(len(set.union(*assignments)), 3)
        for first, second in itertools.combinations(assignments, 2):
            self.assertTrue(first.isdisjoint(second))

    def test_evaluator_prefers_sampler_only_card(self):
        allocator, _ = self.reserve(2, num_sampler_gpus=2, num_samplers=4)
        learner = gpu_indices(allocator, "learner")
        sampler = gpu_indices(allocator, "sampler")
        evaluator = gpu_indices(allocator, "evaluator")
        self.assertEqual(len(sampler), 2)
        self.assertEqual(evaluator, sampler - learner)
        first_sampler_index = allocator.allocations["sampler"][0][0]
        self.assertNotIn(first_sampler_index, learner)
        self.assertTrue(learner.issubset(sampler))

    def test_cpu_learners_allow_sampler_evaluator_sharing(self):
        allocator, _ = self.reserve(1, learner_device="cpu", num_learners=2)
        self.assertFalse(gpu_indices(allocator, "learner"))
        sampler = gpu_indices(allocator, "sampler")
        self.assertEqual(len(sampler), 1)
        self.assertEqual(gpu_indices(allocator, "evaluator"), sampler)

    def test_ddp_cards_remain_exclusive(self):
        allocator, _ = self.reserve(3, num_learners=2)
        learner = gpu_indices(allocator, "learner")
        sampler = gpu_indices(allocator, "sampler")
        self.assertEqual(len(learner), 2)
        self.assertTrue(learner.isdisjoint(sampler))
        self.assertEqual(gpu_indices(allocator, "evaluator"), sampler)
        learner_shares = [share for _, share in allocator.allocations["learner"]]
        self.assertEqual(learner_shares, [1, 1])

    def test_minimum_gpu_counts(self):
        for device, learners in (("gpu", 1), ("gpu", 3), ("cpu", 2)):
            for samplers, evaluators in itertools.product(range(3), repeat=2):
                kwargs = {
                    "learner_device": device,
                    "num_learners": learners,
                    "num_samplers": 4,
                    "num_evaluators": 4,
                    "num_sampler_gpus": samplers,
                    "num_evaluator_gpus": evaluators,
                }
                learner_gpus = learners if device == "gpu" else 0
                minimum = max(learner_gpus, samplers, evaluators)
                if learner_gpus > 1:
                    minimum = learner_gpus + max(samplers, evaluators)
                with self.subTest(**kwargs):
                    allocator, bundles = self.reserve(minimum, **kwargs)
                    actual = sum(bundle.get("GPU", 0) for bundle in bundles)
                    self.assertEqual(actual, minimum)
                    if minimum == 0:
                        for allocations in allocator.allocations.values():
                            self.assertTrue(all(share == 0 for _, share in allocations))
                    if minimum > 0:
                        with self.assertRaisesRegex(ValueError, "Not enough GPUs"):
                            self.reserve(minimum - 1, **kwargs)

    def test_excess_actors_still_use_cpu(self):
        expected = {
            "sampler": ["gpu", "gpu", "cpu", "cpu", "cpu"],
            "evaluator": ["gpu", "cpu", "cpu"],
        }
        kwargs = {
            "num_samplers": 5,
            "num_evaluators": 3,
            "num_evaluators_per_gpu": 1,
        }
        for available_gpus in (1, 2, 3):
            with self.subTest(available_gpus=available_gpus):
                allocator, bundles = self.reserve(available_gpus, **kwargs)
                for role, devices in expected.items():
                    for index, expected_device in enumerate(devices):
                        device, options = allocator.get_allocation(role, index)
                        self.assertEqual(device, expected_device)
                        self.assertEqual(options["num_cpus"], 1)
                        self.assertEqual(options["num_gpus"] > 0, device == "gpu")
                        strategy = options["scheduling_strategy"]
                        self.assertIs(strategy.placement_group, allocator.group)
                        bundle_index = strategy.placement_group_bundle_index
                        expected_index = allocator.allocations[role][index][0]
                        self.assertEqual(bundle_index, expected_index)
                        bundle_gpus = bundles[bundle_index].get("GPU", 0)
                        self.assertEqual(bundle_gpus, int(device == "gpu"))

    def test_shared_fractions_fit_ray_precision(self):
        kwargs = {
            "num_samplers": 7,
            "num_samplers_per_gpu": 7,
            "num_evaluators": 11,
            "num_evaluators_per_gpu": 11,
        }
        allocator, _ = self.reserve(1, **kwargs)
        units = 0
        for role in ("learner", "sampler", "evaluator"):
            for _, share in allocator.allocations[role]:
                self.assertGreater(share, 0)
                self.assertEqual(share, round(share, 4))
                units += round(share * 10_000)
        self.assertLessEqual(units, 10_000)

    def test_gpu_precision_without_shared_normalization(self):
        kwargs = {"learner_device": "cpu", "num_evaluator_gpus": 0}
        allocator, _ = self.reserve(1, num_samplers_per_gpu=10_000, **kwargs)
        for i in range(2):
            device, options = allocator.get_allocation("sampler", i)
            self.assertEqual(device, "gpu")
            self.assertEqual(options["num_gpus"], 0.0001)
        self.assertEqual(allocator.get_allocation("learner")[0], "cpu")


if __name__ == "__main__":
    unittest.main()
