import unittest
from unittest.mock import Mock

import numpy as np

from omnirl.buffers.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


def make_rollout(start, count):
    values = np.arange(start, start + count, dtype=np.float32).reshape(1, count)
    return {
        "states": values[..., None],
        "next_states": values[..., None] + 1,
        "action": np.stack((values, -values), axis=-1),
        "rewards": values,
        "terminals": values % 2 == 0,
        "extra_data": {"hidden_state": values[..., None] * 2},
    }


class ReplayBufferTest(unittest.TestCase):
    def test_replay_ratio_is_optional(self):
        for buffer_class in (ReplayBuffer, PrioritizedReplayBuffer):
            memory_kwargs = {
                "state_dim": 1,
                "state_dtype": np.float32,
                "action_dim": 2,
                "is_continuous": True,
                "batch_size": 5,
                "max_size": 7,
            }
            memory = buffer_class(**memory_kwargs)
            self.assertFalse(memory.needs_backpressure())
            self.assertTrue(memory.supports_prefetch())
            memory.add(make_rollout(0, 5))
            self.assertTrue(memory.ready())
            self.assertTrue(memory.can_sample(0))

    def test_replay_ratio_prefill_and_exact_update_count(self):
        memory_kwargs = {
            "state_dim": 1,
            "state_dtype": np.float32,
            "action_dim": 2,
            "is_continuous": True,
            "batch_size": 256,
            "warmup_size": 1024,
            "max_size": 2048,
            "replay_ratio": 32,
        }
        memory = ReplayBuffer(**memory_kwargs)
        memory.add(make_rollout(0, 1024))
        self.assertFalse(memory.ready())
        self.assertTrue(memory.can_sample(0))
        memory.add(make_rollout(1024, 64))
        for _ in range(8):
            self.assertTrue(memory.ready())
            self.assertTrue(memory.can_sample(0))
            batch = memory.get_batch(0, 1)
            self.assertEqual(len(batch["rewards"]), 256)
            memory.on_learn([{}])
        self.assertFalse(memory.ready())
        self.assertIsNone(memory.get_batch(0, 1))
        self.assertTrue(memory.can_sample(0))

    def test_replay_ratio_allows_collection_during_learning(self):
        cases = ((0.25, 128, 1), (2, 16, 1), (32, 1, 1), (32, 8, 8))
        for buffer_class in (ReplayBuffer, PrioritizedReplayBuffer):
            for ratio, fragment, updates_per_fragment in cases:
                case = {"buffer": buffer_class.__name__, "ratio": ratio}
                with self.subTest(**case, fragment=fragment):
                    memory_kwargs = {
                        "state_dim": 1,
                        "state_dtype": np.float32,
                        "action_dim": 2,
                        "is_continuous": True,
                        "batch_size": 32,
                        "warmup_size": 64,
                        "max_size": 64,
                        "replay_ratio": ratio,
                    }
                    memory = buffer_class(**memory_kwargs)
                    memory.add(make_rollout(0, 64))
                    self.assertFalse(memory.ready())
                    self.assertTrue(memory.can_sample(0))
                    memory.add(make_rollout(64, fragment))

                    for update in range(2 * updates_per_fragment):
                        self.assertTrue(memory.ready())
                        can_collect = update == 0 or update >= updates_per_fragment
                        self.assertEqual(memory.can_sample(0), can_collect)
                        batches = [memory.get_batch(rank, 2) for rank in range(2)]
                        if update == 0:
                            # The next fragment arrives before training completes.
                            memory.add(make_rollout(64 + fragment, fragment))
                            self.assertEqual(memory.replayed_size, 0)
                            self.assertFalse(memory.can_sample(0))

                        train_infos = []
                        for batch in batches:
                            train_info = {
                                "indices": batch.get("indices", []),
                                "generations": batch.get("generations", []),
                                "td_errors": np.ones(len(batch["rewards"])),
                            }
                            train_infos.append(train_info)
                        memory.on_learn(train_infos)
                        self.assertEqual(memory.replayed_size, 32 * (update + 1))

                    self.assertFalse(memory.ready())
                    self.assertIsNone(memory.get_batch(0, 2))
                    self.assertTrue(memory.can_sample(0))

    def test_replay_ratio_progress_with_fragments_and_multiple_ranks(self):
        for buffer_class in (ReplayBuffer, PrioritizedReplayBuffer):
            for ratio in (0.25, 8, 32):
                for fragment in (1, 3, 16):
                    for warmup in (0, 11):
                        case = {
                            "buffer": buffer_class.__name__,
                            "ratio": ratio,
                            "fragment": fragment,
                            "warmup": warmup,
                        }
                        with self.subTest(**case):
                            memory_kwargs = {
                                "state_dim": 1,
                                "state_dtype": np.float32,
                                "action_dim": 2,
                                "is_continuous": True,
                                "batch_size": 5,
                                "warmup_size": warmup,
                                "max_size": 7,
                                "replay_ratio": ratio,
                            }
                            memory = buffer_class(**memory_kwargs)
                            self.assertTrue(memory.needs_backpressure())
                            self.assertFalse(memory.supports_prefetch())
                            message = "sampling and training stopped making progress"
                            for _ in range(200):
                                self.assertTrue(memory.can_sample(0))
                                # Four samplers can receive permission before any add.
                                for _ in range(4):
                                    start = memory.collected_size
                                    rollout = make_rollout(start, fragment)
                                    memory.add(rollout)
                                while memory.ready():
                                    batches = [memory.get_batch(r, 2) for r in range(2)]
                                    sizes = [len(batch["rewards"]) for batch in batches]
                                    self.assertEqual(sizes, [2, 3])
                                    learn_infos = []
                                    for batch in batches:
                                        empty_indices = np.array([], dtype=int)
                                        indices = batch.get("indices", empty_indices)
                                        td_errors = np.ones(len(batch["rewards"]))
                                        learn_info = {
                                            "indices": indices,
                                            "generations": batch.get("generations"),
                                            "td_errors": td_errors,
                                        }
                                        learn_infos.append(learn_info)
                                    memory.on_learn(learn_infos)
                                    prefill_size = max(5, warmup)
                                    new_samples = memory.collected_size - prefill_size
                                    budget = ratio * max(0, new_samples)
                                    self.assertLessEqual(memory.replayed_size, budget)
                                if memory.replayed_size >= 100:
                                    self.assertLess(budget - memory.replayed_size, 5)
                                    break
                            else:
                                self.fail(message)

    def test_wraparound_and_oversized_rollouts(self):
        for buffer_class in (ReplayBuffer, PrioritizedReplayBuffer):
            with self.subTest(buffer=buffer_class.__name__):
                memory_kwargs = {
                    "state_dim": 1,
                    "state_dtype": np.float32,
                    "action_dim": 2,
                    "is_continuous": True,
                    "batch_size": 5,
                    "warmup_size": 6,
                    "max_size": 5,
                }
                memory = buffer_class(**memory_kwargs)
                memory.add(make_rollout(0, 5))
                self.assertFalse(memory.ready())
                self.assertIsNone(memory.get_batch(0, 1))
                learn_info = {
                    "indices": np.arange(5),
                    "generations": np.ones(5, dtype=np.int64),
                    "td_errors": np.arange(1, 6),
                }
                memory.on_learn([learn_info])
                self.assertEqual(memory.replayed_size, 5)

                memory.add(make_rollout(5, 2))
                self.assertTrue(memory.ready())
                np.testing.assert_array_equal(memory.reward_memory, [5, 6, 2, 3, 4])
                if isinstance(memory, PrioritizedReplayBuffer):
                    expected_priorities = np.array([5, 5, 3, 4, 5]) + 1e-6
                    priorities = memory.priority_memory
                    np.testing.assert_allclose(priorities, expected_priorities)

                memory.add(make_rollout(7, 8))
                expected = np.arange(10, 15, dtype=np.float32)
                np.testing.assert_array_equal(memory.state_memory[:, 0], expected)
                next_states = memory.next_state_memory[:, 0]
                np.testing.assert_array_equal(next_states, expected + 1)
                expected_actions = np.stack((expected, -expected), axis=-1)
                np.testing.assert_array_equal(memory.action_memory, expected_actions)
                np.testing.assert_array_equal(memory.reward_memory, expected)
                np.testing.assert_array_equal(memory.terminal_memory, expected % 2 == 0)
                hidden_states = memory.extra_data_memory["hidden_state"][:, 0]
                np.testing.assert_array_equal(hidden_states, expected * 2)
                expected_stats = {"available_data": 5, "collected_data": 15}
                self.assertEqual(memory.stats(), expected_stats)
                if isinstance(memory, PrioritizedReplayBuffer):
                    np.testing.assert_allclose(memory.priority_memory, 5 + 1e-6)

    def test_rank_batches_keep_sample_fields_aligned(self):
        for buffer_class in (ReplayBuffer, PrioritizedReplayBuffer):
            with self.subTest(buffer=buffer_class.__name__):
                memory_kwargs = {
                    "state_dim": 1,
                    "state_dtype": np.float32,
                    "action_dim": 2,
                    "is_continuous": True,
                    "batch_size": 5,
                    "max_size": 5,
                }
                memory = buffer_class(**memory_kwargs)
                memory.add(make_rollout(0, 5))
                for rank, size in enumerate((2, 3)):
                    batch = memory.get_batch(rank, 2)
                    self.assertEqual(batch["states"].shape, (size, 1))
                    rewards = batch["rewards"]
                    np.testing.assert_array_equal(batch["states"][:, 0], rewards)
                    hidden_states = batch["hidden_state"][:, 0]
                    np.testing.assert_array_equal(hidden_states, rewards * 2)
                    expected_actions = np.stack((rewards, -rewards), axis=-1)
                    np.testing.assert_array_equal(batch["actions"], expected_actions)
                    if isinstance(memory, PrioritizedReplayBuffer):
                        np.testing.assert_array_equal(batch["indices"], rewards)
                        np.testing.assert_allclose(batch["weights"], np.ones(size))
                        expected_generations = np.ones(size, dtype=np.int64)
                        generations = batch["generations"]
                        np.testing.assert_array_equal(generations, expected_generations)
                    else:
                        self.assertEqual(np.unique(batch["rewards"]).size, size)
                        self.assertNotIn("weights", batch)
                        self.assertNotIn("indices", batch)
                        self.assertNotIn("generations", batch)

    def test_prioritized_sampling_probabilities_and_weights(self):
        for alpha in (0.0, 0.6, 1.0):
            for capacity, count in ((1, 1), (5, 3), (5, 5)):
                with self.subTest(alpha=alpha, capacity=capacity, count=count):
                    memory_kwargs = {
                        "state_dim": 1,
                        "state_dtype": np.float32,
                        "action_dim": 2,
                        "is_continuous": True,
                        "batch_size": 1,
                        "max_size": capacity,
                        "alpha": alpha,
                    }
                    memory = PrioritizedReplayBuffer(**memory_kwargs)
                    memory.rng = np.random.default_rng(7)
                    memory.add(make_rollout(0, count))
                    errors = np.arange(1, count + 1, dtype=np.float32)
                    learn_info = {
                        "indices": np.arange(count),
                        "generations": np.ones(count, dtype=np.int64),
                        "td_errors": errors,
                    }
                    memory.on_learn([learn_info])

                    indices, sample_info = memory._sample(30000)
                    self.assertTrue(np.all(indices < count))
                    scaled_priorities = (errors + 1e-6).astype(np.float64) ** alpha
                    probabilities = scaled_priorities / scaled_priorities.sum()
                    frequencies = np.bincount(indices, minlength=count) / indices.size
                    np.testing.assert_allclose(frequencies, probabilities, atol=0.01)
                    expected_weights = probabilities[indices] / probabilities.min()
                    expected_weights = expected_weights ** (-memory.beta)
                    weights = sample_info["weights"]
                    np.testing.assert_allclose(weights, expected_weights, rtol=1e-6)
                    self.assertEqual(weights.dtype, np.float32)

                    empty_indices, empty_info = memory._sample(0)
                    self.assertEqual(empty_indices.size, 0)
                    self.assertEqual(empty_info["weights"].size, 0)
                    self.assertEqual(empty_info["generations"].size, 0)

    def test_prioritized_sampling_roundoff_cannot_select_padding(self):
        memory_kwargs = {
            "state_dim": 1,
            "state_dtype": np.float32,
            "action_dim": 2,
            "is_continuous": True,
            "batch_size": 1,
            "max_size": 3,
            "alpha": 1.0,
        }
        memory = PrioritizedReplayBuffer(**memory_kwargs)
        memory.add(make_rollout(0, 3))
        errors = np.array([2**34, 1 + 3 * 2**-17, 5 * 2**34], dtype=np.float32)
        learn_info = {
            "indices": np.arange(3),
            "generations": np.ones(3, dtype=np.int64),
            "td_errors": errors,
        }
        memory.on_learn([learn_info])
        memory.rng = Mock()
        memory.rng.random.return_value = np.array([np.nextafter(1.0, 0.0)])

        indices, sample_info = memory._sample(1)

        np.testing.assert_array_equal(indices, [2])
        self.assertTrue(np.all(np.isfinite(sample_info["weights"])))
        self.assertTrue(np.all(sample_info["weights"] > 0))

    def test_prioritized_updates_filter_stale_rows_before_duplicate_reduction(self):
        memory_kwargs = {
            "state_dim": 1,
            "state_dtype": np.float32,
            "action_dim": 2,
            "is_continuous": True,
            "batch_size": 5,
            "max_size": 5,
            "alpha": 1.0,
        }
        memory = PrioritizedReplayBuffer(**memory_kwargs)
        memory.add(make_rollout(0, 5))
        old_info = {
            "indices": np.array([0, 1, 3]),
            "generations": np.ones(3, dtype=np.int64),
            "td_errors": np.array([99, 98, 4], dtype=np.float32),
        }
        memory.add(make_rollout(5, 2))
        new_info = {
            "indices": np.array([0, 0, 1, 3]),
            "generations": np.array([2, 2, 2, 1], dtype=np.int64),
            "td_errors": np.array([2, 3, -5, 2], dtype=np.float32),
        }
        memory.on_learn([old_info, new_info])

        expected_priorities = np.array([3, 5, 1, 4, 1], dtype=np.float32)
        expected_priorities[[0, 1, 3]] += 1e-6
        np.testing.assert_allclose(memory.priority_memory, expected_priorities)
        self.assertAlmostEqual(memory.max_priority, 5 + 1e-6)
        self.assertAlmostEqual(memory.sum_tree[1], expected_priorities.sum(), places=5)
        self.assertEqual(memory.min_tree[1], 1)
        indices, sample_info = memory._sample(100)
        expected_weights = expected_priorities[indices] ** (-memory.beta)
        np.testing.assert_allclose(sample_info["weights"], expected_weights, rtol=1e-6)

    def test_prioritized_oversized_rollout_invalidates_sampled_generations(self):
        memory_kwargs = {
            "state_dim": 1,
            "state_dtype": np.float32,
            "action_dim": 2,
            "is_continuous": True,
            "batch_size": 5,
            "max_size": 5,
            "beta_anneal_steps": 2,
        }
        memory = PrioritizedReplayBuffer(**memory_kwargs)
        memory.add(make_rollout(0, 5))
        batch = memory.get_batch(0, 1)
        memory.add(make_rollout(5, 12))
        np.testing.assert_array_equal(batch["generations"], np.ones(5, dtype=np.int64))
        np.testing.assert_array_equal(memory.generation_memory, np.full(5, 2))
        learn_info = {
            "indices": batch["indices"],
            "generations": batch["generations"],
            "td_errors": np.full(5, 1000, dtype=np.float32),
        }
        memory.on_learn([learn_info])

        np.testing.assert_array_equal(memory.priority_memory, np.ones(5))
        self.assertEqual(memory.max_priority, 1)
        self.assertEqual(memory.sum_tree[1], 5)
        self.assertEqual(memory.min_tree[1], 1)
        self.assertEqual(memory.replayed_size, 5)
        self.assertAlmostEqual(memory.beta, 0.7)


if __name__ == "__main__":
    unittest.main()
