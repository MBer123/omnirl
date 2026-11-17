import unittest

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
            self.assertFalse(memory.can_sample(0))
            batch = memory.get_batch(0, 1)
            self.assertEqual(len(batch["rewards"]), 256)
            memory.on_learn([{}])
        self.assertFalse(memory.ready())
        self.assertIsNone(memory.get_batch(0, 1))
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
                                    self.assertFalse(memory.can_sample(0))
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
                learn_info = {"indices": np.arange(5), "td_errors": np.arange(1, 6)}
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
                    else:
                        self.assertEqual(np.unique(batch["rewards"]).size, size)
                        self.assertNotIn("weights", batch)
                        self.assertNotIn("indices", batch)


if __name__ == "__main__":
    unittest.main()
