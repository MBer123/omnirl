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
            kwargs = dict(state_dim=1, state_dtype=np.float32, action_dim=2,
                          is_continuous=True, batch_size=5, max_size=7)
            memory = buffer_class(**kwargs)
            self.assertFalse(memory.needs_backpressure())
            self.assertTrue(memory.supports_prefetch())
            memory.add(make_rollout(0, 5))
            self.assertTrue(memory.ready())
            self.assertTrue(memory.can_sample(0))

    def test_replay_ratio_prefill_and_exact_update_count(self):
        memory = ReplayBuffer(state_dim=1, state_dtype=np.float32, action_dim=2,
                              is_continuous=True, batch_size=256, warmup_size=1024,
                              max_size=2048, replay_ratio=32)
        memory.add(make_rollout(0, 1024))
        self.assertFalse(memory.ready())
        self.assertTrue(memory.can_sample(0))
        memory.add(make_rollout(1024, 64))
        for _ in range(8):
            self.assertTrue(memory.ready())
            self.assertFalse(memory.can_sample(0))
            self.assertEqual(len(memory.get_batch(0, 1)["rewards"]), 256)
            memory.on_learn([{}])
        self.assertFalse(memory.ready())
        self.assertIsNone(memory.get_batch(0, 1))
        self.assertTrue(memory.can_sample(0))

    def test_replay_ratio_progress_with_fragments_and_multiple_ranks(self):
        for buffer_class in (ReplayBuffer, PrioritizedReplayBuffer):
            for ratio in (0.25, 8, 32):
                for fragment in (1, 3, 16):
                    for warmup in (0, 11):
                        with self.subTest(buffer=buffer_class.__name__, ratio=ratio,
                                          fragment=fragment, warmup=warmup):
                            memory = buffer_class(
                                state_dim=1, state_dtype=np.float32, action_dim=2,
                                is_continuous=True, batch_size=5, warmup_size=warmup,
                                max_size=7, replay_ratio=ratio)
                            self.assertTrue(memory.needs_backpressure())
                            self.assertFalse(memory.supports_prefetch())
                            for _ in range(200):
                                self.assertTrue(memory.can_sample(0))
                                # Four samplers can receive permission before any add.
                                for _ in range(4):
                                    memory.add(make_rollout(memory.collected_size, fragment))
                                while memory.ready():
                                    self.assertFalse(memory.can_sample(0))
                                    batches = [memory.get_batch(rank, 2) for rank in range(2)]
                                    self.assertEqual([len(b["rewards"]) for b in batches], [2, 3])
                                    memory.on_learn([
                                        {"indices": b.get("indices", np.array([], dtype=int)),
                                         "td_errors": np.ones(len(b["rewards"]))}
                                        for b in batches])
                                    budget = ratio * max(0, memory.collected_size - max(5, warmup))
                                    self.assertLessEqual(memory.replayed_size, budget)
                                if memory.replayed_size >= 100:
                                    self.assertLess(budget - memory.replayed_size, 5)
                                    break
                            else:
                                self.fail("sampling and training stopped making progress")

    def test_wraparound_and_oversized_rollouts(self):
        for buffer_class in (ReplayBuffer, PrioritizedReplayBuffer):
            with self.subTest(buffer=buffer_class.__name__):
                memory = buffer_class(
                    state_dim=1,
                    state_dtype=np.float32,
                    action_dim=2,
                    is_continuous=True,
                    batch_size=5,
                    warmup_size=6,
                    max_size=5,
                )
                memory.add(make_rollout(0, 5))
                self.assertFalse(memory.ready())
                self.assertIsNone(memory.get_batch(0, 1))
                memory.on_learn([
                    {"indices": np.arange(5), "td_errors": np.arange(1, 6)}
                ])
                self.assertEqual(memory.replayed_size, 5)

                memory.add(make_rollout(5, 2))
                self.assertTrue(memory.ready())
                np.testing.assert_array_equal(memory.reward_memory, [5, 6, 2, 3, 4])
                if isinstance(memory, PrioritizedReplayBuffer):
                    np.testing.assert_allclose(
                        memory.priority_memory, np.array([5, 5, 3, 4, 5]) + 1e-6
                    )

                memory.add(make_rollout(7, 8))
                expected = np.arange(10, 15, dtype=np.float32)
                np.testing.assert_array_equal(memory.state_memory[:, 0], expected)
                np.testing.assert_array_equal(memory.next_state_memory[:, 0], expected + 1)
                np.testing.assert_array_equal(
                    memory.action_memory, np.stack((expected, -expected), axis=-1)
                )
                np.testing.assert_array_equal(memory.reward_memory, expected)
                np.testing.assert_array_equal(memory.terminal_memory, expected % 2 == 0)
                np.testing.assert_array_equal(
                    memory.extra_data_memory["hidden_state"][:, 0], expected * 2
                )
                self.assertEqual(
                    memory.stats(), {"available_data": 5, "collected_data": 15}
                )
                if isinstance(memory, PrioritizedReplayBuffer):
                    np.testing.assert_allclose(memory.priority_memory, 5 + 1e-6)

    def test_rank_batches_keep_sample_fields_aligned(self):
        for buffer_class in (ReplayBuffer, PrioritizedReplayBuffer):
            with self.subTest(buffer=buffer_class.__name__):
                memory = buffer_class(
                    state_dim=1,
                    state_dtype=np.float32,
                    action_dim=2,
                    is_continuous=True,
                    batch_size=5,
                    max_size=5,
                )
                memory.add(make_rollout(0, 5))
                for rank, size in enumerate((2, 3)):
                    batch = memory.get_batch(rank, 2)
                    self.assertEqual(batch["states"].shape, (size, 1))
                    np.testing.assert_array_equal(batch["states"][:, 0], batch["rewards"])
                    np.testing.assert_array_equal(
                        batch["hidden_state"][:, 0], batch["rewards"] * 2
                    )
                    np.testing.assert_array_equal(
                        batch["actions"],
                        np.stack((batch["rewards"], -batch["rewards"]), axis=-1),
                    )
                    if isinstance(memory, PrioritizedReplayBuffer):
                        np.testing.assert_array_equal(batch["indices"], batch["rewards"])
                        np.testing.assert_allclose(batch["weights"], np.ones(size))
                    else:
                        self.assertEqual(np.unique(batch["rewards"]).size, size)
                        self.assertNotIn("weights", batch)
                        self.assertNotIn("indices", batch)


if __name__ == "__main__":
    unittest.main()
