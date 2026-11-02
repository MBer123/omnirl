import unittest
from unittest.mock import Mock, patch

import numpy as np

from omnirl.envs.gymnasium import CollectionLoop, build_env


class GymnasiumEnvironmentTest(unittest.TestCase):
    def test_collection_returns_batched_trajectories(self):
        for num_envs in (1, 3):
            for is_continuous in (False, True):
                with self.subTest(num_envs=num_envs, continuous=is_continuous):
                    envs = []
                    expected_states = []
                    expected_next_states = []
                    for i in range(num_envs):
                        observations = np.asarray(
                            [[i, 0], [i, 1], [i, -1], [i, 2]], dtype=np.float32
                        )
                        env = Mock()
                        env.reset.side_effect = [
                            (observations[0], {}), (observations[2], {})
                        ]
                        env.step.side_effect = [
                            (observations[1], 1.0, i % 2 == 0, i % 2 == 1, {}),
                            (observations[3], 2.0, False, False, {}),
                        ]
                        envs.append(env)
                        expected_states.append(observations[[0, 2]])
                        expected_next_states.append(observations[[1, 3]])

                    actions = np.arange(num_envs)
                    if is_continuous:
                        actions = np.stack((actions, -actions), axis=-1).astype(
                            np.float32
                        )
                    policy = Mock()
                    policy.sample_action.return_value = {
                        "env_action": actions + 10,
                        "data": {
                            "action": actions,
                            "log_prob": np.full(num_envs, -0.5, dtype=np.float32),
                        },
                    }
                    collection = CollectionLoop(
                        env_builder=Mock(side_effect=envs), seed=10,
                        num_envs=num_envs,
                    )

                    rollout = collection.collect(policy, num_steps=2)

                    np.testing.assert_array_equal(rollout["states"], expected_states)
                    np.testing.assert_array_equal(
                        rollout["next_states"], expected_next_states
                    )
                    self.assertEqual(rollout["states"].dtype, np.float32)
                    np.testing.assert_array_equal(
                        rollout["rewards"], np.tile([1.0, 2.0], (num_envs, 1))
                    )
                    np.testing.assert_array_equal(
                        rollout["terminals"],
                        [[i % 2 == 0, False] for i in range(num_envs)],
                    )
                    np.testing.assert_array_equal(
                        rollout["dones"], np.tile([True, False], (num_envs, 1))
                    )
                    np.testing.assert_array_equal(
                        rollout["policy_data"]["action"],
                        np.stack((actions, actions), axis=1),
                    )
                    np.testing.assert_array_equal(
                        rollout["policy_data"]["log_prob"],
                        np.full((num_envs, 2), -0.5, dtype=np.float32),
                    )
                    self.assertEqual(policy.sample_action.call_count, 2)

    def test_collection_disables_rendering(self):
        raw_env = Mock()
        config = {"id": "Example-v0", "render_mode": "human"}
        make_target = "omnirl.envs.gymnasium.builder.gym.make"

        with patch(make_target, return_value=raw_env) as make:
            env = build_env(config, role="collection")

        make.assert_called_once_with("Example-v0", render_mode=None)
        self.assertIs(env.env, raw_env)
        self.assertIsNone(env.render_mode)

    def test_evaluation_preserves_rendering(self):
        raw_env = Mock()
        config = {"id": "Example-v0", "render_mode": "human"}
        make_target = "omnirl.envs.gymnasium.builder.gym.make"

        with patch(make_target, return_value=raw_env) as make:
            env = build_env(config, role="evaluation")

        make.assert_called_once_with("Example-v0", render_mode="human")
        self.assertIs(env.env, raw_env)
        self.assertEqual(env.render_mode, "human")

    def test_specification_disables_rendering(self):
        raw_env = Mock()
        config = {"id": "Example-v0", "render_mode": "human"}
        make_target = "omnirl.envs.gymnasium.builder.gym.make"

        with patch(make_target, return_value=raw_env) as make:
            env = build_env(config, role="specification")

        make.assert_called_once_with("Example-v0", render_mode=None)
        self.assertIsNone(env.render_mode)


if __name__ == "__main__":
    unittest.main()
