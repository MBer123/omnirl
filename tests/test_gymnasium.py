import unittest
from unittest.mock import Mock, patch

import numpy as np

from omnirl.envs.gymnasium import (
    CollectionLoop,
    EnvAdapter,
    EnvProvider,
    EvaluationLoop,
)


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
                            (observations[0], {}),
                            (observations[2], {}),
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
                        "action": actions,
                        "env_action": actions + 10,
                        "extra_data": {
                            "log_probs": np.full(num_envs, -0.5, dtype=np.float32),
                            "hidden_state": np.arange(
                                num_envs * 3, dtype=np.float32
                            ).reshape(num_envs, 3),
                        },
                    }
                    collection = CollectionLoop(
                        env_builder=Mock(side_effect=envs),
                        seed=10,
                        num_envs=num_envs,
                    )

                    rollout = collection.run(policy, num_steps=2)

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
                        rollout["action"],
                        np.stack((actions, actions), axis=1),
                    )
                    np.testing.assert_array_equal(
                        rollout["extra_data"]["log_probs"],
                        np.full((num_envs, 2), -0.5, dtype=np.float32),
                    )
                    np.testing.assert_array_equal(
                        rollout["extra_data"]["hidden_state"],
                        np.repeat(
                            np.arange(num_envs * 3, dtype=np.float32).reshape(
                                num_envs, 1, 3
                            ),
                            2,
                            axis=1,
                        ),
                    )
                    self.assertEqual(policy.sample_action.call_count, 2)

    def test_collection_disables_rendering(self):
        raw_env = Mock()
        config = {"id": "Example-v0", "render_mode": "human"}
        make_target = "omnirl.envs.gymnasium.backend.gym.make"

        with patch(make_target, return_value=raw_env) as make:
            env = EnvProvider.build(config, role="collection")

        make.assert_called_once_with("Example-v0", render_mode=None)
        self.assertIsInstance(env, EnvAdapter)
        self.assertIs(env.env, raw_env)

    def test_evaluation_preserves_rendering(self):
        raw_env = Mock()
        config = {"id": "Example-v0", "render_mode": "human"}
        make_target = "omnirl.envs.gymnasium.backend.gym.make"

        with patch(make_target, return_value=raw_env) as make:
            env = EnvProvider.build(config, role="evaluation")

        make.assert_called_once_with("Example-v0", render_mode="human")
        self.assertIsInstance(env, EnvAdapter)
        self.assertIs(env.env, raw_env)

    def test_specification_disables_rendering(self):
        raw_env = Mock()
        config = {"id": "Example-v0", "render_mode": "human"}
        make_target = "omnirl.envs.gymnasium.backend.gym.make"

        with patch(make_target, return_value=raw_env) as make:
            env = EnvProvider.build(config, role="specification")

        make.assert_called_once_with("Example-v0", render_mode=None)
        self.assertIsInstance(env, EnvAdapter)
        self.assertIs(env.env, raw_env)

    def test_provider_forwards_environment_options(self):
        config = {"id": "Example-v0", "render_mode": "human", "max_episode_steps": 7}
        with patch("omnirl.envs.gymnasium.backend.gym.make") as make:
            EnvProvider.build(config, role="collection")
        make.assert_called_once_with(
            "Example-v0", render_mode=None, max_episode_steps=7
        )
        self.assertEqual(config["render_mode"], "human")
        self.assertEqual(config["max_episode_steps"], 7)

    def test_spec_uses_provider_and_closes_environment(self):
        class CustomProvider(EnvProvider):
            pass

        cases = [("CartPole-v1", 4, 2, False), ("Pendulum-v1", 3, 1, True)]
        for env_id, state_dim, action_dim, is_continuous in cases:
            with self.subTest(env_id=env_id):
                config = {"id": env_id}
                env = EnvProvider.build(config)
                try:
                    with patch.object(
                        CustomProvider, "build", autospec=True, return_value=env
                    ) as build:
                        with patch.object(env, "close", wraps=env.close) as close:
                            spec = CustomProvider.get_spec(config)
                            close.assert_called_once_with()
                    build.assert_called_once_with(config, role="specification")
                finally:
                    env.close()

                env_spec = {
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "is_continuous": is_continuous,
                }
                self.assertEqual(set(spec), {"env_spec", "state_spec", "action_spec"})
                self.assertEqual(spec["env_spec"], env_spec)
                self.assertEqual(
                    spec["state_spec"],
                    {**env_spec, "state_dtype": np.dtype(np.float32)},
                )
                for key, value in env_spec.items():
                    self.assertEqual(spec["action_spec"][key], value)
                for name, bound in [("action_low", -2.0), ("action_high", 2.0)]:
                    if is_continuous:
                        np.testing.assert_array_equal(
                            spec["action_spec"][name], [bound]
                        )
                    else:
                        self.assertIsNone(spec["action_spec"][name])

    def test_evaluation_resets_metrics_between_runs(self):
        for render_mode in (None, "human", "rgb_array"):
            with self.subTest(render_mode=render_mode):
                observation = np.zeros(2, dtype=np.float32)
                env = Mock(render_mode=render_mode)
                env.reset.return_value = (observation, {})
                env.step.side_effect = [
                    (observation, 1.0, False, False, {}),
                    (observation, 2.0, True, False, {}),
                    (observation, 5.0, False, True, {}),
                    (observation, 9.0, True, False, {}),
                ]
                policy = Mock()
                policy.choose_action.return_value = 0
                evaluation = EvaluationLoop(env_builder=lambda: env, seed=10)

                first = evaluation.run(policy, num_episodes=2)
                second = evaluation.run(policy, num_episodes=1)

                self.assertEqual(
                    first,
                    {
                        "num_episodes": 2,
                        "metrics": {
                            "return_mean": 4.0,
                            "return_std": 1.0,
                            "return_max": 5.0,
                            "return_min": 3.0,
                        },
                    },
                )
                self.assertEqual(
                    second,
                    {
                        "num_episodes": 1,
                        "metrics": {
                            "return_mean": 9.0,
                            "return_std": 0.0,
                            "return_max": 9.0,
                            "return_min": 9.0,
                        },
                    },
                )
                self.assertEqual(env.reset.call_count, 4)
                self.assertEqual(policy.choose_action.call_count, 4)
                self.assertEqual(
                    env.render.call_count, 4 if render_mode == "human" else 0
                )


if __name__ == "__main__":
    unittest.main()
