import unittest
from unittest.mock import Mock, patch

import numpy as np

from omnirl.envs.gymnasium import CollectionLoop, EvaluationLoop
from omnirl.envs.gymnasium import EnvAdapter, EnvProvider


class GymnasiumEnvironmentTest(unittest.TestCase):
    def test_collection_returns_batched_trajectories(self):
        for num_envs in (1, 3):
            for is_continuous in (False, True):
                with self.subTest(num_envs=num_envs, continuous=is_continuous):
                    envs = []
                    expected_states = []
                    expected_next_states = []
                    for i in range(num_envs):
                        observations = [[i, 0], [i, 1], [i, -1], [i, 2]]
                        observations = np.asarray(observations, dtype=np.float32)
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
                        actions = np.stack((actions, -actions), axis=-1)
                        actions = actions.astype(np.float32)
                    hidden_state = np.arange(num_envs * 3, dtype=np.float32)
                    hidden_state = hidden_state.reshape(num_envs, 3)
                    policy = Mock()
                    policy.sample_action.return_value = {
                        "action": actions,
                        "env_action": actions + 10,
                        "extra_data": {
                            "log_probs": np.full(num_envs, -0.5, dtype=np.float32),
                            "hidden_state": hidden_state,
                        },
                    }
                    env_builder = Mock(side_effect=envs)
                    collection = CollectionLoop(env_builder, seed=10, num_envs=num_envs)

                    rollout = collection.run(policy, num_steps=2)

                    np.testing.assert_array_equal(rollout["states"], expected_states)
                    next_states = rollout["next_states"]
                    np.testing.assert_array_equal(next_states, expected_next_states)
                    self.assertEqual(rollout["states"].dtype, np.float32)
                    expected_rewards = np.tile([1.0, 2.0], (num_envs, 1))
                    expected_terminals = [[i % 2 == 0, False] for i in range(num_envs)]
                    expected_dones = np.tile([True, False], (num_envs, 1))
                    expected_actions = np.stack((actions, actions), axis=1)
                    terminals = rollout["terminals"]
                    np.testing.assert_array_equal(rollout["rewards"], expected_rewards)
                    np.testing.assert_array_equal(terminals, expected_terminals)
                    np.testing.assert_array_equal(rollout["dones"], expected_dones)
                    np.testing.assert_array_equal(rollout["action"], expected_actions)

                    extra_data = rollout["extra_data"]
                    expected_log_probs = np.full((num_envs, 2), -0.5, dtype=np.float32)
                    expected_hidden = np.arange(num_envs * 3, dtype=np.float32)
                    expected_hidden = expected_hidden.reshape(num_envs, 1, 3)
                    expected_hidden = np.repeat(expected_hidden, 2, axis=1)
                    log_probs = extra_data["log_probs"]
                    hidden_state = extra_data["hidden_state"]
                    np.testing.assert_array_equal(log_probs, expected_log_probs)
                    np.testing.assert_array_equal(hidden_state, expected_hidden)
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
        expected_options = {"render_mode": None, "max_episode_steps": 7}
        make.assert_called_once_with("Example-v0", **expected_options)
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
                    build_kwargs = {"autospec": True, "return_value": env}
                    with patch.object(CustomProvider, "build", **build_kwargs) as build:
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
                state_spec = {**env_spec, "state_dtype": np.dtype(np.float32)}
                self.assertEqual(spec["state_spec"], state_spec)
                for key, value in env_spec.items():
                    self.assertEqual(spec["action_spec"][key], value)
                for name, bound in [("action_low", -2.0), ("action_high", 2.0)]:
                    if is_continuous:
                        action_bound = spec["action_spec"][name]
                        np.testing.assert_array_equal(action_bound, [bound])
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
                policy.choose_action.return_value = np.array([0])
                evaluation = EvaluationLoop(env_builder=lambda: env, seed=10)

                first = evaluation.run(policy, num_episodes_per_env=2)
                second = evaluation.run(policy, num_episodes_per_env=1)

                expected_first = {
                    "num_episodes": 2,
                    "metrics": {
                        "return_mean": 4.0,
                        "return_std": 1.0,
                        "return_max": 5.0,
                        "return_min": 3.0,
                    },
                }
                expected_second = {
                    "num_episodes": 1,
                    "metrics": {
                        "return_mean": 9.0,
                        "return_std": 0.0,
                        "return_max": 9.0,
                        "return_min": 9.0,
                    },
                }
                self.assertEqual(first, expected_first)
                self.assertEqual(second, expected_second)
                self.assertEqual(env.reset.call_count, 4)
                self.assertEqual(policy.choose_action.call_count, 4)
                expected_render_count = 4 if render_mode == "human" else 0
                self.assertEqual(env.render.call_count, expected_render_count)

    def test_evaluation_batches_equal_episodes_per_env(self):
        cases = [(2, 10, [2, 3, 2, 3, 3, 3]), (1, None, [2, 3, 3])]
        for num_episodes, seed, returns in cases:
            for is_continuous in (False, True):
                with self.subTest(episodes=num_episodes, continuous=is_continuous):
                    envs = []
                    for i in range(3):
                        observation = np.array([i, 0], dtype=np.float32)
                        env = Mock(render_mode="human")
                        env.reset.return_value = (observation, {})
                        steps = []
                        length = 3 if i == 0 else 1
                        for _ in range(num_episodes):
                            for step in range(1, length + 1):
                                done = step == length
                                next_observation = np.array([i, step], dtype=np.float32)
                                transition = (
                                    next_observation,
                                    float(i + 1),
                                    done and i % 2 == 0,
                                    done and i % 2 == 1,
                                    {},
                                )
                                steps.append(transition)
                        env.step.side_effect = steps
                        envs.append(env)

                    def choose_action(observations):
                        actions = observations[:, 0].astype(np.int64)
                        if is_continuous:
                            actions = np.stack((actions, -actions), axis=-1)
                        return actions

                    policy = Mock()
                    policy.choose_action.side_effect = choose_action
                    env_builder = Mock(side_effect=envs)
                    evaluation = EvaluationLoop(env_builder, seed=seed, num_envs=3)
                    result = evaluation.run(policy, num_episodes)

                    returns = np.asarray(returns, dtype=np.float32)
                    expected_metrics = {
                        "return_mean": float(returns.mean()),
                        "return_std": float(returns.std()),
                        "return_max": float(returns.max()),
                        "return_min": float(returns.min()),
                    }
                    self.assertEqual(result["num_episodes"], 3 * num_episodes)
                    self.assertEqual(result["metrics"], expected_metrics)
                    expected_batches = [[0, 1, 2]] * num_episodes
                    expected_batches += [[0]] * (2 * num_episodes)
                    calls = policy.choose_action.call_args_list
                    self.assertEqual(len(calls), len(expected_batches))
                    for call, expected in zip(calls, expected_batches):
                        np.testing.assert_array_equal(call.args[0][:, 0], expected)
                    policy.sample_action.assert_not_called()
                    for i, env in enumerate(envs):
                        expected_seed = None if seed is None else seed + i
                        reset_kwargs = env.reset.call_args_list[0].kwargs
                        expected_steps = num_episodes * (3 if i == 0 else 1)
                        self.assertEqual(reset_kwargs, {"seed": expected_seed})
                        self.assertEqual(env.reset.call_count, num_episodes + 1)
                        self.assertEqual(env.step.call_count, expected_steps)
                        self.assertEqual(env.render.call_count, env.step.call_count)
                        expected_action = np.array([i, -i]) if is_continuous else i
                        for call in env.step.call_args_list:
                            np.testing.assert_array_equal(call.args[0], expected_action)


if __name__ == "__main__":
    unittest.main()
