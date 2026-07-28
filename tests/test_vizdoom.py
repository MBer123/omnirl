from pathlib import Path
import tempfile
import unittest

import numpy as np

from omnirl.envs.vizdoom import (
    CollectionLoop,
    EnvAdapter,
    EvaluationLoop,
    aggregate_evaluations,
    build_env,
    get_env_spec,
)
from omnirl.envs.vizdoom.protocol import extract_observation


class FakeDoomState:
    """Minimal ViZDoom state carrying a screen buffer.

    Args:
        screen_buffer: Image data returned by the fake game.
    """

    def __init__(self, screen_buffer):
        self.screen_buffer = screen_buffer


class FakeDoomGame:
    """One-step native-game double for adapter protocol tests."""

    def __init__(self):
        self.running = False
        self.finished = False
        self.multiplayer = False
        self.timeout_reached = False
        self.closed = False
        self.seed_value = None
        self.last_action = None
        self.last_frame_skip = None
        self.make_action_calls = 0
        self.ammo = 10.0
        self.kills = 0.0
        self.episode_time = 14
        self.episode_start_time = 14
        self.total_reward = 0.0
        screen_data = [
            [[0, 127, 255], [255, 127, 0]],
            [[64, 128, 192], [32, 96, 160]],
        ]
        screen_buffer = np.asarray(screen_data, dtype=np.uint8)
        self.state = FakeDoomState(screen_buffer)

    def set_seed(self, seed):
        self.seed_value = seed

    def is_running(self):
        return self.running

    def init(self):
        self.running = True
        self.finished = False

    def new_episode(self):
        self.finished = False

    def is_multiplayer_game(self):
        return self.multiplayer

    def get_state(self):
        return None if self.finished else self.state

    def get_game_variable(self, variable):
        if variable == "ammo":
            return self.ammo
        if variable == "kills":
            return self.kills
        raise AssertionError(f"Unexpected game variable: {variable!r}")

    def make_action(self, action, frame_skip):
        self.make_action_calls += 1
        self.last_action = action
        self.last_frame_skip = frame_skip
        self.ammo = 8.0
        self.kills = 1.0
        self.episode_time = 54
        self.total_reward = 5.0
        self.finished = True
        return 3.5

    def is_episode_finished(self):
        return self.finished

    def is_episode_timeout_reached(self):
        return self.timeout_reached

    def get_episode_time(self):
        return self.episode_time

    def get_episode_start_time(self):
        return self.episode_start_time

    def get_total_reward(self):
        return self.total_reward

    def close(self):
        self.running = False
        self.closed = True


class FakeVizDoomLoopEnv:
    """Deterministic two-step episodes for loop and metric tests."""

    render_mode = None

    def __init__(self):
        self.reset_seeds = []
        self.episode = -1
        self.steps = 0

    def reset(self, seed=None, options=None):
        self.reset_seeds.append(seed)
        self.episode += 1
        self.steps = 0
        return np.asarray([self.episode, 0, 1], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        state = [self.episode, self.steps, action]
        observation = np.asarray(state, dtype=np.float32)
        return observation, 1.0, False, self.steps == 2, {}

    def episode_metrics(self):
        if self.episode % 2 == 0:
            return {"kills": 3, "shots": 5, "episode_tics": 400}
        return {"kills": 1, "shots": 3, "episode_tics": 600}

    def render(self):
        return None

    def close(self):
        return None


class VizDoomEnvironmentTest(unittest.TestCase):
    def test_real_defend_center_spec_is_rgb_discrete_and_domain_specific(self):
        spec = get_env_spec({"id": "defend_the_center"})
        expected_metrics = {
            "kills_mean": 0.0,
            "survival_time_mean": 0.0,
            "shots_mean": 0.0,
            "shooting_precision": 0.0,
        }

        self.assertEqual(spec.state_dim, (120, 160, 3))
        self.assertEqual(spec.state_dtype, np.dtype(np.float32))
        self.assertEqual(spec.action_dim, 4)
        self.assertFalse(spec.is_continuous)
        self.assertIsNone(spec.action_low)
        self.assertIsNone(spec.action_high)
        self.assertEqual(spec.default_eval_metrics, expected_metrics)
        self.assertNotIn("return_mean", spec.default_eval_metrics)

    def test_real_defend_center_reset_returns_normalized_hwc_screen(self):
        config = {"id": "defend_the_center", "render_mode": "human"}
        env = build_env(config, role="collection")
        config_path = env.config_path

        try:
            observation, info = env.reset(seed=7)

            self.assertIsInstance(env, EnvAdapter)
            self.assertIsNone(env.render_mode)
            self.assertEqual(env.num_actions, 4)
            self.assertEqual(observation.shape, (120, 160, 3))
            self.assertEqual(observation.dtype, np.dtype(np.float32))
            self.assertGreaterEqual(float(observation.min()), 0.0)
            self.assertLessEqual(float(observation.max()), 1.0)
            self.assertEqual(info, {})
            self.assertNotEqual(config_path.parent.resolve(), Path.cwd().resolve())
        finally:
            env.close()

        self.assertFalse(config_path.exists())

    def test_builder_rejects_unsupported_multi_scenario_and_invalid_config(self):
        with self.assertRaisesRegex(ValueError, "single-agent"):
            build_env({"id": "multi"})

        custom_option = {"id": "defend_the_center", "custom_option": 42}
        invalid_frame_skip = {"id": "defend_the_center", "frame_skip": 0}
        invalid_screen_format = {
            "id": "defend_the_center",
            "screen_format": "BGR24",
        }
        invalid_configs = (
            ({}, "requires an 'id'"),
            (custom_option, "custom_option"),
            (invalid_frame_skip, "positive integer"),
            (invalid_screen_format, "screen_format"),
        )
        for config, message in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, message):
                    build_env(config)

    def test_adapter_tracks_shooting_metrics_and_removes_temp_config(self):
        game = FakeDoomGame()
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "worker-vizdoom.ini"
            config_path.write_text("temporary", encoding="utf-8")
            adapter_kwargs = {
                "game": game,
                "actions": ((0, 0), (1, 0), (0, 1)),
                "frame_skip": 4,
                "render_mode": None,
                "observation_shape": (2, 2, 3),
                "config_path": config_path,
                "ammo_variable": "ammo",
                "killcount_variable": "kills",
            }
            env = EnvAdapter(**adapter_kwargs)

            observation, info = env.reset(seed=11)
            self.assertEqual(game.seed_value, 11)
            self.assertEqual(info, {})
            self.assertEqual(observation.dtype, np.dtype(np.float32))
            np.testing.assert_allclose(observation[0, 0], [0.0, 127 / 255, 1.0])

            action = np.int64(2)
            next_observation, reward, terminated, truncated, info = env.step(action)
            expected_observation = np.zeros((2, 2, 3), dtype=np.float32)
            expected_metrics = {"kills": 1, "shots": 2, "episode_tics": 40}

            np.testing.assert_array_equal(next_observation, expected_observation)
            self.assertEqual(game.last_action, [0, 1])
            self.assertEqual(game.last_frame_skip, 4)
            self.assertEqual(reward, 3.5)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info, {"episode_tics": 40, "total_reward": 5.0})
            self.assertEqual(env.episode_metrics(), expected_metrics)

            env.close()
            self.assertTrue(game.closed)
            self.assertFalse(config_path.exists())

    def test_adapter_rejects_invalid_actions_before_advancing_game(self):
        game = FakeDoomGame()
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "worker-vizdoom.ini"
            adapter_kwargs = {
                "game": game,
                "actions": ((0,), (1,)),
                "frame_skip": 4,
                "render_mode": None,
                "observation_shape": (2, 2, 3),
                "config_path": config_path,
                "ammo_variable": "ammo",
                "killcount_variable": "kills",
            }
            env = EnvAdapter(**adapter_kwargs)

            with self.assertRaisesRegex(ValueError, "must be an integer"):
                env.step(0.5)
            with self.assertRaisesRegex(ValueError, "outside"):
                env.step(2)

            self.assertEqual(game.make_action_calls, 0)
            env.close()

    def test_adapter_rejects_multiplayer_after_native_initialization(self):
        game = FakeDoomGame()
        game.multiplayer = True
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "worker-vizdoom.ini"
            config_path.write_text("temporary", encoding="utf-8")
            adapter_kwargs = {
                "game": game,
                "actions": ((0,), (1,)),
                "frame_skip": 4,
                "render_mode": None,
                "observation_shape": (2, 2, 3),
                "config_path": config_path,
                "ammo_variable": "ammo",
                "killcount_variable": "kills",
            }
            env = EnvAdapter(**adapter_kwargs)

            with self.assertRaisesRegex(ValueError, "single-player"):
                env.reset(seed=3)

            self.assertTrue(game.closed)
            self.assertFalse(config_path.exists())

    def test_observation_protocol_handles_grayscale_and_rejects_bad_shapes(self):
        screen_buffer = np.asarray([[0, 255], [64, 128]], dtype=np.uint8)
        state = FakeDoomState(screen_buffer)

        observation = extract_observation(state, (2, 2, 1))

        self.assertEqual(observation.shape, (2, 2, 1))
        self.assertEqual(observation.dtype, np.dtype(np.float32))
        np.testing.assert_allclose(observation[:, :, 0], state.screen_buffer / 255)
        with self.assertRaisesRegex(ValueError, "Unexpected ViZDoom screen shape"):
            extract_observation(state, (2, 3, 1))

    def test_collection_marks_missing_terminal_screens_and_advances_seed(self):
        env = FakeVizDoomLoopEnv()
        loop = CollectionLoop(lambda: env, seed=17)
        action = {"env_action": 1, "data": {"action": 1}}

        rollout = loop.collect(lambda observation: action, num_steps=3)
        expected_terminals = [False, True, False]

        self.assertEqual(env.reset_seeds, [17, 18])
        np.testing.assert_array_equal(rollout["rewards"], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(rollout["terminals"], expected_terminals)
        np.testing.assert_array_equal(rollout["dones"], expected_terminals)
        np.testing.assert_array_equal(rollout["policy_data"]["action"], [1, 1, 1])
        self.assertEqual(rollout["states"].shape, (3, 3))
        self.assertEqual(rollout["next_states"].shape, (3, 3))

    def test_evaluation_reports_only_vizdoom_shooting_metrics(self):
        env = FakeVizDoomLoopEnv()
        loop = EvaluationLoop(lambda: env, seed=23)

        result = loop.run(lambda observation: 0, num_episodes=2)

        self.assertEqual(env.reset_seeds, [23, 24])
        self.assertEqual(result["num_episodes"], 2)
        self.assertEqual(result["num_kills"], 4)
        self.assertEqual(result["num_shots"], 8)
        self.assertEqual(result["survival_time_sum"], 1000)
        self.assertAlmostEqual(result["metrics"]["kills_mean"], 2.0)
        self.assertAlmostEqual(result["metrics"]["survival_time_mean"], 500.0)
        self.assertAlmostEqual(result["metrics"]["shots_mean"], 4.0)
        self.assertAlmostEqual(result["metrics"]["shooting_precision"], 0.5)
        self.assertNotIn("return_mean", result["metrics"])

    def test_distributed_aggregation_uses_exact_shooting_statistics(self):
        eval_infos = [
            {
                "num_episodes": 2,
                "num_kills": 4,
                "num_shots": 8,
                "survival_time_sum": 900,
                "metrics": {},
            },
            {
                "num_episodes": 1,
                "num_kills": 2,
                "num_shots": 4,
                "survival_time_sum": 300,
                "metrics": {},
            },
        ]

        result = aggregate_evaluations(eval_infos, version=9)

        self.assertEqual(result["version"], 9)
        self.assertEqual(result["num_episodes"], 3)
        self.assertAlmostEqual(result["kills_mean"], 2.0)
        self.assertAlmostEqual(result["survival_time_mean"], 400.0)
        self.assertAlmostEqual(result["shots_mean"], 4.0)
        self.assertAlmostEqual(result["shooting_precision"], 0.5)
        self.assertNotIn("return_mean", result)


if __name__ == "__main__":
    unittest.main()
