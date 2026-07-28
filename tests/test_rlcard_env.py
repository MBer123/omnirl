import inspect
import unittest
from pathlib import Path

import numpy as np

from omnirl.envs.rlcard import (
    CollectionLoop,
    EnvAdapter,
    EvaluationLoop,
    aggregate_evaluations,
    build_env,
    get_env_spec,
)


class FakeBlackjack:
    """Deterministic two-step games used to test the backend protocol.

    Args:
        payoffs: Terminal payoffs assigned to consecutive games.
    """

    num_players = 1
    num_actions = 2

    def __init__(self, payoffs=(1, 0, -1)):
        self.payoffs = tuple(payoffs)
        self.episode = -1
        self.steps = 0
        self.over = False
        self.seed_value = None

    def seed(self, seed=None):
        self.seed_value = seed
        return seed

    def reset(self):
        self.episode += 1
        self.steps = 0
        self.over = False
        return self._state(10 + self.episode), 0

    def step(self, action):
        self.steps += 1
        self.over = self.steps == 2
        return self._state(20 + self.steps + action), 0

    def is_over(self):
        return self.over

    def get_payoffs(self):
        return np.asarray([self.payoffs[self.episode % len(self.payoffs)]])

    @staticmethod
    def _state(score):
        return {
            "obs": np.asarray([score, 7], dtype=np.int64),
            "legal_actions": {0: None, 1: None},
        }


class RLCardEnvironmentTest(unittest.TestCase):
    def test_backend_has_no_rendering_contract(self):
        source_paths = Path("omnirl/envs/rlcard").rglob("*.py")
        source_files = [path.read_text(encoding="utf-8") for path in source_paths]
        sources = "\n".join(source_files)

        self.assertNotIn("render_mode", sources)
        self.assertNotIn("def render(", sources)

    def test_builder_wraps_the_rlcard_environment(self):
        env = build_env({"id": "blackjack"})

        self.assertIsInstance(env, EnvAdapter)
        self.assertNotIn("render_mode", inspect.signature(build_env).parameters)
        self.assertFalse(hasattr(env, "render_mode"))
        self.assertFalse(hasattr(env, "render"))
        self.assertEqual(env.num_players, 1)
        self.assertEqual(env.num_actions, 2)
        self.assertEqual(env.state_shape, [[2]])
        self.assertEqual(env.action_shape, [None])
        self.assertEqual(env.env.__class__.__module__, "rlcard.envs.blackjack")
        self.assertIsNone(env.close())

    def test_real_blackjack_spec(self):
        spec = get_env_spec({"id": "blackjack"})
        expected_metrics = {
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
        }

        self.assertEqual(spec.state_dim, 2)
        self.assertEqual(spec.state_dtype, np.dtype(np.int64))
        self.assertEqual(spec.action_dim, 2)
        self.assertFalse(spec.is_continuous)
        self.assertIsNone(spec.action_low)
        self.assertIsNone(spec.action_high)
        self.assertEqual(spec.default_eval_metrics, expected_metrics)
        self.assertNotIn("return_mean", spec.default_eval_metrics)

    def test_builder_rejects_multi_player_games(self):
        with self.assertRaisesRegex(ValueError, "single-player"):
            build_env({"id": "leduc-holdem"})

    def test_missing_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires an 'id'"):
            build_env({})

    def test_unknown_config_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "custom_option"):
            build_env({"id": "blackjack", "custom_option": 42})

    def test_collection_translates_terminal_payoff(self):
        env = FakeBlackjack(payoffs=(1, -1))
        loop = CollectionLoop(lambda: env, seed=17)

        def sample_action(observation):
            self.assertEqual(observation.shape, (2,))
            return {"env_action": 1, "data": {"action": 1}}

        rollout = loop.collect(sample_action, num_steps=4)
        expected_terminals = [False, True, False, True]
        expected_actions = [1, 1, 1, 1]

        self.assertEqual(env.seed_value, 17)
        np.testing.assert_array_equal(rollout["rewards"], [0.0, 1.0, 0.0, -1.0])
        np.testing.assert_array_equal(rollout["terminals"], expected_terminals)
        np.testing.assert_array_equal(rollout["dones"], expected_terminals)
        actual_actions = rollout["policy_data"]["action"]
        np.testing.assert_array_equal(actual_actions, expected_actions)
        self.assertEqual(rollout["states"].shape, (4, 2))
        self.assertEqual(rollout["next_states"].shape, (4, 2))

    def test_evaluation_reports_outcomes(self):
        env = FakeBlackjack(payoffs=(1, 0, -1))
        loop = EvaluationLoop(lambda: env, seed=23)

        result = loop.run(lambda observation: 0, num_episodes=3)
        expected_counts = {"wins": 1, "draws": 1, "losses": 1}

        self.assertEqual(result["num_episodes"], 3)
        self.assertEqual(result["outcome_counts"], expected_counts)
        self.assertAlmostEqual(result["metrics"]["win_rate"], 1 / 3)
        self.assertAlmostEqual(result["metrics"]["draw_rate"], 1 / 3)
        self.assertAlmostEqual(result["metrics"]["loss_rate"], 1 / 3)

    def test_distributed_aggregation_uses_exact_counts(self):
        eval_infos = [
            {
                "num_episodes": 3,
                "outcome_counts": {"wins": 2, "draws": 1, "losses": 0},
                "metrics": {},
            },
            {
                "num_episodes": 2,
                "outcome_counts": {"wins": 0, "draws": 0, "losses": 2},
                "metrics": {},
            },
        ]

        result = aggregate_evaluations(eval_infos, version=9)

        self.assertEqual(result["version"], 9)
        self.assertEqual(result["num_episodes"], 5)
        self.assertAlmostEqual(result["win_rate"], 0.4)
        self.assertAlmostEqual(result["draw_rate"], 0.2)
        self.assertAlmostEqual(result["loss_rate"], 0.4)

    def test_illegal_action_is_rejected_before_game_step(self):
        loop = CollectionLoop(lambda: FakeBlackjack(), seed=0)
        illegal_action = {"env_action": 3, "data": {"action": 3}}

        with self.assertRaisesRegex(ValueError, "illegal"):
            loop.collect(lambda observation: illegal_action, num_steps=1)


if __name__ == "__main__":
    unittest.main()
