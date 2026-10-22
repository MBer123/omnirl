import unittest
from unittest.mock import Mock, patch

from omnirl.envs.gymnasium import build_env


class GymnasiumEnvironmentTest(unittest.TestCase):
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
