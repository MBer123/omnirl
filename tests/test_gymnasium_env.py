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

    def test_unknown_role_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported environment role"):
            build_env({"id": "Example-v0"}, role="unknown")

    def test_missing_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires an 'id'"):
            build_env({})

    def test_unknown_config_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "custom_option"):
            build_env({"id": "Example-v0", "custom_option": 42})


if __name__ == "__main__":
    unittest.main()
