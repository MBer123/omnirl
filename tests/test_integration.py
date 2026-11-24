import tempfile
import unittest

import ray

from omnirl.algorithms.off_policy import dqn, per_dqn
from omnirl.algorithms.on_policy import ppo
from omnirl.envs import gymnasium
from omnirl.runtime.runner import Runner


class IntegrationTest(unittest.TestCase):
    def run_updates(
        self,
        agent,
        agent_config,
        num_envs,
        fragment_length=1,
        num_buffers=1,
        max_updates=1,
    ):
        with tempfile.TemporaryDirectory() as ckpt_dir:
            runner_kwargs = {
                "agent": agent,
                "environment": gymnasium,
                "agent_config": agent_config,
                "env_config": {"id": "CartPole-v1"},
                "ckpt_dir": ckpt_dir,
                "num_learners": num_buffers,
                "num_samplers": num_buffers,
                "num_buffers": num_buffers,
                "num_envs_per_sampler": num_envs,
                "collection_fragment_length": fragment_length,
                "num_evaluators": 1,
                "num_envs_per_evaluator": num_envs,
                "num_episodes_per_env": 5,
                "max_updates": max_updates,
                "save_interval": 3600,
                "log_interval": 3600,
                "load_checkpoint": False,
                "learner_device": "cpu",
                "learner_seed": 0,
                "sampler_seed": 0,
                "evaluator_seed": 0,
            }
            runner = Runner(**runner_kwargs)
            try:
                runner.fit()
                self.assertEqual(runner.version, max_updates)
                self.assertEqual(len(runner.memories), num_buffers)
                self.assertGreater(runner.trained_samples, 0)
                self.assertEqual(runner.latest_eval_info["num_episodes"], num_envs * 5)
                return runner.trained_samples
            finally:
                ray.shutdown()

    def test_replay_runner_updates(self):
        agent_config = {
            "learner": {"gamma": 0.99, "tau": 0.005, "lr": 0.0003},
            "memory": {
                "batch_size": 4,
                "warmup_size": 4,
                "max_size": 32,
            },
            "policy": {
                "epsilon_start": 1.0,
                "epsilon_end": 0.1,
                "decay_steps": 10,
                "random_warmup": True,
            },
            "network": {
                "layer_type": "MLP",
                "mlp_hidden_dims": [16, 16],
            },
        }

        for num_envs, fragment_length in ((1, 1), (4, 3)):
            with self.subTest(num_envs=num_envs, fragment_length=fragment_length):
                self.run_updates(dqn, agent_config, num_envs, fragment_length)

        agent_config["memory"]["replay_ratio"] = 2
        train_kwargs = {"num_envs": 2, "num_buffers": 2, "max_updates": 2}
        for agent in (dqn, per_dqn):
            with self.subTest(agent=agent.__name__, num_buffers=2):
                trained_samples = self.run_updates(agent, agent_config, **train_kwargs)
                self.assertEqual(trained_samples, 8)

    def test_ppo_runner_updates(self):
        agent_config = {
            "learner": {"k_epochs": 1, "num_minibatches": 1},
            "memory": {
                "batch_size": 6,
            },
            "network": {"layer_type": "MLP", "mlp_hidden_dims": [8, 8]},
        }
        rollout_kwargs = {"num_envs": 4, "fragment_length": 3}
        trained_samples = self.run_updates(ppo, agent_config, **rollout_kwargs)
        self.assertEqual(trained_samples, 6)

        rollout_kwargs.update(num_buffers=2, max_updates=2)
        trained_samples = self.run_updates(ppo, agent_config, **rollout_kwargs)
        self.assertEqual(trained_samples, 12)


if __name__ == "__main__":
    unittest.main()
