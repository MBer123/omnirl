import tempfile
import unittest

import ray

from omnirl.algorithms import dqn, ppo
from omnirl.envs import gymnasium
from omnirl.runtime.runner import Runner


class IntegrationTest(unittest.TestCase):
    def test_dqn_runner_one_update(self):
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

        for num_envs in (1, 4):
            with self.subTest(num_envs=num_envs):
                self.run_one_update(dqn, agent_config, num_envs)

    def test_ppo_runner_one_update(self):
        agent_config = {
            "learner": {"k_epochs": 1, "num_minibatches": 1},
            "memory": {
                "unroll_length": 2,
                "max_chunks": 3,
            },
            "network": {"layer_type": "MLP", "mlp_hidden_dims": [8, 8]},
        }
        self.run_one_update(ppo, agent_config, num_envs=4)

    def run_one_update(self, agent, agent_config, num_envs):
        with tempfile.TemporaryDirectory() as ckpt_dir:
            runner_kwargs = {
                "agent": agent,
                "environment": gymnasium,
                "agent_config": agent_config,
                "env_config": {"id": "CartPole-v1"},
                "ckpt_dir": ckpt_dir,
                "num_learners": 1,
                "num_samplers": 1,
                "num_envs_per_sampler": num_envs,
                "num_evaluators": 1,
                "num_episodes": 1,
                "max_updates": 1,
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
                self.assertEqual(runner.version, 1)
                self.assertGreater(runner.trained_samples, 0)
            finally:
                ray.shutdown()


if __name__ == "__main__":
    unittest.main()
