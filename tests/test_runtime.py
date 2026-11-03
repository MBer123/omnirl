import tempfile
import threading
import unittest

import numpy as np
import ray
import torch

from omnirl.buffers import PrioritizedReplayBuffer, ReplayBuffer, RolloutBuffer
from omnirl.envs.gymnasium import aggregate_evaluations
from omnirl.runtime.runner import Runner
from omnirl.runtime.weight_store import WeightStore


@ray.remote
class MetricEvaluator:
    """Evaluator double returning fixed metrics.

    Args:
        metrics: Evaluation metrics returned by the double.
        num_episodes: Number of episodes represented by the metrics.
    """

    def __init__(self, metrics, num_episodes):
        self.metrics = metrics
        self.num_episodes = num_episodes

    def evaluate(self, weights, version):
        return {
            "version": version,
            "num_episodes": self.num_episodes,
            "metrics": self.metrics,
        }


class RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(num_cpus=8, include_dashboard=False, log_to_driver=False)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_weight_store(self):
        weights = {"weight": torch.tensor([1.0])}
        store = WeightStore.remote(weights, 0)

        self.assertIsNone(ray.get(store.get.remote(0))["weights"])
        ray.get(store.set.remote({"weight": torch.tensor([2.0])}, 1))

        result = ray.get(store.get.remote(0))
        self.assertEqual(result["version"], 1)
        self.assertEqual(result["weights"]["weight"].item(), 2.0)

    def test_evaluation_metrics(self):
        runner = Runner.__new__(Runner)
        runner.stop_event = threading.Event()
        runner.eval_version = -1
        runner.latest_eval_info = {}
        runner.weight_store = WeightStore.remote({"weight": torch.tensor([1.0])}, 0)
        first_metrics = {
            "return_mean": 2.0,
            "return_std": 1.0,
            "return_max": 3.0,
            "return_min": 1.0,
        }
        second_metrics = {
            "return_mean": 5.0,
            "return_std": 0.0,
            "return_max": 5.0,
            "return_min": 5.0,
        }
        first_evaluator = MetricEvaluator.remote(first_metrics, 2)
        second_evaluator = MetricEvaluator.remote(second_metrics, 1)
        runner.evaluators = [first_evaluator, second_evaluator]
        runner.aggregate_evaluations = aggregate_evaluations

        result = runner._start_evaluators(is_train=False)

        self.assertEqual(result["version"], 0)
        self.assertEqual(result["num_episodes"], 3)
        self.assertAlmostEqual(result["return_mean"], 3.0)
        self.assertAlmostEqual(result["return_std"], (8 / 3) ** 0.5)
        self.assertEqual(result["return_max"], 5.0)
        self.assertEqual(result["return_min"], 1.0)

    def test_replay_buffer(self):
        memory_kwargs = {
            "state_dim": 4,
            "state_dtype": np.float32,
            "action_dim": 2,
            "batch_size": 2,
            "warmup_size": 2,
            "max_size": 8,
        }
        memory = ReplayBuffer.remote(**memory_kwargs)

        rollout = {
            "states": np.zeros((1, 3, 4), dtype=np.float32),
            "next_states": np.ones((1, 3, 4), dtype=np.float32),
            "rewards": np.arange(3, dtype=np.float32).reshape(1, 3),
            "terminals": np.zeros((1, 3), dtype=np.bool_),
            "action": np.zeros((1, 3), dtype=np.int64),
            "extra_data": {
                "hidden_state": np.array([[[0, 0], [1, -1], [2, -2]]], dtype=np.float32),
            },
        }
        ray.get(memory.add.remote(rollout))

        self.assertTrue(ray.get(memory.ready.remote()))
        batch = ray.get(memory.get_batch.remote(0, 1))
        self.assertEqual(batch["states"].shape, (2, 4))
        self.assertEqual(batch["actions"].shape, (2,))
        self.assertNotIn("extra_data", batch)
        np.testing.assert_array_equal(
            batch["hidden_state"],
            np.stack((batch["rewards"], -batch["rewards"]), axis=-1),
        )

        ray.get(memory.on_learn.remote([{}]))
        self.assertTrue(ray.get(memory.ready.remote()))

    def test_prioritized_replay_buffer(self):
        memory_kwargs = {
            "state_dim": 4,
            "state_dtype": np.float32,
            "action_dim": 2,
            "batch_size": 2,
            "warmup_size": 2,
            "max_size": 8,
        }
        memory = PrioritizedReplayBuffer.remote(**memory_kwargs)

        rollout = {
            "states": np.zeros((1, 3, 4), dtype=np.float32),
            "next_states": np.ones((1, 3, 4), dtype=np.float32),
            "rewards": np.arange(3, dtype=np.float32).reshape(1, 3),
            "terminals": np.zeros((1, 3), dtype=np.bool_),
            "action": np.zeros((1, 3), dtype=np.int64),
            "extra_data": {
                "hidden_state": np.array([[[0, 0], [1, -1], [2, -2]]], dtype=np.float32),
            },
        }
        ray.get(memory.add.remote(rollout))

        batch = ray.get(memory.get_batch.remote(0, 1))
        self.assertEqual(batch["weights"].shape, (2,))
        self.assertNotIn("extra_data", batch)
        np.testing.assert_array_equal(
            batch["hidden_state"],
            np.stack((batch["rewards"], -batch["rewards"]), axis=-1),
        )
        learn_info = {
            "indices": batch["indices"],
            "td_errors": np.ones(2, dtype=np.float32),
        }
        ray.get(memory.on_learn.remote([learn_info]))
        self.assertTrue(ray.get(memory.ready.remote()))

        weighted_memory_kwargs = {
            "state_dim": 1,
            "state_dtype": np.float32,
            "action_dim": 2,
            "alpha": 1.0,
            "beta": 0.4,
            "beta_anneal_steps": 1,
            "batch_size": 4,
            "warmup_size": 0,
            "max_size": 4,
        }
        weighted_memory = PrioritizedReplayBuffer.remote(**weighted_memory_kwargs)
        rollout = {
            "states": np.zeros((1, 4, 1), dtype=np.float32),
            "next_states": np.ones((1, 4, 1), dtype=np.float32),
            "rewards": np.zeros((1, 4), dtype=np.float32),
            "terminals": np.zeros((1, 4), dtype=np.bool_),
            "action": np.zeros((1, 4), dtype=np.int64),
            "extra_data": {},
        }
        ray.get(weighted_memory.add.remote(rollout))

        priorities = np.asarray([1, 2, 3, 4], dtype=np.float32) + 1e-6
        learn_info = {
            "indices": np.arange(4),
            "td_errors": priorities - 1e-6,
        }
        ray.get(weighted_memory.on_learn.remote([learn_info]))
        weighted_batch = ray.get(weighted_memory.get_batch.remote(0, 1))
        probabilities = priorities / priorities.sum()
        raw_weights = (4 * probabilities[weighted_batch["indices"]]) ** -1
        max_weight = (4 * probabilities.min()) ** -1
        actual_weights = weighted_batch["weights"]
        expected_weights = raw_weights / max_weight
        np.testing.assert_allclose(actual_weights, expected_weights, rtol=1e-6)

    def test_prioritized_replay_buffer_duplicate_updates(self):
        buffer_class = PrioritizedReplayBuffer.__ray_metadata__.modified_class
        memory_kwargs = {
            "state_dim": 1,
            "state_dtype": np.float32,
            "action_dim": 2,
            "batch_size": 2,
            "max_size": 4,
        }
        memory = buffer_class(**memory_kwargs)

        learn_info = {
            "indices": np.asarray([1, 1]),
            "td_errors": np.asarray([3.0, 1.0], dtype=np.float32),
        }
        memory.on_learn([learn_info])

        self.assertAlmostEqual(memory.priority_memory[1], 3.0 + 1e-6)

    def test_tqc_quantile_huber_loss_reduction(self):
        from omnirl.algorithms.tqc import Learner as TQCLearner

        learner_class = TQCLearner.__ray_metadata__.modified_class
        learner = learner_class.__new__(learner_class)
        learner.device = torch.device("cpu")
        learner.num_quantiles = 2

        current = torch.tensor([[0.0, 2.0]])
        target = torch.tensor([[1.0, 3.0, 5.0]])
        td = target.unsqueeze(1) - current.unsqueeze(2)
        abs_td = td.abs()
        huber = torch.where(abs_td <= 1.0, 0.5 * td.pow(2), abs_td - 0.5)
        taus = torch.tensor([0.25, 0.75]).view(1, 2, 1)
        expected = (torch.abs(taus - (td < 0).float()) * huber).sum(dim=1).mean()

        actual = learner._quantile_huber_loss(current, target)

        torch.testing.assert_close(actual, expected)

    def test_rollout_buffer(self):
        memory_kwargs = {
            "state_dim": 4,
            "state_dtype": np.float32,
            "action_dim": 2,
            "unroll_length": 2,
            "max_chunks": 1,
        }
        memory = RolloutBuffer.remote(**memory_kwargs)

        rollout = {
            "version": 0,
            "states": np.zeros((1, 2, 4), dtype=np.float32),
            "next_states": np.ones((1, 2, 4), dtype=np.float32),
            "rewards": np.zeros((1, 2), dtype=np.float32),
            "terminals": np.zeros((1, 2), dtype=np.bool_),
            "dones": np.zeros((1, 2), dtype=np.bool_),
            "action": np.zeros((1, 2), dtype=np.int64),
            "extra_data": {
                "log_probs": np.zeros((1, 2), dtype=np.float32),
            },
        }
        ray.get(memory.add.remote(rollout))

        self.assertTrue(ray.get(memory.ready.remote()))
        batch = ray.get(memory.get_batch.remote(0, 1))
        self.assertEqual(batch["states"].shape, (1, 2, 4))
        self.assertEqual(batch["log_probs"].shape, (1, 2))
        self.assertNotIn("extra_data", batch)

    def test_representative_learners(self):
        from omnirl.algorithms.ddpg import Learner as DDPGLearner
        from omnirl.algorithms.dqn import Learner as DQNLearner
        from omnirl.algorithms.impala import Learner as ImpalaLearner
        from omnirl.algorithms.ppo import Learner as PPOLearner
        from omnirl.algorithms.sac import Learner as SACLearner
        from omnirl.algorithms.td3 import Learner as TD3Learner
        from omnirl.algorithms.tqc import Learner as TQCLearner

        replay_batch = {
            "states": np.random.randn(8, 4).astype(np.float32),
            "actions": np.random.randint(0, 2, size=8, dtype=np.int64),
            "rewards": np.random.randn(8).astype(np.float32),
            "next_states": np.random.randn(8, 4).astype(np.float32),
            "terminals": np.zeros(8, dtype=np.bool_),
        }
        rollout_batch = {
            "states": np.random.randn(2, 4, 4).astype(np.float32),
            "actions": np.random.randint(0, 2, size=(2, 4), dtype=np.int64),
            "rewards": np.random.randn(2, 4).astype(np.float32),
            "next_states": np.random.randn(2, 4, 4).astype(np.float32),
            "terminals": np.zeros((2, 4), dtype=np.bool_),
            "dones": np.zeros((2, 4), dtype=np.bool_),
            "log_probs": np.zeros((2, 4), dtype=np.float32),
        }
        actor_critic_network_kwargs = {
            "actor": {
                "layer_type": "MLP",
                "mlp_hidden_dims": [16, 16],
            },
            "critic": {
                "layer_type": "MLP",
                "mlp_hidden_dims": [16, 16],
            },
        }

        with tempfile.TemporaryDirectory() as ckpt_dir:
            base_learner_kwargs = {
                "state_dim": 4,
                "action_dim": 2,
                "ckpt_dir": ckpt_dir,
                "device": "cpu",
            }
            dqn_kwargs = {
                **base_learner_kwargs,
                "network_kwargs": {
                    "layer_type": "MLP",
                    "mlp_hidden_dims": [16, 16],
                },
            }
            ppo_kwargs = {
                **base_learner_kwargs,
                "k_epochs": 1,
                "num_minibatches": 2,
                "network_kwargs": actor_critic_network_kwargs,
            }
            actor_critic_kwargs = {
                **base_learner_kwargs,
                "network_kwargs": actor_critic_network_kwargs,
            }
            dqn = DQNLearner.remote(**dqn_kwargs)
            ppo = PPOLearner.remote(**ppo_kwargs)
            impala = ImpalaLearner.remote(**actor_critic_kwargs)

            continuous_batch = dict(replay_batch)
            continuous_actions = np.random.uniform(-1, 1, size=(8, 2))
            continuous_batch["actions"] = continuous_actions.astype(np.float32)
            continuous_kwargs = {
                **actor_critic_kwargs,
                "is_continuous": True,
            }
            sac = SACLearner.remote(**continuous_kwargs)
            learn_tasks = [
                dqn.learn.remote(replay_batch),
                ppo.learn.remote(rollout_batch),
                impala.learn.remote(rollout_batch),
                sac.learn.remote(continuous_batch),
            ]
            results = ray.get(learn_tasks)
            for learner in [dqn, ppo, impala, sac]:
                ray.kill(learner)

            tqc_kwargs = {
                **continuous_kwargs,
                "num_critics": 2,
                "num_quantiles": 5,
                "drop_per_net": 1,
            }
            td3 = TD3Learner.remote(**continuous_kwargs)
            tqc = TQCLearner.remote(**tqc_kwargs)
            ddpg = DDPGLearner.remote(**continuous_kwargs)
            discrete_sac = SACLearner.remote(**actor_critic_kwargs)
            learn_tasks = [
                td3.learn.remote(continuous_batch),
                tqc.learn.remote(continuous_batch),
                ddpg.learn.remote(continuous_batch),
                discrete_sac.learn.remote(replay_batch),
            ]
            results.extend(ray.get(learn_tasks))
            ray.kill(td3)
            ray.kill(tqc)
            ray.kill(ddpg)
            ray.kill(discrete_sac)

        for result in results:
            self.assertEqual(result["version"], 1)
            for value in result["log"].values():
                self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
