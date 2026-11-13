import numpy as np

from omnirl.envs.base import BaseEvaluationLoop


class EvaluationLoop(BaseEvaluationLoop):
    """
    Run evaluation episodes using the Gymnasium interaction protocol.

    Args:
        env_builder: Callable that builds a new environment instance.
        seed: Random seed used for the initial environment reset.
    """

    default_metrics = {
        "return_mean": 0,
        "return_std": 0,
        "return_max": 0,
        "return_min": 0,
    }

    def __init__(self, env_builder, seed=None):
        self.env = env_builder()
        self.env.reset(seed=seed)

    def run(self, policy, num_episodes):
        """Evaluate a policy and return environment metrics."""
        episode_returns = []

        for _ in range(num_episodes):
            observation, _ = self.env.reset()
            episode_return = 0.0
            done = False

            while not done:
                if self.env.render_mode == "human":
                    self.env.render()

                action = policy.choose_action(observation)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_return += float(reward)

            episode_returns.append(episode_return)

        returns = np.asarray(episode_returns, dtype=np.float32)
        return {
            "num_episodes": int(num_episodes),
            "metrics": {
                "return_mean": float(returns.mean()),
                "return_std": float(returns.std()),
                "return_max": float(returns.max()),
                "return_min": float(returns.min()),
            },
        }

    @staticmethod
    def aggregate(eval_infos):
        """Aggregate metrics returned by Gymnasium evaluation workers."""
        num_episodes = 0
        return_sum = 0.0
        return_square_sum = 0.0
        return_max = float("-inf")
        return_min = float("inf")

        for info in eval_infos:
            metrics = info["metrics"]
            count = info["num_episodes"]
            mean = metrics["return_mean"]
            std = metrics["return_std"]

            num_episodes += count
            return_sum += mean * count
            return_square_sum += (std**2 + mean**2) * count
            return_max = max(return_max, metrics["return_max"])
            return_min = min(return_min, metrics["return_min"])

        return_mean = return_sum / num_episodes
        return_square_mean = return_square_sum / num_episodes
        return_variance = max(0.0, return_square_mean - return_mean**2)
        return_std = return_variance**0.5

        return {
            "num_episodes": num_episodes,
            "return_mean": float(return_mean),
            "return_std": float(return_std),
            "return_max": float(return_max),
            "return_min": float(return_min),
        }
