import numpy as np

from omnirl.envs.base import BaseEvaluationLoop


class EvaluationLoop(BaseEvaluationLoop):
    """
    Run evaluation episodes using the Gymnasium interaction protocol.

    Args:
        env_builder: Callable that builds a new environment instance.
        seed: Random seed used for the initial environment reset.
        num_envs: Number of environments sharing each policy inference batch.
    """

    default_metrics = {
        "return_mean": 0,
        "return_std": 0,
        "return_max": 0,
        "return_min": 0,
    }

    def __init__(self, env_builder, seed=None, num_envs=1):
        self.envs = [env_builder() for _ in range(num_envs)]
        for i, env in enumerate(self.envs):
            env.reset(seed=None if seed is None else seed + i)

    def run(self, policy, num_episodes_per_env):
        """Evaluate the requested number of episodes in each environment."""
        episode_returns = []
        num_envs = len(self.envs)
        remaining = [num_episodes_per_env] * num_envs
        active = [i for i in range(num_envs) if remaining[i]]
        observations = [None] * num_envs
        current_returns = [0.0] * num_envs
        for i in active:
            observation, _ = self.envs[i].reset()
            observations[i] = np.array(observation, copy=True)

        while active:
            batch = np.stack([observations[i] for i in active])
            actions = policy.choose_action(batch)
            for i, action in zip(active, actions):
                env = self.envs[i]
                if env.render_mode == "human":
                    env.render()

                observation, reward, terminated, truncated, _ = env.step(action)
                current_returns[i] += float(reward)
                if terminated or truncated:
                    episode_returns.append(current_returns[i])
                    current_returns[i] = 0.0
                    remaining[i] -= 1
                    if remaining[i]:
                        observation, _ = env.reset()
                observations[i] = np.array(observation, copy=True)
            active = [i for i in active if remaining[i]]

        returns = np.asarray(episode_returns, dtype=np.float32)
        return {
            "num_episodes": len(episode_returns),
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
