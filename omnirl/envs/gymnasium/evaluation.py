import numpy as np


class MetricTracker:
    """Track and summarize Gymnasium evaluation returns."""

    default_metrics = {
        "return_mean": 0,
        "return_std": 0,
        "return_max": 0,
        "return_min": 0,
    }

    def __init__(self):
        self.episode_returns = []
        self.episode_return = 0.0

    def reset(self):
        """Clear metrics collected during the previous evaluation."""
        self.episode_returns = []
        self.episode_return = 0.0

    def step(self, reward, info):
        """Update metrics after one environment step."""
        self.episode_return += float(reward)

    def end_episode(self):
        """Store metrics for the completed episode."""
        self.episode_returns.append(self.episode_return)
        self.episode_return = 0.0

    def summary(self):
        """Return aggregated evaluation metrics."""
        returns = np.asarray(self.episode_returns, dtype=np.float32)
        return {
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "return_max": float(returns.max()),
            "return_min": float(returns.min()),
        }


class EvaluationLoop:
    """
    Run evaluation episodes using the Gymnasium interaction protocol.

    Args:
        env_builder: Callable that builds a new environment instance.
        seed: Random seed used for the initial environment reset.
    """

    def __init__(self, env_builder, seed=None):
        self.env = env_builder()
        self.env.reset(seed=seed)
        self.metric_tracker = MetricTracker()

    def run(self, policy, num_episodes):
        """Evaluate an action callback and return environment metrics."""
        self.metric_tracker.reset()

        for _ in range(num_episodes):
            observation, _ = self.env.reset()
            done = False

            while not done:
                if self.env.render_mode == "human":
                    self.env.render()

                action = policy.choose_action(observation)
                observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.metric_tracker.step(reward, info)

            self.metric_tracker.end_episode()

        return {
            "num_episodes": int(num_episodes),
            "metrics": self.metric_tracker.summary(),
        }


def aggregate_evaluations(eval_infos, version):
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
        "version": version,
        "num_episodes": num_episodes,
        "return_mean": float(return_mean),
        "return_std": float(return_std),
        "return_max": float(return_max),
        "return_min": float(return_min),
    }
