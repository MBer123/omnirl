from .protocol import episode_seed


class MetricTracker:
    """Track kills, survival, and ammunition efficiency."""

    default_metrics = {
        "kills_mean": 0.0,
        "survival_time_mean": 0.0,
        "shots_mean": 0.0,
        "shooting_precision": 0.0,
    }

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear metrics collected during the previous evaluation."""
        self.num_episodes = 0
        self.num_kills = 0
        self.num_shots = 0
        self.survival_time_sum = 0

    def end_episode(self, episode_metrics):
        """Accumulate task metrics for one completed shooting episode."""
        kills = int(episode_metrics["kills"])
        shots = int(episode_metrics["shots"])
        episode_tics = int(episode_metrics["episode_tics"])
        if kills < 0 or shots < 0 or episode_tics < 0:
            raise ValueError("ViZDoom episode metrics must be non-negative.")

        self.num_episodes += 1
        self.num_kills += kills
        self.num_shots += shots
        self.survival_time_sum += episode_tics

    def statistics(self):
        """Return sufficient statistics for exact distributed aggregation."""
        return {
            "num_kills": self.num_kills,
            "num_shots": self.num_shots,
            "survival_time_sum": self.survival_time_sum,
        }

    def summary(self):
        """Return task-specific evaluation metrics."""
        if self.num_episodes == 0:
            return dict(self.default_metrics)

        return {
            "kills_mean": float(self.num_kills / self.num_episodes),
            "survival_time_mean": float(
                self.survival_time_sum / self.num_episodes
            ),
            "shots_mean": float(self.num_shots / self.num_episodes),
            "shooting_precision": float(
                self.num_kills / self.num_shots if self.num_shots else 0.0
            ),
        }


class EvaluationLoop:
    """
    Evaluate a policy on native single-player ViZDoom shooting episodes.

    Args:
        env_builder: Callable that builds a new environment instance.
        seed: Base seed used to derive a distinct seed for every episode.
    """

    def __init__(self, env_builder, seed=None):
        self.env = env_builder()
        self.base_seed = seed
        self.metric_tracker = MetricTracker()

    def run(self, choose_action, num_episodes):
        """Evaluate a policy using shooting-domain metrics only."""
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive.")

        self.metric_tracker.reset()
        for episode in range(num_episodes):
            seed = episode_seed(self.base_seed, episode)
            observation, _ = self.env.reset(seed=seed)
            done = False

            while not done:
                if self.env.render_mode == "human":
                    self.env.render()

                action = choose_action(observation)
                observation, _, terminated, truncated, _ = self.env.step(action)
                done = bool(terminated) or bool(truncated)

            self.metric_tracker.end_episode(self.env.episode_metrics())

        return {
            "num_episodes": self.metric_tracker.num_episodes,
            **self.metric_tracker.statistics(),
            "metrics": self.metric_tracker.summary(),
        }


def aggregate_evaluations(eval_infos, version):
    """Aggregate exact shooting statistics across evaluator workers."""
    num_episodes = sum(int(info["num_episodes"]) for info in eval_infos)
    if num_episodes <= 0:
        raise ValueError("ViZDoom evaluation requires at least one episode.")

    num_kills = sum(int(info["num_kills"]) for info in eval_infos)
    num_shots = sum(int(info["num_shots"]) for info in eval_infos)
    survival_time_sum = sum(
        int(info["survival_time_sum"]) for info in eval_infos
    )

    return {
        "version": version,
        "num_episodes": num_episodes,
        "kills_mean": float(num_kills / num_episodes),
        "survival_time_mean": float(survival_time_sum / num_episodes),
        "shots_mean": float(num_shots / num_episodes),
        "shooting_precision": float(
            num_kills / num_shots if num_shots else 0.0
        ),
    }
