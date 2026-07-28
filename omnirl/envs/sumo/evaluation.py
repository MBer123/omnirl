from .protocol import episode_seed, extract_observation, extract_queue_length
from .tripinfo import TRIP_METRICS


class MetricTracker:
    """Aggregate traffic-domain metrics without reporting episode return."""

    default_metrics = {
        "delay_mean": 0.0,
        "trip_time_mean": 0.0,
        "waiting_time_mean": 0.0,
        "queue_length_mean": 0.0,
    }

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear metrics collected during the previous evaluation."""
        self.trip_metric_sums = {name: 0.0 for name in TRIP_METRICS}
        self.num_trips = 0
        self.queue_length_sum = 0.0
        self.num_queue_samples = 0
        self.num_episodes = 0

    def step(self, info, traffic_signal_ids):
        """Accumulate one per-signal queue snapshot."""
        self.queue_length_sum += extract_queue_length(
            info, traffic_signal_ids
        )
        self.num_queue_samples += 1

    def end_episode(self, trip_statistics):
        """Accumulate vehicle-level metrics from one fixed-horizon episode."""
        for name in self.trip_metric_sums:
            self.trip_metric_sums[name] += float(
                trip_statistics["trip_metric_sums"][name]
            )
        self.num_trips += int(trip_statistics["num_trips"])
        self.num_episodes += 1

    def statistics(self):
        """Return exact sufficient statistics for distributed aggregation."""
        return {
            "trip_metric_sums": dict(self.trip_metric_sums),
            "num_trips": self.num_trips,
            "queue_length_sum": self.queue_length_sum,
            "num_queue_samples": self.num_queue_samples,
        }

    def summary(self):
        """Return vehicle- and queue-sample-weighted benchmark means."""
        metrics = dict(self.default_metrics)
        if self.num_trips:
            metrics.update(
                {
                    "delay_mean": self.trip_metric_sums["delay"]
                    / self.num_trips,
                    "trip_time_mean": self.trip_metric_sums["trip_time"]
                    / self.num_trips,
                    "waiting_time_mean": self.trip_metric_sums["waiting_time"]
                    / self.num_trips,
                }
            )
        if self.num_queue_samples:
            metrics["queue_length_mean"] = (
                self.queue_length_sum / self.num_queue_samples
            )
        return metrics


class EvaluationLoop:
    """
    Evaluate a policy on single-agent SUMO traffic simulations.

    Args:
        env_builder: Callable that builds a new environment instance.
        seed: Base seed used to derive a distinct seed for every episode.
    """

    def __init__(self, env_builder, seed=None):
        self.env = env_builder()
        self.base_seed = seed
        self.metric_tracker = MetricTracker()

    def run(self, choose_action, num_episodes):
        """Evaluate a policy and return only traffic-domain metrics."""
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive.")

        self.metric_tracker.reset()
        for episode in range(num_episodes):
            seed = episode_seed(self.base_seed, episode)
            observation, _ = self.env.reset(seed=seed)
            observation = extract_observation(observation)
            done = False

            while not done:
                if self.env.render_mode == "human":
                    self.env.render()

                action = choose_action(observation)
                observation, _, terminated, truncated, info = self.env.step(action)
                observation = extract_observation(observation)
                done = bool(terminated) or bool(truncated)
                self.metric_tracker.step(info, self.env.traffic_signal_ids)

            trip_statistics = self.env.finish_episode_metrics()
            self.metric_tracker.end_episode(trip_statistics)

        return {
            "num_episodes": self.metric_tracker.num_episodes,
            **self.metric_tracker.statistics(),
            "metrics": self.metric_tracker.summary(),
        }


def aggregate_evaluations(eval_infos, version):
    """Aggregate exact traffic metric sums returned by evaluation workers."""
    num_episodes = 0
    num_trips = 0
    num_queue_samples = 0
    queue_length_sum = 0.0
    trip_metric_sums = {name: 0.0 for name in TRIP_METRICS}

    for info in eval_infos:
        num_episodes += int(info["num_episodes"])
        num_trips += int(info["num_trips"])
        num_queue_samples += int(info["num_queue_samples"])
        queue_length_sum += float(info["queue_length_sum"])
        for name in trip_metric_sums:
            trip_metric_sums[name] += float(info["trip_metric_sums"][name])

    tracker = MetricTracker()
    tracker.trip_metric_sums = trip_metric_sums
    tracker.num_trips = num_trips
    tracker.queue_length_sum = queue_length_sum
    tracker.num_queue_samples = num_queue_samples
    metrics = tracker.summary()

    return {
        "version": version,
        "num_episodes": num_episodes,
        **metrics,
    }
