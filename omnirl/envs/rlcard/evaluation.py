from .protocol import (
    extract_observation,
    seed_and_reset,
    terminal_payoff,
    validate_action,
    validate_player,
)


class MetricTracker:
    """Track and summarize RLCard evaluation outcomes."""

    default_metrics = {
        "win_rate": 0.0,
        "draw_rate": 0.0,
        "loss_rate": 0.0,
    }

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear metrics collected during the previous evaluation."""
        self.wins = 0
        self.draws = 0
        self.losses = 0

    @property
    def num_episodes(self):
        """Return the number of completed episodes."""
        return self.wins + self.draws + self.losses

    def end_episode(self, payoff):
        """Store metrics for the completed episode."""
        if payoff > 0:
            self.wins += 1
        elif payoff < 0:
            self.losses += 1
        else:
            self.draws += 1

    def counts(self):
        """Return exact sufficient statistics for distributed aggregation."""
        return {
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
        }

    def summary(self):
        """Return aggregated evaluation metrics."""
        count = self.num_episodes
        if count == 0:
            return dict(self.default_metrics)
        return {
            "win_rate": self.wins / count,
            "draw_rate": self.draws / count,
            "loss_rate": self.losses / count,
        }


class EvaluationLoop:
    """
    Run evaluation episodes using the RLCard interaction protocol.

    Args:
        env_builder: Callable that builds a new environment instance.
        seed: Random seed used for the initial environment reset.
    """

    def __init__(self, env_builder, seed=None):
        self.env = env_builder()
        if seed is not None:
            self.env.seed(seed)
        self.metric_tracker = MetricTracker()

    def run(self, choose_action, num_episodes):
        """Evaluate an action callback and return environment metrics."""
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive.")

        self.metric_tracker.reset()

        for _ in range(num_episodes):
            state = seed_and_reset(self.env)

            while not self.env.is_over():
                observation = extract_observation(state)
                action = validate_action(state, choose_action(observation))
                state, player_id = self.env.step(action)
                validate_player(player_id)

            self.metric_tracker.end_episode(terminal_payoff(self.env))

        return {
            "num_episodes": self.metric_tracker.num_episodes,
            "outcome_counts": self.metric_tracker.counts(),
            "metrics": self.metric_tracker.summary(),
        }


def aggregate_evaluations(eval_infos, version):
    """Aggregate metrics returned by RLCard evaluation workers."""
    wins = 0
    draws = 0
    losses = 0

    for info in eval_infos:
        counts = info["outcome_counts"]
        worker_wins = int(counts["wins"])
        worker_draws = int(counts["draws"])
        worker_losses = int(counts["losses"])
        worker_episodes = worker_wins + worker_draws + worker_losses
        if worker_episodes != int(info["num_episodes"]):
            raise ValueError(
                "RLCard evaluator outcome counts do not match num_episodes."
            )

        wins += worker_wins
        draws += worker_draws
        losses += worker_losses

    num_episodes = wins + draws + losses
    if num_episodes == 0:
        metrics = dict(MetricTracker.default_metrics)
    else:
        metrics = {
            "win_rate": wins / num_episodes,
            "draw_rate": draws / num_episodes,
            "loss_rate": losses / num_episodes,
        }

    return {
        "version": version,
        "num_episodes": num_episodes,
        **metrics,
    }
