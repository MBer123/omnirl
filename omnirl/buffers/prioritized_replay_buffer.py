import numpy as np
import ray


@ray.remote(num_cpus=1)
class PrioritizedReplayBuffer:
    """
    Prioritized replay buffer for experience replay.

    Args:
        state_dim: Dimension of the environment state space.
        state_dtype: Data type of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        alpha: Priority exponent alpha=0 reduces to uniform replay.
        beta: Initial importance-sampling exponent.
        beta_anneal_steps: Number of learner updates used to anneal beta.
        batch_size: Mini-batch size sampled from the buffer.
        warmup_size: Minimum number of transitions required before training.
        max_size: Maximum number of replay transitions stored in the buffer.
        collection_fragment_length: Number of steps collected per environment.
    """

    def __init__(
        self,
        state_dim: int | tuple[int, int, int],
        state_dtype: np.dtype | type,
        action_dim: int,
        is_continuous: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_anneal_steps: int = 1000000,
        batch_size: int = 64,
        warmup_size: int = 0,
        max_size: int = 1000000,
        collection_fragment_length: int = 1,
    ):
        self.batch_size = batch_size
        self.warmup_size = warmup_size
        self.max_size = max_size

        self.alpha = alpha
        self.beta_start = beta
        self.beta = beta
        self.beta_anneal_steps = beta_anneal_steps
        self.max_priority = 1.0

        self.collected_size = 0
        self.available_size = 0
        self.replayed_size = 0

        state_dim = (state_dim,) if isinstance(state_dim, int) else tuple(state_dim)

        self.state_memory = np.zeros((max_size, *state_dim), dtype=state_dtype)
        if is_continuous:
            self.action_memory = np.zeros((max_size, action_dim), dtype=np.float32)
        else:
            self.action_memory = np.zeros(max_size, dtype=np.int64)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.next_state_memory = np.zeros((max_size, *state_dim), dtype=state_dtype)
        self.terminal_memory = np.zeros(max_size, dtype=np.bool_)
        self.priority_memory = np.zeros(max_size, dtype=np.float32)
        self.extra_data_memory = {}

    def add(self, rollout):
        """Store a rollout with leading [num_envs, num_steps] dimensions."""
        count = rollout["rewards"].size
        size = min(count, self.max_size)
        offset = count - size
        start = (self.collected_size + offset) % self.max_size
        end = min(start + size, self.max_size)
        first_size = end - start
        remaining_size = size - first_size

        # Initialize extra data storage.
        extra_data = rollout["extra_data"]
        if self.collected_size == 0:
            for name, values in extra_data.items():
                memory = np.empty((self.max_size, *values.shape[2:]), dtype=values.dtype)
                self.extra_data_memory[name] = memory

        state_shape = self.state_memory.shape[1:]
        action_shape = self.action_memory.shape[1:]
        states = rollout["states"].reshape(count, *state_shape)[offset:]
        actions = rollout["action"].reshape(count, *action_shape)[offset:]
        rewards = rollout["rewards"].reshape(count)[offset:]
        next_states = rollout["next_states"].reshape(count, *state_shape)[offset:]
        terminals = rollout["terminals"].reshape(count)[offset:]

        self.state_memory[start:end] = states[:first_size]
        self.action_memory[start:end] = actions[:first_size]
        self.reward_memory[start:end] = rewards[:first_size]
        self.next_state_memory[start:end] = next_states[:first_size]
        self.terminal_memory[start:end] = terminals[:first_size]
        self.priority_memory[start:end] = self.max_priority

        self.state_memory[:remaining_size] = states[first_size:]
        self.action_memory[:remaining_size] = actions[first_size:]
        self.reward_memory[:remaining_size] = rewards[first_size:]
        self.next_state_memory[:remaining_size] = next_states[first_size:]
        self.terminal_memory[:remaining_size] = terminals[first_size:]
        self.priority_memory[:remaining_size] = self.max_priority
        for name, values in extra_data.items():
            values = values.reshape(count, *values.shape[2:])[offset:]
            memory = self.extra_data_memory[name]
            memory[start:end] = values[:first_size]
            memory[:remaining_size] = values[first_size:]

        self.collected_size += count
        self.available_size = min(self.max_size, self.collected_size)

    def get_batch(self, rank, world_size):
        """Sample a rank-specific mini-batch from the replay buffer."""
        if not self.ready():
            return None

        batch_per_worker = self.batch_size // world_size
        start = rank * batch_per_worker
        end = self.batch_size if rank == world_size - 1 else start + batch_per_worker
        size = end - start

        priorities = self.priority_memory[: self.available_size]
        scaled_priorities = priorities**self.alpha
        total = scaled_priorities.sum()
        probs = scaled_priorities / total

        indices = np.random.choice(self.available_size, size, replace=True, p=probs)

        weights = (self.available_size * probs[indices]) ** (-self.beta)
        max_weight = (self.available_size * probs.min()) ** (-self.beta)
        weights /= max_weight

        extra_data = {}
        for name, memory in self.extra_data_memory.items():
            extra_data[name] = memory[indices]

        batch = {
            "states": self.state_memory[indices],
            "actions": self.action_memory[indices],
            "rewards": self.reward_memory[indices],
            "next_states": self.next_state_memory[indices],
            "terminals": self.terminal_memory[indices],
            "indices": indices,
            "weights": weights.astype(np.float32),
            **extra_data,
        }
        return batch

    def on_learn(self, train_infos):
        """Update internal state after the learner finishes one training step."""
        self.replayed_size += self.batch_size
        anneal = self.replayed_size / (self.batch_size * self.beta_anneal_steps)
        progress = min(1.0, anneal)

        self.beta = self.beta_start + progress * (1.0 - self.beta_start)

        indices = np.concatenate([info["indices"] for info in train_infos], axis=0)
        td_errors = np.concatenate([info["td_errors"] for info in train_infos], axis=0)

        indices = indices.reshape(-1)
        priorities = np.abs(td_errors).reshape(-1) + 1e-6
        unique_indices, inverse = np.unique(indices, return_inverse=True)
        unique_priorities = np.zeros(unique_indices.size, dtype=np.float32)
        np.maximum.at(unique_priorities, inverse, priorities)
        self.priority_memory[unique_indices] = unique_priorities

        batch_max_priority = unique_priorities.max()
        if batch_max_priority > self.max_priority:
            self.max_priority = batch_max_priority

    def ready(self):
        """Return True if the buffer has enough samples to start training."""
        batch_ok = self.available_size >= self.batch_size
        warmup_ok = self.collected_size >= self.warmup_size
        return batch_ok and warmup_ok

    def needs_backpressure(self):
        """Return False because new samples can overwrite old replay data."""
        return False

    def can_sample(self, version):
        """Allow collection regardless of replay occupancy or policy version."""
        return True

    def supports_prefetch(self):
        """Return True because batches can be sampled before the last update."""
        return True

    def stats(self):
        """Return statistics describing the current state of the replay buffer."""
        return {
            "available_data": self.available_size,
            "collected_data": self.collected_size,
        }
