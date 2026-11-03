import numpy as np
import ray


@ray.remote(num_cpus=1)
class RolloutBuffer:
    """
    Rollout buffer for experience store.

    Args:
        state_dim: Dimension of the environment state space.
        state_dtype: Data type of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        unroll_length: Number of transitions contained in each rollout segment.
        max_chunks: Maximum number of rollout chunks stored in the buffer.
        track_version: Whether to reject stale rollouts and track learner versions.
    """

    def __init__(
        self,
        state_dim: int | tuple[int, int, int],
        state_dtype: np.dtype | type,
        action_dim: int,
        is_continuous: bool = False,
        unroll_length: int = 32,
        max_chunks: int = 64,
        track_version: bool = True,
    ):
        self.unroll_length = unroll_length
        self.max_chunks = max_chunks
        self.track_version = track_version

        self.collected_chunks = 0
        self.available_chunks = 0

        self.is_ready = False
        self.version = -1

        state_dim = (state_dim,) if isinstance(state_dim, int) else tuple(state_dim)
        chunk_shape = (max_chunks, unroll_length)

        self.state_memory = np.zeros((*chunk_shape, *state_dim), dtype=state_dtype)
        if is_continuous:
            self.action_memory = np.zeros((*chunk_shape, action_dim), dtype=np.float32)
        else:
            self.action_memory = np.zeros(chunk_shape, dtype=np.int64)
        self.reward_memory = np.zeros(chunk_shape, dtype=np.float32)
        self.next_state_memory = np.zeros((*chunk_shape, *state_dim), dtype=state_dtype)
        self.terminal_memory = np.zeros(chunk_shape, dtype=np.bool_)
        self.done_memory = np.zeros(chunk_shape, dtype=np.bool_)
        self.extra_data_memory = {}

    def add(self, rollout):
        """Store a rollout with leading [num_envs, unroll_length] dimensions."""
        if self.track_version and rollout["version"] < self.version:
            return

        if self.is_ready:
            return

        size = min(rollout["rewards"].shape[0], self.max_chunks - self.available_chunks)
        start = self.available_chunks
        end = start + size

        # Initialize extra data storage.
        extra_data = rollout["extra_data"]
        if self.collected_chunks == 0:
            chunk_shape = (self.max_chunks, self.unroll_length)
            for name, values in extra_data.items():
                memory = np.empty((*chunk_shape, *values.shape[2:]), dtype=values.dtype)
                self.extra_data_memory[name] = memory

        self.state_memory[start:end] = rollout["states"][:size]
        self.action_memory[start:end] = rollout["action"][:size]
        self.reward_memory[start:end] = rollout["rewards"][:size]
        self.next_state_memory[start:end] = rollout["next_states"][:size]
        self.terminal_memory[start:end] = rollout["terminals"][:size]
        self.done_memory[start:end] = rollout["dones"][:size]
        for name, values in extra_data.items():
            self.extra_data_memory[name][start:end] = values[:size]

        self.collected_chunks += size
        self.available_chunks = end

        if self.available_chunks == self.max_chunks:
            self.is_ready = True

    def get_batch(self, rank, world_size):
        """Return collected transitions from the rollout buffer."""
        if not self.ready():
            return None

        chunk_per_worker = self.max_chunks // world_size
        start = rank * chunk_per_worker
        end = self.max_chunks if rank == world_size - 1 else start + chunk_per_worker

        extra_data = {}
        for name, memory in self.extra_data_memory.items():
            extra_data[name] = memory[start:end]

        batch = {
            "states": self.state_memory[start:end],
            "actions": self.action_memory[start:end],
            "rewards": self.reward_memory[start:end],
            "next_states": self.next_state_memory[start:end],
            "terminals": self.terminal_memory[start:end],
            "dones": self.done_memory[start:end],
            **extra_data,
        }
        return batch

    def on_learn(self, train_infos):
        """Update internal state after the learner finishes one training step."""
        self._clear()
        if self.track_version:
            self.version = train_infos[0]["version"]

    def ready(self):
        """Return True if the buffer has enough samples to start training."""
        return self.is_ready

    def supports_prefetch(self):
        """Return False because batches must be collected after the last update."""
        return False

    def _clear(self):
        """Clear all stored transitions in the rollout buffer."""
        self.available_chunks = 0
        self.is_ready = False

    def stats(self):
        """Return statistics describing the current state of the rollout buffer."""
        return {
            "available_data": self.available_chunks * self.unroll_length,
            "collected_data": self.collected_chunks * self.unroll_length,
        }
