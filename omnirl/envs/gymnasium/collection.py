import numpy as np

from omnirl.envs.base import BaseCollectionLoop


class CollectionLoop(BaseCollectionLoop):
    """
    Collect transitions using the Gymnasium interaction protocol.

    Args:
        env_builder: Callable that builds a new environment instance.
        seed: Random seed used for the initial environment reset.
        num_envs: Number of environments sharing each policy inference batch.
    """

    def __init__(self, env_builder, seed=None, num_envs=1):
        self.envs = [env_builder() for _ in range(num_envs)]
        self.observations = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=None if seed is None else seed + i)
            self.observations.append(np.array(obs, copy=True))

    def run(self, policy, num_steps):
        """Return a rollout with leading [num_envs, num_steps] dimensions."""
        fields = ("states", "action", "rewards", "next_states", "terminals", "dones")
        rollouts = [{name: [] for name in fields} for _ in self.envs]
        rollouts = [{**rollout, "extra_data": {}} for rollout in rollouts]

        for _ in range(num_steps):
            observations = np.stack(self.observations)
            decision = policy.sample_action(observations)

            for i, env in enumerate(self.envs):
                action = decision["env_action"][i]
                next_observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                rollout = rollouts[i]
                rollout["states"].append(observations[i])
                rollout["action"].append(np.array(decision["action"][i], copy=True))
                rollout["rewards"].append(reward)
                rollout["next_states"].append(np.array(next_observation, copy=True))
                rollout["terminals"].append(terminated)
                rollout["dones"].append(done)
                for name, values in decision["extra_data"].items():
                    value = np.array(values[i], copy=True)
                    rollout["extra_data"].setdefault(name, []).append(value)

                if done:
                    next_observation, _ = env.reset()
                self.observations[i] = np.array(next_observation, copy=True)

        rollout = {}
        for name in fields:
            values = [item[name] for item in rollouts]
            rollout[name] = np.asarray(values)

        extra_data = {}
        for name in rollouts[0]["extra_data"]:
            values = [item["extra_data"][name] for item in rollouts]
            extra_data[name] = np.asarray(values)
        rollout["extra_data"] = extra_data
        return rollout
