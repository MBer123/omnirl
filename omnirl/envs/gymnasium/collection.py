import numpy as np


class CollectionLoop:
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
            obs = env.reset(seed=None if seed is None else seed + i)[0]
            self.observations.append(np.array(obs, copy=True))

    def collect(self, policy, num_steps):
        """Return a rollout with leading [num_envs, num_steps] dimensions."""
        fields = ("states", "rewards", "next_states", "terminals", "dones")
        rollouts = [{name: [] for name in fields} for _ in self.envs]
        rollouts = [{**rollout, "policy_data": {}} for rollout in rollouts]

        for _ in range(num_steps):
            observations = np.stack(self.observations)
            decision = policy.sample_action(observations)

            for i, env in enumerate(self.envs):
                action = decision["env_action"][i]
                next_observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                rollout = rollouts[i]
                rollout["states"].append(observations[i])
                rollout["rewards"].append(reward)
                rollout["next_states"].append(np.array(next_observation, copy=True))
                rollout["terminals"].append(terminated)
                rollout["dones"].append(done)
                for name, values in decision.get("data", {}).items():
                    value = np.array(values[i], copy=True)
                    rollout["policy_data"].setdefault(name, []).append(value)

                if done:
                    next_observation, _ = env.reset()
                self.observations[i] = np.array(next_observation, copy=True)

        rollout = {}
        for name in fields:
            values = [item[name] for item in rollouts]
            rollout[name] = np.asarray(values)

        policy_data = {}
        for name in rollouts[0]["policy_data"]:
            values = [item["policy_data"][name] for item in rollouts]
            policy_data[name] = np.asarray(values)
        rollout["policy_data"] = policy_data
        return rollout
