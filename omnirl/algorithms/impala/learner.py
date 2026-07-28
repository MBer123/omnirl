import os

import ray
import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.nn.parallel import DistributedDataParallel as DDP

from omnirl.networks.actors import StochasticActorNetwork
from omnirl.networks.critics import ValueCriticNetwork
from omnirl.utils.seed import set_global_seed


@ray.remote
class Learner:
    """
    IMPALA learner agent.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        network_kwargs: Keyword arguments used to build the networks.
        ckpt_dir: Directory path to save model checkpoints.
        gamma: Discount factor.
        ent_coef: Entropy regularization coefficient used to encourage exploration.
        vf_coef: Coefficient for the critic loss in the total optimization objective.
        rho_bar: Clipping threshold for importance sampling ratios in V-trace.
        c_bar: Clipping threshold for trace coefficients in V-trace.
        actor_lr: Learning rate for the actor optimizer.
        critic_lr: Learning rate for the critic optimizer.
        max_grad_norm: Maximum gradient norm used for gradient clipping.
        device: Torch device to run the networks on.
        seed: Random seed used to ensure consistent results across runs.
        rank: Distributed learner rank.
        world_size: Total number of distributed learners.
        master_addr: Master address used to initialize distributed training.
        master_port: Master port used to initialize distributed training.
    """

    def __init__(
        self,
        state_dim: int | tuple[int, int, int],
        action_dim: int,
        is_continuous: bool = False,
        network_kwargs: dict | None = None,
        ckpt_dir: str = "checkpoints",
        gamma: float = 0.99,
        ent_coef: float = 0.02,
        vf_coef: float = 0.5,
        rho_bar: float = 1.0,
        c_bar: float = 1.0,
        actor_lr: float = 0.0003,
        critic_lr: float = 0.0003,
        max_grad_norm: float = 0.5,
        device: str | None = None,
        seed: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
    ):
        # Limit PyTorch CPU threads
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        # Initialize random seed
        if seed is not None:
            set_global_seed(seed)

        # Create checkpoint directory
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Initialize device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:0" if device == "gpu" else "cpu")

        # Initialize distributed training
        self.rank = rank
        is_distributed = world_size > 1

        if is_distributed and not torch_dist.is_initialized():
            backend = "nccl" if self.device.type == "cuda" else "gloo"
            torch_dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=self.rank,
                world_size=world_size,
            )

        # Initialize parameters
        self.gamma = gamma
        self.rho_bar = rho_bar
        self.c_bar = c_bar
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.is_continuous = is_continuous
        self.max_grad_norm = max_grad_norm

        # Initialize actor and critic networks
        self.version = 0
        network_kwargs = dict(network_kwargs or {})
        actor_kwargs = dict(network_kwargs.get("actor", network_kwargs))
        critic_kwargs = dict(network_kwargs.get("critic", network_kwargs))
        actor_layer_type = actor_kwargs.pop("layer_type")
        critic_layer_type = critic_kwargs.pop("layer_type")
        log_std_min = actor_kwargs.pop("log_std_min", -20.0)
        log_std_max = actor_kwargs.pop("log_std_max", 2.0)

        actor = StochasticActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            is_continuous=is_continuous,
            layer_type=actor_layer_type,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            network_kwargs=actor_kwargs,
        )
        critic = ValueCriticNetwork(
            state_dim=state_dim,
            layer_type=critic_layer_type,
            network_kwargs=critic_kwargs,
        )

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        self.actor.train()
        self.critic.train()

        if is_distributed:
            ddp_kwargs = {"device_ids": [0]} if self.device.type == "cuda" else {}
            self.train_actor = DDP(self.actor, **ddp_kwargs)
            self.train_critic = DDP(self.critic, **ddp_kwargs)
        else:
            self.train_actor = self.actor
            self.train_critic = self.critic

        # Initialize optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def _state_dict(self, model):
        """Return a detached CPU copy of model state_dict."""
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    def get_weights(self):
        """Export actor network weights."""
        return self._state_dict(self.actor)

    def get_version(self):
        """Export the current learner model version."""
        return self.version

    def _evaluate_action(self, states_tensor, actions_tensor):
        """Evaluate action log probabilities and entropies under the current policy."""
        if self.is_continuous:
            action_mean, action_log_std = self.train_actor(states_tensor)

            action_std = action_log_std.exp()
            dist = Normal(action_mean, action_std)

            dist_entropy = dist.entropy().sum(dim=-1)
            log_probs = dist.log_prob(actions_tensor).sum(dim=-1)
        else:
            action_logits = self.train_actor(states_tensor)

            dist = Categorical(logits=action_logits)

            dist_entropy = dist.entropy()
            log_probs = dist.log_prob(actions_tensor)

        state_values = self.train_critic(states_tensor)

        return {
            "log_probs": log_probs.view(-1),
            "dist_entropy": dist_entropy.view(-1),
            "state_values": state_values.view(-1),
        }

    def learn(self, batch):
        """Sample a batch from rollout buffer and update actor and critic networks."""
        states_tensor = torch.tensor(batch["states"], device=self.device)
        actions_tensor = torch.tensor(batch["actions"], device=self.device)
        rewards_tensor = torch.tensor(batch["rewards"], device=self.device)
        next_states_tensor = torch.tensor(batch["next_states"], device=self.device)
        terminals_tensor = torch.tensor(batch["terminals"], device=self.device)
        dones_tensor = torch.tensor(batch["dones"], device=self.device)
        log_probs_tensor = torch.tensor(batch["log_probs"], device=self.device)

        states_tensor = states_tensor.float()
        next_states_tensor = next_states_tensor.float()
        terminals_tensor = terminals_tensor.float()
        dones_tensor = dones_tensor.float()

        B, T = rewards_tensor.shape

        flat_states = states_tensor.view(B * T, *states_tensor.shape[2:])
        flat_next_states = next_states_tensor.view(B * T, *next_states_tensor.shape[2:])
        if self.is_continuous:
            flat_actions = actions_tensor.view(B * T, -1)
        else:
            flat_actions = actions_tensor.reshape(B * T)

        results = self._evaluate_action(flat_states, flat_actions)
        new_log_probs = results["log_probs"].view(B, T)
        dist_entropy = results["dist_entropy"].view(B, T)
        state_values = results["state_values"].view(B, T)

        # Compute V-trace
        with torch.no_grad():
            next_state_values = self.critic(flat_next_states).squeeze(-1)
            next_state_values = next_state_values.view(B, T)

            state_values_detached = state_values.detach()

            log_rhos = new_log_probs.detach() - log_probs_tensor
            rhos = torch.exp(log_rhos)

            clipped_rhos = torch.clamp(rhos, max=self.rho_bar)
            clipped_cs = torch.clamp(rhos, max=self.c_bar)

            next_state_values = next_state_values * (1.0 - terminals_tensor)
            td_target = rewards_tensor + self.gamma * next_state_values
            deltas = clipped_rhos * (td_target - state_values_detached)

            vs = torch.zeros_like(rewards_tensor, device=self.device)
            acc = torch.zeros(B, device=self.device)

            for t in reversed(range(T)):
                discount = self.gamma * clipped_cs[:, t] * (1.0 - dones_tensor[:, t])
                acc = deltas[:, t] + discount * acc
                vs[:, t] = state_values_detached[:, t] + acc

            next_vs = torch.zeros_like(vs, device=self.device)
            if T > 1:
                done_t = dones_tensor[:, :-1].bool()
                bootstrap_vs = next_state_values[:, :-1]
                future_vs = vs[:, 1:]
                next_vs[:, :-1] = torch.where(done_t, bootstrap_vs, future_vs)
            next_vs[:, -1] = next_state_values[:, -1]

            bootstrap_value = self.gamma * (1.0 - terminals_tensor) * next_vs
            value_residual = bootstrap_value - state_values_detached
            pg_advantages = clipped_rhos * (rewards_tensor + value_residual)

        actor_loss = -(new_log_probs * pg_advantages).mean()
        critic_loss = F.mse_loss(state_values, vs)
        entropy_loss = dist_entropy.mean()

        loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss

        value_error_var = torch.var(vs - state_values.detach())
        return_var = torch.var(vs)
        explained_var = 1 - value_error_var / (return_var + 1e-8)
        rho_mean = clipped_rhos.mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Update model version
        self.version += 1

        return {
            "log": {
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "entropy": entropy_loss.item(),
                "rho_mean": rho_mean.item(),
                "explained_var": explained_var.item(),
            },
            "memory": {"version": self.version},
            "version": self.get_version(),
            "trained_samples": int(B * T),
        }

    def save_checkpoint(self):
        """Save actor, critic and optimizer into checkpoint file."""
        if self.rank != 0:
            return

        path = os.path.join(self.ckpt_dir, "checkpoint.ckpt")

        ckpt = {
            "actor": self._state_dict(self.actor),
            "critic": self._state_dict(self.critic),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "version": self.version,
        }
        torch.save(ckpt, path)

        eval_path = os.path.join(self.ckpt_dir, f"agent_{self.version}.pth")
        torch.save(self._state_dict(self.actor), eval_path)

    def load_checkpoint(self):
        """Load actor, critic and optimizer from checkpoint file."""
        path = os.path.join(self.ckpt_dir, "checkpoint.ckpt")
        ckpt = torch.load(path, map_location=self.device)

        # Load actor and critic networks
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        # Load optimizer state
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])

        # Load model version
        self.version = ckpt["version"]

        return {"version": self.get_version(), "weights": self.get_weights()}

    def load_model(self, model_path):
        """Load actor from pre-trained model weights."""
        path = os.path.join(self.ckpt_dir, model_path)
        state_dict = torch.load(path, map_location=self.device)

        # Load actor network
        self.actor.load_state_dict(state_dict)

        # Load model version
        self.version = int(model_path.removeprefix("agent_").removesuffix(".pth"))
