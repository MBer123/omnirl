import copy
import os

import numpy as np
import ray
import torch
import torch.distributed as torch_dist
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.parallel import DistributedDataParallel as DDP

from omnirl.networks.actors import StochasticActorNetwork
from omnirl.networks.critics import DeepQNetwork, QCriticNetwork
from omnirl.utils.seed import set_global_seed


@ray.remote
class Learner:
    """
    SAC learner agent.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        network_kwargs: Keyword arguments used to build the networks.
        ckpt_dir: Directory path to save model checkpoints.
        gamma: Discount factor.
        tau: Soft update coefficient for target networks.
        alpha: Initial temperature coefficient.
        target_entropy: Target entropy used for automatic temperature tuning.
        actor_lr: Learning rate for the actor optimizer.
        critic_lr: Learning rate for both critic optimizers.
        alpha_lr: Learning rate for the temperature optimizer.
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
        tau: float = 0.005,
        alpha: float = 0.1,
        target_entropy: float | None = None,
        actor_lr: float = 0.0003,
        critic_lr: float = 0.0003,
        alpha_lr: float = 0.0003,
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
        self.is_distributed = world_size > 1

        if self.is_distributed and not torch_dist.is_initialized():
            backend = "nccl" if self.device.type == "cuda" else "gloo"
            torch_dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=self.rank,
                world_size=world_size,
            )

        # Initialize parameters
        self.gamma = gamma
        self.tau = tau
        self.is_continuous = is_continuous

        if target_entropy is None:
            entropy = -float(action_dim) if is_continuous else 0.6 * np.log(action_dim)
            self.target_entropy = entropy
        else:
            self.target_entropy = target_entropy

        # Initialize online network and target network
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

        critic_cls = QCriticNetwork if is_continuous else DeepQNetwork
        critic_1 = critic_cls(
            state_dim=state_dim,
            action_dim=action_dim,
            layer_type=critic_layer_type,
            network_kwargs=critic_kwargs,
        )
        critic_2 = critic_cls(
            state_dim=state_dim,
            action_dim=action_dim,
            layer_type=critic_layer_type,
            network_kwargs=critic_kwargs,
        )

        self.actor = actor.to(self.device)

        self.critic_1 = critic_1.to(self.device)
        self.critic_2 = critic_2.to(self.device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)

        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()

        self.target_critic_1.eval()
        self.target_critic_2.eval()

        if self.is_distributed:
            ddp_kwargs = {"device_ids": [0]} if self.device.type == "cuda" else {}
            self.train_actor = DDP(self.actor, **ddp_kwargs)
            self.train_critic_1 = DDP(self.critic_1, **ddp_kwargs)
            self.train_critic_2 = DDP(self.critic_2, **ddp_kwargs)
            self.target_critic_1.load_state_dict(self.critic_1.state_dict())
            self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        else:
            self.train_actor = self.actor
            self.train_critic_1 = self.critic_1
            self.train_critic_2 = self.critic_2

        # Initialize temperature
        log_alpha = np.log(alpha)
        self.log_alpha = torch.tensor(log_alpha, device=self.device, requires_grad=True)

        # Initialize optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def _update_network_parameters(self, source_net, target_net):
        """Soft-update target network parameters from online network parameters."""
        with torch.no_grad():
            zip_params = zip(source_net.parameters(), target_net.parameters())
            for source_params, target_params in zip_params:
                params = self.tau * source_params + (1 - self.tau) * target_params
                target_params.data.copy_(params)

    def _state_dict(self, model):
        """Return a detached CPU copy of model state_dict."""
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    def get_weights(self):
        """Export actor network weights."""
        return self._state_dict(self.actor)

    def get_version(self):
        """Export the current learner model version."""
        return self.version

    def learn(self, batch):
        """Sample a batch from replay buffer and update the online network."""
        states_tensor = torch.tensor(batch["states"], device=self.device)
        actions_tensor = torch.tensor(batch["actions"], device=self.device)
        rewards_tensor = torch.tensor(batch["rewards"], device=self.device)
        next_states_tensor = torch.tensor(batch["next_states"], device=self.device)
        terminals_tensor = torch.tensor(batch["terminals"], device=self.device)

        states_tensor = states_tensor.float()
        next_states_tensor = next_states_tensor.float()

        batch_size = rewards_tensor.shape[0]

        with torch.no_grad():
            # Compute next log probabilities
            if self.is_continuous:
                action_mean, action_log_std = self.actor(next_states_tensor)
                action_std = action_log_std.exp()
                dist = Normal(action_mean, action_std)

                sampled_next_actions = dist.sample()
                next_actions = torch.tanh(sampled_next_actions)

                next_log_probs = dist.log_prob(sampled_next_actions)
                squash_correction = torch.log(1 - next_actions.pow(2) + 1e-6)
                next_log_probs = next_log_probs - squash_correction
                next_log_probs = next_log_probs.sum(dim=-1)
                next_q1 = self.target_critic_1(next_states_tensor, next_actions)
                next_q2 = self.target_critic_2(next_states_tensor, next_actions)
                next_q1 = next_q1.squeeze(-1)
                next_q2 = next_q2.squeeze(-1)
            else:
                action_logits = self.actor(next_states_tensor)
                next_log_probs = F.log_softmax(action_logits, dim=-1)
                next_q1 = self.target_critic_1(next_states_tensor)
                next_q2 = self.target_critic_2(next_states_tensor)

            # Compute soft state value
            next_q = torch.min(next_q1, next_q2)
            soft_next_v = next_q - torch.exp(self.log_alpha) * next_log_probs

            # Compute expectation for discrete action space
            if not self.is_continuous:
                next_probs = torch.exp(next_log_probs)
                soft_next_v = torch.sum(next_probs * soft_next_v, dim=-1)

            # Compute target values
            soft_next_v[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * soft_next_v

        # Q(s, a) predicted by online network
        if self.is_continuous:
            q1 = self.train_critic_1(states_tensor, actions_tensor)
            q2 = self.train_critic_2(states_tensor, actions_tensor)
        else:
            action_indices = actions_tensor.long().unsqueeze(-1)
            q1 = self.train_critic_1(states_tensor).gather(1, action_indices)
            q2 = self.train_critic_2(states_tensor).gather(1, action_indices)
        q1 = q1.squeeze(-1)
        q2 = q2.squeeze(-1)

        # Update critic network
        critic_1_loss = F.mse_loss(q1, target)
        critic_2_loss = F.mse_loss(q2, target)
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Compute log probabilities using re-sampling techology
        if self.is_continuous:
            action_mean, action_log_std = self.train_actor(states_tensor)
            action_std = torch.exp(action_log_std)
            dist = Normal(action_mean, action_std)

            sampled_actions = dist.rsample()
            new_actions = torch.tanh(sampled_actions)

            log_probs = dist.log_prob(sampled_actions)
            squash_correction = torch.log(1 - new_actions.pow(2) + 1e-6)
            log_probs = log_probs - squash_correction
            log_probs = log_probs.sum(dim=-1)
        else:
            action_logits = self.train_actor(states_tensor)
            log_probs = F.log_softmax(action_logits, dim=-1)

        # Freeze critic during actor update
        for p in self.critic_1.parameters():
            p.requires_grad_(False)
        for p in self.critic_2.parameters():
            p.requires_grad_(False)

        # Compute action value
        if self.is_continuous:
            q1 = self.critic_1(states_tensor, new_actions).squeeze(-1)
            q2 = self.critic_2(states_tensor, new_actions).squeeze(-1)
        else:
            q1 = self.critic_1(states_tensor)
            q2 = self.critic_2(states_tensor)
        q = torch.min(q1, q2)

        # Update actor network
        actor_loss = torch.exp(self.log_alpha).detach() * log_probs - q
        if not self.is_continuous:
            actor_loss = torch.sum(torch.exp(log_probs) * actor_loss, dim=-1)
        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic after actor update
        for p in self.critic_1.parameters():
            p.requires_grad_(True)
        for p in self.critic_2.parameters():
            p.requires_grad_(True)

        # Soft update target network
        self._update_network_parameters(self.critic_1, self.target_critic_1)
        self._update_network_parameters(self.critic_2, self.target_critic_2)

        # Update temperature
        temp_loss = -self.log_alpha * (log_probs.detach() + self.target_entropy)
        if not self.is_continuous:
            probs = torch.exp(log_probs).detach()
            temp_loss = torch.sum(probs * temp_loss, dim=-1)
        temp_loss = temp_loss.mean()

        self.alpha_optimizer.zero_grad()
        temp_loss.backward()
        if self.is_distributed:
            torch_dist.all_reduce(self.log_alpha.grad)
            self.log_alpha.grad.div_(torch_dist.get_world_size())
        self.alpha_optimizer.step()

        # Update model version
        self.version += 1

        return {
            "log": {
                "actor_loss": actor_loss.item(),
                "critic_loss": ((critic_1_loss + critic_2_loss) / 2).item(),
                "temp_loss": temp_loss.item(),
            },
            "memory": {"version": self.version},
            "version": self.get_version(),
            "trained_samples": int(batch_size),
        }

    def save_checkpoint(self):
        """Save actor, critic and optimizer into checkpoint file."""
        if self.rank != 0:
            return

        path = os.path.join(self.ckpt_dir, "checkpoint.ckpt")

        ckpt = {
            "actor": self._state_dict(self.actor),
            "critic_1": self._state_dict(self.critic_1),
            "critic_2": self._state_dict(self.critic_2),
            "target_critic_1": self._state_dict(self.target_critic_1),
            "target_critic_2": self._state_dict(self.target_critic_2),
            "log_alpha": self.log_alpha,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "version": self.version,
        }
        torch.save(ckpt, path)

        eval_path = os.path.join(self.ckpt_dir, f"agent_{self.version}.pth")
        torch.save(self._state_dict(self.actor), eval_path)

    def load_checkpoint(self):
        """Load actor, critic and optimizer from checkpoint file."""
        path = os.path.join(self.ckpt_dir, "checkpoint.ckpt")
        ckpt = torch.load(path, map_location=self.device)

        # Load both online and target network
        self.actor.load_state_dict(ckpt["actor"])

        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.target_critic_1.load_state_dict(ckpt["target_critic_1"])
        self.target_critic_2.load_state_dict(ckpt["target_critic_2"])

        # Load alpha parameter
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))

        # Load actor, critic and alpha optimizer state
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(ckpt["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(ckpt["critic_2_optimizer"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])

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
