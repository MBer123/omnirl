import copy
import os

import numpy as np
import ray
import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.parallel import DistributedDataParallel as DDP

from omnirl.algorithms.base import BaseLearner
from omnirl.networks.actors import StochasticActorNetwork
from omnirl.networks.critics import QuantileCriticNetwork
from omnirl.utils.seed import set_global_seed


@ray.remote
class Learner(BaseLearner):
    """
    TQC learner agent.

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
        num_critics: Number of distributional critic networks.
        num_quantiles: Number of quantiles.
        drop_per_net: Number of top quantiles dropped from each critic.
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
        num_critics: int = 5,
        num_quantiles: int = 25,
        drop_per_net: int = 2,
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
        self.num_critics = num_critics
        self.num_quantiles = num_quantiles
        self.drop_per_net = drop_per_net

        if target_entropy is None:
            entropy = -float(action_dim)
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
            is_continuous=True,
            layer_type=actor_layer_type,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            network_kwargs=actor_kwargs,
        )
        critics = [
            QuantileCriticNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                layer_type=critic_layer_type,
                num_quantiles=num_quantiles,
                network_kwargs=critic_kwargs,
            )
            for _ in range(num_critics)
        ]
        online_list = [critic.to(self.device) for critic in critics]
        target_list = [copy.deepcopy(critic).to(self.device) for critic in critics]

        self.actor = actor.to(self.device)
        self.critics = nn.ModuleList(online_list)
        self.target_critics = nn.ModuleList(target_list)

        self.actor.train()
        self.critics.train()
        self.target_critics.eval()

        if self.is_distributed:
            ddp_kwargs = {"device_ids": [0]} if self.device.type == "cuda" else {}
            self.train_actor = DDP(self.actor, **ddp_kwargs)
            self.train_critics = [DDP(critic, **ddp_kwargs) for critic in self.critics]
            for critic, target_critic in zip(self.critics, self.target_critics):
                target_critic.load_state_dict(critic.state_dict())
        else:
            self.train_actor = self.actor
            self.train_critics = self.critics

        # Initialize temperature
        log_alpha = np.log(alpha)
        self.log_alpha = torch.tensor(log_alpha, device=self.device, requires_grad=True)

        # Initialize optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizers = []
        for critic in self.critics:
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=critic_lr))
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
        """Export online network weights."""
        return self._state_dict(self.actor)

    def get_version(self):
        """Export the current learner model version."""
        return self.version

    def _quantile_huber_loss(self, current_quantiles, target_quantiles):
        """Huber loss for two quantiles."""
        td = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        abs_td = td.abs()

        huber = torch.where(abs_td <= 1.0, 0.5 * td.pow(2), abs_td - 0.5)

        arange = torch.arange(self.num_quantiles, device=self.device)
        taus = ((arange + 0.5) / self.num_quantiles).view(1, self.num_quantiles, 1)

        loss = torch.abs(taus - (td.detach() < 0).float()) * huber
        return loss.sum(dim=1).mean()

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
            action_mean, action_log_std = self.actor(next_states_tensor)
            action_std = action_log_std.exp()
            dist = Normal(action_mean, action_std)

            sampled_next_actions = dist.sample()
            next_actions = torch.tanh(sampled_next_actions)

            next_log_probs = dist.log_prob(sampled_next_actions)
            squash_correction = torch.log(1 - next_actions.pow(2) + 1e-6)
            next_log_probs = next_log_probs - squash_correction
            next_log_probs = next_log_probs.sum(dim=-1)

            # Next quantiles
            next_quantiles = []
            for target_critic in self.target_critics:
                next_quantiles.append(target_critic(next_states_tensor, next_actions))
            next_quantiles = torch.cat(next_quantiles, dim=-1)
            next_quantiles, _ = torch.sort(next_quantiles, dim=-1)

            # Drop top k quantiles to avoid over-estimating
            num_drop = self.drop_per_net * self.num_critics
            if num_drop > 0:
                next_quantiles = next_quantiles[..., :-num_drop]

            alpha = torch.exp(self.log_alpha)
            soft_next_quantiles = next_quantiles - alpha * next_log_probs.unsqueeze(-1)

            # Compute target values
            soft_next_quantiles[terminals_tensor] = 0.0
            rewards = rewards_tensor.unsqueeze(-1)
            target_quantiles = rewards + self.gamma * soft_next_quantiles

        # Quantiles predicted by online network
        quantiles_list = []
        for critic in self.train_critics:
            quantiles_list.append(critic(states_tensor, actions_tensor))

        # Update critic network
        critic_loss = 0.0
        for quantiles in quantiles_list:
            huber_loss = self._quantile_huber_loss(quantiles, target_quantiles)
            critic_loss = critic_loss + huber_loss
        critic_loss = critic_loss / self.num_critics

        for optimizer in self.critic_optimizers:
            optimizer.zero_grad()
        critic_loss.backward()
        for optimizer in self.critic_optimizers:
            optimizer.step()

        # Compute log probabilities using re-sampling techology
        action_mean, action_log_std = self.train_actor(states_tensor)
        action_std = torch.exp(action_log_std)
        dist = Normal(action_mean, action_std)

        sampled_actions = dist.rsample()
        new_actions = torch.tanh(sampled_actions)

        log_probs = dist.log_prob(sampled_actions)
        squash_correction = torch.log(1 - new_actions.pow(2) + 1e-6)
        log_probs = log_probs - squash_correction
        log_probs = log_probs.sum(dim=-1)

        # Freeze critics during actor update
        for critic in self.critics:
            for p in critic.parameters():
                p.requires_grad_(False)

        # Compute action value
        q = []
        for critic in self.critics:
            q.append(critic(states_tensor, new_actions))
        q = torch.cat(q, dim=-1).mean(dim=-1)

        # Update actor network
        actor_loss = torch.exp(self.log_alpha).detach() * log_probs - q
        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critics after actor update
        for critic in self.critics:
            for p in critic.parameters():
                p.requires_grad_(True)

        # Soft update target network
        for critic, target_critic in zip(self.critics, self.target_critics):
            self._update_network_parameters(critic, target_critic)

        # Update temperature
        temp_loss = -self.log_alpha * (log_probs.detach() + self.target_entropy)
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
                "critic_loss": critic_loss.item(),
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
            "critics": [self._state_dict(c) for c in self.critics],
            "target_critics": [self._state_dict(c) for c in self.target_critics],
            "log_alpha": self.log_alpha,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizers": [op.state_dict() for op in self.critic_optimizers],
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

        critic_zip = zip(self.critics, ckpt["critics"])
        for critic, state_dict in critic_zip:
            critic.load_state_dict(state_dict)

        target_critic_zip = zip(self.target_critics, ckpt["target_critics"])
        for target_critic, state_dict in target_critic_zip:
            target_critic.load_state_dict(state_dict)

        # Load alpha parameter
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))

        # Load actor, critic and alpha optimizer state
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])

        optimizer_zip = zip(self.critic_optimizers, ckpt["critic_optimizers"])
        for optimizer, state_dict in optimizer_zip:
            optimizer.load_state_dict(state_dict)
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
