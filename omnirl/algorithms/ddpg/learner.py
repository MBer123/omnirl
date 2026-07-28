import copy
import os

import ray
import torch
import torch.distributed as torch_dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from omnirl.networks.actors import DeterministicActorNetwork
from omnirl.networks.critics import QCriticNetwork
from omnirl.utils.seed import set_global_seed


@ray.remote
class Learner:
    """
    DDPG learner agent.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        network_kwargs: Keyword arguments used to build the networks.
        ckpt_dir: Directory path to save model checkpoints.
        gamma: Discount factor.
        tau: Soft update coefficient for target networks.
        actor_lr: Learning rate for the actor optimizer.
        critic_lr: Learning rate for the critic optimizer.
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
        actor_lr: float = 0.0003,
        critic_lr: float = 0.0003,
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
        self.tau = tau

        # Initialize online network and target network
        self.version = 0
        network_kwargs = dict(network_kwargs or {})
        actor_kwargs = dict(network_kwargs.get("actor", network_kwargs))
        critic_kwargs = dict(network_kwargs.get("critic", network_kwargs))
        actor_layer_type = actor_kwargs.pop("layer_type")
        critic_layer_type = critic_kwargs.pop("layer_type")

        actor = DeterministicActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            layer_type=actor_layer_type,
            network_kwargs=actor_kwargs,
        )
        critic = QCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            layer_type=critic_layer_type,
            is_continuous=True,
            network_kwargs=critic_kwargs,
        )

        self.actor = actor.to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)

        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)

        self.actor.train()
        self.critic.train()

        self.target_actor.eval()
        self.target_critic.eval()

        if is_distributed:
            ddp_kwargs = {"device_ids": [0]} if self.device.type == "cuda" else {}
            self.train_actor = DDP(self.actor, **ddp_kwargs)
            self.train_critic = DDP(self.critic, **ddp_kwargs)
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
        else:
            self.train_actor = self.actor
            self.train_critic = self.critic

        # Initialize optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

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
            # Compute target values using the target network
            next_actions = self.target_actor(next_states_tensor)
            q_next = self.target_critic(next_states_tensor, next_actions).view(-1)
            q_next[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_next

        # Q(s, a) predicted by online network
        q = self.train_critic(states_tensor, actions_tensor).view(-1)

        # Update critic network
        critic_loss = F.mse_loss(q, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic during actor update
        for p in self.critic.parameters():
            p.requires_grad_(False)

        # Update actor network
        new_actions = self.train_actor(states_tensor)
        actor_loss = -torch.mean(self.critic(states_tensor, new_actions))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic after actor update
        for p in self.critic.parameters():
            p.requires_grad_(True)

        # Soft update target network
        self._update_network_parameters(self.actor, self.target_actor)
        self._update_network_parameters(self.critic, self.target_critic)

        # Update model version
        self.version += 1

        return {
            "log": {
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
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
            "target_actor": self._state_dict(self.target_actor),
            "critic": self._state_dict(self.critic),
            "target_critic": self._state_dict(self.target_critic),
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

        # Load both online and target network
        self.actor.load_state_dict(ckpt["actor"])
        self.target_actor.load_state_dict(ckpt["target_actor"])

        self.critic.load_state_dict(ckpt["critic"])
        self.target_critic.load_state_dict(ckpt["target_critic"])

        # Load actor and critic optimizer state
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
