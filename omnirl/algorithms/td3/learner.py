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
    TD3 learner agent.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        network_kwargs: Keyword arguments used to build the networks.
        ckpt_dir: Directory path to save model checkpoints.
        gamma: Discount factor.
        tau: Soft update coefficient for target networks.
        policy_noise: Standard deviation of Gaussian noise added to target actions.
        noise_clip: Maximum absolute value used to clip target policy noise.
        delay_time: Update interval for the actor and target networks.
        actor_lr: Learning rate for the actor optimizer.
        critic_lr: Learning rate for both critic optimizers.
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
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        delay_time: int = 2,
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
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.delay_time = delay_time

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
        critic_1 = QCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            layer_type=critic_layer_type,
            is_continuous=True,
            network_kwargs=critic_kwargs,
        )
        critic_2 = QCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            layer_type=critic_layer_type,
            is_continuous=True,
            network_kwargs=critic_kwargs,
        )

        self.actor = actor.to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)

        self.critic_1 = critic_1.to(self.device)
        self.critic_2 = critic_2.to(self.device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)

        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()

        self.target_actor.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()

        if is_distributed:
            ddp_kwargs = {"device_ids": [0]} if self.device.type == "cuda" else {}
            self.train_actor = DDP(self.actor, **ddp_kwargs)
            self.train_critic_1 = DDP(self.critic_1, **ddp_kwargs)
            self.train_critic_2 = DDP(self.critic_2, **ddp_kwargs)
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic_1.load_state_dict(self.critic_1.state_dict())
            self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        else:
            self.train_actor = self.actor
            self.train_critic_1 = self.critic_1
            self.train_critic_2 = self.critic_2

        # Initialize optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

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
            action_noise = torch.randn_like(next_actions) * self.policy_noise
            action_noise = torch.clamp(action_noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + action_noise, -1, 1)

            next_q1 = self.target_critic_1(next_states_tensor, next_actions).view(-1)
            next_q2 = self.target_critic_2(next_states_tensor, next_actions).view(-1)
            next_q1[terminals_tensor] = 0.0
            next_q2[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * torch.min(next_q1, next_q2)

        # Q(s, a) predicted by online network
        q1 = self.train_critic_1(states_tensor, actions_tensor).view(-1)
        q2 = self.train_critic_2(states_tensor, actions_tensor).view(-1)

        # Update critic network
        critic_1_loss = F.mse_loss(q1, target)
        critic_2_loss = F.mse_loss(q2, target)
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delay update actor
        self.version += 1
        if self.version % self.delay_time != 0:
            return {
                "log": {
                    "actor_loss": 0,
                    "critic_loss": ((critic_1_loss + critic_2_loss) / 2).item(),
                },
                "memory": {"version": self.version},
                "version": self.get_version(),
                "trained_samples": int(batch_size),
            }

        # Freeze critic_1 during actor update
        for p in self.critic_1.parameters():
            p.requires_grad_(False)

        # Update actor network
        new_actions = self.train_actor(states_tensor)
        actor_loss = -torch.mean(self.critic_1(states_tensor, new_actions))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic_1 after actor update
        for p in self.critic_1.parameters():
            p.requires_grad_(True)

        # Soft update target network
        self._update_network_parameters(self.actor, self.target_actor)
        self._update_network_parameters(self.critic_1, self.target_critic_1)
        self._update_network_parameters(self.critic_2, self.target_critic_2)

        return {
            "log": {
                "actor_loss": actor_loss.item(),
                "critic_loss": ((critic_1_loss + critic_2_loss) / 2).item(),
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
            "critic_1": self._state_dict(self.critic_1),
            "critic_2": self._state_dict(self.critic_2),
            "target_critic_1": self._state_dict(self.target_critic_1),
            "target_critic_2": self._state_dict(self.target_critic_2),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
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

        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.target_critic_1.load_state_dict(ckpt["target_critic_1"])
        self.target_critic_2.load_state_dict(ckpt["target_critic_2"])

        # Load actor and critic optimizer state
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(ckpt["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(ckpt["critic_2_optimizer"])

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
