import copy
import os

import ray
import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from omnirl.algorithms.base import BaseLearner
from omnirl.networks.critics import QuantileQNetwork
from omnirl.utils.seed import set_global_seed


@ray.remote
class Learner(BaseLearner):
    """
    QR-DQN learner agent.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        network_kwargs: Keyword arguments used to build the network.
        ckpt_dir: Directory path to save model checkpoints.
        gamma: Discount factor.
        tau: Soft update coefficient for target network.
        lr: Learning rate for the q net optimizer.
        max_grad_norm: Maximum gradient norm used for clipping.
        num_quantiles: Number of quantiles.
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
        lr: float = 0.0003,
        max_grad_norm: float = 10.0,
        num_quantiles: int = 25,
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
        self.max_grad_norm = max_grad_norm
        self.num_quantiles = num_quantiles

        # Initialize online network and target network
        self.version = 0
        network_kwargs = dict(network_kwargs or {})
        layer_type = network_kwargs.pop("layer_type")
        q_net = QuantileQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            layer_type=layer_type,
            num_quantiles=num_quantiles,
            network_kwargs=network_kwargs,
        )

        self.q_eval = q_net.to(self.device)
        self.q_target = copy.deepcopy(self.q_eval).to(self.device)

        self.q_eval.train()
        self.q_target.eval()

        if is_distributed:
            ddp_kwargs = {"device_ids": [0]} if self.device.type == "cuda" else {}
            self.train_q_eval = DDP(self.q_eval, **ddp_kwargs)
            self.q_target.load_state_dict(self.q_eval.state_dict())
        else:
            self.train_q_eval = self.q_eval

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=lr)

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
        return self._state_dict(self.q_eval)

    def get_version(self):
        """Export the current learner model version."""
        return self.version

    def _quantile_huber_loss(self, current_quantiles, target_quantiles):
        """Compute the quantile Huber loss."""
        td = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        abs_td = td.abs()

        huber = torch.where(abs_td <= 1.0, 0.5 * td.pow(2), abs_td - 0.5)

        arange = torch.arange(self.num_quantiles, device=self.device)
        taus = ((arange + 0.5) / self.num_quantiles).view(1, self.num_quantiles, 1)

        loss = torch.abs(taus - (td.detach() < 0).float()) * huber
        return loss.sum(dim=2).mean()

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
        batch_indices = torch.arange(batch_size, device=self.device)

        with torch.no_grad():
            # Compute target quantiles using the target network
            next_quantiles = self.q_target(next_states_tensor)
            next_q_values = next_quantiles.mean(dim=-1)
            next_actions = next_q_values.argmax(dim=-1)
            next_action_quantiles = next_quantiles[batch_indices, next_actions]

            # Bellman target quantiles
            rewards = rewards_tensor.unsqueeze(1)
            discount_mask = (~terminals_tensor).float().unsqueeze(1)

            next_quantile_targets = self.gamma * discount_mask * next_action_quantiles
            target_quantiles = rewards + next_quantile_targets

        # Quantiles predicted by online network
        quantiles = self.train_q_eval(states_tensor)[batch_indices, actions_tensor]

        # Quantile Huber loss
        loss = self._quantile_huber_loss(quantiles, target_quantiles)

        # Optimize the online network
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_eval.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Soft update target network
        self._update_network_parameters(self.q_eval, self.q_target)

        # Update model version
        self.version += 1

        return {
            "log": {"loss": loss.item()},
            "memory": {},
            "version": self.get_version(),
            "trained_samples": int(batch_size),
        }

    def save_checkpoint(self):
        """Save q_eval, q_target, and optimizer into checkpoint file."""
        if self.rank != 0:
            return

        path = os.path.join(self.ckpt_dir, "checkpoint.ckpt")

        ckpt = {
            "q_eval": self._state_dict(self.q_eval),
            "q_target": self._state_dict(self.q_target),
            "optimizer": self.optimizer.state_dict(),
            "version": self.version,
        }
        torch.save(ckpt, path)

        eval_path = os.path.join(self.ckpt_dir, f"agent_{self.version}.pth")
        torch.save(self._state_dict(self.q_eval), eval_path)

    def load_checkpoint(self):
        """Load q_eval, q_target, and optimizer from checkpoint file."""
        path = os.path.join(self.ckpt_dir, "checkpoint.ckpt")
        ckpt = torch.load(path, map_location=self.device)

        # Load both online and target network
        self.q_eval.load_state_dict(ckpt["q_eval"])
        self.q_target.load_state_dict(ckpt["q_target"])

        # Load optimizer state
        self.optimizer.load_state_dict(ckpt["optimizer"])

        # Load model version
        self.version = ckpt["version"]

        return {"version": self.get_version(), "weights": self.get_weights()}

    def load_model(self, model_path):
        """Load q_eval from pre-trained model weights."""
        path = os.path.join(self.ckpt_dir, model_path)
        state_dict = torch.load(path, map_location=self.device)

        # Load online network
        self.q_eval.load_state_dict(state_dict)

        # Load model version
        self.version = int(model_path.removeprefix("agent_").removesuffix(".pth"))
