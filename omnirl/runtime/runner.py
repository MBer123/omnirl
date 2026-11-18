import os
import time
from collections import deque
from datetime import timedelta
from functools import partial
import threading

import ray

from omnirl.utils.file_writer import FileWriter
from omnirl.utils.logger import setup_logger

from .evaluator import Evaluator
from .resources import ResourceAllocator
from .sampler import Sampler
from .weight_store import WeightStore


class Runner:
    """
    Distributed training runner.

    Args:
        agent: Algorithm module that provides Learner, Policy, and Memory.
        environment: Environment backend module used for interaction and evaluation.
        agent_config: Configuration for agent modules.
        env_config: Configuration for environment modules.
        ckpt_dir: Directory path to save model checkpoints.
        eval_model: Checkpoint file name of the model used for evaluation.
        num_learners: Number of learner actors.
        num_samplers: Number of sampler actors.
        num_envs_per_sampler: Number of environments sharing each sampler policy.
        collection_fragment_length: Steps collected per environment per transfer.
        num_evaluators: Number of evaluator actors.
        num_episodes_per_evaluator: Number of evaluation episodes per evaluator.
        num_buffers: Buffers sharing configured batch, capacity, and warm-up sizes.
        max_updates: Maximum number of learner updates.
        prefetch_depth: Maximum number of learner updates kept in flight.
        save_interval: Time in seconds between checkpoint saves.
        log_interval: Time in seconds between logging events.
        log_name: Name of the training log file.
        load_checkpoint: Whether to resume training from an existing checkpoint.
        learner_device: Device selector used to run the learner.
        num_sampler_gpus: Number of GPUs reserved for samplers.
        num_samplers_per_gpu: Maximum number of samplers sharing each GPU.
        num_evaluator_gpus: Number of GPUs reserved for evaluators.
        num_evaluators_per_gpu: Maximum number of evaluators sharing each GPU.
        learner_seed: Random seed used for learner.
        sampler_seed: Random seed used for sampler.
        evaluator_seed: Random seed used for evaluator.
    """

    def __init__(
        self,
        agent,
        environment,
        agent_config,
        env_config,
        ckpt_dir: str = "checkpoints",
        eval_model: str | None = None,
        num_learners: int = 1,
        num_samplers: int = 1,
        num_envs_per_sampler: int = 1,
        collection_fragment_length: int = 1,
        num_evaluators: int = 1,
        num_episodes_per_evaluator: int = 10,
        num_buffers: int = 1,
        max_updates: int = 10000000,
        prefetch_depth: int = 1,
        save_interval: int = 3600,
        log_interval: int = 60,
        log_name: str = "log.csv",
        load_checkpoint: bool = True,
        learner_device: str = "cpu",
        num_sampler_gpus: int = 0,
        num_samplers_per_gpu: int = 1,
        num_evaluator_gpus: int = 0,
        num_evaluators_per_gpu: int = 1,
        learner_seed: int | None = None,
        sampler_seed: int | None = None,
        evaluator_seed: int | None = None,
    ):
        # Initialize ray
        if not ray.is_initialized():
            runtime_env = {
                "working_dir": ".",
                "excludes": ["**/*.ckpt", "**/*.csv", "**/*.pth"],
            }
            ray.init(runtime_env=runtime_env)

        # Initialize resource allocation
        self.resources = ResourceAllocator(
            num_learners,
            num_samplers,
            num_evaluators,
            num_buffers,
            learner_device,
            num_sampler_gpus,
            num_samplers_per_gpu,
            num_evaluator_gpus,
            num_evaluators_per_gpu,
        )
        self.resources.reserve(timeout_seconds=30)

        # Initialize logging and checkpoint utilities
        self.log = setup_logger()
        ckpt = os.path.abspath(ckpt_dir)
        self.ckpt_path = os.path.join(ckpt, "checkpoint.ckpt")
        self.fw = FileWriter(ckpt_dir=ckpt, log_name=log_name)

        # Read the environment description through the backend interface
        env_config = dict(env_config)
        env_provider = environment.EnvProvider
        env_meta = env_provider.get_spec(env_config)
        env_spec = env_meta["env_spec"]
        state_spec = env_meta["state_spec"]
        action_spec = env_meta["action_spec"]
        eval_metrics = environment.EvaluationLoop.default_metrics

        # Load agent module arguments from config
        learner_kwargs = dict(agent_config.get("learner") or {})
        memory_kwargs = dict(agent_config.get("memory") or {})
        policy_kwargs = dict(agent_config.get("policy") or {})
        network_kwargs = dict(agent_config.get("network") or {})

        # Initialize memory
        for name in ("batch_size", "max_size", "warmup_size"):
            if name in memory_kwargs:
                memory_kwargs[name] //= num_buffers
        memory_kwargs["collection_fragment_length"] = collection_fragment_length

        self.memories = []
        for i in range(num_buffers):
            _, options = self.resources.get_allocation("memory", i)
            memory = agent.Memory.options(**options)
            self.memories.append(memory.remote(**state_spec, **memory_kwargs))
        supports_prefetch = ray.get(self.memories[0].supports_prefetch.remote())

        # Initialize learner
        self.learners = []
        for i in range(num_learners):
            learner_device, options = self.resources.get_allocation("learner", i)
            self.learners.append(
                agent.Learner.options(**options).remote(
                    network_kwargs=network_kwargs,
                    ckpt_dir=ckpt,
                    device=learner_device,
                    seed=learner_seed,
                    rank=i,
                    world_size=num_learners,
                    **env_spec,
                    **learner_kwargs,
                )
            )
        self.learner = self.learners[0]
        if eval_model is not None:
            ray.get(self.learner.load_model.remote(eval_model))

        # Initialize weight store
        _, options = self.resources.get_allocation("weight_store")
        init_weight = ray.get(self.learner.get_weights.remote())
        init_version = ray.get(self.learner.get_version.remote())
        weight_store = WeightStore.options(**options)
        self.weight_store = weight_store.remote(init_weight, init_version)

        # Initialize samplers
        self.samplers = []
        role = "collection"
        env_builder = partial(env_provider.build, env_config=env_config, role=role)
        for i in range(num_samplers):
            device, options = self.resources.get_allocation("sampler", i)
            offset = 1000 + i * num_envs_per_sampler
            seed = None if sampler_seed is None else sampler_seed + offset
            self.samplers.append(
                Sampler.options(**options).remote(
                    policy_cls=agent.Policy,
                    collection_cls=environment.CollectionLoop,
                    env_builder=env_builder,
                    action_spec=action_spec,
                    network_kwargs=network_kwargs,
                    policy_kwargs=policy_kwargs,
                    num_envs_per_sampler=num_envs_per_sampler,
                    collection_fragment_length=collection_fragment_length,
                    device=device,
                    seed=seed,
                )
            )

        # Initialize evaluators
        self.evaluators = []
        role = "evaluation"
        env_builder = partial(env_provider.build, env_config=env_config, role=role)
        for i in range(num_evaluators):
            device, options = self.resources.get_allocation("evaluator", i)
            offset = 2000 + i
            seed = None if evaluator_seed is None else evaluator_seed + offset
            self.evaluators.append(
                Evaluator.options(**options).remote(
                    policy_cls=agent.Policy,
                    evaluation_cls=environment.EvaluationLoop,
                    env_builder=env_builder,
                    action_spec=action_spec,
                    network_kwargs=network_kwargs,
                    policy_kwargs=policy_kwargs,
                    num_episodes_per_evaluator=num_episodes_per_evaluator,
                    device=device,
                    seed=seed,
                )
            )

        # Initialize training configuration
        self.max_updates = max_updates
        self.prefetch_depth = max(1, prefetch_depth) if supports_prefetch else 1
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.load_checkpoint = load_checkpoint

        # Initialize training state
        self.version = init_version
        self.eval_version = -1
        self.trained_samples = 0
        self.inflight_updates = deque()

        # Initialize cached statistics
        self.latest_train_info = {}
        self.latest_eval_info = {
            "version": 0,
            "num_episodes": 0,
            **eval_metrics,
        }

        # Initialize timing state
        self.start_time = 0.0
        self.last_log_time = 0.0
        self.next_log_time = float("inf")
        self.next_save_time = float("inf")
        self.stop_event = threading.Event()
        self.aggregate_evaluations = environment.EvaluationLoop.aggregate

        # Initialize throughput counters
        self.last_collected_data = 0
        self.last_trained_samples = 0

    @classmethod
    def from_config(cls, agent, environment, runner_config, agent_config):
        """Create a Runner instance from parsed configuration dictionaries."""
        seed_param = runner_config.get("seed", {})
        env_param = dict(runner_config["env"])
        env_param.pop("module")

        # Load runner and environment arguments
        runner_kwargs = {
            "agent": agent,
            "environment": environment,
            "agent_config": agent_config["modules"],
            "env_config": env_param,
            **runner_config.get("runner", {}),
        }

        # Load learner, sampler, evaluator, and memory arguments
        for role in ("learner", "sampler", "evaluator", "memory"):
            params = dict(runner_config.get(role) or {})
            runner_kwargs.update(params)

        # Load seed arguments
        seed_mapping = {
            "learner": "learner_seed",
            "sampler": "sampler_seed",
            "evaluator": "evaluator_seed",
        }
        for src_key, dst_key in seed_mapping.items():
            if src_key in seed_param:
                runner_kwargs[dst_key] = seed_param[src_key]

        return cls(**runner_kwargs)

    def _log_str(self, string):
        """Log a centered banner string."""
        self.log.info("-" * 59)
        self.log.info(f"|{string:^57}|")
        self.log.info("-" * 59)

    def _log_groups(self, groups):
        """Format grouped metrics and write to logger."""
        lines = self._build_log_lines(groups)
        for line in lines:
            self.log.info(line)

    def _build_log_lines(self, groups):
        """Build aligned text lines for grouped metric logging."""
        key_width = 22
        val_width = 30
        total_width = key_width + val_width + 7

        lines = ["-" * total_width]

        for group_name, metrics in groups:
            if not metrics:
                continue

            lines.append(f"| {group_name:<{key_width + val_width + 3}} |")
            for key, value in metrics.items():
                value = f"{value:.4f}" if isinstance(value, float) else str(value)
                lines.append(f"|    {key:<{key_width - 3}} | {value:<{val_width}} |")

        lines.append("-" * total_width)
        return lines

    def _load_checkpoint(self):
        """Load learner checkpoint and synchronize weights."""
        if self.load_checkpoint and os.path.exists(self.ckpt_path):
            remote = [learner.load_checkpoint.remote() for learner in self.learners]
            results = ray.get(remote)[0]

            self.version = results["version"]
            weights = results["weights"]
            ray.get(self.weight_store.set.remote(weights, self.version))

            self._log_str("RESUME FROM CHECKPOINT")

    def _start_samplers(self):
        """Launch asynchronous rollout tasks on all sampler actors."""
        for i, sampler in enumerate(self.samplers):
            memory = self.memories[i % len(self.memories)]
            sampler.rollout.remote(memory, self.weight_store)

    def _start_evaluators(self, is_train=True):
        """Run evaluators in parallel and aggregate evaluation metrics."""
        while not self.stop_event.is_set() or not is_train:
            results = ray.get(self.weight_store.get.remote(self.eval_version))
            version = results["version"]
            weights = results["weights"]

            if weights is None:
                self.stop_event.wait(1.0)
                continue

            eval_refs = []
            for evaluator in self.evaluators:
                eval_refs.append(evaluator.evaluate.remote(weights, version))
            eval_infos = ray.get(eval_refs)

            avg_info = {"version": version, **self.aggregate_evaluations(eval_infos)}

            self.latest_eval_info = avg_info
            self.eval_version = version

            if not is_train:
                break

        return avg_info

    def _submit_update(self):
        """Request batches and submit one learner update without waiting for it."""
        batch_refs = []
        num_buffers = len(self.memories)
        world_size = len(self.learners) // num_buffers
        for i in range(len(self.learners)):
            memory = self.memories[i % num_buffers]
            rank = i // num_buffers
            batch_refs.append(memory.get_batch.remote(rank, world_size))

        lb_zip = zip(self.learners, batch_refs)
        learn_list = [learner.learn.remote(batch_ref) for learner, batch_ref in lb_zip]

        weights_ref = self.learner.get_weights.remote()
        self.inflight_updates.append((learn_list, weights_ref))

    def _can_submit(self):
        """Return True if prefetch capacity and update budget allow another update."""
        inflight_size = len(self.inflight_updates)
        not_full = inflight_size < self.prefetch_depth
        reach_limit = self.version + inflight_size >= self.max_updates
        return not_full and not reach_limit

    def _train_one_step(self):
        """Keep learner updates in flight and finish the oldest one."""
        if not self.inflight_updates:
            ready = ray.get([memory.ready.remote() for memory in self.memories])
            if not all(ready):
                time.sleep(0.01)
                return

        while self._can_submit():
            self._submit_update()

        learn_list, weights_ref = self.inflight_updates.popleft()
        infos = ray.get(learn_list)

        train_info = infos[0]
        avg_log = {}
        for key in train_info["log"]:
            avg_log[key] = sum(info["log"][key] for info in infos) / len(infos)
        self.latest_train_info = avg_log

        self.version = train_info["version"]
        self.trained_samples += sum(info["trained_samples"] for info in infos)

        self.weight_store.set.remote(weights_ref, self.version)
        num_buffers = len(self.memories)
        for i, memory in enumerate(self.memories):
            memory_infos = [info["memory"] for info in infos[i::num_buffers]]
            memory.on_learn.remote(memory_infos)

    def _log_info(self, force=False):
        """Collect runtime statistics, evaluate policy, and write training logs."""
        log_start = time.perf_counter()
        if log_start < self.next_log_time and not force:
            return

        stats = ray.get([memory.stats.remote() for memory in self.memories])
        collected_data = sum(info["collected_data"] for info in stats)

        dt = log_start - self.last_log_time
        wall_time = log_start - self.start_time

        sampler_fps = (collected_data - self.last_collected_data) / dt
        learner_fps = (self.trained_samples - self.last_trained_samples) / dt

        groups = [
            (
                "time/",
                {
                    "updates": f"{self.version} / {self.max_updates}",
                    "wall_time": str(timedelta(seconds=int(wall_time))),
                    "collected_samples": collected_data,
                    "trained_samples": self.trained_samples,
                    "sampler_fps": int(sampler_fps),
                    "learner_fps": int(learner_fps),
                },
            ),
            ("train/", self.latest_train_info),
            ("eval/", self.latest_eval_info),
        ]

        self._log_groups(groups)

        self.fw.write(
            {
                "wall_time": wall_time,
                "updates": self.version,
                "collected_samples": collected_data,
                "trained_samples": self.trained_samples,
                "sampler_fps": sampler_fps,
                "learner_fps": learner_fps,
                **self.latest_eval_info,
                **self.latest_train_info,
            }
        )

        self.last_log_time = log_start
        self.last_collected_data = collected_data
        self.last_trained_samples = self.trained_samples
        self.next_log_time = log_start + self.log_interval

    def _save_checkpoint(self):
        """Save learner checkpoint when the save interval is reached."""
        now = time.perf_counter()
        if now < self.next_save_time:
            return

        ray.get(self.learner.save_checkpoint.remote())
        self._log_str("SAVE CHECKPOINTS")
        self.next_save_time = now + self.save_interval

    def fit(self):
        """Run the full distributed training loop."""
        self._log_str("START TRAINING")
        self._load_checkpoint()
        self._start_samplers()

        self.stop_event.clear()
        eval_thread = threading.Thread(target=self._start_evaluators, daemon=True)
        eval_thread.start()

        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time

        self.next_log_time = self.start_time + self.log_interval
        self.next_save_time = self.start_time + self.save_interval

        try:
            while self.version < self.max_updates:
                self._train_one_step()
                self._log_info()
                self._save_checkpoint()

            self._log_info(force=True)
            self._log_str("FINISH TRAINING")

        finally:
            self.stop_event.set()
            eval_thread.join()
            for sampler in self.samplers:
                ray.kill(sampler)
            ray.get(self.learner.save_checkpoint.remote())
            self.fw.close()

    def eval(self):
        """Run distributed evaluation on all evaluators and report results."""
        self._log_str("START EVALUATION")
        self.latest_eval_info = self._start_evaluators(is_train=False)
        groups = [("eval/", self.latest_eval_info)]
        self._log_groups(groups)
        self._log_str("FINISH EVALUATION")

        return groups
