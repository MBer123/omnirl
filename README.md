# OmniRL

English | [简体中文](https://github.com/MBer123/omnirl/blob/main/README.zh-CN.md)

OmniRL is a distributed reinforcement learning framework built on Python, Ray, and PyTorch. It provides a common runtime for asynchronous sampling, replay and rollout storage, distributed learning, versioned weight synchronization, evaluation, logging, and checkpoint recovery. The framework includes 13 value-based and actor-critic algorithms for discrete and continuous control.

## Architecture

```text
Environment Collection ──► Samplers ──► Memory ──► Learners ──► WeightStore
                               ▲                                     │
                               └───────── Algorithm Policy ◄─────────┘

Environment Evaluation ◄── Evaluators ◄── Algorithm Policy ◄── WeightStore
```

- `omnirl.runtime` orchestrates Ray actors for sampling, learning, evaluation, and versioned weight synchronization.
- `omnirl.buffers` provides rollout, replay, and prioritized replay storage.
- `omnirl.algorithms` contains algorithm-specific learners and inference policies.
- `omnirl.networks` contains reusable network architectures for value-based and actor-critic algorithms.
- `omnirl.envs` owns environment construction, specifications, collection and evaluation loops, and metric aggregation.

Every algorithm package exposes the same `Learner`, `Policy`, and `Memory` interface. The runtime connects these components to the selected environment backend, keeping algorithm and environment implementations independent.

## Implemented Algorithms

| Algorithm | Family | Action space |
| --- | --- | --- |
| ✅ [Deep Q-Network (DQN)](https://arxiv.org/abs/1312.5602) | Value-based, off-policy | Discrete |
| ✅ [Double Deep Q-Network (DDQN)](https://arxiv.org/abs/1509.06461) | Value-based, off-policy | Discrete |
| ✅ [Dueling Double Deep Q-Network (Dueling DQN)](https://arxiv.org/abs/1511.06581) | Value-based, off-policy | Discrete |
| ✅ [Prioritized Experience Replay Deep Q-Network (PER-DQN)](https://arxiv.org/abs/1511.05952) | Value-based, off-policy | Discrete |
| ✅ [Categorical 51-Atom Deep Q-Network (C51)](https://arxiv.org/abs/1707.06887) | Distributional value-based, off-policy | Discrete |
| ✅ [Quantile Regression Deep Q-Network (QR-DQN)](https://arxiv.org/abs/1710.10044) | Distributional value-based, off-policy | Discrete |
| ✅ [Advantage Actor-Critic (A2C)](https://arxiv.org/abs/1602.01783) | Actor-critic, on-policy | Discrete / Continuous |
| ✅ [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) | Actor-critic, on-policy | Discrete / Continuous |
| ✅ [Importance Weighted Actor-Learner Architectures (IMPALA)](https://arxiv.org/abs/1802.01561) | Actor-critic, off-policy | Discrete / Continuous |
| ✅ [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971) | Actor-critic, off-policy | Continuous |
| ✅ [Twin Delayed Deep Deterministic Policy Gradient (TD3)](https://arxiv.org/abs/1802.09477) | Actor-critic, off-policy | Continuous |
| ✅ [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) | Actor-critic, off-policy | Discrete / Continuous |
| ✅ [Truncated Quantile Critics (TQC)](https://arxiv.org/abs/2005.04269) | Distributional actor-critic, off-policy | Continuous |

## Installation

OmniRL supports Python 3.10–3.12. Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux and macOS:

```bash
source .venv/bin/activate
```

Install OmniRL and its shared dependencies:

```bash
python -m pip install --upgrade pip
pip install omnirl-framework
```

The PyPI distribution is named `omnirl-framework`; the Python import package remains `omnirl`. Install the Gymnasium dependency:

```bash
pip install "gymnasium>=1.0,<2"
```

The BipedalWalker and CarRacing examples require the Box2D extra:

```bash
pip install "gymnasium[box2d]>=1.0,<2"
```

For GPU training, install the PyTorch build that matches the operating system and CUDA runtime, then verify CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Getting Started

An OmniRL run uses two YAML files:

- The runner configuration defines the environment, Ray worker counts, device, seed, checkpoints, and logging.
- The agent configuration defines the algorithm module and its learner, memory, policy, and network parameters.

Parameter templates for all 13 algorithms are in [examples/algorithms](https://github.com/MBer123/omnirl/tree/main/examples/algorithms).

All three environment examples use PPO. Each directory contains `ppo.yaml`, `train.yaml`, and `test.yaml`:

| Example directory | Input | Action space | Network |
| --- | --- | --- | --- |
| `examples/cartpole/` | State vector | Discrete | MLP |
| `examples/bipedal_walker/` | State vector | Continuous | MLP |
| `examples/car_racing/` | RGB image | Continuous | CNN |

These files and `run.py` are not included in the PyPI package, so first get the repository:

```bash
git clone https://github.com/MBer123/omnirl.git
cd omnirl
```

Start a single-machine PPO run on `CartPole-v1`:

```bash
python run.py --mode train --runner-config examples/cartpole/train.yaml --agent-config examples/cartpole/ppo.yaml
```

The example uses one CPU learner, one sampler, and one evaluator, with a target of 100 learner updates.

`run.py` connects to an existing Ray cluster when available; otherwise, it starts a local Ray runtime. For multi-node execution, start the head node:

```bash
ray start --head --port=6379
```

Join each worker, then launch `run.py` from the head node:

```bash
ray start --address=<head-ip>:6379
```

When `runner.load_checkpoint` is `true`, restarting the same training command resumes from `checkpoint.ckpt` in `runner.ckpt_dir`.

### Outputs

Each run writes the following artifacts to `runner.ckpt_dir`:

- `checkpoint.ckpt`: complete learner and optimizer state for resuming training.
- `agent_<version>.pth`: exported policy weights.
- `log.csv`: runtime, evaluation, and learner metrics.
- `run_config.yaml`: merged runner and algorithm configuration used by the run.

## Evaluation

Set `runner.ckpt_dir` and `runner.eval_model` in the matching `test.yaml`, then run:

```bash
python run.py --mode test --runner-config examples/cartpole/test.yaml --agent-config examples/cartpole/ppo.yaml
```

The example loads `agent_100.pth`. If you change the training update target, update `runner.eval_model` accordingly.

The selected environment backend defines the evaluation metrics. The total number of evaluated episodes is `runner.num_evaluators * runner.num_episodes`.

When a backend supports interactive rendering, set `env.render_mode` to `human` and `runner.num_evaluators` to `1` in the test configuration.

## Adding an Algorithm

Create a package under `omnirl/algorithms/on_policy/<name>/` or `omnirl/algorithms/off_policy/<name>/` that exports the shared runtime interface:

```python
from .learner import Learner
from .policy import Policy
from omnirl.buffers import ReplayBuffer as Memory

__all__ = ["Learner", "Memory", "Policy"]
```

The implementation should provide:

- `Learner`: a class inheriting `BaseLearner` from `omnirl.algorithms.base`, wrapped as a Ray actor, that implements learning, checkpointing, and versioned weight access. Existing learners can be used as interface examples.
- `Policy`: a local inference class inheriting `BasePolicy` from `omnirl.algorithms.base`, with deterministic `choose_action(observation)` for a single observation and exploratory `sample_action(observations)` for a batch. Sampling inputs and outputs must retain the leading batch dimension, including when the batch size is 1.
- `Memory`: one of the buffers in `omnirl.buffers`, or a compatible new Ray actor.
- Network definitions under `omnirl.networks` when the algorithm needs a new architecture.
- An agent YAML whose `agent.module` points to `omnirl.algorithms.on_policy.<name>` or `omnirl.algorithms.off_policy.<name>` and whose `modules` section supplies the constructor parameters.

The environment backend and distributed runtime do not need to be changed. Add module-interface and learner-update tests before adding the algorithm to the table above.

## Adding an Environment Backend

Create a package under `omnirl/envs/<name>/` that exports:

```python
build_env
get_env_spec
CollectionLoop
EvaluationLoop
aggregate_evaluations
```

`get_env_spec` describes observation shape and dtype, action dimensions and bounds, and default evaluation metrics. `CollectionLoop` turns policy decisions into rollout data, while `EvaluationLoop` and `aggregate_evaluations` produce environment-specific metrics.

Set the backend module in a runner configuration:

```yaml
env:
  module: omnirl.envs.gymnasium
  id: CartPole-v1
  render_mode: null
```

`module` selects the backend; the remaining `env` mapping is passed through unchanged.

Algorithms and runtime actors communicate only through this interface, so neither needs backend-specific imports or configuration fields.

## Notes

- When training on CPU, set `device.learner` to `cpu` and keep `runner.num_learners` at `1`.
- Size `runner.num_samplers`, `runner.num_evaluators`, and `runner.num_learners` for the available hardware. As a starting point, keep their sum below 90% of the available CPU threads.
- Run the test suite with `python -m unittest discover -s tests -v`.
- OmniRL is a research framework. Multi-seed algorithm validation and scaling results reflect the current experimental conditions and do not constitute production guarantees.

## License

Copyright 2026 Huang Chenbiao (MBer123).

OmniRL is licensed under the [Apache License 2.0](https://github.com/MBer123/omnirl/blob/main/LICENSE).
