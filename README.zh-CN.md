# OmniRL

[English](https://github.com/MBer123/omnirl/blob/main/README.md) | 简体中文

OmniRL 是一个基于 Python、Ray 和 PyTorch 的分布式强化学习框架。框架支持异步采样、经验存储与回放、分布式学习、权重同步、评估、日志记录和检查点恢复，内置 13 种 Value-based 和 Actor-Critic 算法，覆盖离散与连续控制任务。

## 架构

```text
Environment Collection ──► Samplers ──► Memory ──► Learners ──► WeightStore
                               ▲                                     │
                               └───────── Algorithm Policy ◄─────────┘

Environment Evaluation ◄── Evaluators ◄── Algorithm Policy ◄── WeightStore
```

- `omnirl.runtime` 负责调度采样、学习、评估和权重同步所需的 Ray Actor。
- `omnirl.buffers` 提供 Rollout Buffer、Replay Buffer 和 Prioritized Replay Buffer。
- `omnirl.algorithms` 包含各算法的 `Learner` 和 `Policy` 实现。
- `omnirl.networks` 提供可复用的 Value-based 与 Actor-Critic 网络结构。
- `omnirl.envs` 负责环境构建、环境规格定义、数据采集、评估和指标聚合。

所有算法包都导出统一的 `Learner`、`Policy` 和 `Memory` 接口。运行时通过这些接口组合算法与环境模块，使两者可以独立实现和扩展。

## 已实现算法

| 算法 | 类型 | 动作空间 |
| --- | --- | --- |
| ✅ [Deep Q-Network (DQN)](https://arxiv.org/abs/1312.5602) | Value-based，Off-policy | 离散 |
| ✅ [Double Deep Q-Network (DDQN)](https://arxiv.org/abs/1509.06461) | Value-based，Off-policy | 离散 |
| ✅ [Dueling Double Deep Q-Network (Dueling DQN)](https://arxiv.org/abs/1511.06581) | Value-based，Off-policy | 离散 |
| ✅ [Prioritized Experience Replay Deep Q-Network (PER-DQN)](https://arxiv.org/abs/1511.05952) | Value-based，Off-policy | 离散 |
| ✅ [Categorical 51-Atom Deep Q-Network (C51)](https://arxiv.org/abs/1707.06887) | Distributional Value-based，Off-policy | 离散 |
| ✅ [Quantile Regression Deep Q-Network (QR-DQN)](https://arxiv.org/abs/1710.10044) | Distributional Value-based，Off-policy | 离散 |
| ✅ [Advantage Actor-Critic (A2C)](https://arxiv.org/abs/1602.01783) | Actor-Critic，On-policy | 离散 / 连续 |
| ✅ [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) | Actor-Critic，On-policy | 离散 / 连续 |
| ✅ [Importance Weighted Actor-Learner Architectures (IMPALA)](https://arxiv.org/abs/1802.01561) | Actor-Critic，Off-policy | 离散 / 连续 |
| ✅ [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971) | Actor-Critic，Off-policy | 连续 |
| ✅ [Twin Delayed Deep Deterministic Policy Gradient (TD3)](https://arxiv.org/abs/1802.09477) | Actor-Critic，Off-policy | 连续 |
| ✅ [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) | Actor-Critic，Off-policy | 离散 / 连续 |
| ✅ [Truncated Quantile Critics (TQC)](https://arxiv.org/abs/2005.04269) | Distributional Actor-Critic，Off-policy | 连续 |

## 安装

OmniRL 支持 Python 3.10–3.12。首先创建虚拟环境：

```bash
python -m venv .venv
```

在 Windows 上激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

在 Linux 和 macOS 上：

```bash
source .venv/bin/activate
```

安装 OmniRL 及其基础依赖：

```bash
python -m pip install --upgrade pip
pip install omnirl-framework
```

PyPI 包名为 `omnirl-framework`，Python 导入包名仍为 `omnirl`。安装 Gymnasium 环境依赖：

```bash
pip install "gymnasium>=1.0,<2"
```

BipedalWalker 与 CarRacing 示例需要 Box2D 附加依赖：

```bash
pip install "gymnasium[box2d]>=1.0,<2"
```

如需使用 GPU 训练，请安装与操作系统和 CUDA 运行时相匹配的 PyTorch 版本，然后确认 CUDA 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 快速开始

运行 OmniRL 需要两份 YAML 配置：

- Runner 配置用于定义环境、Ray Worker 数量、设备、随机种子、检查点和日志。
- Agent 配置用于定义算法模块，以及 `Learner`、`Memory`、`Policy` 和网络参数。

全部 13 种算法的参数模板见 [examples/algorithms](https://github.com/MBer123/omnirl/tree/main/examples/algorithms)。

三个环境示例都使用 PPO，每个目录包含 `ppo.yaml`、`train.yaml` 和 `test.yaml`：

| 示例目录 | 输入 | 动作空间 | 网络 |
| --- | --- | --- | --- |
| `examples/cartpole/` | 状态向量 | 离散 | MLP |
| `examples/bipedal_walker/` | 状态向量 | 连续 | MLP |
| `examples/car_racing/` | RGB 图像 | 连续 | CNN |

`run.py` 与这些示例文件不随 PyPI 包安装，运行前先获取仓库：

```bash
git clone https://github.com/MBer123/omnirl.git
cd omnirl
```

在单机上启动 `CartPole-v1` 的 PPO 训练：

```bash
python run.py --mode train --runner-config examples/cartpole/train.yaml --agent-config examples/cartpole/ppo.yaml
```

该示例默认使用一个 CPU Learner、一个 Sampler 和一个 Evaluator，执行 100 次参数更新。

如果检测到可用的 Ray 集群，`run.py` 会直接连接；否则会启动本地 Ray 运行时。多节点运行时，先启动 Head 节点：

```bash
ray start --head --port=6379
```

让各 Worker 节点加入集群，然后在 Head 节点上启动 `run.py`：

```bash
ray start --address=<head-ip>:6379
```

当 `runner.load_checkpoint` 为 `true` 时，重新执行相同的训练命令会从 `runner.ckpt_dir` 中的 `checkpoint.ckpt` 恢复训练。

### 输出

每次运行都会将以下文件写入 `runner.ckpt_dir`：

- `checkpoint.ckpt`：用于恢复训练的完整 `Learner` 状态和优化器状态。
- `agent_<version>.pth`：导出的策略权重。
- `log.csv`：运行时、评估和 `Learner` 指标。
- `run_config.yaml`：本次运行实际使用的 Runner 与 Agent 合并配置文件。

## 评估

在对应的 `test.yaml` 中设置 `runner.ckpt_dir` 和 `runner.eval_model`，然后运行：

```bash
python run.py --mode test --runner-config examples/cartpole/test.yaml --agent-config examples/cartpole/ppo.yaml
```

示例默认加载 `agent_100.pth`；调整训练更新次数后，请同步修改 `runner.eval_model`。

评估指标由所选环境模块定义。总评估回合数为 `runner.num_evaluators * runner.num_episodes`。

当环境模块支持交互式渲染时，请在测试配置中将 `env.render_mode` 设置为 `human`，并将 `runner.num_evaluators` 设置为 `1`。

## 添加算法

在 `omnirl/algorithms/<name>/` 下创建算法包，并导出统一的运行时接口：

```python
from .learner import Learner
from .policy import Policy
from omnirl.buffers import ReplayBuffer as Memory

__all__ = ["Learner", "Memory", "Policy"]
```

算法模块需要提供：

- `Learner`：负责参数更新、检查点保存和带版本权重访问的 Ray Actor，可参考现有实现的接口。
- `Policy`：提供确定性 `choose_action(observation)` 和探索性 `sample_action(observations)` 方法的本地推理类。前者接收单个观测，后者接收批量观测；采样输入和输出必须保留首维 batch 维度，即使 batch 大小为 1。
- `Memory`：`omnirl.buffers` 中的一种 Buffer，或符合相同接口的新 Ray Actor。
- 如果算法需要新的网络架构，请在 `omnirl.networks` 下定义网络。
- 一份 Agent YAML 配置，其中 `agent.module` 指向 `omnirl.algorithms.<name>`，`modules` 部分提供各组件的构造参数。

无需修改环境模块和分布式运行时。将新算法加入上表前，请添加模块接口测试和学习器更新测试。

## 添加环境模块

在 `omnirl/envs/<name>/` 下创建一个包，并导出：

```python
build_env
get_env_spec
CollectionLoop
EvaluationLoop
aggregate_evaluations
```

`get_env_spec` 用于描述观测的形状与数据类型、动作的维度与边界以及默认评估指标。`CollectionLoop` 将策略输出转换为 Rollout 数据；`EvaluationLoop` 和 `aggregate_evaluations` 负责生成并聚合环境特定的评估指标。

在 Runner 配置中指定环境模块：

```yaml
env:
  module: omnirl.envs.gymnasium
  id: CartPole-v1
  render_mode: null
```

`module` 用于选择环境模块；`env` 中的其余配置会原样传递给该模块。

算法和运行时 Actor 只依赖这组统一接口，无需导入具体的环境模块，也无需包含环境模块专用配置。

## 注意事项

- 使用 CPU 训练时，请将 `device.learner` 设置为 `cpu`，并保持 `runner.num_learners` 为 `1`。
- 请根据可用硬件调整 `runner.num_samplers`、`runner.num_evaluators` 和 `runner.num_learners`。初次配置时，建议将三者总数控制在可用 CPU 线程数的 90% 以下。
- 使用 `python -m unittest discover -s tests -v` 运行测试套件。
- OmniRL 面向强化学习研究。基于多个随机种子的算法验证和扩展性结果仅代表当前实验条件，不构成生产级保证。

## 许可证

Copyright 2026 Huang Chenbiao (MBer123).

本项目采用 [Apache License 2.0](https://github.com/MBer123/omnirl/blob/main/LICENSE) 开源许可证。
