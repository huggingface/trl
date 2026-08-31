# TRL - 基于 Transformers 的强化学习库 (Transformers Reinforcement Learning)

<div style="text-align: center">
    <picture>
        <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl_banner_light.png">
        <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl_banner_dark.png" alt="TRL Banner">
    </picture>
</div>

<hr> <br>

<h3 align="center">
    <p>用于基座大模型后训练（Post-Training）的全栈强化学习与对齐库</p>
</h3>

<p align="center">
    <a href="https://github.com/huggingface/trl/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/huggingface/trl.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/trl/index"><img alt="Documentation" src="https://img.shields.io/website?label=documentation&url=https%3A%2F%2Fhuggingface.co%2Fdocs%2Ftrl%2Findex&down_color=red&down_message=offline&up_color=blue&up_message=online"></a>
    <a href="https://github.com/huggingface/trl/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/trl.svg"></a>
    <a href="https://huggingface.co/trl-lib"><img alt="Hugging Face Hub" src="https://img.shields.io/badge/🤗%20Hub-trl--lib-yellow"></a>
</p>

<p align="center">
    <a href="README.md">English</a> · <b>简体中文</b>
</p>

## 🎉 最新动态

**⚗️ DistillationTrainer 现已进入稳定版：** [`DistillationTrainer`](https://huggingface.co/docs/trl/distillation_trainer) 正式升级为稳定 API —— 支持在线（On-policy）知识蒸馏，通过节省显存的分块 JSD（Jensen-Shannon Divergence）损失函数与 vLLM 加速生成，精准拟合教师模型的完整下个 Token 分布。

## 概述

TRL 是一个专为基座大模型后训练（Post-Training）打造的前沿库，集成了监督微调（SFT）、群体相对策略优化（GRPO）以及直接偏好优化（DPO）等先进对齐技术。TRL 构建于 [🤗 Transformers](https://github.com/huggingface/transformers) 生态系统之上，支持多种模型架构与多模态输入，并能在各种规模的硬件集群上弹性扩展。

## 核心亮点

- **丰富的 Trainer 训练器**：涵盖多种主流微调与对齐算法，包括 [`SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer)、[`GRPOTrainer`](https://huggingface.co/docs/trl/grpo_trainer)、[`DPOTrainer`](https://huggingface.co/docs/trl/dpo_trainer)、[`KTOTrainer`](https://huggingface.co/docs/trl/kto_trainer) 等。

- **高效与高可扩展性**：
  - 深度集成 [🤗 Accelerate](https://github.com/huggingface/accelerate)，支持通过 [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)、[DeepSpeed ZeRO](https://github.com/deepspeedai/DeepSpeed) 和 FSDP 从单卡无缝扩展至多节点分布式集群。
  - 全面整合 [🤗 PEFT](https://github.com/huggingface/peft)，支持模型量化与 LoRA / QLoRA，使消费级显卡也能轻松微调超大参数模型。
  - 集成 [🦥 Unsloth](https://github.com/unslothai/unsloth) 优化内核，大幅加速模型训练速度。

- **命令行工具 (CLI)**：提供直观简洁的 CLI 界面，无需编写代码即可快速启动模型微调。

## 安装指南

### Python 官方包安装

通过 `pip` 安装稳定版：

```bash
pip install trl
```

### 源码安装

若需使用官方尚未发布的最新特性，可直接从源码安装：

```bash
pip install git+https://github.com/huggingface/trl.git
```

### 源码克隆

若需要使用仓库中的丰富示例脚本：

```bash
git clone https://github.com/huggingface/trl.git
```

## 快速上手

为了在训练中获得更高的灵活性与控制力，TRL 提供了专门的 Trainer 类，用于在自定义数据集上后训练语言模型或 PEFT 适配器。TRL 中的每个 Trainer 都是对 🤗 Transformers Trainer 的轻量封装，原生支持 DDP、DeepSpeed ZeRO 和 FSDP 等分布式训练方式。

### `SFTTrainer` (监督微调)

以下是使用 [`SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer) 进行指令监督微调的基础示例：

```python
from trl import SFTTrainer
from datasets import load_dataset

dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
)
trainer.train()
```

### `GRPOTrainer` (群体相对策略优化)

[`GRPOTrainer`](https://huggingface.co/docs/trl/grpo_trainer) 实现了 [GRPO 算法（Group Relative Policy Optimization）](https://huggingface.co/papers/2402.03300)。相比传统 PPO 算法，GRPO 显著降低了显存占用（无需 Critic 价值模型），正是训练 [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) 核心推理能力的关键对齐技术。

```python
from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
```

> [!NOTE]
> 对于推理（Reasoning）模型训练，建议使用 `reasoning_accuracy_reward()` 奖励函数以获得更佳的推理链激发效果。

### `DPOTrainer` (直接偏好优化)

[`DPOTrainer`](https://huggingface.co/docs/trl/dpo_trainer) 实现了广受欢迎的 [DPO 算法（Direct Preference Optimization）](https://huggingface.co/papers/2305.18290)。该算法被广泛应用于 [Llama 3](https://huggingface.co/papers/2407.21783) 等顶级开源模型的后训练对齐。基础示例：

```python
from datasets import load_dataset
from trl import DPOTrainer

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = DPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=dataset,
)
trainer.train()
```

### `KTOTrainer` (Kahneman-Tversky 优化)

[`KTOTrainer`](https://huggingface.co/docs/trl/kto_trainer) 实现了 [KTO 算法（Kahneman-Tversky Optimization）](https://huggingface.co/papers/2402.01306)。它允许模型直接从简单的二元标签（好 / 坏，满意 / 不满意）进行对齐，而无需成对的偏好比较数据。基础示例：

```python
from datasets import load_dataset
from trl import KTOTrainer

dataset = load_dataset("trl-lib/kto-mix-14k", split="train")

trainer = KTOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=dataset,
)
trainer.train()
```

### `RewardTrainer` (奖励模型训练)

以下是使用 [`RewardTrainer`](https://huggingface.co/docs/trl/reward_trainer) 训练独立 Reward Model 奖励模型的基础示例：

```python
from trl import RewardTrainer
from datasets import load_dataset

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = RewardTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset,
)
trainer.train()
```

## 命令行工具 (CLI)

可以使用 TRL 命令行工具（CLI）无需编写代码直接启动监督微调（SFT）或偏好优化（DPO/KTO）：

**SFT (监督微调):**

```bash
trl sft --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset_name trl-lib/Capybara \
    --output_dir Qwen2.5-0.5B-SFT
```

**DPO (直接偏好优化):**

```bash
trl dpo --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name argilla/Capybara-Preferences \
    --output_dir Qwen2.5-0.5B-DPO 
```

**KTO (二元反馈对齐):**

```bash
trl kto --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name trl-lib/kto-mix-14k \
    --output_dir Qwen2.5-0.5B-KTO
```

了解更多 CLI 参数与用法请参阅 [CLI 官方文档](https://huggingface.co/docs/trl/clis) 或运行 `--help`。

## 本地开发与贡献

如果您希望向 `trl` 提交贡献或进行二次定制开发，请阅读[贡献指南 (CONTRIBUTING.md)](https://github.com/huggingface/trl/blob/main/CONTRIBUTING.md) 并进行可编辑模式安装：

```bash
git clone https://github.com/huggingface/trl.git
cd trl/
pip install -e .[dev]
```

## 实验性功能 (Experimental)

`trl.experimental` 模块提供了一个最小化的孵化试验区，用于存放处于早期迭代的实验性特性。此目录下的 API 可能会在任何后续版本中变动或移除，无需提前通知。

示例：

```python
from trl.experimental.new_trainer import NewTrainer
```

详情参阅[实验性功能文档](https://huggingface.co/docs/trl/experimental_overview)。

## 引用 TRL

如果您在学术研究或开源项目中使用了 TRL，请使用以下 BibTeX 条目进行引用：

```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```

## 开源许可证

本项目源码基于 [Apache-2.0 开源许可证](LICENSE) 发布。
