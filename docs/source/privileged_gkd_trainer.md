# Privileged GKD 训练器

[![model badge](https://img.shields.io/badge/All_models-Privileged_GKD-blue)](https://huggingface.co/models?other=trl)

## 概述

`PrivilegedSelfDistillTrainer` 是对 [`experimental.gkd.GKDTrainer`] 的扩展，用于带特权信息（PI, privileged information）的 on-policy 自蒸馏。

相比原始 GKD：

1. **Student 分支**只看到 prompt + completion。
2. **Teacher 分支**看到 prompt + PI + completion。
3. **损失函数**仍然是在 completion token 上计算 generalized JSD，因此训练行为与 GKD 接近，同时允许 teacher 利用额外上下文。

当你可以为每条样本准备额外指导信息（例如 critique、专家提示、草稿改进建议），且这些信息不希望在推理时暴露给模型时，这个训练器很有用。

> [!NOTE]
> 该训练器属于 experimental API，后续可能在无提前通知的情况下调整。

## 使用建议

训练数据需要包含常规对话字段，以及一个额外 PI 字段（默认键名：`privileged_messages`）。

```python
from datasets import Dataset
from trl.experimental.gkd import PrivilegedGKDConfig, PrivilegedSelfDistillTrainer

dataset = Dataset.from_list(
    [
        {
            "messages": [
                {"role": "user", "content": "Explain Newton's method."},
                {"role": "assistant", "content": "Newton's method iteratively uses local slope information."},
            ],
            "privileged_messages": [
                {"role": "user", "content": "补充收敛条件"},
                {"role": "assistant", "content": "在根附近且导数不为 0 时可实现二阶收敛。"},
            ],
        }
    ]
)

config = PrivilegedGKDConfig(
    output_dir="privileged-gkd-model",
    lmbda=1.0,
    share_student_as_teacher=True,
    privileged_key="privileged_messages",
    max_privileged_length=512,
)

trainer = PrivilegedSelfDistillTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    teacher_model="Qwen/Qwen2-0.5B-Instruct",
    args=config,
    train_dataset=dataset,
)
trainer.train()
```

关键参数：

* `privileged_key`：数据集中 PI 文本的字段名。
* `max_privileged_length`：PI tokenization 后的最大长度。
* `share_student_as_teacher`：teacher 与 student 是否共享同一套模型参数。
* `rollout_log_steps`：按 step 打印 student/teacher rollout 日志（`0` 为关闭）。
* `rollout_log_samples`：每次日志打印的样本数量。
* `rollout_log_max_new_tokens`：teacher rollout 日志生成长度上限。
* `rollout_log_max_chars`：日志文本最大字符数（超长截断）。
* 标准 GKD 参数（`lmbda`、`beta`、`temperature`、`max_new_tokens`）继承自 [`experimental.gkd.GKDConfig`]。

## 期望数据格式

每条样本应包含：

* `messages`：对话训练样本（ChatML 风格）。
* `privileged_messages`（或你在配置里指定的键名）：仅供 teacher 分支使用的 PI 信息。

最小示例：

```python
{
    "messages": [
        {"role": "user", "content": "What is quicksort?"},
        {"role": "assistant", "content": "Quicksort partitions the array around a pivot."},
    ],
    "privileged_messages": [
        {"role": "user", "content": "补充复杂度"},
        {"role": "assistant", "content": "平均 O(nlogn)，最坏 O(n^2)。"},
        {"role": "user", "content": "补充优化建议"},
        {"role": "assistant", "content": "可使用随机 pivot 降低最坏情况概率。"},
    ],
}
```

## 示例脚本

当前提供两个脚本：

* [`examples/scripts/privileged_gkd.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/privileged_gkd.py)：基于 Hub 数据集进行完整训练。
* [`examples/scripts/privileged_gkd_smoke.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/privileged_gkd_smoke.py)：使用内存数据进行单步 smoke 验证。

## PrivilegedSelfDistillTrainer API

[[autodoc]] experimental.gkd.PrivilegedSelfDistillTrainer
    - train
    - save_model
    - push_to_hub

## PrivilegedGKDConfig API

[[autodoc]] experimental.gkd.PrivilegedGKDConfig

## DataCollatorForPrivilegedGKD API

[[autodoc]] experimental.gkd.DataCollatorForPrivilegedGKD
