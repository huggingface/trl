# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""
最小可复现 smoke 训练脚本（1 step）：

python examples/scripts/privileged_gkd_smoke.py \
    --output_dir /tmp/privileged-gkd-smoke \
    --max_steps 1
"""

import argparse

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from trl.experimental.gkd import PrivilegedGKDConfig, PrivilegedSelfDistillTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal Privileged GKD smoke training.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model path used for both student and teacher.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/privileged-gkd-smoke",
        help="Output directory for trainer artifacts.",
    )
    parser.add_argument("--max_steps", type=int, default=1, help="Number of training steps for smoke run.")
    return parser.parse_args()


def _capture_trainable_param_slice(
    model: torch.nn.Module, max_elems: int = 2048
) -> tuple[str, int, torch.Tensor]:
    # 只抽样一个可训练参数切片，避免复制整层大参数导致额外显存开销
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        flat = param.detach().float().view(-1)
        if flat.numel() == 0:
            continue
        slice_len = min(max_elems, flat.numel())
        return name, slice_len, flat[:slice_len].clone()
    raise ValueError("未找到可训练参数，无法做参数更新检查。")


def main() -> None:
    args = parse_args()

    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "解释快速排序"},
                    {"role": "assistant", "content": "快速排序通过 pivot 划分并递归排序。"},
                ],
                "privileged_messages": [
                    {"role": "user", "content": "请补充复杂度信息"},
                    {"role": "assistant", "content": "平均 O(nlogn)，最坏 O(n^2)。"},
                    {"role": "user", "content": "再补充优化建议"},
                    {"role": "assistant", "content": "可使用随机 pivot 降低最坏情况概率。"},
                ],
            },
            {
                "messages": [
                    {"role": "user", "content": "解释牛顿法"},
                    {"role": "assistant", "content": "牛顿法通过切线迭代逼近方程根。"},
                ],
                "privileged_messages": [
                    {"role": "user", "content": "补充收敛条件"},
                    {"role": "assistant", "content": "在根附近且导数不为 0 时可二阶收敛。"},
                ],
            },
        ]
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = PrivilegedGKDConfig(
        output_dir=args.output_dir,
        report_to="none",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        eval_strategy="no",
        save_strategy="no",
        max_steps=args.max_steps,
        max_length=128,
        max_new_tokens=1024,
        lmbda=1.0,
        share_student_as_teacher=True,
        privileged_key="privileged_messages",
        rollout_log_steps=1,
        rollout_log_samples=1,
        rollout_log_max_new_tokens=64,
        rollout_log_max_chars=1200,
        debug_log_loss_steps=1,
        debug_log_grad_norm=True,
    )

    trainer = PrivilegedSelfDistillTrainer(
        model=args.model_name_or_path,
        teacher_model=args.model_name_or_path,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    tracked_name, tracked_slice_len, tracked_before = _capture_trainable_param_slice(trainer.model)
    train_result = trainer.train()
    metrics = train_result.metrics

    tracked_after = dict(trainer.model.named_parameters())[tracked_name].detach().float().view(-1)[:tracked_slice_len]
    delta = tracked_after - tracked_before
    print(
        "PARAM_UPDATE_CHECK",
        {
            "param": tracked_name,
            "slice_len": tracked_slice_len,
            "delta_l2": float(delta.norm().item()),
            "max_abs_delta": float(delta.abs().max().item()),
        },
    )
    print("SMOKE_RUN_OK", metrics)


if __name__ == "__main__":
    main()
