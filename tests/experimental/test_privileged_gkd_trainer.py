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

import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from trl.experimental.gkd import (
    DataCollatorForPrivilegedGKD,
    PrivilegedGKDConfig,
    PrivilegedSelfDistillTrainer,
)

from ..testing_utils import TrlTestCase


class TestDataCollatorForPrivilegedGKD(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def test_collator_returns_pi_tensors(self):
        collator = DataCollatorForPrivilegedGKD(
            tokenizer=self.tokenizer,
            max_length=128,
            privileged_key="privileged_messages",
            max_privileged_length=32,
        )
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "什么是量子纠缠？"},
                    {"role": "assistant", "content": "量子纠缠是量子态之间的关联。"},
                ],
                "privileged_messages": [
                    {"role": "user", "content": "补充非定域性解释。"},
                    {"role": "assistant", "content": "纠缠态存在超越经典局域模型的关联。"},
                ],
            },
            {
                "messages": [
                    {"role": "user", "content": "解释一下贝叶斯公式。"},
                    {"role": "assistant", "content": "贝叶斯公式用于更新后验概率。"},
                ],
                "privileged_messages": [
                    {"role": "user", "content": "请给出符号定义。"},
                    {"role": "assistant", "content": "定义先验、似然和后验三个项。"},
                    {"role": "user", "content": "再给个例子。"},
                    {"role": "assistant", "content": "可用医疗诊断中的条件概率作为示例。"},
                ],
            },
        ]
        batch = collator(examples)
        assert "pi_input_ids" in batch
        assert "pi_attention_mask" in batch
        assert batch["pi_input_ids"].shape == batch["pi_attention_mask"].shape
        assert batch["pi_input_ids"].shape[0] == len(examples)

    def test_collator_rejects_nested_privileged_groups(self):
        collator = DataCollatorForPrivilegedGKD(
            tokenizer=self.tokenizer,
            max_length=128,
            privileged_key="privileged_messages",
            max_privileged_length=32,
        )
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "解释牛顿法。"},
                    {"role": "assistant", "content": "牛顿法使用局部线性近似。"},
                ],
                "privileged_messages": [
                    [{"role": "user", "content": "这是不再支持的多组格式。"}],
                    [{"role": "assistant", "content": "应改为单组 message 列表。"}],
                ],
            }
        ]
        with pytest.raises(ValueError, match="必须是字符串或消息列表"):
            collator(examples)


class TestPrivilegedSelfDistillTrainer(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def test_compute_loss_runs(self):
        train_dataset = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "给一个快速排序的思路。"},
                        {"role": "assistant", "content": "快速排序通过分治法递归排序。"},
                    ],
                    "privileged_messages": [
                        {"role": "user", "content": "请补充 pivot 策略。"},
                        {"role": "assistant", "content": "可选随机 pivot 或三数取中。"},
                    ],
                },
                {
                    "messages": [
                        {"role": "user", "content": "牛顿法为什么收敛快？"},
                        {"role": "assistant", "content": "在满足条件时牛顿法具有二阶收敛。"},
                    ],
                    "privileged_messages": [
                        {"role": "user", "content": "补充收敛域。"},
                        {"role": "assistant", "content": "初始点需位于足够接近根的邻域。"},
                        {"role": "user", "content": "补充条件。"},
                        {"role": "assistant", "content": "函数需二阶可导且导数不为零。"},
                    ],
                },
            ]
        )

        args = PrivilegedGKDConfig(
            output_dir=self.tmp_dir,
            report_to="none",
            per_device_train_batch_size=2,
            max_length=128,
            lmbda=0.0,
            max_new_tokens=16,
        )
        trainer = PrivilegedSelfDistillTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=args,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
        )

        batch = next(iter(trainer.get_train_dataloader()))
        batch = trainer._prepare_inputs(batch)
        loss = trainer.compute_loss(trainer.model, batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)

    def test_completion_mask_for_offpolicy_and_onpolicy(self):
        prompts = torch.zeros((1, 4), dtype=torch.long)

        offpolicy_inputs = {
            "prompts": prompts,
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
            "labels": torch.tensor([[-100, -100, -100, -100, 101, 102]], dtype=torch.long),
        }
        offpolicy_mask = PrivilegedSelfDistillTrainer._get_completion_mask(offpolicy_inputs)
        assert torch.equal(offpolicy_mask, torch.tensor([[False, False, False, False, True, True]]))

        onpolicy_inputs = {
            "prompts": prompts,
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
            "labels": torch.tensor([[11, 12, 13, 14, 101, 102]], dtype=torch.long),
        }
        onpolicy_mask = PrivilegedSelfDistillTrainer._get_completion_mask(onpolicy_inputs)
        assert torch.equal(onpolicy_mask, torch.tensor([[False, False, False, False, True, True]]))

    def test_training_step_with_rollout_logs_runs(self):
        train_dataset = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "解释什么是二分查找。"},
                        {"role": "assistant", "content": "二分查找每次将搜索区间折半。"},
                    ],
                    "privileged_messages": [
                        {"role": "user", "content": "补充前置条件。"},
                        {"role": "assistant", "content": "要求输入序列已排序。"},
                    ],
                }
            ]
        )

        args = PrivilegedGKDConfig(
            output_dir=self.tmp_dir,
            report_to="none",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=1,
            max_length=128,
            max_new_tokens=16,
            lmbda=1.0,
            rollout_log_steps=1,
            rollout_log_samples=1,
            rollout_log_max_new_tokens=8,
            eval_strategy="no",
            save_strategy="no",
        )

        trainer = PrivilegedSelfDistillTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=args,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
        )
        train_result = trainer.train()
        assert "train_loss" in train_result.metrics
