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

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from datasets import Dataset

from trl.experimental.self_distillation.base_self_distillation_trainer import (
    BaseSelfDistillationTrainer,
    DistillationLogits,
)
from trl.experimental.self_distillation.self_distillation_config import SelfDistillationConfig

from ..testing_utils import TrlTestCase


class MinimalSelfDistillationTrainer(BaseSelfDistillationTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del inputs, num_items_in_batch
        anchor = next(model.parameters())
        return anchor.sum() * 0.0


class FakeTextTokenizer:
    def __call__(self, text, **kwargs):
        del kwargs
        token_map = {
            "short prompt": [1, 2, 3],
            "long prompt": [10, 11, 12, 13, 14],
        }
        return {"input_ids": [token_map[prompt] for prompt in text]}


class FakeChatProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, conversation, add_generation_prompt, tokenize, return_dict, **kwargs):
        self.calls.append(
            {
                "conversation": conversation,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "return_dict": return_dict,
                "kwargs": kwargs,
            }
        )
        return {"input_ids": [[21, 22, 23, 24]]}


class TestBaseSelfDistillationTrainer(TrlTestCase):
    @staticmethod
    def _make_loss_test_trainer(**args_overrides):
        trainer = object.__new__(MinimalSelfDistillationTrainer)
        args = {
            "distillation_topk": None,
            "full_logit_distillation": False,
            "distillation_alpha": 1.0,
            "distillation_add_tail": False,
            "distillation_is_clip": None,
        }
        args.update(args_overrides)
        trainer.args = SimpleNamespace(**args)
        trainer.loss_type = "dapo"
        trainer.max_completion_length = 2
        trainer.accelerator = SimpleNamespace(gather=lambda tensor: tensor)
        trainer._metrics = {
            "train": defaultdict(list),
            "eval": defaultdict(list),
        }
        trainer._name = "Minimal Self Distillation"
        return trainer

    def test_teacher_model_kind_live_uses_student_model(self):
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2."]})
        training_args = SelfDistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            teacher_model_kind="live",
            report_to="none",
        )

        trainer = MinimalSelfDistillationTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        assert trainer.teacher_model is trainer.model

    @pytest.mark.parametrize("teacher_model_kind", ["base", "ema"])
    def test_teacher_model_kind_base_and_ema_use_frozen_teacher_copy(self, teacher_model_kind):
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2."]})
        training_args = SelfDistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            teacher_model_kind=teacher_model_kind,
            report_to="none",
        )

        trainer = MinimalSelfDistillationTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        assert trainer.teacher_model is not trainer.model
        assert trainer.teacher_model.training is False

        student_param = next(trainer.model.parameters())
        teacher_param = next(trainer.teacher_model.parameters())
        assert teacher_param.requires_grad is False
        assert teacher_param.data_ptr() != student_param.data_ptr()

    def test_tokenize_prompts_truncates_text_prompts_from_left(self):
        trainer = object.__new__(MinimalSelfDistillationTrainer)
        trainer.processing_class = FakeTextTokenizer()
        trainer.max_prompt_length = 3

        prompt_ids = trainer._tokenize_prompts(["long prompt", "short prompt"])

        assert prompt_ids == [[12, 13, 14], [1, 2, 3]]

    def test_tokenize_prompts_for_conversational_prompts_forwards_chat_template_kwargs(self):
        trainer = object.__new__(MinimalSelfDistillationTrainer)
        trainer.processing_class = FakeChatProcessor()
        trainer.max_prompt_length = 2
        trainer.chat_template_kwargs = {"enable_thinking": False}

        prompt_ids = trainer._tokenize_prompts([[{"role": "user", "content": "Solve 2+2."}]])

        assert prompt_ids == [[23, 24]]
        assert trainer.processing_class.calls == [
            {
                "conversation": [[{"role": "user", "content": "Solve 2+2."}]],
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "kwargs": {"enable_thinking": False},
            }
        ]

    def test_prepare_inputs_reuses_buffered_generation_batches_within_window(self):
        trainer = object.__new__(MinimalSelfDistillationTrainer)
        trainer.model = SimpleNamespace(training=True)
        trainer.args = SimpleNamespace(steps_per_generation=2)
        trainer.num_iterations = 1
        trainer._step = 0
        trainer._buffered_inputs = None
        trainer.callback_handler = SimpleNamespace(callbacks=[])
        trainer.state = SimpleNamespace()
        trainer.control = SimpleNamespace()
        trainer.processing_class = None
        trainer._prepare_training_batch = Mock(
            side_effect=[
                {"value": torch.tensor([[1.0], [2.0]])},
                {"value": torch.tensor([[3.0], [4.0]])},
            ]
        )

        first_batch = trainer._prepare_inputs([{"prompt": "first"}])

        trainer._step = 1
        second_batch = trainer._prepare_inputs([{"prompt": "second"}])

        trainer._step = 2
        third_batch = trainer._prepare_inputs([{"prompt": "third"}])

        assert first_batch["value"].item() == 1.0
        assert second_batch["value"].item() == 2.0
        assert third_batch["value"].item() == 3.0
        assert trainer._prepare_training_batch.call_count == 2

    def test_compute_self_distillation_loss_ignores_masked_completion_tokens(self):
        trainer = self._make_loss_test_trainer(
            full_logit_distillation=True,
            distillation_alpha=0.0,
        )
        model = SimpleNamespace(training=True)

        # Token 0 is active and has a known non-zero divergence.
        # Token 1 is intentionally very different but masked out, so it must not affect the loss.
        student_probs = torch.tensor([[[0.8, 0.2], [0.01, 0.99]]], dtype=torch.float32)
        teacher_probs = torch.tensor([[[0.5, 0.5], [0.99, 0.01]]], dtype=torch.float32)
        distillation_logits = DistillationLogits(
            completion_ids=torch.tensor([[0, 1]], dtype=torch.long),
            completion_mask=torch.tensor([[1, 1]], dtype=torch.long),
            response_mask=torch.tensor([[1, 0]], dtype=torch.long),
            student_logits=student_probs.log(),
            teacher_logits=teacher_probs.log(),
        )

        loss = trainer._compute_self_distillation_loss(model, {}, distillation_logits)

        expected_active_token_loss = teacher_probs[0, 0, 0] * (
            teacher_probs[0, 0, 0].log() - student_probs[0, 0, 0].log()
        ) + teacher_probs[0, 0, 1] * (teacher_probs[0, 0, 1].log() - student_probs[0, 0, 1].log())
        torch.testing.assert_close(loss, expected_active_token_loss)
        torch.testing.assert_close(
            torch.tensor(trainer._metrics["train"]["self_distillation/distillation_loss"]),
            expected_active_token_loss.unsqueeze(0),
        )

    def test_compute_self_distillation_loss_applies_importance_sampling_clip(self):
        trainer = self._make_loss_test_trainer(distillation_is_clip=2.0)
        model = SimpleNamespace(training=True)

        student_token_probs = torch.tensor([[0.2, 0.4]], dtype=torch.float32)
        teacher_token_probs = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        old_token_probs = torch.tensor([[0.05, 0.4]], dtype=torch.float32)
        clip_coeff = trainer.args.distillation_is_clip

        distillation_logits = DistillationLogits(
            completion_ids=torch.tensor([[0, 1]], dtype=torch.long),
            completion_mask=torch.tensor([[1, 1]], dtype=torch.long),
            response_mask=torch.tensor([[1, 1]], dtype=torch.long),
            student_logits=torch.log(torch.tensor([[[0.2, 0.8], [0.6, 0.4]]], dtype=torch.float32)),
            teacher_logits=torch.log(torch.tensor([[[0.5, 0.5], [0.5, 0.5]]], dtype=torch.float32)),
        )

        loss = trainer._compute_self_distillation_loss(
            model,
            {"old_per_token_logps": old_token_probs.log()},
            distillation_logits,
        )

        raw_per_token_loss = (student_token_probs.log() - teacher_token_probs.log()) * student_token_probs.log()
        clipped_ratio = torch.minimum(
            student_token_probs / old_token_probs, torch.full_like(student_token_probs, clip_coeff)
        )
        expected_loss = (raw_per_token_loss * clipped_ratio).mean()

        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(
            torch.tensor(trainer._metrics["train"]["self_distillation/distillation_loss"]),
            expected_loss.unsqueeze(0),
        )
