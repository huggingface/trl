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
from datasets import Dataset, load_dataset
from transformers import TrainerCallback

from trl.experimental.sdpo import SDPOConfig, SDPOTrainer

from ..testing_utils import TrlTestCase


class TeacherContextCaptureCallback(TrainerCallback):
    def __init__(self):
        self.captured_teacher_input_text = None
        self.captured_self_distillation_mask = None

    def on_teacher_context_built(
        self, processing_class=None, teacher_input_ids=None, self_distillation_mask=None, **kwargs
    ):
        if self.captured_teacher_input_text is None and teacher_input_ids is not None:
            self.captured_teacher_input_text = processing_class.decode(teacher_input_ids[0], skip_special_tokens=True)
        if self.captured_self_distillation_mask is None and self_distillation_mask is not None:
            self.captured_self_distillation_mask = self_distillation_mask.detach().cpu()


class TestSDPOTrainer(TrlTestCase):
    def test_training_with_required_dataset_columns(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2."],
                "privileged_context": ["Your earlier answer used the wrong format."],
            }
        )

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            generation_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            report_to="none",
            distillation_weight=1.0,
            distillation_alpha=1.0,
            distillation_topk=None,
            distillation_is_clip=None,
            include_environment_feedback=True,
            success_reward_threshold=1.0,
            max_steps=1,
            num_train_epochs=1,
        )

        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=lambda **kwargs: [0.0] * len(kwargs["prompts"]),
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_training(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
            distillation_weight=1.0,
            distillation_alpha=0.5,
            distillation_topk=5,
            full_logit_distillation=True,
            distillation_is_clip=None,
        )
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:
                assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12), f"Parameter {n} has not changed."

    def test_training_without_successful_rollouts(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
            distillation_weight=1.0,
            distillation_alpha=1.0,
            distillation_topk=None,
            distillation_is_clip=None,
            success_reward_threshold=1.0,
        )

        def zero_reward(**kwargs):
            prompts = kwargs["prompts"]
            return [0.0] * len(prompts)

        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=zero_reward,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_training_with_hybrid_policy_loss_mode(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
            distillation_weight=1.0,
            distillation_alpha=0.5,
            distillation_topk=5,
            full_logit_distillation=True,
            distillation_is_clip=None,
            sdpo_policy_loss_mode="hybrid",
        )
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_training_with_teacher_regularization_none(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
            distillation_weight=1.0,
            distillation_alpha=0.5,
            distillation_topk=5,
            full_logit_distillation=True,
            distillation_is_clip=None,
            teacher_regularization="none",
        )
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.teacher_model is None
        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_training_rejects_non_reverse_token_level_distillation(self):
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [
                        {"role": "system", "content": "You are a careful assistant."},
                        {"role": "user", "content": "Try the puzzle again."},
                    ]
                ],
                "privileged_context": ["Your earlier answer violated the format requirements."],
            }
        )

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            generation_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            report_to="none",
            distillation_weight=1.0,
            distillation_alpha=0.5,
            distillation_topk=5,
            full_logit_distillation=False,
            distillation_is_clip=None,
            success_reward_threshold=1.0,
            include_environment_feedback=True,
            max_steps=1,
            num_train_epochs=1,
        )

        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=lambda **kwargs: [0.0] * len(kwargs["prompts"]),
            args=training_args,
            train_dataset=dataset,
        )

        with pytest.raises(ValueError, match="Only reverse KL"):
            trainer.train()

    def test_training_with_conversational_prompts_preserves_context(self):
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [
                        {"role": "system", "content": "You are a careful assistant."},
                        {"role": "user", "content": "Solve 2+2."},
                    ]
                ]
            }
        )

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            generation_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            report_to="none",
            distillation_weight=1.0,
            distillation_alpha=1.0,
            distillation_topk=None,
            distillation_is_clip=None,
            success_reward_threshold=0.5,
            dont_reprompt_on_self_success=False,
            num_train_epochs=1,
            max_steps=1,
        )

        def alternating_reward(**kwargs):
            prompts = kwargs["prompts"]
            return [1.0 if i % 2 == 0 else 0.0 for i in range(len(prompts))]

        capture_callback = TeacherContextCaptureCallback()
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=alternating_reward,
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_teacher_input_text is not None
        assert "careful assistant" in capture_callback.captured_teacher_input_text
        assert "Solve 2+2" in capture_callback.captured_teacher_input_text
        assert capture_callback.captured_self_distillation_mask is not None
        assert capture_callback.captured_self_distillation_mask[0].item() == 1.0

    def test_training_with_feedback_only_reprompts_teacher(self):
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [
                        {"role": "system", "content": "You are a careful assistant."},
                        {"role": "user", "content": "Try the puzzle again."},
                    ]
                ],
                "privileged_context": ["Your earlier answer violated the format requirements."],
            }
        )

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            generation_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            report_to="none",
            distillation_weight=1.0,
            distillation_alpha=1.0,
            distillation_topk=None,
            distillation_is_clip=None,
            success_reward_threshold=1.0,
            include_environment_feedback=True,
            num_train_epochs=1,
            max_steps=1,
        )

        def zero_reward(**kwargs):
            prompts = kwargs["prompts"]
            return [0.0] * len(prompts)

        capture_callback = TeacherContextCaptureCallback()
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=zero_reward,
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_teacher_input_text is not None
        assert "format requirements" in capture_callback.captured_teacher_input_text
        assert capture_callback.captured_self_distillation_mask is not None
        assert capture_callback.captured_self_distillation_mask[0].item() == 1.0
