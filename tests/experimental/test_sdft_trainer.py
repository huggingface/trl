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
from transformers import TrainerCallback
from transformers.utils import is_peft_available

from trl.experimental.sdft import SDFTConfig, SDFTTrainer

from ..testing_utils import TrlTestCase, require_liger_kernel, require_peft, require_torch_accelerator


if is_peft_available():
    from peft import LoraConfig


class SelfDistillationCaptureCallback(TrainerCallback):
    def __init__(self):
        self.captured_generation_prompts = None
        self.captured_old_per_token_logps = None
        self.captured_prompt_ids = None
        self.generation_batch_build_count = 0

    def on_generation_prompts_selected(self, generation_prompts=None, **kwargs):
        if self.captured_generation_prompts is None and generation_prompts is not None:
            self.captured_generation_prompts = generation_prompts

    def on_self_distillation_batch_prepared(self, old_per_token_logps=None, prompt_ids=None, **kwargs):
        if self.captured_old_per_token_logps is None and old_per_token_logps is not None:
            self.captured_old_per_token_logps = old_per_token_logps.detach().cpu()
        if self.captured_prompt_ids is None and prompt_ids is not None:
            self.captured_prompt_ids = prompt_ids.detach().cpu()

    def on_generation_batch_built(self, **kwargs):
        self.generation_batch_build_count += 1


class TestSDFTTrainer(TrlTestCase):
    @staticmethod
    def _trainable_param_snapshot(model):
        return {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}

    @staticmethod
    def _assert_any_trainable_param_changed(model, previous_trainable_params):
        assert any(
            not torch.allclose(previous_param, model.get_parameter(name), rtol=1e-12, atol=1e-12)
            for name, previous_param in previous_trainable_params.items()
        )

    def test_train(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Name the capital of France."],
                "privileged_context": [
                    "Example answer: 4.",
                    "Example answer: Paris.",
                ],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            report_to="none",
        )

        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )
        previous_trainable_params = self._trainable_param_snapshot(trainer.model)

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        self._assert_any_trainable_param_changed(trainer.model, previous_trainable_params)

    @require_liger_kernel
    @require_torch_accelerator
    def test_liger_loss_matches_non_liger_loss(self):
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2."], "privileged_context": ["Example answer: 4."]})
        common = dict(
            output_dir=self.tmp_dir,
            report_to="none",
            per_device_train_batch_size=1,
            max_completion_length=3,
            num_generations=1,
            distillation_mode="full_logits",
            distillation_is_clip=None,
            num_loss_tokens_to_skip=1,
        )

        ref_trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=SDFTConfig(use_liger_kernel=False, **common),
            train_dataset=dataset,
        )
        liger_trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=SDFTConfig(use_liger_kernel=True, **common),
            train_dataset=dataset,
        )

        liger_trainer.model.load_state_dict(ref_trainer.model.state_dict())
        torch.manual_seed(0)
        with torch.no_grad():
            for param in ref_trainer.teacher_model.parameters():
                param.add_(0.5 * torch.randn_like(param))
        liger_trainer.teacher_model.load_state_dict(ref_trainer.teacher_model.state_dict())

        device = next(ref_trainer.model.parameters()).device
        batch = {
            "prompt_ids": torch.tensor([[10, 11], [12, 13]], device=device),
            "prompt_mask": torch.tensor([[1, 1], [1, 1]], device=device),
            "completion_ids": torch.tensor([[14, 15, 16], [17, 18, 19]], device=device),
            "completion_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], device=device),
            "teacher_input_ids": torch.tensor([[20, 21, 22, 14, 15, 16], [23, 24, 25, 17, 18, 19]], device=device),
            "teacher_attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], device=device),
            "self_distillation_mask": torch.tensor([1.0, 0.0], device=device),
        }

        ref_trainer.model.eval()
        liger_trainer.model.eval()
        with torch.no_grad():
            ref_loss = ref_trainer.compute_loss(ref_trainer.model, batch).item()
            liger_loss = liger_trainer.compute_loss(liger_trainer.model, batch).item()

        torch.testing.assert_close(
            torch.tensor(liger_loss),
            torch.tensor(ref_loss),
            rtol=2e-2,
            atol=1e-6,
        )

    def test_train_rejects_none_privileged_context(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2."],
                "privileged_context": [None],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            report_to="none",
        )

        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        with pytest.raises(ValueError, match="`privileged_context` must not be None"):
            trainer.train()

    def test_train_with_generate_from_teacher(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Solve 3+3."],
                "privileged_context": [
                    "Teacher hint: answer with 4 and explain briefly.",
                    "Teacher hint: answer with 6 and explain briefly.",
                ],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            generate_from_teacher=True,
            report_to="none",
        )

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_generation_prompts == [
            "Solve 2+2.\n\nTeacher hint: answer with 4 and explain briefly."
        ]
        student_prompt_text = trainer.processing_class.decode(
            capture_callback.captured_prompt_ids[0],
            skip_special_tokens=True,
        )
        assert "Teacher hint" not in student_prompt_text
        assert "Solve 2+2." in student_prompt_text

    def test_train_with_chat_template_kwargs(self):
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [{"role": "user", "content": "Solve 2+2."}],
                    [{"role": "user", "content": "Solve 3+3."}],
                ],
                "privileged_context": [
                    "Teacher hint: answer with 4.",
                    "Teacher hint: answer with 6.",
                ],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            chat_template_kwargs={"enable_thinking": False},
            report_to="none",
        )

        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = self._trainable_param_snapshot(trainer.model)

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        self._assert_any_trainable_param_changed(trainer.model, previous_trainable_params)

    @require_peft
    def test_train_with_peft_model(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Name the capital of France."],
                "privileged_context": [
                    "Example answer: 4.",
                    "Example answer: Paris.",
                ],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            report_to="none",
        )

        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
            ),
        )

        previous_trainable_params = self._trainable_param_snapshot(trainer.model)

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        self._assert_any_trainable_param_changed(trainer.model, previous_trainable_params)

    @require_peft
    def test_train_with_peft_model_and_ema_teacher_sync(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Name the capital of France."],
                "privileged_context": [
                    "Example answer: 4.",
                    "Example answer: Paris.",
                ],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=2,
            num_generations=1,
            teacher_model_kind="ema",
            teacher_update_rate=0.05,
            teacher_sync_steps=1,
            report_to="none",
        )

        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
            ),
        )
        previous_trainable_params = self._trainable_param_snapshot(trainer.model)

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        self._assert_any_trainable_param_changed(trainer.model, previous_trainable_params)

    def test_train_populates_old_log_probs_for_distillation_clipping_when_misaligned(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Solve 3+3."],
                "privileged_context": [
                    "Example answer: 4.",
                    "Example answer: 6.",
                ],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=3,
            steps_per_generation=2,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            report_to="none",
        )

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_old_per_token_logps is not None

    def test_train_with_generate_from_teacher_skips_old_log_probs_for_distillation_clipping(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Solve 3+3."],
                "privileged_context": [
                    "Teacher hint: answer with 4.",
                    "Teacher hint: answer with 6.",
                ],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=3,
            steps_per_generation=2,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            generate_from_teacher=True,
            report_to="none",
        )

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_old_per_token_logps is None

    def test_train_reuses_buffered_generation_batches(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Solve 3+3."],
                "privileged_context": [
                    "Example answer: 4.",
                    "Example answer: 6.",
                ],
            }
        )

        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            steps_per_generation=2,
            max_completion_length=8,
            max_steps=2,
            num_generations=1,
            report_to="none",
        )

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.generation_batch_build_count == 1
