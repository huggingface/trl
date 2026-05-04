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
from transformers import AutoModelForCausalLM, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.utils import is_peft_available

from trl.data_utils import maybe_apply_chat_template
from trl.experimental.sdft import SDFTConfig, SDFTTrainer

from ..testing_utils import TrlTestCase, require_peft


if is_peft_available():
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

    from trl.experimental.self_distillation.peft_adapter_ema_callback import PEFTAdapterEMACallback


class SelfDistillationCaptureCallback(TrainerCallback):
    def __init__(self):
        self.captured_generation_prompt_text = None
        self.captured_old_per_token_logps = None
        self.generation_batch_build_count = 0

    def on_generation_prompts_selected(self, generation_prompt_text=None, **kwargs):
        if self.captured_generation_prompt_text is None and generation_prompt_text is not None:
            self.captured_generation_prompt_text = generation_prompt_text[0]

    def on_self_distillation_batch_prepared(self, old_per_token_logps=None, **kwargs):
        if self.captured_old_per_token_logps is None and old_per_token_logps is not None:
            self.captured_old_per_token_logps = old_per_token_logps.detach().cpu()

    def on_generation_batch_built(self, **kwargs):
        self.generation_batch_build_count += 1


class TestSDFTTrainer(TrlTestCase):
    def test_training_rejects_none_privileged_context(self):
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
        )

        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        with pytest.raises(ValueError, match="`privileged_context` must not be None"):
            trainer.train()

    def test_training_with_generate_from_teacher(self):
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
        )

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_generation_prompt_text is not None
        assert "Solve 2+2." in capture_callback.captured_generation_prompt_text
        assert "Teacher hint" in capture_callback.captured_generation_prompt_text

    def test_training_with_chat_template_kwargs(self):
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
        )

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        expected_prompt = maybe_apply_chat_template(
            {"prompt": dataset[0]["prompt"]},
            trainer.processing_class,
            **training_args.chat_template_kwargs,
        )["prompt"]

        trainer.train()

        assert capture_callback.captured_generation_prompt_text == expected_prompt

    @require_peft
    def test_training_with_peft_model(self):
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

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    @require_peft
    def test_training_with_peft_model_and_sync_ref_model(self):
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
            sync_ref_model=True,
            ref_model_mixup_alpha=0.05,
            ref_model_sync_steps=1,
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

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    @require_peft
    def test_peft_adapter_ema_callback(self):
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            device_map="cpu",
        )
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
            r=8,
        )
        model = get_peft_model(model, lora_config, adapter_name="default")

        update_rate = 0.5
        callback = PEFTAdapterEMACallback(
            model=model,
            teacher_adapter_name="teacher",
            update_rate=update_rate,
            sync_steps=1,
        )

        # Initialize and verify teacher adapter was created with zero weights
        callback._initialize_teacher_adapter()
        assert "teacher" in model.peft_config
        assert callback.shadow_weights is not None

        teacher_state = get_peft_model_state_dict(model, adapter_name="teacher")
        for key, param in teacher_state.items():
            assert torch.all(param == 0), f"Teacher param {key} should be zero-initialized"

        # Verify shadow weights keys match student state dict keys
        student_state = {k: v.clone() for k, v in get_peft_model_state_dict(model, adapter_name="default").items()}
        assert set(callback.shadow_weights.keys()) == set(student_state.keys())

        # Simulate a training step and verify EMA update
        args = TrainingArguments(output_dir=self.tmp_dir)
        state = TrainerState(global_step=1)
        control = TrainerControl()
        callback.on_step_end(args, state, control)

        # shadow = (1 - rate) * 0 + rate * student = rate * student
        for key in callback.shadow_weights:
            expected = update_rate * student_state[key]
            torch.testing.assert_close(callback.shadow_weights[key], expected)

        # Verify teacher adapter received the shadow weights
        teacher_state = get_peft_model_state_dict(model, adapter_name="teacher")
        for key in teacher_state:
            torch.testing.assert_close(teacher_state[key].float(), callback.shadow_weights[key])

    def test_training_populates_old_log_probs_for_distillation_clipping_when_misaligned(self):
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

    def test_training_reuses_buffered_generation_batches(self):
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
