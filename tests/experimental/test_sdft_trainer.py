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

import torch
from datasets import Dataset
from transformers import TrainerCallback
from transformers.utils import is_peft_available

from trl.experimental.sdft import SDFTConfig, SDFTTrainer

from ..testing_utils import TrlTestCase


if is_peft_available():
    from peft import LoraConfig


class GenerationPromptCaptureCallback(TrainerCallback):
    def __init__(self):
        self.captured_generation_prompt_text = None

    def on_generation_prompts_selected(self, generation_prompt_text=None, **kwargs):
        if self.captured_generation_prompt_text is None and generation_prompt_text is not None:
            self.captured_generation_prompt_text = generation_prompt_text[0]


class TestSDFTTrainer(TrlTestCase):
    def test_training_with_required_dataset_columns(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Name the capital of France."],
                "privileged_context": [
                    "Solve 2+2. Example answer: 4.",
                    "Name the capital of France. Example answer: Paris.",
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
            ref_model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {name: param.clone() for name, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for name, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(name)
            if param.sum() != 0:
                assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12), (
                    f"Parameter {name} has not changed."
                )

    def test_training_with_generate_from_teacher(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Solve 3+3."],
                "privileged_context": [
                    "Solve 2+2. Teacher hint: answer with 4 and explain briefly.",
                    "Solve 3+3. Teacher hint: answer with 6 and explain briefly.",
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
            generate_from_teacher=True,
        )

        capture_callback = GenerationPromptCaptureCallback()
        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            ref_model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_generation_prompt_text is not None
        assert "Teacher hint" in capture_callback.captured_generation_prompt_text

    def test_training_with_peft_model_and_no_explicit_ref_model(self):
        if not is_peft_available():
            self.skipTest("PEFT is not available")

        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Name the capital of France."],
                "privileged_context": [
                    "Solve 2+2. Example answer: 4.",
                    "Name the capital of France. Example answer: Paris.",
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
            ref_model=None,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
            ),
        )

        assert trainer.ref_model is None

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
