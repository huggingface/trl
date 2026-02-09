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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig
from transformers.utils import is_peft_available

from trl.experimental.nash_md import NashMDConfig, NashMDTrainer
from trl.experimental.nash_md.nash_md_trainer import GeometricMixtureWrapper
from trl.models.utils import create_reference_model

from ..testing_utils import TrlTestCase, require_llm_blender, require_peft
from .testing_utils import RandomPairwiseJudge


if is_peft_available():
    from peft import LoraConfig, get_peft_model


class TestGeometricMixtureWrapper(TrlTestCase):
    def setup_method(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32").to(self.device)
        self.ref_model = create_reference_model(self.model).to(self.device)
        self.generation_config = GenerationConfig.from_pretrained(model_id)
        self.mixture_coef = 0.5
        self.wrapper = GeometricMixtureWrapper(
            self.model, self.ref_model, self.generation_config, mixture_coef=self.mixture_coef
        )

    def test_forward(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        output = self.wrapper(input_ids=input_ids, attention_mask=attention_mask)

        assert output is not None
        assert hasattr(output, "logits")
        assert output.logits.shape == (1, 5, self.model.config.vocab_size)

    def test_mixture_coefficient(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            ref_model_output = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            wrapper_output = self.wrapper(input_ids=input_ids, attention_mask=attention_mask)

        expected_logits = torch.nn.functional.log_softmax(
            self.mixture_coef * ref_model_output.logits + (1 - self.mixture_coef) * model_output.logits, dim=-1
        )

        torch.testing.assert_close(wrapper_output.logits, expected_logits)

    def test_prepare_inputs_for_generation(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        inputs = self.wrapper.prepare_inputs_for_generation(input_ids, attention_mask=attention_mask, use_cache=True)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert not inputs.get("use_cache", False)


class TestNashMDTrainer(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="float32")
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    def test_nash_md_trainer_training(self, config_name):
        training_args = NashMDConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
            learning_rate=9e-1,
            eval_strategy="steps",
            report_to="none",
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", config_name)

        trainer = NashMDTrainer(
            model=self.model,
            ref_model=self.ref_model,
            reward_funcs=self.reward_model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
        )

        trainer.train()

        # Check if training loss is available
        assert "train_loss" in trainer.state.log_history[-1]

    @require_peft
    def test_training_with_peft(self):
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        training_args = NashMDConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            eval_strategy="steps",
            report_to="none",
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        trainer = NashMDTrainer(
            model=self.model,
            reward_funcs=self.reward_model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
            peft_config=lora_config,
        )

        trainer.train()

        # Check if training loss is available
        assert "train_loss" in trainer.state.log_history[-1]

    @require_peft
    def test_training_with_peft_and_ref_model(self):
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        training_args = NashMDConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            eval_strategy="steps",
            report_to="none",
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        trainer = NashMDTrainer(
            model=self.model,
            ref_model=self.ref_model,
            reward_funcs=self.reward_model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
            peft_config=lora_config,
        )

        trainer.train()

        # Check if training loss is available
        assert "train_loss" in trainer.state.log_history[-1]

    @require_peft
    def test_training_pre_pefted_model_implicit_ref_with_reward_model(self):
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
        # self.model from setUp is a base AutoModelForCausalLM
        peft_model_instance = get_peft_model(self.model, lora_config)

        training_args = NashMDConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,  # Keep small for quick test
            max_steps=2,  # Few steps
            learning_rate=5.0e-7,
            eval_strategy="no",
            report_to="none",
            remove_unused_columns=False,  # Important for the dummy dataset
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")["train"]

        trainer = NashMDTrainer(
            model=peft_model_instance,  # Pass the already PEFT model
            ref_model=None,  # Implicit reference from peft_model_instance's base
            reward_funcs=self.reward_model,  # To trigger GeometricMixtureWrapper path
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dummy_dataset,
            # peft_config is not passed, as model is already PEFT
        )

        trainer.train()

        assert "train_loss" in trainer.state.log_history[-1]

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    @require_llm_blender
    def test_nash_md_trainer_judge_training(self, config_name):
        training_args = NashMDConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
            learning_rate=9e-1,
            eval_strategy="steps",
            report_to="none",
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", config_name)
        judge = RandomPairwiseJudge()

        trainer = NashMDTrainer(
            model=self.model,
            ref_model=self.ref_model,
            judge=judge,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
        )

        trainer.train()

        # Check if training loss is available
        assert "train_loss" in trainer.state.log_history[-1]
