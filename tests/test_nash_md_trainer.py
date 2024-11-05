# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import tempfile
import unittest

from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl import NashMDConfig, NashMDTrainer, PairRMJudge

from .testing_utils import require_llm_blender


if is_peft_available():
    from peft import LoraConfig, get_peft_model


class TestNashMDTrainer(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/pythia-14m", num_labels=1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @parameterized.expand([("standard_prompt_only",), ("conversational_prompt_only",)])
    def test_nash_md_trainer_training(self, config_name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = NashMDConfig(
                output_dir=tmp_dir,
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
                reward_model=self.reward_model,
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

            trainer.train()

            # Check if training loss is available
            self.assertIn("train_loss", trainer.state.log_history[-1])

    @require_peft
    def test_training_with_peft(self):
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = NashMDConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                learning_rate=5.0e-7,
                eval_strategy="steps",
                report_to="none",
            )
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

            trainer = NashMDTrainer(
                model=self.model,
                reward_model=self.reward_model,
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                peft_config=lora_config,
            )

            trainer.train()

            # Check if training loss is available
            self.assertIn("train_loss", trainer.state.log_history[-1])

    @require_peft
    def test_training_with_peft_and_ref_model(self):
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = NashMDConfig(
                output_dir=tmp_dir,
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
                reward_model=self.reward_model,
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                peft_config=lora_config,
            )

            trainer.train()

            # Check if training loss is available
            self.assertIn("train_loss", trainer.state.log_history[-1])

    @require_peft
    def test_training_with_peft_model_and_peft_config(self):
        model_lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(self.model, model_lora_config)
        # we want only the "train adapter" to be trained
        lora_train_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = NashMDConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                learning_rate=5.0e-7,
                eval_strategy="steps",
                report_to="none",
            )
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

            trainer = NashMDTrainer(
                model=model,
                reward_model=self.reward_model,
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                peft_config=lora_train_config,
            )

            trainer.train()

            # Check if training loss is available
            self.assertIn("train_loss", trainer.state.log_history[-1])

    @parameterized.expand([("standard_prompt_only",), ("conversational_prompt_only",)])
    @require_llm_blender
    def test_nash_md_trainer_judge_training(self, config_name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = NashMDConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                report_to="none",
            )
            dummy_dataset = load_dataset("trl-internal-testing/zen", config_name)
            judge = PairRMJudge()

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
            self.assertIn("train_loss", trainer.state.log_history[-1])
