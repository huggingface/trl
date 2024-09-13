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

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl import OnlineDPOConfig, OnlineDPOTrainer


if is_peft_available():
    from peft import LoraConfig


class TestOnlineDPOTrainer(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/pythia-14m", num_labels=1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # fmt: off
        dummy_dataset_dict = {
            "prompt": [
                "hello",
                "how are you",
                "What is your name?",
                "What is your name?",
                "Which is the best programming language?",
                "Which is the best programming language?",
                "Which is the best programming language?",
                "[INST] How is the stock price? [/INST]",
                "[INST] How is the stock price? [/INST] ",
            ],
            "chosen": [
                "hi nice to meet you",
                "I am fine",
                "My name is Mary",
                "My name is Mary",
                "Python",
                "Python",
                "Python",
                "$46 as of 10am EST",
                "46 as of 10am EST",
            ],
            "rejected": [
                "leave me alone",
                "I am not fine",
                "Whats it to you?",
                "I dont have a name",
                "Javascript",
                "C++",
                "Java",
                " $46 as of 10am EST",
                " 46 as of 10am EST",
            ],
        }
        # fmt: on
        self.dummy_dataset = Dataset.from_dict(dummy_dataset_dict)

    def test_training(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = OnlineDPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                learning_rate=5.0e-7,
                eval_strategy="steps",
                report_to="none",
            )

            trainer = OnlineDPOTrainer(
                model=self.model,
                reward_model=self.reward_model,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=self.dummy_dataset,
                eval_dataset=self.dummy_dataset,
            )

            trainer.train()

            # Check if training loss is available
            self.assertIn("train_loss", trainer.state.log_history[-1])

    def test_training_with_ref_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = OnlineDPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                learning_rate=5.0e-7,
                eval_strategy="steps",
                report_to="none",
            )

            trainer = OnlineDPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=self.dummy_dataset,
                eval_dataset=self.dummy_dataset,
            )

            trainer.train()

            # Check if training loss is available
            self.assertIn("train_loss", trainer.state.log_history[-1])

    @require_peft
    def test_training_with_peft(self):
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = OnlineDPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                learning_rate=5.0e-7,
                eval_strategy="steps",
                report_to="none",
            )

            trainer = OnlineDPOTrainer(
                model=self.model,
                reward_model=self.reward_model,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=self.dummy_dataset,
                eval_dataset=self.dummy_dataset,
                peft_config=lora_config,
            )

            trainer.train()

            # Check if training loss is available
            self.assertIn("train_loss", trainer.state.log_history[-1])

    @require_peft
    def test_training_with_peft_and_ref_model(self):
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = OnlineDPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                learning_rate=5.0e-7,
                eval_strategy="steps",
                report_to="none",
            )

            trainer = OnlineDPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=self.dummy_dataset,
                eval_dataset=self.dummy_dataset,
                peft_config=lora_config,
            )

            trainer.train()

            # Check if training loss is available
            self.assertIn("train_loss", trainer.state.log_history[-1])
