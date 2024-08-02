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

from trl import OnlineDPOConfig, OnlineDPOTrainer


class TestOnlineDPOTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        cls.reward_model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/pythia-14m", num_labels=1)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token

    def _get_dummy_model_and_tokenizer(self):
        # Return dummy model and tokenizer. This is a placeholder.
        return self.model, self.tokenizer, self.reward_model

    def _init_dummy_dataset(self):
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
        return Dataset.from_dict(dummy_dataset_dict)

    def test_online_dpo_trainer_training(self):
        model, tokenizer, reward_model = self._get_dummy_model_and_tokenizer()
        dummy_dataset = self._init_dummy_dataset()

        def tokenize(element):
            outputs = tokenizer(
                element["prompt"],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        dummy_dataset = dummy_dataset.map(
            tokenize,
            remove_columns=dummy_dataset.column_names,
            batched=True,
            num_proc=4,  # multiprocessing.cpu_count(),
            load_from_cache_file=False,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = OnlineDPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
            )

            trainer = OnlineDPOTrainer(
                model=model,
                ref_model=model,
                reward_model=reward_model,
                config=training_args,
                tokenizer=tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
            )

            trainer.train()

            # Check if training loss is available
            self.assertIn("loss/policy_avg", trainer.state.log_history[-1])
