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

from trl import VASTrainer, VASConfig
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


class TestVASTrainer(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        self.value_model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=1)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/pythia-14m", num_labels=1)
        self.reward_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
        self.reward_tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
        self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @parameterized.expand([("standard_prompt_only",)])
    def test_training(self, config_name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = VASConfig(
                output_dir=tmp_dir,
                lam=0.95,
                batch_size=1,
                total_episodes=10,
                learning_rate=5.0e-7,
                per_device_eval_batch_size=2,
                report_to="none",
            )
            dummy_dataset = load_dataset("trl-internal-testing/zen", config_name)
            def prepare_dataset(dataset, tokenizer):
                """pre-tokenize the dataset before training; only collate during training"""

                def tokenize(element):
                    outputs = tokenizer(
                        element["prompt"],
                        padding=False,
                    )
                    return {"input_ids": outputs["input_ids"]}

                return dataset.map(
                    tokenize,
                    batched=True,
                    remove_columns=dataset.column_names,
                    num_proc=training_args.dataset_num_proc,
                )


            dummy_dataset['train'] = prepare_dataset(dummy_dataset['train'], self.tokenizer)
            dummy_dataset['test'] = prepare_dataset(dummy_dataset['test'], self.tokenizer)
            trainer = VASTrainer(
                ref_policy=self.ref_model,
                reward_model=self.reward_model,
                value_model=self.value_model,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                processing_class=self.tokenizer,
                config=training_args,
            )
            trainer.train()

            self.assertIn("loss/value_loss", trainer.state.log_history[-1])
