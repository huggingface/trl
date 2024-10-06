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

import numpy as np
import pytest
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from transformers.testing_utils import require_bitsandbytes, require_peft
from parameterized import parameterized
from trl import CGPOConfig, CGPOTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from trl.trainer.cgpo_trainer import MixtureOfConstraintJudges
import random


class CGPOTrainerTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=1)
        self.moj = MixtureOfConstraintJudges()

        # Ensure the tokenizer has a chat template
        if not hasattr(self.tokenizer, "chat_template") or self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    @parameterized.expand(["crraft", "crpg", "codpo"])
    def test_policy_optimization(self, rlhf_optimizer):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = CGPOConfig(
                output_dir=tmp_dir,
                rlhf_optimizer=rlhf_optimizer,
                k=4,
                max_kl=5.0,
                temperature=0.9,
                max_new_tokens=4,
                per_device_train_batch_size=4,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

            trainer = CGPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                mixture_of_judges=self.moj,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            assert trainer.state.log_history[-1]["train_loss"] is not None

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)
