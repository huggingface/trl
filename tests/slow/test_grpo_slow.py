# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import gc
import tempfile
import unittest

import pytest
import torch
from accelerate.utils.memory import release_memory
from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.testing_utils import require_liger_kernel, require_torch_accelerator

from trl import GRPOConfig, GRPOTrainer

from .testing_constants import MODELS_TO_TEST


@pytest.mark.slow
@require_torch_accelerator
class GRPOTrainerSlowTester(unittest.TestCase):
    def setUp(self):
        self.train_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        self.eval_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="test")
        self.max_length = 128

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    @parameterized.expand(MODELS_TO_TEST)
    @require_liger_kernel
    def test_training_with_liger_grpo_loss(self, model_name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=3,
                num_generations=3,
                use_liger_loss=True,
                max_completion_length=self.max_length,
                report_to="none",
                logging_strategy="no",
            )

            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

            trainer = GRPOTrainer(
                model=model,
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                processing_class=tokenizer,
            )
            from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

            assert isinstance(trainer.liger_grpo_loss, LigerFusedLinearGRPOLoss)

            previous_trainable_params = {n: param.clone() for n, param in model.named_parameters()}

            trainer.train()

            for n, param in previous_trainable_params.items():
                new_param = model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

        release_memory(model, trainer)
