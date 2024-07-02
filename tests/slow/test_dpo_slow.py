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
import gc
import itertools
import tempfile
import unittest

import torch
from accelerate.utils.memory import release_memory
from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from trl import DPOConfig, DPOTrainer, is_peft_available

from ..testing_utils import require_bitsandbytes, require_peft, require_torch_gpu
from .testing_constants import DPO_LOSS_TYPES, DPO_PRECOMPUTE_LOGITS, GRADIENT_CHECKPOINTING_KWARGS, MODELS_TO_TEST


if is_peft_available():
    from peft import LoraConfig, PeftModel


@require_torch_gpu
class DPOTrainerSlowTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset("trl-internal-testing/mlabonne-chatml-dpo-pairs-copy", split="train[:10%]")
        cls.peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
        )
        cls.max_length = 128

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    @parameterized.expand(list(itertools.product(MODELS_TO_TEST, DPO_LOSS_TYPES, DPO_PRECOMPUTE_LOGITS)))
    def test_dpo_bare_model(self, model_id, loss_type, pre_compute_logits):
        """
        A test that tests the simple usage of `DPOTrainer` using a bare model in full precision.
        """
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=2,
                remove_unused_columns=False,
                gradient_accumulation_steps=2,
                learning_rate=9e-1,
                eval_strategy="steps",
                fp16=True,
                logging_strategy="no",
                report_to="none",
                beta=0.1,
                loss_type=loss_type,
                precompute_ref_log_probs=pre_compute_logits,
                max_length=self.max_length,
            )

            # dpo train lora model
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                tokenizer=tokenizer,
                train_dataset=self.dataset,
                eval_dataset=self.dataset,
            )

            # train the model
            trainer.train()

            # save trained model or adapter
            trainer.save_model()

        release_memory(model, trainer)

    @parameterized.expand(
        list(
            itertools.product(
                MODELS_TO_TEST,
                DPO_LOSS_TYPES,
                DPO_PRECOMPUTE_LOGITS,
                GRADIENT_CHECKPOINTING_KWARGS,
            )
        )
    )
    @require_peft
    def test_dpo_peft_model(self, model_id, loss_type, pre_compute_logits, gradient_checkpointing_kwargs):
        """
        A test that tests the simple usage of `DPOTrainer` using a peft model in full precision + different scenarios of gradient checkpointing.
        """
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=2,
                remove_unused_columns=False,
                gradient_accumulation_steps=2,
                learning_rate=9e-1,
                eval_strategy="steps",
                fp16=True,
                logging_strategy="no",
                report_to="none",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
                generate_during_eval=False,
                loss_type=loss_type,
                precompute_ref_log_probs=pre_compute_logits,
                beta=0.1,
                max_length=self.max_length,
            )

            # dpo train lora model
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                tokenizer=tokenizer,
                train_dataset=self.dataset,
                eval_dataset=self.dataset,
                peft_config=self.peft_config,
            )

            assert isinstance(trainer.model, PeftModel)
            assert trainer.ref_model is None

            # train the model
            trainer.train()

            # save trained model or adapter
            trainer.save_model()

        release_memory(model, trainer)

    @parameterized.expand(
        list(
            itertools.product(
                MODELS_TO_TEST,
                DPO_LOSS_TYPES,
                DPO_PRECOMPUTE_LOGITS,
                GRADIENT_CHECKPOINTING_KWARGS,
            )
        )
    )
    @require_bitsandbytes
    @require_peft
    def test_dpo_peft_model_qlora(self, model_id, loss_type, pre_compute_logits, gradient_checkpointing_kwargs):
        """
        A test that tests the simple usage of `DPOTrainer` using QLoRA + different scenarios of gradient checkpointing.
        """
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=2,
                remove_unused_columns=False,
                gradient_accumulation_steps=2,
                learning_rate=9e-1,
                eval_strategy="steps",
                fp16=True,
                logging_strategy="no",
                report_to="none",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
                beta=0.1,
                generate_during_eval=False,
                loss_type=loss_type,
                precompute_ref_log_probs=pre_compute_logits,
                max_length=self.max_length,
            )

            # dpo train lora model
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                tokenizer=tokenizer,
                train_dataset=self.dataset,
                eval_dataset=self.dataset,
                peft_config=self.peft_config,
            )

            assert isinstance(trainer.model, PeftModel)
            assert trainer.ref_model is None

            # train the model
            trainer.train()

            # save trained model or adapter
            trainer.save_model()

        release_memory(model, trainer)
