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

from trl import SFTConfig, SFTTrainer, is_peft_available
from trl.models.utils import setup_chat_format

from ..testing_utils import require_bitsandbytes, require_peft, require_torch_gpu, require_torch_multi_gpu
from .testing_constants import DEVICE_MAP_OPTIONS, GRADIENT_CHECKPOINTING_KWARGS, MODELS_TO_TEST, PACKING_OPTIONS


if is_peft_available():
    from peft import LoraConfig, PeftModel


@require_torch_gpu
class SFTTrainerSlowTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_dataset = load_dataset("imdb", split="train[:10%]")
        cls.eval_dataset = load_dataset("imdb", split="test[:10%]")
        cls.dataset_text_field = "text"
        cls.max_seq_length = 128
        cls.peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    @parameterized.expand(list(itertools.product(MODELS_TO_TEST, PACKING_OPTIONS)))
    def test_sft_trainer_str(self, model_name, packing):
        """
        Simply tests if passing a simple str to `SFTTrainer` loads and runs the trainer
        as expected.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(
                output_dir=tmp_dir,
                logging_strategy="no",
                report_to="none",
                per_device_train_batch_size=2,
                max_steps=10,
                packing=packing,
                dataset_text_field=self.dataset_text_field,
                max_seq_length=self.max_seq_length,
            )

            trainer = SFTTrainer(
                model_name,
                args=args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            trainer.train()

    @parameterized.expand(list(itertools.product(MODELS_TO_TEST, PACKING_OPTIONS)))
    def test_sft_trainer_transformers(self, model_name, packing):
        """
        Simply tests if passing a transformers model to `SFTTrainer` loads and runs the trainer
        as expected.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(
                output_dir=tmp_dir,
                logging_strategy="no",
                report_to="none",
                per_device_train_batch_size=2,
                max_steps=10,
                packing=packing,
                dataset_text_field=self.dataset_text_field,
                max_seq_length=self.max_seq_length,
            )

            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            trainer = SFTTrainer(
                model,
                args=args,
                tokenizer=tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            trainer.train()

        release_memory(model, trainer)

    @parameterized.expand(list(itertools.product(MODELS_TO_TEST, PACKING_OPTIONS)))
    @require_peft
    def test_sft_trainer_peft(self, model_name, packing):
        """
        Simply tests if passing a transformers model + peft config to `SFTTrainer` loads and runs the trainer
        as expected.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(
                output_dir=tmp_dir,
                logging_strategy="no",
                report_to="none",
                per_device_train_batch_size=2,
                max_steps=10,
                fp16=True,
                packing=packing,
                dataset_text_field=self.dataset_text_field,
                max_seq_length=self.max_seq_length,
            )

            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            trainer = SFTTrainer(
                model,
                args=args,
                tokenizer=tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                peft_config=self.peft_config,
            )

            assert isinstance(trainer.model, PeftModel)

            trainer.train()

        release_memory(model, trainer)

    @parameterized.expand(list(itertools.product(MODELS_TO_TEST, PACKING_OPTIONS)))
    def test_sft_trainer_transformers_mp(self, model_name, packing):
        """
        Simply tests if passing a transformers model to `SFTTrainer` loads and runs the trainer
        as expected in mixed precision.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(
                output_dir=tmp_dir,
                logging_strategy="no",
                report_to="none",
                per_device_train_batch_size=2,
                max_steps=10,
                fp16=True,  # this is sufficient to enable amp
                packing=packing,
                dataset_text_field=self.dataset_text_field,
                max_seq_length=self.max_seq_length,
            )

            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            trainer = SFTTrainer(
                model,
                args=args,
                tokenizer=tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            trainer.train()

        release_memory(model, trainer)

    @parameterized.expand(list(itertools.product(MODELS_TO_TEST, PACKING_OPTIONS, GRADIENT_CHECKPOINTING_KWARGS)))
    def test_sft_trainer_transformers_mp_gc(self, model_name, packing, gradient_checkpointing_kwargs):
        """
        Simply tests if passing a transformers model to `SFTTrainer` loads and runs the trainer
        as expected in mixed precision + different scenarios of gradient_checkpointing.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(
                output_dir=tmp_dir,
                logging_strategy="no",
                report_to="none",
                per_device_train_batch_size=2,
                max_steps=10,
                packing=packing,
                dataset_text_field=self.dataset_text_field,
                max_seq_length=self.max_seq_length,
                fp16=True,  # this is sufficient to enable amp
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            )

            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            trainer = SFTTrainer(
                model,
                args=args,
                tokenizer=tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            trainer.train()

        release_memory(model, trainer)

    @parameterized.expand(list(itertools.product(MODELS_TO_TEST, PACKING_OPTIONS, GRADIENT_CHECKPOINTING_KWARGS)))
    @require_peft
    def test_sft_trainer_transformers_mp_gc_peft(self, model_name, packing, gradient_checkpointing_kwargs):
        """
        Simply tests if passing a transformers model + PEFT to `SFTTrainer` loads and runs the trainer
        as expected in mixed precision + different scenarios of gradient_checkpointing.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(
                output_dir=tmp_dir,
                logging_strategy="no",
                report_to="none",
                per_device_train_batch_size=2,
                max_steps=10,
                packing=packing,
                dataset_text_field=self.dataset_text_field,
                max_seq_length=self.max_seq_length,
                fp16=True,  # this is sufficient to enable amp
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            )

            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            trainer = SFTTrainer(
                model,
                args=args,
                tokenizer=tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                peft_config=self.peft_config,
            )

            assert isinstance(trainer.model, PeftModel)

            trainer.train()

        release_memory(model, trainer)

    @parameterized.expand(
        list(itertools.product(MODELS_TO_TEST, PACKING_OPTIONS, GRADIENT_CHECKPOINTING_KWARGS, DEVICE_MAP_OPTIONS))
    )
    @require_torch_multi_gpu
    def test_sft_trainer_transformers_mp_gc_device_map(
        self, model_name, packing, gradient_checkpointing_kwargs, device_map
    ):
        """
        Simply tests if passing a transformers model to `SFTTrainer` loads and runs the trainer
        as expected in mixed precision + different scenarios of gradient_checkpointing (single, multi-gpu, etc).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(
                output_dir=tmp_dir,
                logging_strategy="no",
                report_to="none",
                per_device_train_batch_size=2,
                max_steps=10,
                packing=packing,
                dataset_text_field=self.dataset_text_field,
                max_seq_length=self.max_seq_length,
                fp16=True,  # this is sufficient to enable amp
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            )

            model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            trainer = SFTTrainer(
                model,
                args=args,
                tokenizer=tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )

            trainer.train()

        release_memory(model, trainer)

    @parameterized.expand(list(itertools.product(MODELS_TO_TEST, PACKING_OPTIONS, GRADIENT_CHECKPOINTING_KWARGS)))
    @require_peft
    @require_bitsandbytes
    def test_sft_trainer_transformers_mp_gc_peft_qlora(self, model_name, packing, gradient_checkpointing_kwargs):
        """
        Simply tests if passing a transformers model + PEFT + bnb to `SFTTrainer` loads and runs the trainer
        as expected in mixed precision + different scenarios of gradient_checkpointing.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(
                output_dir=tmp_dir,
                logging_strategy="no",
                report_to="none",
                per_device_train_batch_size=2,
                max_steps=10,
                packing=packing,
                dataset_text_field=self.dataset_text_field,
                max_seq_length=self.max_seq_length,
                fp16=True,  # this is sufficient to enable amp
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            )

            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            trainer = SFTTrainer(
                model,
                args=args,
                tokenizer=tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                peft_config=self.peft_config,
            )

            assert isinstance(trainer.model, PeftModel)

            trainer.train()

        release_memory(model, trainer)

    @parameterized.expand(list(itertools.product(MODELS_TO_TEST, PACKING_OPTIONS)))
    @require_peft
    @require_bitsandbytes
    def test_sft_trainer_with_chat_format_qlora(self, model_name, packing):
        """
        Simply tests if using setup_chat_format with a transformers model + peft + bnb config to `SFTTrainer` loads and runs the trainer
        as expected.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_dataset = load_dataset("trl-internal-testing/dolly-chatml-sft", split="train")

            args = SFTConfig(
                packing=packing,
                max_seq_length=self.max_seq_length,
                output_dir=tmp_dir,
                logging_strategy="no",
                report_to="none",
                per_device_train_batch_size=2,
                max_steps=10,
                fp16=True,
            )

            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            model, tokenizer = setup_chat_format(model, tokenizer)

            trainer = SFTTrainer(
                model,
                args=args,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                peft_config=self.peft_config,
            )

            assert isinstance(trainer.model, PeftModel)

            trainer.train()

        release_memory(model, trainer)
