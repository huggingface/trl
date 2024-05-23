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
from functools import partial

import torch
from accelerate import Accelerator
from datasets import Dataset
from parameterized import parameterized
from pytest import mark
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from trl import KTOConfig, KTOTrainer
from trl.trainer.kto_trainer import _get_kl_dataset, _process_tokens, _tokenize

from .testing_utils import require_no_wandb, require_peft


class KTOTrainerTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        cls.ref_model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token

        # get t5 as seq2seq example:
        model_id = "trl-internal-testing/tiny-T5ForConditionalGeneration-correct-vocab"
        cls.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        cls.t5_ref_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        cls.t5_tokenizer = AutoTokenizer.from_pretrained(model_id)

        # get embedding model
        model_id = "facebook/bart-base"
        cls.embedding_model = AutoModel.from_pretrained(model_id)
        cls.embedding_tokenizer = AutoTokenizer.from_pretrained(model_id)

    def _init_dummy_dataset(self):
        # fmt: off
        dummy_dataset_dict = {
            "prompt": [
                "Hey, hello",
                "How are you",
                "What is your name?",
                "What is your name?",
                "Which is the best programming language?",
                "Which is the best programming language?",
                "Which is the best programming language?",
            ],
            "completion": [
                "hi nice to meet you",
                "leave me alone",
                "I don't have a name",
                "My name is Mary",
                "Python",
                "C++",
                "Java",
            ],
            "label": [
                True,
                False,
                False,
                True,
                True,
                False,
                False,
            ],
        }
        # fmt: on
        return Dataset.from_dict(dummy_dataset_dict)

    @parameterized.expand(
        [
            ["gpt2", "kto", True, True],
            ["gpt2", "kto", True, False],
            # ["t5", True],
            ["gpt2", "kto", False, True],
            ["gpt2", "kto", False, False],
            # ["t5", False],
            ["gpt2", "bco", True, True],
            ["gpt2", "bco", True, False],
            ["gpt2", "bco", False, True],
            ["gpt2", "bco", False, False],
        ]
    )
    def test_kto_trainer(self, name, loss_type, pre_compute, eval_dataset):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = KTOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                precompute_ref_log_probs=pre_compute,
                loss_type=loss_type,
            )

            dummy_dataset = self._init_dummy_dataset()

            if name == "gpt2":
                model = self.model
                ref_model = self.ref_model
                tokenizer = self.tokenizer
            elif name == "t5":
                model = self.t5_model
                ref_model = self.t5_ref_model
                tokenizer = self.t5_tokenizer

            trainer = KTOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                tokenizer=tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset if eval_dataset else None,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    self.assertFalse(torch.equal(param, new_param))

    def test_tokenize_and_process_tokens(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = KTOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
            )

            dummy_dataset = self._init_dummy_dataset()

            trainer = KTOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
            )

            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenized_dataset = dummy_dataset.map(
                    _tokenize,
                    fn_kwargs={"tokenizer": trainer.tokenizer},
                    batched=True,
                    batch_size=2,
                )
                self.assertListEqual(tokenized_dataset["prompt"], dummy_dataset["prompt"])
                self.assertListEqual(tokenized_dataset["completion"], dummy_dataset["completion"])
                self.assertListEqual(tokenized_dataset["label"], dummy_dataset["label"])
                self.assertListEqual(tokenized_dataset["prompt_input_ids"][0], [10814, 11])
                self.assertListEqual(tokenized_dataset["prompt_attention_mask"][0], [1, 1])
                self.assertListEqual(tokenized_dataset["answer_input_ids"][0], [5968, 1219, 72, 3621, 284, 1826, 345])
                self.assertListEqual(tokenized_dataset["answer_attention_mask"][0], [1, 1, 1, 1, 1, 1, 1])

                # Test reversal of (prompt, completion) pairs for KL dataset
                tokenized_kl_dataset = tokenized_dataset.map(_get_kl_dataset, batched=True, batch_size=2)
                self.assertListEqual(
                    tokenized_kl_dataset["prompt_input_ids"][0], tokenized_dataset["prompt_input_ids"][0]
                )
                self.assertListEqual(
                    tokenized_kl_dataset["prompt_attention_mask"][0], tokenized_dataset["prompt_attention_mask"][0]
                )
                self.assertListEqual(
                    tokenized_kl_dataset["answer_input_ids"][0], tokenized_dataset["answer_input_ids"][1]
                )
                self.assertListEqual(
                    tokenized_kl_dataset["answer_attention_mask"][0], tokenized_dataset["answer_attention_mask"][1]
                )

                fn_kwargs = {
                    "prefix": "",
                    "is_encoder_decoder": trainer.is_encoder_decoder,
                    "tokenizer": trainer.tokenizer,
                    "max_length": trainer.max_length,
                    "truncation_mode": trainer.truncation_mode,
                    "label_pad_token_id": trainer.label_pad_token_id,
                    "max_prompt_length": trainer.max_prompt_length,
                }
                processed_dataset = tokenized_dataset.map(_process_tokens, fn_kwargs=fn_kwargs, num_proc=2)
                self.assertListEqual(processed_dataset["prompt"], dummy_dataset["prompt"])
                self.assertListEqual(processed_dataset["completion"], dummy_dataset["completion"])
                self.assertListEqual(processed_dataset["label"], dummy_dataset["label"])
                self.assertListEqual(processed_dataset["prompt_input_ids"][0], [50256, 10814, 11])
                self.assertListEqual(processed_dataset["prompt_attention_mask"][0], [1, 1, 1])
                self.assertListEqual(
                    processed_dataset["completion_input_ids"][0],
                    [50256, 10814, 11, 5968, 1219, 72, 3621, 284, 1826, 345, 50256],
                )
                self.assertListEqual(
                    processed_dataset["completion_attention_mask"][0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                )
                self.assertListEqual(
                    processed_dataset["completion_labels"][0],
                    [-100, -100, -100, 5968, 1219, 72, 3621, 284, 1826, 345, 50256],
                )

    def test_kto_trainer_without_providing_ref_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = KTOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
            )

            dummy_dataset = self._init_dummy_dataset()

            trainer = KTOTrainer(
                model=self.model,
                ref_model=None,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    self.assertFalse(torch.equal(param, new_param))

    def test_kto_trainer_bco_udm(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = KTOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                loss_type="bco",
            )

            dummy_dataset = self._init_dummy_dataset()

            def embed_prompt(input_ids, attention_mask, model):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                return outputs.last_hidden_state.mean(dim=1)

            embedding_model = Accelerator().prepare_model(self.embedding_model)
            embedding_func = partial(embed_prompt, model=embedding_model)

            trainer = KTOTrainer(
                model=self.model,
                ref_model=None,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
                embedding_func=embedding_func,
                embedding_tokenizer=self.embedding_tokenizer,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    self.assertFalse(torch.equal(param, new_param))

    @require_peft
    @mark.peft_test
    def test_kto_trainer_without_providing_ref_model_with_lora(self):
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = KTOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
            )

            dummy_dataset = self._init_dummy_dataset()

            trainer = KTOTrainer(
                model=self.model,
                ref_model=None,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
                peft_config=lora_config,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # check the params have changed
            for n, param in previous_trainable_params.items():
                if "lora" in n:
                    new_param = trainer.model.get_parameter(n)
                    # check the params have changed - ignore 0 biases
                    if param.sum() != 0:
                        self.assertFalse(torch.equal(param, new_param))

    @require_no_wandb
    def test_kto_trainer_generate_during_eval_no_wandb(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = KTOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                generate_during_eval=True,
            )

            dummy_dataset = self._init_dummy_dataset()

            with self.assertRaisesRegex(
                ValueError,
                expected_regex="`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install with `pip install wandb` to resolve.",
            ):
                KTOTrainer(
                    model=self.model,
                    ref_model=None,
                    args=training_args,
                    tokenizer=self.tokenizer,
                    train_dataset=dummy_dataset,
                    eval_dataset=dummy_dataset,
                )

    @require_peft
    @mark.peft_test
    def test_kto_lora_save(self):
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # lora model
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        model_peft = get_peft_model(model, lora_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = KTOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
            )

            dummy_dataset = self._init_dummy_dataset()

            # kto train lora model with a lora config
            trainer = KTOTrainer(
                model=model_peft,
                ref_model=None,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
                peft_config=lora_config,
            )

            # train the model
            trainer.train()

            # save peft adapter
            trainer.save_model()

            # assert that the model is loaded without giving OSError
            try:
                AutoModelForCausalLM.from_pretrained(tmp_dir)
            except OSError:
                self.fail("Loading the saved peft adapter failed")
