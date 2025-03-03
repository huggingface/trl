# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import torch
from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.testing_utils import require_peft

from trl import KTOConfig, KTOTrainer
from trl.trainer.kto_trainer import _get_kl_dataset, _process_tokens, _tokenize

from .testing_utils import require_no_wandb


class KTOTrainerTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # get t5 as seq2seq example:
        model_id = "trl-internal-testing/tiny-T5ForConditionalGeneration"
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.t5_ref_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.t5_tokenizer = AutoTokenizer.from_pretrained(model_id)

    @parameterized.expand(
        [
            ("qwen", "standard_preference", "kto", True, True),
            # ("t5", "standard_implicit_prompt_preference", "kto", True, False), # KTO broken for enc-dec
            ("qwen", "standard_unpaired_preference", "kto", False, True),
            # ("t5", "conversational_preference", "kto", False, False),
            ("qwen", "conversational_implicit_prompt_preference", "apo_zero_unpaired", True, True),
            # ("t5", "conversational_unpaired_preference", "apo_zero_unpaired", True, False),
            ("qwen", "standard_unpaired_preference", "apo_zero_unpaired", False, True),
            # ("t5", "conversational_unpaired_preference", "apo_zero_unpaired", False, False),
        ]
    )
    def test_kto_trainer(self, name, config_name, loss_type, pre_compute, eval_dataset):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = KTOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps" if eval_dataset else "no",
                beta=0.1,
                precompute_ref_log_probs=pre_compute,
                loss_type=loss_type,
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", config_name)

            if name == "qwen":
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
                processing_class=tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"] if eval_dataset else None,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the parameters have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
                    self.assertFalse(torch.equal(param, new_param))

    def test_kto_trainer_with_ref_model_is_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = KTOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

            with self.assertRaises(ValueError):
                KTOTrainer(
                    model=self.model,
                    ref_model=self.model,  # ref_model can't be the same as model
                    args=training_args,
                    processing_class=self.tokenizer,
                    train_dataset=dummy_dataset["train"],
                )

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
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

            trainer = KTOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

            train_dataset = dummy_dataset["train"]
            tokenized_dataset = train_dataset.map(
                _tokenize,
                fn_kwargs={"tokenizer": trainer.tokenizer},
                batched=True,
                batch_size=2,
            )
            self.assertListEqual(tokenized_dataset["prompt"], train_dataset["prompt"])
            self.assertListEqual(tokenized_dataset["completion"], train_dataset["completion"])
            self.assertListEqual(tokenized_dataset["label"], train_dataset["label"])
            self.assertListEqual(tokenized_dataset["prompt_input_ids"][0], [46518, 374, 2664, 1091])
            self.assertListEqual(tokenized_dataset["prompt_attention_mask"][0], [1, 1, 1, 1])
            self.assertListEqual(tokenized_dataset["answer_input_ids"][0], [27261, 13])
            self.assertListEqual(tokenized_dataset["answer_attention_mask"][0], [1, 1])

            # Test corruption of (prompt, completion) pairs for KL dataset
            for batch_size in [2, 3]:
                tokenized_kl_dataset = tokenized_dataset.map(_get_kl_dataset, batched=True, batch_size=batch_size)

                # Verify that the "answer_input_ids" have been modified, meaning the new "answer_input_ids" differ
                # from the original ones. However, when the length of the dataset modulo batch_size equals 1,
                # the last batch remains unaltered. This is a rare scenario that does not impact the training
                # process, so we exclude it from testing by iterating only up to len - 1.
                for i in range(len(tokenized_kl_dataset["answer_input_ids"]) - 1):
                    self.assertListEqual(
                        tokenized_dataset["prompt_input_ids"][i],
                        tokenized_kl_dataset["prompt_input_ids"][i],
                    )
                    self.assertListEqual(
                        tokenized_dataset["prompt_attention_mask"][i],
                        tokenized_kl_dataset["prompt_attention_mask"][i],
                    )
                    self.assertNotEqual(
                        tokenized_dataset["answer_input_ids"][i],
                        tokenized_kl_dataset["answer_input_ids"][i],
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
            self.assertListEqual(processed_dataset["prompt"], train_dataset["prompt"])
            self.assertListEqual(processed_dataset["completion"], train_dataset["completion"])
            self.assertListEqual(processed_dataset["label"], train_dataset["label"])
            self.assertListEqual(processed_dataset["prompt_input_ids"][0], [46518, 374, 2664, 1091])
            self.assertListEqual(processed_dataset["prompt_attention_mask"][0], [1, 1, 1, 1])
            self.assertListEqual(
                processed_dataset["completion_input_ids"][0], [46518, 374, 2664, 1091, 27261, 13, 151645]
            )
            self.assertListEqual(processed_dataset["completion_attention_mask"][0], [1, 1, 1, 1, 1, 1, 1])
            self.assertListEqual(
                processed_dataset["completion_labels"][0], [-100, -100, -100, -100, 27261, 13, 151645]
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
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

            trainer = KTOTrainer(
                model=self.model,
                ref_model=None,
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the parameters have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
                    self.assertFalse(torch.equal(param, new_param))

    @require_peft
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
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

            trainer = KTOTrainer(
                model=self.model,
                ref_model=None,
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                peft_config=lora_config,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the parameters have changed
            for n, param in previous_trainable_params.items():
                if "lora" in n:
                    new_param = trainer.model.get_parameter(n)
                    if param.sum() != 0:  # ignore 0 biases
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
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

            with self.assertRaisesRegex(
                ValueError,
                expected_regex="`generate_during_eval=True` requires Weights and Biases or Comet to be installed."
                " Please install `wandb` or `comet-ml` to resolve.",
            ):
                KTOTrainer(
                    model=self.model,
                    ref_model=None,
                    args=training_args,
                    processing_class=self.tokenizer,
                    train_dataset=dummy_dataset["train"],
                    eval_dataset=dummy_dataset["test"],
                )

    @require_peft
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
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

            # kto train lora model with a lora config
            trainer = KTOTrainer(
                model=model_peft,
                ref_model=None,
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
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

    def test_compute_metrics(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        ref_model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        tokenizer.pad_token = tokenizer.eos_token

        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

        def dummy_compute_metrics(*args, **kwargs):
            return {"test": 0.0}

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = KTOConfig(
                output_dir=tmp_dir,
                remove_unused_columns=False,
                per_device_train_batch_size=2,
                do_eval=True,
                eval_strategy="steps",
                eval_steps=1,
                per_device_eval_batch_size=2,
                report_to="none",
            )

            trainer = KTOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                compute_metrics=dummy_compute_metrics,
            )

            trainer.train()

            self.assertEqual(trainer.state.log_history[-2]["eval_test"], 0.0)
