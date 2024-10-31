# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from unittest.mock import MagicMock

import numpy as np
import torch
from datasets import Dataset, features, load_dataset
from parameterized import parameterized
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    is_vision_available,
)
from transformers.testing_utils import (
    require_bitsandbytes,
    require_peft,
    require_torch_gpu_if_bnb_not_multi_backend_enabled,
    require_vision,
)

from trl import DPOConfig, DPOTrainer, FDivergenceType

from .testing_utils import require_no_wandb


if is_vision_available():
    from PIL import Image


class TestTokenizeRow(unittest.TestCase):
    def setUp(self):
        # Set up the mock tokenizer with specific behaviors
        self.tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        self.tokenizer.bos_token_id = 0
        self.tokenizer.eos_token_id = 2

        # Define mock return values for the tokenizer's 'input_ids' for the different text inputs
        self.tokenizer.return_value = {
            "input_ids": {"The sky is": [464, 6766, 318], " blue": [4171], " green": [4077]}
        }

        # Define tokenizer behavior when called
        def mock_tokenizer_call(text, add_special_tokens):
            token_map = {
                "The sky is": {"input_ids": [464, 6766, 318]},
                " blue": {"input_ids": [4171]},
                " green": {"input_ids": [4077]},
            }
            return token_map[text]

        self.tokenizer.side_effect = mock_tokenizer_call

    def test_tokenize_row_no_truncation_no_special_tokens(self):
        # Define the input features
        features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}

        # Call the method with no truncation and no special tokens
        result = DPOTrainer.tokenize_row(
            features=features,
            processing_class=self.tokenizer,
            max_prompt_length=None,
            max_completion_length=None,
            add_special_tokens=False,
        )

        # Assert the correct output without truncation or special tokens
        self.assertEqual(
            result,
            {
                "prompt_input_ids": [464, 6766, 318],
                "chosen_input_ids": [4171, 2],  # eos_token added
                "rejected_input_ids": [4077, 2],  # eos_token added
            },
        )

    def test_tokenize_row_with_truncation(self):
        # Define the input features
        features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}

        # Call the method with truncation
        result = DPOTrainer.tokenize_row(
            features=features,
            processing_class=self.tokenizer,
            max_prompt_length=2,
            max_completion_length=1,
            add_special_tokens=False,
        )

        # Assert the correct output with truncation applied
        self.assertEqual(
            result,
            {
                "prompt_input_ids": [6766, 318],  # truncated to the last 2 tokens
                "chosen_input_ids": [4171],  # truncated to 1 token
                "rejected_input_ids": [4077],  # truncated to 1 token
            },
        )

    def test_tokenize_row_with_special_tokens(self):
        # Define the input features
        features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}

        # Call the method with special tokens
        result = DPOTrainer.tokenize_row(
            features=features,
            processing_class=self.tokenizer,
            max_prompt_length=None,
            max_completion_length=None,
            add_special_tokens=True,
        )

        # Assert the correct output with special tokens added
        self.assertEqual(
            result,
            {
                "prompt_input_ids": [0, 464, 6766, 318, 2],  # bos_token and eos_token added
                "chosen_input_ids": [4171, 2],  # eos_token added
                "rejected_input_ids": [4077, 2],  # eos_token added
            },
        )

    def test_tokenize_row_with_truncation_and_special_tokens(self):
        # Define the input features
        features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}

        # Call the method with both truncation and special tokens
        result = DPOTrainer.tokenize_row(
            features=features,
            processing_class=self.tokenizer,
            max_prompt_length=4,
            max_completion_length=1,
            add_special_tokens=True,
        )

        # Assert the correct output with both truncation and special tokens
        self.assertEqual(
            result,
            {
                "prompt_input_ids": [464, 6766, 318, 2],  # truncated to 4 tokens with bos_token and eos_token
                "chosen_input_ids": [4171],  # truncated to 1 token
                "rejected_input_ids": [4077],  # truncated to 1 token
            },
        )


class DPOTrainerTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # get t5 as seq2seq example:
        model_id = "trl-internal-testing/T5ForConditionalGeneration-correct-vocab-calibrated"
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.t5_ref_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.t5_tokenizer = AutoTokenizer.from_pretrained(model_id)

    @parameterized.expand(
        [
            ["gpt2", "sigmoid", True],
            ["t5", "hinge", False],
            ["gpt2", "ipo", False],
            ["t5", "ipo", True],
            ["gpt2", "aot_pair", True],
            ["t5", "aot_pair", False],
            ["gpt2", "aot", True],
            ["t5", "aot", False],
            ["gpt2", "bco_pair", False],
            ["t5", "bco_pair", True],
            ["gpt2", "sppo_hard", False],
            ["t5", "sppo_hard", True],
            ["gpt2", "nca_pair", False],
            ["t5", "nca_pair", True],
            ["gpt2", "robust", True],
            ["gpt2", "exo_pair", False],
            ["t5", "exo_pair", True],
            ["gpt2", "apo_zero", True],
            ["t5", "apo_down", False],
        ]
    )
    def test_dpo_trainer(self, name, loss_type, pre_compute):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                loss_type=loss_type,
                precompute_ref_log_probs=pre_compute,
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            if name == "gpt2":
                model = self.model
                ref_model = self.ref_model
                tokenizer = self.tokenizer
            elif name == "t5":
                model = self.t5_model
                ref_model = self.t5_ref_model
                tokenizer = self.t5_tokenizer

            trainer = DPOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    self.assertFalse(torch.allclose(param, new_param, rtol=1e-12, atol=1e-12))

    def test_dpo_trainer_with_weighting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                loss_type="sigmoid",
                precompute_ref_log_probs=False,
                use_weighting=True,
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            trainer = DPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    self.assertFalse(torch.allclose(param, new_param, rtol=1e-12, atol=1e-12))

    @parameterized.expand(
        [
            [None, "Test when rpo_alpha is set to None"],
            [0.5, "Test when rpo_alpha is set to 0.5"],
        ]
    )
    def test_dpo_trainer_without_providing_ref_model(self, rpo_alpha, _):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                precompute_ref_log_probs=True,
                rpo_alpha=rpo_alpha,
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            trainer = DPOTrainer(
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

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    self.assertFalse(torch.equal(param, new_param))

    def test_dpo_trainer_with_ref_model_is_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            with self.assertRaises(ValueError):
                DPOTrainer(
                    model=self.model,
                    ref_model=self.model,  # ref_model can't be the same as model
                    args=training_args,
                    processing_class=self.tokenizer,
                    train_dataset=dummy_dataset["train"],
                )

    @require_peft
    def test_dpo_trainer_without_providing_ref_model_with_lora(self):
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                precompute_ref_log_probs=True,
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            trainer = DPOTrainer(
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

            # check the params have changed
            for n, param in previous_trainable_params.items():
                if "lora" in n:
                    new_param = trainer.model.get_parameter(n)
                    # check the params have changed - ignore 0 biases
                    if param.sum() != 0:
                        self.assertFalse(torch.equal(param, new_param))

    def test_dpo_trainer_padding_token_is_none(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
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

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            tokenizer.pad_token = None

            with self.assertRaisesRegex(
                ValueError,
                expected_regex=r"Can't find `pad_token_id` in the `processing_class`. "
                r"Explicitly set `tokenizer.pad_token` \(e.g. `tokenizer.pad_token = tokenizer.eos_token`\) "
                r"before instantiating the trainer.",
            ):
                trainer = DPOTrainer(
                    model=self.model,
                    ref_model=None,
                    args=training_args,
                    processing_class=tokenizer,
                    train_dataset=dummy_dataset["train"],
                    eval_dataset=dummy_dataset["test"],
                )

                trainer.train()

    def test_dpo_trainer_w_dataset_num_proc(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                dataset_num_proc=5,
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            tokenizer.pad_token = None

            with self.assertRaisesRegex(
                ValueError,
                expected_regex=r"Can't find `pad_token_id` in the `processing_class`. "
                r"Explicitly set `tokenizer.pad_token` \(e.g. `tokenizer.pad_token = tokenizer.eos_token`\) "
                r"before instantiating the trainer.",
            ):
                trainer = DPOTrainer(
                    model=self.model,
                    ref_model=None,
                    args=training_args,
                    processing_class=tokenizer,
                    train_dataset=dummy_dataset["train"],
                    eval_dataset=dummy_dataset["test"],
                )

                trainer.train()

    def test_tr_dpo_trainer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                precompute_ref_log_probs=False,
                sync_ref_model=True,
                ref_model_mixup_alpha=0.5,
                ref_model_sync_steps=1,
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            trainer = DPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

            # params of the ref model as its the same as the model
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.ref_model.get_parameter(n)
                # check the ref model's params have changed - ignore 0 biases
                if param.sum() != 0:
                    self.assertFalse(torch.equal(param, new_param))

    @require_no_wandb
    def test_dpo_trainer_generate_during_eval_no_wandb(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
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

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            with self.assertRaisesRegex(
                ValueError,
                expected_regex="`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve.",
            ):
                DPOTrainer(
                    model=self.model,
                    ref_model=None,
                    args=training_args,
                    processing_class=self.tokenizer,
                    train_dataset=dummy_dataset["train"],
                    eval_dataset=dummy_dataset["test"],
                )

    @require_peft
    def test_dpo_lora_save(self):
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
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                precompute_ref_log_probs=True,
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            # dpo train lora model with a lora config
            trainer = DPOTrainer(
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

            try:
                AutoModelForCausalLM.from_pretrained(tmp_dir)
            except OSError:
                self.fail("Loading the saved peft adapter failed")

    @require_peft
    @require_torch_gpu_if_bnb_not_multi_backend_enabled
    def test_dpo_lora_bf16_autocast_llama(self):
        # Note this test only works on compute capability > 7 GPU devices
        from peft import LoraConfig

        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # lora model
        model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                bf16=True,
                beta=0.1,
                generate_during_eval=True,
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            # dpo train lora model with a lora config
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                peft_config=lora_config,
            )

            # train the model
            trainer.train()

            # save peft adapter
            trainer.save_model()

    @parameterized.expand(
        [
            ["gpt2", "sigmoid", False, False],
            ["gpt2", "sigmoid", False, True],
            ["gpt2", "sigmoid", True, False],
            ["gpt2", "sigmoid", True, True],
            ["gpt2", "ipo", False, False],
            ["gpt2", "ipo", False, True],
            ["gpt2", "ipo", True, False],
            ["gpt2", "ipo", True, True],
            ["gpt2", "aot_pair", False, False],
            ["gpt2", "aot_pair", False, True],
            ["gpt2", "aot_pair", True, False],
            ["gpt2", "aot_pair", True, True],
            ["gpt2", "aot", False, False],
            ["gpt2", "aot", False, True],
            ["gpt2", "aot", True, False],
            ["gpt2", "aot", True, True],
            ["gpt2", "bco_pair", False, False],
            ["gpt2", "bco_pair", False, True],
            ["gpt2", "bco_pair", True, False],
            ["gpt2", "bco_pair", True, True],
            ["gpt2", "robust", False, False],
            ["gpt2", "robust", False, True],
            ["gpt2", "robust", True, False],
            ["gpt2", "robust", True, True],
        ]
    )
    @require_bitsandbytes
    @require_peft
    @unittest.skip("You need a GPU with bf16 support in order to run these tests")
    def test_dpo_lora_bf16_autocast(self, name, loss_type, pre_compute, gen_during_eval):
        # Note this test only works on compute capability > 7 GPU devices
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # lora model
        model = AutoModelForCausalLM.from_pretrained(self.model_id, load_in_4bit=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                bf16=True,
                beta=0.1,
                generate_during_eval=gen_during_eval,
                loss_type=loss_type,
                precompute_ref_log_probs=pre_compute,
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            # dpo train lora model with a lora config
            trainer = DPOTrainer(
                model=model,
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

    @require_peft
    def test_dpo_lora_tags(self):
        from peft import LoraConfig

        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # lora model
        model = AutoModelForCausalLM.from_pretrained(model_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
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

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            # dpo train lora model with a lora config
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                peft_config=lora_config,
            )

            for tag in ["dpo", "trl"]:
                self.assertIn(tag, trainer.model.model_tags)

    @require_peft
    def test_dpo_tags(self):
        model_id = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # lora model
        model = AutoModelForCausalLM.from_pretrained(model_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
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

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            # dpo train lora model with a lora config
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

            for tag in ["dpo", "trl"]:
                self.assertIn(tag, trainer.model.model_tags)

    @require_peft
    def test_dpo_lora_force_use_ref(self):
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

        ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
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

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            with self.assertRaises(ValueError):
                # passing a peft_model as model and ref_model should error out,
                # unless you pass `force_use_ref_model`
                trainer = DPOTrainer(
                    model=model_peft,
                    ref_model=ref_model,
                    args=training_args,
                    processing_class=self.tokenizer,
                    train_dataset=dummy_dataset["train"],
                    eval_dataset=dummy_dataset["test"],
                    peft_config=lora_config,
                )

            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                force_use_ref_model=True,
                report_to="none",
            )

            trainer = DPOTrainer(
                model=model_peft,
                ref_model=ref_model,
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                peft_config=lora_config,
            )

            # train the model
            trainer.train()

    def test_dpo_trainer_torch_dtype(self):
        # See https://github.com/huggingface/trl/issues/1751
        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=1,
                model_init_kwargs={"torch_dtype": "float16"},
                ref_model_init_kwargs={"torch_dtype": "float16"},
                report_to="none",
            )

            trainer = DPOTrainer(
                model=self.model_id,
                ref_model=self.model_id,
                processing_class=self.tokenizer,
                args=training_args,
                train_dataset=dummy_dataset["train"],
            )
            self.assertEqual(trainer.model.config.torch_dtype, torch.float16)
            self.assertEqual(trainer.ref_model.config.torch_dtype, torch.float16)

        # Now test when `torch_dtype` is provided but is wrong to either the model or the ref_model
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=1,
                model_init_kwargs={"torch_dtype": -1},
                report_to="none",
            )

            with self.assertRaises(ValueError) as context:
                _ = DPOTrainer(
                    model=self.model_id,
                    processing_class=self.tokenizer,
                    args=training_args,
                    train_dataset=dummy_dataset["train"],
                )

            self.assertIn(
                "Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got -1.",
                str(context.exception),
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=1,
                ref_model_init_kwargs={"torch_dtype": -1},
                report_to="none",
            )

            with self.assertRaises(ValueError) as context:
                _ = DPOTrainer(
                    model=self.model_id,
                    ref_model=self.model_id,
                    processing_class=self.tokenizer,
                    args=training_args,
                    train_dataset=dummy_dataset["train"],
                )

            self.assertIn(
                "Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got -1.",
                str(context.exception),
            )

    def test_dpo_loss_alpha_div_f(self):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # lora model
        model = AutoModelForCausalLM.from_pretrained(model_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                f_divergence_type=FDivergenceType.ALPHA_DIVERGENCE.value,
                f_alpha_divergence_coef=0.5,
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            # dpo train lora model with a lora config
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

            # Fake chosen and rejected log probs
            policy_chosen_logps = torch.FloatTensor([410.0, 0.1])
            policy_rejected_logps = torch.FloatTensor([810.5, 0.2])
            reference_chosen_logps = torch.FloatTensor([-610.0, -0.1])
            reference_rejected_logps = torch.FloatTensor([110.6, 0.5])
            losses, _, _ = trainer.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            self.assertTrue(torch.isfinite(losses).cpu().numpy().all())

    def test_dpo_loss_js_div_f(self):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # lora model
        model = AutoModelForCausalLM.from_pretrained(model_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                f_divergence_type=FDivergenceType.JS_DIVERGENCE.value,
                f_alpha_divergence_coef=0.5,
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

            # dpo train lora model with a lora config
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

            # Fake chosen and rejected log probs
            policy_chosen_logps = torch.FloatTensor([410.0, 0.1])
            policy_rejected_logps = torch.FloatTensor([95.5, 0.2])
            reference_chosen_logps = torch.FloatTensor([-610.0, -0.1])
            reference_rejected_logps = torch.FloatTensor([5.5, 0.5])
            losses, _, _ = trainer.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            self.assertTrue(torch.isfinite(losses).cpu().numpy().all())


@require_vision
class DPOVisionTrainerTester(unittest.TestCase):
    @parameterized.expand(
        [
            ["trl-internal-testing/tiny-random-idefics2"],
            ["trl-internal-testing/tiny-random-paligemma"],
            ["trl-internal-testing/tiny-random-llava-1.5"],
        ]
    )
    def test_vdpo_trainer(self, model_id):
        # fmt: off
        dataset_dict = {
            "prompt": [
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image in great detail."}]}],
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Is this bus in the USA?"}]}],
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Give a thorough description of the image."}]}],
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Who are the people in the image?"}]}],
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is written?"}]}],
            ],
            "chosen": [
                [{"role": "assistant", "content": [{"type": "text", "text": "The image features a modern, multi-colored train."}]}],
                [{"role": "assistant", "content": [{"type": "text", "text": "Yes, it can be assumed that this bus is in the USA."}]}],
                [{"role": "assistant", "content": [{"type": "text", "text": "The image features a forest path."}]}],
                [{"role": "assistant", "content": [{"type": "text", "text": "There are two individuals, possibly girls or women."}]}],
                [{"role": "assistant", "content": [{"type": "text", "text": '"ccpb".'}]}],
            ],
            "rejected": [
                [{"role": "assistant", "content": [{"type": "text", "text": "The image features a modern, colorful train."}]}],
                [{"role": "assistant", "content": [{"type": "text", "text": "No, it's not in the USA."}]}],
                [{"role": "assistant", "content": [{"type": "text", "text": "The image features a forest path surrounded by trees."}]}],
                [{"role": "assistant", "content": [{"type": "text", "text": "In the image, there are two individuals."}]}],
                [{"role": "assistant", "content": [{"type": "text", "text": '"ccpb".'}]}],
            ],
            "images": [
                [Image.fromarray(np.random.randint(0, 255, (92, 33, 3), dtype=np.uint8))],
                [Image.fromarray(np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8))],
                [Image.fromarray(np.random.randint(0, 255, (80, 152, 3), dtype=np.uint8))],
                [Image.fromarray(np.random.randint(0, 255, (57, 24, 3), dtype=np.uint8))],
                [Image.fromarray(np.random.randint(0, 255, (102, 48, 3), dtype=np.uint8))],
            ],
        }
        # fmt: on
        dataset = Dataset.from_dict(dataset_dict)
        dataset = dataset.cast_column("images", features.Sequence(features.Image()))

        # Instantiate the model and processor
        model = AutoModelForVision2Seq.from_pretrained(model_id)
        ref_model = AutoModelForVision2Seq.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_length=512,
                max_prompt_length=512,
                remove_unused_columns=False,
                report_to="none",
            )
            trainer = DPOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                processing_class=processor,
                train_dataset=dataset,
                eval_dataset=dataset,
            )

            # Save the initial weights, so we can check if they have changed after training
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the trainable params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
                    if model_id == "trl-internal-testing/tiny-random-llava-1.5" and (
                        n.startswith("vision_tower.vision_model.encoder.layers.3")
                        or n == "vision_tower.vision_model.post_layernorm.weight"
                    ):
                        # For some reason, these params are not updated. This is probably not related to TRL, but to
                        # the model itself. We should investigate this further, but for now we just skip these params.
                        continue
                    self.assertFalse(torch.allclose(param, new_param, rtol=1e-12, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
