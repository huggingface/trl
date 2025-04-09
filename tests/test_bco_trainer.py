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

import tempfile
import unittest
from functools import partial

import torch
from accelerate import Accelerator
from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl import BCOConfig, BCOTrainer
from trl.trainer.bco_trainer import _process_tokens, _tokenize

from .testing_utils import require_no_wandb, require_sklearn


if is_peft_available():
    from peft import LoraConfig


class BCOTrainerTester(unittest.TestCase):
    @parameterized.expand([("standard_unpaired_preference"), ("conversational_unpaired_preference")])
    @require_sklearn
    def test_train(self, config_name):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = BCOConfig(
                output_dir=tmp_dir,
                remove_unused_columns=False,  # warning raised if not set to False
                learning_rate=0.1,  # increase the learning rate to speed up the test
                report_to="none",
            )

            trainer = BCOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the parameters have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
                    self.assertFalse(torch.equal(param.cpu(), new_param.cpu()))

    @require_sklearn
    def test_train_with_precompute(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = BCOConfig(
                output_dir=tmp_dir,
                remove_unused_columns=False,  # warning raised if not set to False
                learning_rate=0.1,  # increase the learning rate to speed up the test
                precompute_ref_log_probs=True,
                report_to="none",
            )

            trainer = BCOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the parameters have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
                    self.assertFalse(torch.equal(param.cpu(), new_param.cpu()))

    @require_sklearn
    def test_train_eval(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = BCOConfig(
                output_dir=tmp_dir,
                remove_unused_columns=False,  # warning raised if not set to False
                eval_strategy="steps",
                eval_steps=3,
                report_to="none",
            )

            trainer = BCOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
            )

            trainer.train()

    @require_sklearn
    def test_init_with_ref_model_is_model(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = BCOConfig(
                output_dir=tmp_dir,
                remove_unused_columns=False,  # warning raised if not set to False
                report_to="none",
            )

            with self.assertRaises(ValueError):
                BCOTrainer(
                    model=model,
                    ref_model=model,  # ref_model can't be the same as model
                    args=training_args,
                    processing_class=tokenizer,
                    train_dataset=dataset,
                )

    @require_sklearn
    def test_tokenize_and_process_tokens(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = BCOConfig(
                output_dir=tmp_dir,
                remove_unused_columns=False,  # warning raised if not set to False
                report_to="none",
            )

            trainer = BCOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dataset,
            )

            tokenized_dataset = dataset.map(
                _tokenize,
                fn_kwargs={"tokenizer": trainer.tokenizer},
                batched=True,
                batch_size=2,
            )
            self.assertListEqual(tokenized_dataset["prompt"], dataset["prompt"])
            self.assertListEqual(tokenized_dataset["completion"], dataset["completion"])
            self.assertListEqual(tokenized_dataset["label"], dataset["label"])
            self.assertListEqual(tokenized_dataset["prompt_input_ids"][0], [46518, 374, 2664, 1091])
            self.assertListEqual(tokenized_dataset["prompt_attention_mask"][0], [1, 1, 1, 1])
            self.assertListEqual(tokenized_dataset["answer_input_ids"][0], [27261, 13])
            self.assertListEqual(tokenized_dataset["answer_attention_mask"][0], [1, 1])

            fn_kwargs = {
                "prefix": "",
                "is_encoder_decoder": trainer.is_encoder_decoder,
                "tokenizer": trainer.tokenizer,
                "max_length": trainer.max_length,
                "truncation_mode": trainer.truncation_mode,
                "label_pad_token_id": trainer.label_pad_token_id,
                "max_prompt_length": trainer.max_prompt_length,
            }
            processed_dataset = tokenized_dataset.map(_process_tokens, fn_kwargs=fn_kwargs)
            self.assertListEqual(processed_dataset["prompt"], dataset["prompt"])
            self.assertListEqual(processed_dataset["completion"], dataset["completion"])
            self.assertListEqual(processed_dataset["label"], dataset["label"])
            self.assertListEqual(processed_dataset["prompt_input_ids"][0], [46518, 374, 2664, 1091])
            self.assertListEqual(processed_dataset["prompt_attention_mask"][0], [1, 1, 1, 1])
            self.assertListEqual(
                processed_dataset["completion_input_ids"][0], [46518, 374, 2664, 1091, 27261, 13, 151645]
            )
            self.assertListEqual(processed_dataset["completion_attention_mask"][0], [1, 1, 1, 1, 1, 1, 1])
            self.assertListEqual(
                processed_dataset["completion_labels"][0], [-100, -100, -100, -100, 27261, 13, 151645]
            )

    @require_sklearn
    def test_train_without_providing_ref_model(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = BCOConfig(
                output_dir=tmp_dir,
                remove_unused_columns=False,  # warning raised if not set to False
                learning_rate=0.1,  # increase the learning rate to speed up the test
                report_to="none",
            )

            trainer = BCOTrainer(
                model=model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the parameters have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
                    self.assertFalse(torch.equal(param.cpu(), new_param.cpu()))

    @require_sklearn
    def test_train_udm(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Get embedding model
        embedding_model_id = "trl-internal-testing/tiny-BartModel"
        embedding_model = AutoModel.from_pretrained(embedding_model_id)
        embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)

        def embed_prompt(input_ids, attention_mask, model):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            return outputs.last_hidden_state.mean(dim=1)

        embedding_model = Accelerator().prepare_model(embedding_model)
        embedding_func = partial(embed_prompt, model=embedding_model)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = BCOConfig(
                output_dir=tmp_dir,
                remove_unused_columns=False,  # warning raised if not set to False
                learning_rate=0.1,  # increase the learning rate to speed up the test
                report_to="none",
            )

            trainer = BCOTrainer(
                model=model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dataset,
                embedding_func=embedding_func,
                embedding_tokenizer=embedding_tokenizer,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the parameters have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
                    self.assertFalse(torch.equal(param.cpu(), new_param.cpu()))

    @require_sklearn
    @require_peft
    def test_train_without_providing_ref_model_with_lora(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = BCOConfig(
                output_dir=tmp_dir,
                remove_unused_columns=False,  # warning raised if not set to False
                learning_rate=0.1,  # increase the learning rate to speed up the test
                report_to="none",
            )

            trainer = BCOTrainer(
                model=model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dataset,
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
                        self.assertFalse(torch.equal(param.cpu(), new_param.cpu()))

    @require_sklearn
    @require_no_wandb
    def test_generate_during_eval_no_wandb(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = BCOConfig(
                output_dir=tmp_dir,
                remove_unused_columns=False,  # warning raised if not set to False
                eval_strategy="steps",
                eval_steps=3,
                generate_during_eval=True,
                report_to="none",
            )

            with self.assertRaisesRegex(
                ValueError,
                expected_regex="`generate_during_eval=True` requires Weights and Biases or Comet to be installed."
                " Please install `wandb` or `comet-ml` to resolve.",
            ):
                BCOTrainer(
                    model=model,
                    args=training_args,
                    processing_class=tokenizer,
                    train_dataset=dataset["train"],
                    eval_dataset=dataset["test"],
                )

    @require_sklearn
    @require_peft
    def test_lora_train_and_save(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = BCOConfig(
                output_dir=tmp_dir,
                remove_unused_columns=False,  # warning raised if not set to False
                report_to="none",
            )

            trainer = BCOTrainer(
                model=model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dataset["train"],
                peft_config=lora_config,
            )

            # train the model
            trainer.train()

            # save peft adapter
            trainer.save_model()

            # assert that the model is loaded without giving OSError
            AutoModelForCausalLM.from_pretrained(tmp_dir)

    @require_sklearn
    def test_compute_metrics(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

        def dummy_compute_metrics(*args, **kwargs):
            return {"test": 0.0}

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = BCOConfig(
                output_dir=tmp_dir,
                remove_unused_columns=False,  # warning raised if not set to False
                eval_strategy="steps",
                eval_steps=3,
                report_to="none",
            )

            trainer = BCOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                compute_metrics=dummy_compute_metrics,
            )

            trainer.train()

            self.assertEqual(trainer.state.log_history[-2]["eval_test"], 0.0)
