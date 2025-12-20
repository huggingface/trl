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

import pytest
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from trl.experimental.kto import KTOConfig, KTOTrainer
from trl.experimental.kto.kto_trainer import _get_kl_dataset, _process_tokens, _tokenize

from ..testing_utils import TrlTestCase, require_liger_kernel, require_no_wandb, require_peft


class TestKTOTrainer(TrlTestCase):
    def setup_method(self):
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

    @pytest.mark.parametrize(
        "name, config_name, loss_type, pre_compute, eval_dataset",
        [
            ("qwen", "standard_preference", "kto", True, True),
            # ("t5", "standard_implicit_prompt_preference", "kto", True, False), # KTO broken for enc-dec
            ("qwen", "standard_unpaired_preference", "kto", False, True),
            # ("t5", "conversational_preference", "kto", False, False),
            ("qwen", "conversational_implicit_prompt_preference", "apo_zero_unpaired", True, True),
            # ("t5", "conversational_unpaired_preference", "apo_zero_unpaired", True, False),
            ("qwen", "standard_unpaired_preference", "apo_zero_unpaired", False, True),
            # ("t5", "conversational_unpaired_preference", "apo_zero_unpaired", False, False),
        ],
    )
    def test_kto_trainer(self, name, config_name, loss_type, pre_compute, eval_dataset):
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the parameters have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.equal(param, new_param)

    def test_kto_trainer_with_ref_model_is_model(self):
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            report_to="none",
        )

        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

        with pytest.raises(ValueError):
            KTOTrainer(
                model=self.model,
                ref_model=self.model,  # ref_model can't be the same as model
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset["train"],
            )

    def test_tokenize_and_process_tokens(self):
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
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
            fn_kwargs={"tokenizer": trainer.processing_class},
            batched=True,
            batch_size=2,
        )
        assert tokenized_dataset["prompt"][:] == train_dataset["prompt"][:]
        assert tokenized_dataset["completion"][:] == train_dataset["completion"][:]
        assert tokenized_dataset["label"][:] == train_dataset["label"][:]
        assert tokenized_dataset["prompt_input_ids"][0] == [46518, 374, 2664, 1091]
        assert tokenized_dataset["prompt_attention_mask"][0] == [1, 1, 1, 1]
        assert tokenized_dataset["answer_input_ids"][0] == [27261, 13]
        assert tokenized_dataset["answer_attention_mask"][0] == [1, 1]

        # Test corruption of (prompt, completion) pairs for KL dataset
        for batch_size in [2, 3]:
            tokenized_kl_dataset = tokenized_dataset.map(_get_kl_dataset, batched=True, batch_size=batch_size)

            # Verify that the "answer_input_ids" have been modified, meaning the new "answer_input_ids" differ
            # from the original ones. However, when the length of the dataset modulo batch_size equals 1,
            # the last batch remains unaltered. This is a rare scenario that does not impact the training
            # process, so we exclude it from testing by iterating only up to len - 1.
            for i in range(len(tokenized_kl_dataset["answer_input_ids"]) - 1):
                assert tokenized_dataset["prompt_input_ids"][i] == tokenized_kl_dataset["prompt_input_ids"][i]
                assert (
                    tokenized_dataset["prompt_attention_mask"][i] == tokenized_kl_dataset["prompt_attention_mask"][i]
                )
                assert tokenized_dataset["answer_input_ids"][i] != tokenized_kl_dataset["answer_input_ids"][i]

        fn_kwargs = {
            "prefix": "",
            "is_encoder_decoder": trainer.is_encoder_decoder,
            "tokenizer": trainer.processing_class,
            "max_length": trainer.max_length,
            "truncation_mode": trainer.truncation_mode,
            "label_pad_token_id": trainer.label_pad_token_id,
            "max_prompt_length": trainer.max_prompt_length,
        }
        processed_dataset = tokenized_dataset.map(_process_tokens, fn_kwargs=fn_kwargs, num_proc=2)
        assert processed_dataset["prompt"][:] == train_dataset["prompt"][:]
        assert processed_dataset["completion"][:] == train_dataset["completion"][:]
        assert processed_dataset["label"][:] == train_dataset["label"][:]
        assert processed_dataset["prompt_input_ids"][0] == [46518, 374, 2664, 1091]
        assert processed_dataset["prompt_attention_mask"][0] == [1, 1, 1, 1]
        assert processed_dataset["completion_input_ids"][0] == [46518, 374, 2664, 1091, 27261, 13, 151645]
        assert processed_dataset["completion_attention_mask"][0] == [1, 1, 1, 1, 1, 1, 1]
        assert processed_dataset["completion_labels"][0] == [-100, -100, -100, -100, 27261, 13, 151645]

    def test_kto_trainer_without_providing_ref_model(self):
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the parameters have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.equal(param, new_param)

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

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the parameters have changed
        for n, param in previous_trainable_params.items():
            if "lora" in n:
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
                    assert not torch.equal(param, new_param)

    @require_no_wandb
    def test_kto_trainer_generate_during_eval_no_wandb(self):
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
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

        with pytest.raises(
            ValueError,
            match="`generate_during_eval=True` requires Weights and Biases or Comet to be installed."
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

    @require_liger_kernel
    def test_kto_trainer_with_liger(self):
        """Test KTO trainer with Liger kernel enabled."""
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            report_to="none",
            use_liger_kernel=True,  # Enable Liger kernel
        )

        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

        trainer = KTOTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dummy_dataset["train"],
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            # check the params have changed - ignore 0 biases
            if param.sum() != 0:
                assert not torch.equal(param, new_param)

    def test_compute_metrics(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        ref_model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        tokenizer.pad_token = tokenizer.eos_token

        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

        def dummy_compute_metrics(*args, **kwargs):
            return {"test": 0.0}

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
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

        assert trainer.state.log_history[-2]["eval_test"] == 0.0
