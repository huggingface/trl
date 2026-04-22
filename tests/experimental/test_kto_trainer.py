# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import multiprocess
import pytest
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.kto import KTOConfig, KTOTrainer
from trl.experimental.kto.kto_trainer import _get_kl_dataset, _process_tokens

from ..testing_utils import TrlTestCase, require_liger_kernel, require_peft


class TestKTOTrainer(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="float32")
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @pytest.mark.parametrize(
        "config_name, loss_type, pre_compute, eval_dataset",
        [
            ("standard_preference", "kto", True, True),
            ("standard_unpaired_preference", "kto", False, True),
            ("conversational_implicit_prompt_preference", "apo_zero_unpaired", True, True),
            ("standard_unpaired_preference", "apo_zero_unpaired", False, True),
        ],
    )
    def test_kto_trainer(self, config_name, loss_type, pre_compute, eval_dataset):
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

        trainer = KTOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            processing_class=self.tokenizer,
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
        # Pytest/CI often starts background threads before tests run. Under Python 3.12+,
        # using "fork" in a multi-threaded process emits a DeprecationWarning and may deadlock.
        # Force "spawn" to keep this multiprocessing test safe while still exercising `num_proc=2`.
        multiprocess.set_start_method("spawn", force=True)

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
        train_dataset = dummy_dataset["train"]

        trainer = KTOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=dummy_dataset["test"],
        )

        # Verify the tokenization step via trainer.train_dataset (aligned with DPO test style).
        # prompt_input_ids must start with the tokenized prompt text.
        prompt_ids = self.tokenizer(train_dataset["prompt"][0])["input_ids"]
        assert trainer.train_dataset[0]["prompt_input_ids"][: len(prompt_ids)] == prompt_ids
        assert all(m == 1 for m in trainer.train_dataset[0]["prompt_attention_mask"])
        # completion_input_ids = (BOS +) prompt + answer + EOS
        assert trainer.train_dataset[0]["completion_input_ids"][-1] == self.tokenizer.eos_token_id
        # completion_labels: -100 for the prompt part, actual token IDs for the answer+EOS part
        n_prompt = len(trainer.train_dataset[0]["prompt_input_ids"])
        assert trainer.train_dataset[0]["completion_labels"][:n_prompt] == [-100] * n_prompt
        assert (
            trainer.train_dataset[0]["completion_labels"][n_prompt:]
            == trainer.train_dataset[0]["completion_input_ids"][n_prompt:]
        )

        # Test corruption of (prompt, completion) pairs for KL dataset.
        # _get_kl_dataset shifts answer_input_ids by one within each batch; prompt_input_ids are unchanged.
        for batch_size in [2, 3]:
            kl_dataset = trainer.train_dataset.map(_get_kl_dataset, batched=True, batch_size=batch_size)

            # Verify that the "answer_input_ids" have been modified, meaning the new "answer_input_ids" differ
            # from the original ones. However, when the length of the dataset modulo batch_size equals 1,
            # the last batch remains unaltered. This is a rare scenario that does not impact the training
            # process, so we exclude it from testing by iterating only up to len - 1.
            for i in range(len(kl_dataset["answer_input_ids"]) - 1):
                assert trainer.train_dataset["prompt_input_ids"][i] == kl_dataset["prompt_input_ids"][i]
                assert trainer.train_dataset["prompt_attention_mask"][i] == kl_dataset["prompt_attention_mask"][i]
                assert trainer.train_dataset["answer_input_ids"][i] != kl_dataset["answer_input_ids"][i]

        # Test _process_tokens directly with a synthetic tokenized dataset (exercises num_proc and prefix).
        raw = {
            "prompt_input_ids": [[100, 200, 300], [400, 500]],
            "prompt_attention_mask": [[1, 1, 1], [1, 1]],
            "answer_input_ids": [[600, 700], [800, 900, 1000]],
            "answer_attention_mask": [[1, 1], [1, 1, 1]],
            "label": [True, False],
        }
        fn_kwargs = {"prefix": "", "tokenizer": self.tokenizer, "max_length": trainer.max_length}
        processed_dataset = Dataset.from_dict(raw).map(_process_tokens, fn_kwargs=fn_kwargs, num_proc=2)
        for i in range(len(processed_dataset)):
            assert processed_dataset[i]["label"] == raw["label"][i]
            # completion_input_ids ends with EOS
            assert processed_dataset[i]["completion_input_ids"][-1] == self.tokenizer.eos_token_id
            # completion_labels: -100 for prompt, answer+EOS token IDs for the rest
            n_prompt = len(processed_dataset[i]["prompt_input_ids"])
            assert processed_dataset[i]["completion_labels"][:n_prompt] == [-100] * n_prompt
            assert (
                processed_dataset[i]["completion_labels"][n_prompt:]
                == processed_dataset[i]["completion_input_ids"][n_prompt:]
            )

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
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32")
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
