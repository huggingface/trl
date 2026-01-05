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
import re
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from accelerate.utils.memory import release_memory
from datasets import Dataset, features, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    is_vision_available,
)
from transformers.testing_utils import backend_empty_cache, get_device_properties, torch_device
from transformers.utils import is_peft_available

from trl import DPOConfig, DPOTrainer, FDivergenceType

from .testing_utils import (
    TrlTestCase,
    require_bitsandbytes,
    require_liger_kernel,
    require_no_wandb,
    require_peft,
    require_torch_accelerator,
    require_torch_gpu_if_bnb_not_multi_backend_enabled,
    require_vision,
)


if is_vision_available():
    from PIL import Image

if is_peft_available():
    from peft import LoraConfig, PeftModel


class TestTokenizeRow(TrlTestCase):
    def setup_method(self):
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
        assert result == {
            "prompt_input_ids": [464, 6766, 318],
            "chosen_input_ids": [4171, 2],  # eos_token added
            "rejected_input_ids": [4077, 2],  # eos_token added
        }

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
        assert result == {
            "prompt_input_ids": [6766, 318],  # truncated to the last 2 tokens
            "chosen_input_ids": [4171],  # truncated to 1 token
            "rejected_input_ids": [4077],  # truncated to 1 token
        }

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
        assert result == {
            "prompt_input_ids": [0, 464, 6766, 318, 2],  # bos_token and eos_token added
            "chosen_input_ids": [4171, 2],  # eos_token added
            "rejected_input_ids": [4077, 2],  # eos_token added
        }

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
        assert result == {
            "prompt_input_ids": [464, 6766, 318, 2],  # truncated to 4 tokens with bos_token and eos_token
            "chosen_input_ids": [4171],  # truncated to 1 token
            "rejected_input_ids": [4077],  # truncated to 1 token
        }


class TestDPOTrainer(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def test_train(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            learning_rate=9e-1,
            report_to="none",
        )
        trainer = DPOTrainer(
            model=model_id,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the parameters have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        "loss_type",
        [
            "sigmoid",
            "hinge",
            "ipo",
            "exo_pair",
            "nca_pair",
            "robust",
            "bco_pair",
            "sppo_hard",
            "aot",
            "aot_pair",
            "discopop",
            "apo_zero",
            "apo_down",
        ],
    )
    def test_train_loss_types(self, loss_type):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            learning_rate=9e-1,
            loss_type=loss_type,
            report_to="none",
        )
        trainer = DPOTrainer(
            model=model_id,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the parameters have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)

    @require_liger_kernel
    def test_train_encoder_decoder_liger(self):
        model_id = "trl-internal-testing/tiny-BartModel"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            learning_rate=9e-1,
            report_to="none",
            use_liger_kernel=True,
        )
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the parameters have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)

    def test_dpo_trainer_with_weighting(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            learning_rate=9e-1,
            use_weighting=True,
            report_to="none",
        )

        trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the parameters have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)

    def test_train_with_multiple_loss_types(self):
        """
        Tests multi-loss combinations, loss type inference, and weight configuration. MPO combines DPO (sigmoid), BCO
        (bco_pair), and SFT (sft) losses.
        """
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            learning_rate=9e-1,
            loss_type=["sigmoid", "bco_pair", "sft"],
            loss_weights=[0.8, 0.2, 1.0],
            report_to="none",
        )
        trainer = DPOTrainer(
            model=model_id,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset,
        )

        # Test that training works
        trainer.train()
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Verify SFT loss is computed in the first test too
        with torch.no_grad():
            batch = next(iter(trainer.get_train_dataloader()))
            loss, metrics = trainer.get_batch_loss_metrics(trainer.model, batch)
            assert "nll_loss" in metrics  # SFT loss should be computed

    def test_wrong_loss_weights_length(self):
        with pytest.raises(ValueError, match="Length of loss_weights list"):
            DPOConfig(
                output_dir=self.tmp_dir,
                loss_type=["sigmoid", "bco_pair"],
                loss_weights=[1.0, 0.5, 0.1],  # Wrong length
            )

    @pytest.mark.parametrize("rpo_alpha", [None, 0.5])
    def test_dpo_trainer_without_providing_ref_model(self, rpo_alpha):
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the parameters have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.equal(param, new_param)

    def test_dpo_trainer_with_ref_model_is_model(self):
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            report_to="none",
        )

        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        with pytest.raises(ValueError):
            DPOTrainer(
                model=self.model,
                ref_model=self.model,  # ref_model can't be the same as model
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset["train"],
            )

    def test_precompute_ref_batch_size(self):
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=2,
            precompute_ref_log_probs=True,
            precompute_ref_batch_size=4,
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

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the parameters have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)

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

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the parameters have changed
        for n, param in previous_trainable_params.items():
            if "lora" in n:
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
                    assert not torch.equal(param, new_param)

    def test_dpo_trainer_w_dataset_num_proc(self):
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
            learning_rate=9e-1,
            eval_strategy="steps",
            beta=0.1,
            dataset_num_proc=2,
            report_to="none",
        )

        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
        )

        trainer.train()

    def test_tr_dpo_trainer(self):
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the parameters have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.ref_model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.equal(param, new_param)

    @require_no_wandb
    def test_dpo_trainer_generate_during_eval_no_wandb(self):
        training_args = DPOConfig(
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

        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        with pytest.raises(
            ValueError,
            match="`generate_during_eval=True` requires Weights and Biases, MLFlow or Comet to be installed."
            " Please install `wandb`, `mlflow` or `comet-ml` to resolve.",
        ):
            DPOTrainer(
                model=self.model,
                ref_model=None,
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

    @require_bitsandbytes
    @require_peft
    @require_torch_gpu_if_bnb_not_multi_backend_enabled
    def test_dpo_lora_bf16_autocast_llama(self):
        # Note this test only works on compute capability > 7 GPU devices
        from peft import LoraConfig
        from transformers import BitsAndBytesConfig

        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # lora model
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            remove_unused_columns=False,
            gradient_accumulation_steps=4,
            learning_rate=9e-1,
            eval_strategy="steps",
            bf16=True,
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

        # train the model
        trainer.train()

        # save peft adapter
        trainer.save_model()

    @pytest.mark.parametrize(
        "loss_type, pre_compute, gen_during_eval",
        [
            ("sigmoid", False, False),
            ("sigmoid", False, True),
            ("sigmoid", True, False),
            ("sigmoid", True, True),
            ("ipo", False, False),
            ("ipo", False, True),
            ("ipo", True, False),
            ("ipo", True, True),
            ("aot_pair", False, False),
            ("aot_pair", False, True),
            ("aot_pair", True, False),
            ("aot_pair", True, True),
            ("aot", False, False),
            ("aot", False, True),
            ("aot", True, False),
            ("aot", True, True),
            ("bco_pair", False, False),
            ("bco_pair", False, True),
            ("bco_pair", True, False),
            ("bco_pair", True, True),
            ("robust", False, False),
            ("robust", False, True),
            ("robust", True, False),
            ("robust", True, True),
        ],
    )
    @require_bitsandbytes
    @require_peft
    @pytest.mark.skipif(
        get_device_properties()[0] == "cuda" and get_device_properties()[1] < 8,
        reason="Skipping because bf16 not supported on CUDA GPU with capability < 8.0",
    )
    def test_dpo_lora_bf16_autocast(self, loss_type, pre_compute, gen_during_eval):
        from peft import LoraConfig
        from transformers import BitsAndBytesConfig

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # lora model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
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

        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
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

        training_args = DPOConfig(
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
            assert tag in trainer.model.model_tags

    @require_peft
    def test_dpo_tags(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # lora model
        model = AutoModelForCausalLM.from_pretrained(model_id)

        training_args = DPOConfig(
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
            assert tag in trainer.model.model_tags

    def test_dpo_trainer_dtype(self):
        # See https://github.com/huggingface/trl/issues/1751
        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            model_init_kwargs={"dtype": "float16"},
            ref_model_init_kwargs={"dtype": "float16"},
            report_to="none",
        )

        trainer = DPOTrainer(
            model=self.model_id,
            ref_model=self.model_id,
            processing_class=self.tokenizer,
            args=training_args,
            train_dataset=dummy_dataset["train"],
        )
        assert trainer.model.config.dtype == torch.float16
        assert trainer.ref_model.config.dtype == torch.float16

        # Now test when `dtype` is provided but is wrong to either the model or the ref_model
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            model_init_kwargs={"dtype": -1},
            report_to="none",
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Invalid `dtype` passed to the config. Expected either 'auto' or a string representing a valid `torch.dtype` (e.g., 'float32'), but got -1."
            ),
        ):
            _ = DPOTrainer(
                model=self.model_id,
                processing_class=self.tokenizer,
                args=training_args,
                train_dataset=dummy_dataset["train"],
            )

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            ref_model_init_kwargs={"dtype": -1},
            report_to="none",
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Invalid `dtype` passed to the config. Expected either 'auto' or a string representing a valid `torch.dtype` (e.g., 'float32'), but got -1."
            ),
        ):
            _ = DPOTrainer(
                model=self.model_id,
                ref_model=self.model_id,
                processing_class=self.tokenizer,
                args=training_args,
                train_dataset=dummy_dataset["train"],
            )

    def test_dpo_loss_alpha_div_f(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # lora model
        model = AutoModelForCausalLM.from_pretrained(model_id)
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
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
        assert torch.isfinite(losses).cpu().numpy().all()

    def test_dpo_loss_js_div_f(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # lora model
        model = AutoModelForCausalLM.from_pretrained(model_id)

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
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
        assert torch.isfinite(losses).cpu().numpy().all()

    def test_dpo_trainer_use_logits_to_keep(self):
        model_id = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id)

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
            learning_rate=9e-1,
            eval_strategy="steps",
            beta=0.1,
            use_logits_to_keep=True,
            rpo_alpha=0.5,
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

        training_args.use_logits_to_keep = False
        trainer2 = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
        )

        # Fake batch
        prompt_input_ids = torch.randint(1, 1000, (2, 10))
        chosen_input_ids = torch.randint(1, 1000, (2, 5))
        rejected_input_ids = torch.randint(1, 1000, (2, 7))
        prompt_attention_mask = torch.ones_like(prompt_input_ids)
        chosen_attention_mask = torch.ones_like(chosen_input_ids)
        rejected_attention_mask = torch.ones_like(rejected_input_ids)

        batch = {
            "prompt_input_ids": prompt_input_ids.to(model.device),
            "chosen_input_ids": chosen_input_ids.to(model.device),
            "rejected_input_ids": rejected_input_ids.to(model.device),
            "prompt_attention_mask": prompt_attention_mask.to(model.device),
            "chosen_attention_mask": chosen_attention_mask.to(model.device),
            "rejected_attention_mask": rejected_attention_mask.to(model.device),
        }

        output = trainer.concatenated_forward(model, batch)
        output2 = trainer2.concatenated_forward(model, batch)

        np.testing.assert_allclose(output["nll_loss"].item(), output2["nll_loss"].item(), atol=1e-5)
        np.testing.assert_allclose(
            output["mean_chosen_logits"].item(), output2["mean_chosen_logits"].item(), atol=1e-5
        )
        np.testing.assert_allclose(
            output["mean_rejected_logits"].item(), output2["mean_rejected_logits"].item(), atol=1e-5
        )

        for i in range(output["chosen_logps"].shape[0]):
            np.testing.assert_allclose(output["chosen_logps"][i].item(), output2["chosen_logps"][i].item(), atol=1e-5)
            np.testing.assert_allclose(
                output["rejected_logps"][i].item(), output2["rejected_logps"][i].item(), atol=1e-5
            )

        trainer.train()

    def test_dpo_trainer_with_tools(self):
        model_id = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id)

        # Define dummy test tools
        def get_current_temperature(location: str):
            """
            Gets the temperature at a given location.

            Args:
                location: The location to get the temperature for
            """
            return 22.0

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            tools=[get_current_temperature],
        )

        dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_preference")

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
        )
        # We don't run the training, but at this stage, the dataset is supposed to be pre-processed. When
        # pre-processing, we expect the available tools to be explicitly mentioned in the system prompt. That's
        # what we're checking here
        assert "get_current_temperature" in tokenizer.decode(trainer.train_dataset["prompt_input_ids"][0])

    def test_padding_free(self):
        model_id = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        # Normally, we need `attn_implementation="flash_attention_2"` to that the model returns correct logits.
        # Without it, the logits may be incorrect, but that's fine here. This test focuses only on the inner logic
        # of padding_free.
        model = AutoModelForCausalLM.from_pretrained(model_id)

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=9e-1,
            per_device_train_batch_size=2,
            padding_free=True,
            report_to="none",
        )

        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        trainer = DPOTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dummy_dataset["train"],
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        # Check that the parameters have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)

    def test_compute_metrics(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        ref_model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        tokenizer.pad_token = tokenizer.eos_token

        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        def dummy_compute_metrics(*args, **kwargs):
            return {"test": 0.0}

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            do_eval=True,
            eval_strategy="steps",
            eval_steps=3,
            per_device_eval_batch_size=2,
            report_to="none",
        )

        trainer = DPOTrainer(
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

    def test_train_with_length_desensitization(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            learning_rate=9e-1,
            ld_alpha=0.5,
            report_to="none",
        )
        trainer = DPOTrainer(
            model=model_id,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the parameters have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        "beta, loss_type",
        [
            (0.1, "sigmoid"),
            (0.1, "apo_zero"),
            (0.1, "apo_down"),
            (0.1, "sppo_hard"),
            (0.1, "nca_pair"),
            (0.5, "sigmoid"),
            (0.5, "apo_zero"),
            (0.5, "apo_down"),
            (0.5, "sppo_hard"),
            (0.5, "nca_pair"),
        ],
    )
    @require_liger_kernel
    def test_dpo_trainer_with_liger(self, beta, loss_type):
        """Test DPO trainer with Liger loss enabled across supported loss types.

        This test verifies that:
        1. Training runs successfully with Liger loss
        2. Model parameters update as expected
        3. Loss values are reasonable and finite
        4. Training works with both default and custom beta values
        """
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            do_eval=True,
            eval_steps=1,
            learning_rate=9e-1,
            eval_strategy="steps",
            beta=beta,
            use_liger_kernel=True,  # Enable Liger kernel
            loss_type=loss_type,
            report_to="none",
        )

        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,  # Add reference model
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
        )

        # Store initial parameters
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        train_output = trainer.train()

        # Verify training completed successfully
        assert train_output is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Verify loss is finite
        assert np.isfinite(trainer.state.log_history[-1]["train_loss"])

        # Check parameters have been updated
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            # Only check non-zero parameters
            if param.sum() != 0:
                assert not torch.equal(param, new_param)
                # Verify new parameters are finite
                assert torch.isfinite(new_param).all()

        # Verify model can still do forward pass after training
        dummy_batch = next(iter(trainer.get_train_dataloader()))
        model_inputs = {
            "input_ids": dummy_batch["prompt_input_ids"],
            "attention_mask": dummy_batch["prompt_attention_mask"],
        }
        with torch.no_grad():
            output = trainer.model(**model_inputs)
        assert output is not None
        assert "loss" not in output.keys()

    def test_train_with_iterable_dataset(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset(
            "trl-internal-testing/zen",
            "standard_preference",
            split="train",
            streaming=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            max_steps=3,
            report_to="none",
        )
        trainer = DPOTrainer(
            model=model_id,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the parameters have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)


@require_vision
class TestDPOVisionTrainer(TrlTestCase):
    @pytest.mark.parametrize(
        "model_id",
        [
            # "trl-internal-testing/tiny-Idefics2ForConditionalGeneration",  high memory peak, skipped for now
            "trl-internal-testing/tiny-LlavaForConditionalGeneration",
            "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
            "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
        ],
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
        model = AutoModelForImageTextToText.from_pretrained(model_id)
        ref_model = AutoModelForImageTextToText.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            remove_unused_columns=False,
            learning_rate=0.01,  # increase learning rate to speed up test
            max_prompt_length=None,  # don't truncate to avoid issues with patch tokens
            max_length=None,
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

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the trainable params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                if model_id in [
                    "trl-internal-testing/tiny-LlavaForConditionalGeneration",
                    "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
                ] and (
                    "vision_tower.vision_model.encoder.layers.1" in n
                    or "vision_tower.vision_model.post_layernorm.weight" in n
                ):
                    # For some reason, these params are not updated. This is probably not related to TRL, but to
                    # the model itself. We should investigate this further, but for now we just skip these params.
                    continue
                assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12), f"Param {n} is not updated"


class TestDPOConfig(TrlTestCase):
    @pytest.mark.parametrize("as_string", [False, True])
    @pytest.mark.parametrize("f_divergence_type", list(FDivergenceType))
    def test_f_divergence_type(self, f_divergence_type, as_string: bool):
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            report_to="none",
            f_divergence_type=f_divergence_type.value if as_string else f_divergence_type,
        )

        # Internal normalization: keep Enum member
        assert isinstance(training_args.f_divergence_type, FDivergenceType)
        assert training_args.f_divergence_type == f_divergence_type

        # Serialization: TrainingArguments.to_dict should yield the enum's string value
        configparser_dict = training_args.to_dict()
        assert configparser_dict["f_divergence_type"] == f_divergence_type.value


@pytest.mark.slow
@require_torch_accelerator
@require_peft
class TestDPOTrainerSlow(TrlTestCase):
    def setup_method(self):
        self.dataset = load_dataset("trl-internal-testing/zen", "standard_preference")
        self.peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.max_length = 128

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    @pytest.mark.parametrize("pre_compute_logits", [True, False])
    @pytest.mark.parametrize("loss_type", ["sigmoid", "ipo"])
    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
            "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        ],
    )
    def test_dpo_bare_model(self, model_id, loss_type, pre_compute_logits):
        """
        A test that tests the simple usage of `DPOTrainer` using a bare model in full precision.
        """
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=2,
            remove_unused_columns=False,
            gradient_accumulation_steps=2,
            learning_rate=9e-1,
            eval_strategy="steps",
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
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            processing_class=tokenizer,
        )

        # train the model
        trainer.train()

        # save trained model or adapter
        trainer.save_model()

        release_memory(model, trainer)

    @pytest.mark.parametrize(
        "gradient_checkpointing_kwargs", [None, {"use_reentrant": False}, {"use_reentrant": True}]
    )
    @pytest.mark.parametrize("pre_compute_logits", [True, False])
    @pytest.mark.parametrize("loss_type", ["sigmoid", "ipo"])
    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
            "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        ],
    )
    @require_peft
    def test_dpo_peft_model(self, model_id, loss_type, pre_compute_logits, gradient_checkpointing_kwargs):
        """
        A test that tests the simple usage of `DPOTrainer` using a peft model in full precision + different scenarios
        of gradient checkpointing.
        """
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
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
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            processing_class=tokenizer,
            peft_config=self.peft_config,
        )

        assert isinstance(trainer.model, PeftModel)
        assert trainer.ref_model is None

        # train the model
        trainer.train()

        # save trained model or adapter
        trainer.save_model()

        release_memory(model, trainer)

    @pytest.mark.parametrize(
        "gradient_checkpointing_kwargs", [None, {"use_reentrant": False}, {"use_reentrant": True}]
    )
    @pytest.mark.parametrize("pre_compute_logits", [True, False])
    @pytest.mark.parametrize("loss_type", ["sigmoid", "ipo"])
    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
            "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        ],
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
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
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
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            processing_class=tokenizer,
            peft_config=self.peft_config,
        )

        assert isinstance(trainer.model, PeftModel)
        assert trainer.ref_model is None

        # train the model
        trainer.train()

        # save trained model or adapter
        trainer.save_model()

        release_memory(model, trainer)
