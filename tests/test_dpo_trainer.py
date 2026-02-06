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

import pytest
import torch
import transformers
from datasets import load_dataset
from packaging.version import Version
from packaging.version import parse as parse_version
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_peft_available

from trl import DPOConfig, DPOTrainer
from trl.trainer.dpo_trainer import DataCollatorForPreference

from .testing_utils import (
    TrlTestCase,
    require_ampere_or_newer,
    require_bitsandbytes,
    require_kernels,
    require_liger_kernel,
    require_peft,
    require_vision,
)


if is_peft_available():
    from peft import LoraConfig, get_peft_model


class TestDataCollatorForPreference(TrlTestCase):
    def test_padding_and_masks(self):
        collator = DataCollatorForPreference(pad_token_id=0)
        examples = [
            {"prompt_ids": [1, 2, 3], "chosen_ids": [4, 5], "rejected_ids": [6]},
            {"prompt_ids": [7, 8], "chosen_ids": [9, 10], "rejected_ids": [11, 12, 13]},
        ]
        result = collator(examples)

        expected_input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],  # prompt + chosen (example 1)
                [7, 8, 9, 10, 0],  # prompt + chosen (example 2, padded)
                [1, 2, 3, 6, 0],  # prompt + rejected (example 1, padded)
                [7, 8, 11, 12, 13],  # prompt + rejected (example 2)
            ]
        )
        expected_attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        expected_completion_mask = torch.tensor(
            [
                [0, 0, 0, 1, 1],  # chosen completion (example 1)
                [0, 0, 1, 1, 0],  # chosen completion (example 2, padded)
                [0, 0, 0, 1, 0],  # rejected completion (example 1, padded)
                [0, 0, 1, 1, 1],  # rejected completion (example 2)
            ]
        )

        assert set(result.keys()) == {"input_ids", "attention_mask", "completion_mask"}
        torch.testing.assert_close(result["input_ids"], expected_input_ids)
        torch.testing.assert_close(result["attention_mask"], expected_attention_mask)
        torch.testing.assert_close(result["completion_mask"], expected_completion_mask)

    def test_optional_reference_logps(self):
        collator = DataCollatorForPreference(pad_token_id=0)
        examples = [
            {
                "prompt_ids": [1, 2],
                "chosen_ids": [3],
                "rejected_ids": [4],
                "ref_chosen_logps": 0.1,
                "ref_rejected_logps": 0.2,
            },
            {
                "prompt_ids": [5],
                "chosen_ids": [6, 7],
                "rejected_ids": [8, 9],
                "ref_chosen_logps": 0.3,
                "ref_rejected_logps": 0.4,
            },
        ]
        result = collator(examples)

        expected_ref_chosen_logps = torch.tensor([0.1, 0.3])
        expected_ref_rejected_logps = torch.tensor([0.2, 0.4])

        assert set(result.keys()) == {
            "input_ids",
            "attention_mask",
            "completion_mask",
            "ref_chosen_logps",
            "ref_rejected_logps",
        }
        torch.testing.assert_close(result["ref_chosen_logps"], expected_ref_chosen_logps)
        torch.testing.assert_close(result["ref_rejected_logps"], expected_ref_rejected_logps)

    def test_with_pad_to_multiple_of(self):
        collator = DataCollatorForPreference(pad_token_id=0, pad_to_multiple_of=5)
        examples = [
            {"prompt_ids": [1], "chosen_ids": [2], "rejected_ids": [3]},
            {"prompt_ids": [4, 5], "chosen_ids": [6, 7], "rejected_ids": [8, 9]},
        ]
        result = collator(examples)

        expected_input_ids = torch.tensor(
            [
                [1, 2, 0, 0, 0],  # prompt + chosen (example 1, padded to multiple of 5)
                [4, 5, 6, 7, 0],  # prompt + chosen (example 2)
                [1, 3, 0, 0, 0],  # prompt + rejected (example 1, padded to multiple of 5)
                [4, 5, 8, 9, 0],  # prompt + rejected (example 2)
            ]
        )

        assert set(result.keys()) == {"input_ids", "attention_mask", "completion_mask"}
        torch.testing.assert_close(result["input_ids"], expected_input_ids)


class TestDPOTrainer(TrlTestCase):
    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            "trl-internal-testing/tiny-GptOssForCausalLM",
        ],
    )
    def test_train(self, model_id):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(model=model_id, args=training_args, train_dataset=dataset)

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    # Special case for harmony
    def test_train_gpt_oss(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/harmony", "preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-GptOssForCausalLM", args=training_args, train_dataset=dataset
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_train_model(self):
        # Instantiate the model
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            dtype="float32",
        )

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(model=model, args=training_args, train_dataset=dataset)

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

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
            "aot_unpaired",
            "apo_zero",
            "apo_down",
            "discopop",
            "sft",
        ],
    )
    def test_train_loss_types(self, loss_type):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            loss_type=loss_type,
            label_smoothing=1e-3 if loss_type == "exo_pair" else 0.0,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
            eval_strategy="steps",
            eval_steps=3,
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_train_multi_loss_types(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            loss_type=["sigmoid", "bco_pair", "sft"],  # this specific combination is used in MPO
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_train_with_wpo(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
            use_weighting=True,
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_train_with_ld(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
            ld_alpha=0.5,
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @pytest.mark.parametrize(
        "f_divergence_type",
        ["reverse_kl", "forward_kl", "js_divergence", "alpha_divergence"],
    )
    def test_train_with_f_divergence(self, f_divergence_type):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
            f_divergence_type=f_divergence_type,
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_train_with_explicit_ref_model(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            report_to="none",
        )
        # When specifying a ref model, it's usually because we want it to be a different checkpoint, but for testing
        # purposes we will just just use the same checkpoint
        ref_model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32"
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            ref_model=ref_model,
            args=training_args,
            train_dataset=dataset,
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"
            new_ref_param = trainer.ref_model.get_parameter(n)
            torch.testing.assert_close(param, new_ref_param), f"Reference model parameter {n} has changed"

    def test_training_with_sync_ref_model(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            sync_ref_model=True,
            ref_model_sync_steps=2,  # reduce sync steps to ensure a sync happens
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
        assert trainer.ref_model is not None
        previous_ref_params = {n: param.clone() for n, param in trainer.ref_model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
            new_ref_param = trainer.ref_model.get_parameter(n)
            assert not torch.equal(previous_ref_params[n], new_ref_param), f"Ref Parameter {n} has not changed."

    def test_train_model_dtype(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            model_init_kwargs={"dtype": torch.float16},
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            # For some reasonn model.layers.0.input_layernorm.weight doesn't change in GitHub Actions but does
            # locally. We ignore this parameter for now
            if "layernorm" in n:
                continue
            new_param = trainer.model.get_parameter(n)
            # Check the torch dtype
            assert new_param.dtype == torch.float16
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_peft
    def test_train_dense_with_peft_config_lora(self):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=1.0,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )

        trainer = DPOTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model parameters to be the same
                torch.testing.assert_close(param, new_param), f"Parameter {n} has changed"
            elif "base_layer" not in n:  # We expect the peft parameters to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_peft
    def test_train_moe_with_peft_config(self):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-GptOssForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )

        trainer = DPOTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(target_parameters=["mlp.experts.down_proj", "mlp.experts.gate_up_proj"]),
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model parameters to be the same
                torch.testing.assert_close(param, new_param), f"Parameter {n} has changed"
            elif "base_layer" not in n:  # We expect the peft parameters to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_peft
    def test_train_peft_model(self):
        # Get the base model
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")

        # Get the base model parameter names
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Turn the model into a peft model
        lora_config = LoraConfig()
        model = get_peft_model(model, lora_config)

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=1.0,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(model=model, args=training_args, train_dataset=dataset)

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model parameters to be the same
                torch.testing.assert_close(param, new_param), f"Parameter {n} has changed"
            elif "base_layer" not in n and "ref" not in n:  # and the peft params to be different (except base and ref)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    # In practice, this test is the same as `test_train_dense_with_peft_config_lora`, since gradient checkpointing is
    # enabled by default in `DPOTrainer`. We keep it as a regression guard: if the default ever changes, we still
    # explicitly test PEFT + gradient checkpointing, which has caused issues in the past.
    @require_peft
    def test_train_with_peft_config_and_gradient_checkpointing(self):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            gradient_checkpointing=True,
            report_to="none",
        )

        trainer = DPOTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model parameters to be the same
                torch.testing.assert_close(param, new_param), f"Parameter {n} has changed"
            elif "base_layer" not in n:  # We expect the peft parameters to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_liger_kernel
    def test_train_with_liger(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            use_liger_kernel=True,
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_train_with_iterable_dataset(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train", streaming=True)

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_steps=3,
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_kernels
    @require_ampere_or_newer  # Flash attention 2 requires Ampere or newer GPUs
    def test_train_padding_free(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            padding_free=True,
            model_init_kwargs={"attn_implementation": "kernels-community/flash-attn2"},
            bf16=True,  # flash_attention_2 only supports bf16 and fp16
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_train_with_chat_template_kwargs(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )

        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        # The following template is a simplified version of the Qwen chat template, where an additional argument
        # `role_capital` is used to control the capitalization of roles.
        tokenizer.chat_template = '{%- if messages[0]["role"] == "system" -%}    {{ "<|im_start|>" + ("SYSTEM" if role_capital else "system") + "\\n" + messages[0]["content"] + "<|im_end|>\\n" }}{%- else -%}    {{ "<|im_start|>" + ("SYSTEM" if role_capital else "system") + "\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n" }}{%- endif -%}{%- for message in messages -%}    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) -%}        {{ "<|im_start|>" + (message.role.upper() if role_capital else message.role) + "\\n" + message.content + "<|im_end|>\\n" }}    {%- elif message.role == "assistant" -%}        {{ "<|im_start|>" + ("ASSISTANT" if role_capital else "assistant") }}        {%- if message.content -%}            {{ "\\n" + message.content }}        {%- endif -%}        {{ "<|im_end|>\\n" }}    {%- elif message.role == "tool" -%}        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") -%}            {{ "<|im_start|>" + ("USER" if role_capital else "user") }}        {%- endif -%}        {{ "\\n<tool_response>\\n" + message.content + "\\n</tool_response>" }}        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") -%}            {{ "<|im_end|>\\n" }}        {%- endif -%}    {%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}    {{ "<|im_start|>" + ("ASSISTANT" if role_capital else "assistant") + "\\n" }}{%- endif -%}'

        dataset = dataset.add_column(
            "chat_template_kwargs", [{"role_capital": bool(i % 2)} for i in range(len(dataset))]
        )
        assert "chat_template_kwargs" in dataset.features

        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        # Assert trainer uses the same chat template as tokenizer
        assert trainer.processing_class.chat_template == tokenizer.chat_template

        # Assert chat_template is applied
        for i in range(2):
            role = "SYSTEM" if i else "system"
            system_prompt = (
                f"<|im_start|>{role}\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>"
            )
            system_prompt_ids = trainer.processing_class(system_prompt)["input_ids"]
            assert trainer.train_dataset[i]["prompt_ids"][: len(system_prompt_ids)] == system_prompt_ids

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_train_toolcall_data(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/toolcall", "preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_train_with_eval(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        # Initialize the trainer
        training_args = DPOConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        # Train the model
        trainer.train()

        # Check that the eval loss is not None
        assert trainer.state.log_history[0]["eval_loss"] is not None

    def test_train_with_multiple_eval_dataset(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        # Initialize the trainer
        training_args = DPOConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset={"data1": dataset["test"], "data2": dataset["test"]},
        )
        # Train the model
        trainer.train()

        # Check that the eval losses are not None
        assert trainer.state.log_history[-3]["eval_data1_loss"] is not None
        assert trainer.state.log_history[-2]["eval_data2_loss"] is not None

    def test_train_with_compute_metrics(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        def dummy_compute_metrics(eval_pred):
            return {"my_metric": 0.123}

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            eval_strategy="steps",
            eval_steps=3,
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=dummy_compute_metrics,
        )

        # Train the model
        trainer.train()

        # Check that the custom metric is logged
        assert trainer.state.log_history[-2]["eval_my_metric"] == 0.123

    # In practice, this test is the same as `test_train`, since gradient checkpointing is enabled by default in
    # `DPOTrainer`. We keep it as a regression guard: if the default ever changes, we still explicitly test gradient
    # checkpointing, which has caused issues in the past.
    def test_train_with_gradient_checkpointing(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            gradient_checkpointing=True,
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_tag_added(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
        )

        for tag in ["dpo", "trl"]:
            assert tag in trainer.model.model_tags

    @require_peft
    def test_tag_added_peft(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        for tag in ["dpo", "trl"]:
            assert tag in trainer.model.model_tags

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
            # "trl-internal-testing/tiny-Idefics2ForConditionalGeneration",  high memory peak, skipped for now
            # "trl-internal-testing/tiny-Idefics3ForConditionalGeneration",  high memory peak, skipped for now
            "trl-internal-testing/tiny-LlavaForConditionalGeneration",
            "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
            "trl-internal-testing/tiny-Qwen2VLForConditionalGeneration",
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            # "trl-internal-testing/tiny-SmolVLMForConditionalGeneration", seems not to support bf16 properly
            pytest.param(
                "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration",
                marks=[
                    pytest.mark.skipif(
                        Version(transformers.__version__) < Version("4.57.0"),
                        reason="Qwen3-VL series were introduced in transformers-4.57.0",
                    ),
                    pytest.mark.xfail(
                        Version(transformers.__version__) >= Version("5.0.0"),
                        reason="Blocked by upstream transformers bug (transformers#43334)",
                    ),
                ],
            ),
        ],
    )
    @require_vision
    def test_train_vlm(self, model_id):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            report_to="none",
        )
        trainer = DPOTrainer(model=model_id, args=training_args, train_dataset=dataset)

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            # For some reason, these params are not updated. This is probably not related to TRL, but to
            # the model itself. We should investigate this further, but for now we just skip these params.
            # fmt: off
            if (
                model_id == "trl-internal-testing/tiny-Gemma3ForConditionalGeneration" and "model.vision_tower.vision_model.head" in n or
                model_id == "trl-internal-testing/tiny-LlavaForConditionalGeneration" and "model.vision_tower.vision_model.post_layernorm" in n or
                model_id == "trl-internal-testing/tiny-LlavaForConditionalGeneration" and "vision_tower.vision_model.encoder.layers.1" in n or
                model_id == "trl-internal-testing/tiny-LlavaNextForConditionalGeneration" and "model.vision_tower.vision_model.post_layernorm" in n or
                model_id == "trl-internal-testing/tiny-LlavaNextForConditionalGeneration" and "vision_tower.vision_model.encoder.layers.1" in n or
                model_id == "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration" and "model.visual.deepstack_merger_list" in n
            ):
            # fmt: on
                continue
            assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12), f"Param {n} is not updated"

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        ],
    )
    @pytest.mark.xfail(
        parse_version(transformers.__version__) < parse_version("4.57.0"),
        reason="Mixing text-only and image+text examples is only supported in transformers >= 4.57.0",
        strict=False,
    )
    @require_vision
    def test_train_vlm_multi_image(self, model_id):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen-multi-image", "conversational_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            per_device_train_batch_size=1,  # VLM training is memory intensive, reduce batch size to avoid OOM
            report_to="none",
        )
        trainer = DPOTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12), f"Param {n} is not updated"

    # Gemma 3n uses a timm encoder, making it difficult to create a smaller variant for testing.
    # To ensure coverage, we run tests on the full model but mark them as slow to exclude from default runs.
    @pytest.mark.slow
    @require_vision
    @pytest.mark.skip(reason="Model google/gemma-3n-E2B-it is gated and requires HF token")
    def test_train_vlm_gemma_3n(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_preference", split="train")

        # Initialize the trainer
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            per_device_train_batch_size=1,  # VLM training is memory intensive, reduce batch size to avoid OOM
            model_init_kwargs={"dtype": "bfloat16"},
            report_to="none",
        )
        trainer = DPOTrainer(model="google/gemma-3n-E2B-it", args=training_args, train_dataset=dataset)

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if "model.audio_tower" in n or "model.embed_audio" in n:
                # The audio embedding parameters are not updated because this dataset contains no audio data
                continue
            assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12), f"Param {n} is not updated"

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        ],
    )
    @pytest.mark.parametrize(
        "dataset_config",
        ["conversational_preference", "standard_preference"],
    )
    @require_vision
    def test_train_vlm_text_only_data(self, model_id, dataset_config):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", dataset_config, split="train")

        # Initialize the trainer
        training_args = DPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = DPOTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n.startswith("model.visual"):
                torch.testing.assert_close(param, new_param, rtol=1e-12, atol=1e-12), f"Param {n} is updated"
            else:
                assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12), f"Param {n} is not updated"

    @require_peft
    @require_bitsandbytes
    def test_peft_with_quantization(self):
        # Get the base model
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="float32",
            quantization_config=quantization_config,
        )

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer with the already configured PeftModel
        training_args = DPOConfig(output_dir=self.tmp_dir, learning_rate=0.1, report_to="none")
        trainer = DPOTrainer(model=model, args=training_args, train_dataset=dataset, peft_config=LoraConfig())

        # Save initial parameters to check they change during training
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        # Check that training completed successfully
        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-1]["mean_token_accuracy"] is not None

        # Check the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            # In bitsandbytes, bias parameters are automatically cast to the input dtype during the forward pass if
            # their dtype doesn’t match. This causes the module to change unexpectedly during the first forward pass of
            # the training. To handle this, we cast these specific bias parameters to float32 before comparison.
            # https://github.com/bitsandbytes-foundation/bitsandbytes/blob/45553f7392e524eacf400b132cfe01261f6477be/bitsandbytes/nn/modules.py#L518
            # We still need to investigate why the compute dtype ends up being different than for these parameters.
            if n in [
                "base_model.model.model.layers.1.self_attn.k_proj.bias",
                "base_model.model.model.layers.1.self_attn.q_proj.base_layer.bias",
                "base_model.model.model.layers.1.self_attn.v_proj.base_layer.bias",
            ]:
                param = param.float()

            if "lora" not in n:  # We expect the base model parameters to be the same
                torch.testing.assert_close(param, new_param), f"Parameter {n} has changed"
            elif "lora" in n:  # We expect the peft parameters to be different
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"
            else:
                raise ValueError(f"Unexpected parameter {n} in model: {trainer.model}")
