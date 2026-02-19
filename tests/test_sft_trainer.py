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

import gc
import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest
import torch
import transformers
from accelerate.utils.memory import release_memory
from datasets import load_dataset
from packaging.version import Version
from packaging.version import parse as parse_version
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from transformers.testing_utils import backend_empty_cache, torch_device
from transformers.utils import is_peft_available

from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling, dft_loss

from .testing_utils import (
    TrlTestCase,
    ignore_warnings,
    require_ampere_or_newer,
    require_bitsandbytes,
    require_kernels,
    require_liger_kernel,
    require_peft,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    require_vision,
)


if is_peft_available():
    import peft
    from peft import (
        LoraConfig,
        PeftModel,
        PrefixTuningConfig,
        PromptEncoderConfig,
        PromptTuningConfig,
        TaskType,
        get_peft_model,
    )


class TestDFTLoss(TrlTestCase):
    def test_dft_loss(self):
        batch_size = 2
        seq_len = 3
        vocab_size = 2
        # All tokens have the same probability
        logits = torch.fill(torch.empty(batch_size, seq_len, vocab_size), torch.rand(1).item())
        outputs = MagicMock()
        outputs.logits = logits
        labels = torch.tensor([[1, 0, 0], [0, 1, -100]])
        ce_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100, reduction="mean"
        )
        # We need to account for the logits shift operation so we don't consider the first tokens
        # in each row of the batch
        num_items_in_batch = 3
        # Dft loss
        predicted_dft_loss = dft_loss(outputs, labels, num_items_in_batch)
        # If we have just two tokens in our vocab and all logits are the same,
        # dft scales the ce_loss per token by 0.5. So the dft_loss should be ce_loss/2
        torch.testing.assert_close(ce_loss / 2.0, predicted_dft_loss, atol=1e-4, rtol=1e-4)


class TestDataCollatorForLanguageModeling(TrlTestCase):
    def test_basic_padding(self):
        """Test basic padding functionality without completion masks."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_completion_mask(self):
        """Test completion mask functionality."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [
            {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
            {"input_ids": [4, 5], "completion_mask": [0, 1]},
        ]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, 2, 3], [-100, 5, -100]]))

    def test_completion_only_loss_disabled(self):
        """Test behavior when completion_only_loss is disabled."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, completion_only_loss=False)
        examples = [
            {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
            {"input_ids": [4, 5], "completion_mask": [0, 1]},
        ]

        result = collator(examples)

        # Labels should not be masked when completion_only_loss=False
        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_padding_free_mode(self):
        """Test padding-free mode where sequences are concatenated."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "position_ids", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, 2, 3, -100, 5]]))

    def test_padding_free_with_completion_mask(self):
        """Test padding-free mode with completion masks."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
        examples = [
            {"input_ids": [1, 2, 3], "completion_mask": [0, 0, 1]},
            {"input_ids": [4, 5], "completion_mask": [1, 1]},
        ]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "position_ids", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, -100, 3, -100, 5]]))

    def test_packing(self):
        """Test that when using packing with position_ids, attention_mask is dropped with fa2."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)

        # Simulate packed sequences with position_ids that restart (typical of BFD packing)
        examples = [
            {"input_ids": [1, 2, 3, 4, 5, 6], "seq_lengths": [3, 3]},
            {"input_ids": [7, 8, 9, 10, 11], "seq_lengths": [4, 1]},
        ]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "position_ids", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, 2, 3, -100, 5, 6, -100, 8, 9, 10, -100]]))

    def test_pad_to_multiple_of(self):
        """Test padding to multiple of specified value."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, pad_to_multiple_of=4)
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3, -100], [4, 5, -100, -100]]))

    def test_pad_to_multiple_of_and_padding_free(self):
        """Test padding to multiple of specified value."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True, pad_to_multiple_of=4)
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "position_ids", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 0, 0, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, 2, 3, -100, 5, -100, -100, -100]]))

    def test_custom_position_ids_but_no_padding_free(self):
        """Test that custom position_ids are ignored if padding_free is False."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [{"input_ids": [1, 2, 3], "seq_lengths": [1, 2]}, {"input_ids": [4, 5], "seq_lengths": [2]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_single_example(self):
        """Test collator with a single example."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [{"input_ids": [1, 2, 3, 4]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1, 1]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3, 4]]))

    def test_different_pad_token_id(self):
        """Test with different pad token ID."""
        collator = DataCollatorForLanguageModeling(pad_token_id=999)
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 999]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_assistant_masks(self):
        """Test handling of assistant masks in examples."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [
            {"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 1]},
            {"input_ids": [4, 5], "assistant_masks": [0, 1]},
        ]

        result = collator(examples)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, 2, 3], [-100, 5, -100]]))

    def test_single_example_single_doc(self):
        batch_seq_lengths = [[5]]
        result = DataCollatorForLanguageModeling.get_position_ids_from_packed_seq_lengths(batch_seq_lengths)
        assert len(result) == 1
        assert torch.equal(result[0], torch.arange(5))

    def test_single_example_multiple_docs(self):
        batch_seq_lengths = [[3, 2]]
        result = DataCollatorForLanguageModeling.get_position_ids_from_packed_seq_lengths(batch_seq_lengths)
        assert len(result) == 1
        # First sequence: 0, 1, 2; second sequence: 0, 1
        assert torch.equal(result[0], torch.tensor([0, 1, 2, 0, 1]))

    def test_multiple_examples(self):
        batch_seq_lengths = [[2, 2], [3]]
        result = DataCollatorForLanguageModeling.get_position_ids_from_packed_seq_lengths(batch_seq_lengths)
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([0, 1, 0, 1]))
        assert torch.equal(result[1], torch.arange(3))


class TestSFTTrainer(TrlTestCase):
    def test_init_with_training_arguments(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        args = TrainingArguments(output_dir=self.tmp_dir, report_to="none")
        SFTTrainer(model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=args, train_dataset=dataset)

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Cohere2ForCausalLM",
            pytest.param(
                "trl-internal-testing/tiny-Glm4MoeForCausalLM",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.0.0"),
                    reason="GLM4 tokenizer requires transformers>=5.0.0",
                ),
            ),
            "trl-internal-testing/tiny-GptOssForCausalLM",
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
        ],
    )
    def test_train(self, model_id):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model_id, args=training_args, train_dataset=dataset)

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
        dataset = load_dataset("trl-internal-testing/harmony", "language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)

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

    def test_train_dft_loss(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            loss_type="dft",
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
            eval_strategy="steps",
            eval_steps=3,
        )
        trainer = SFTTrainer(
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

    def test_train_moe_model_with_aux_loss(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            report_to="none",
            model_init_kwargs={"output_router_logits": True},
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM", args=training_args, train_dataset=dataset
        )
        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss and aux loss are not None
        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-1]["aux_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_train_with_formatting_func(self):
        # Dummy formatting function
        def formatting_prompts_func(example):
            chosen, rejected = example["chosen"], example["rejected"]
            return f"### Chosen: {chosen}\n### Rejected: {rejected}"

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            formatting_func=formatting_prompts_func,
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

    def test_train_model_dtype(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            model_init_kwargs={"dtype": torch.float16},
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = SFTTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")

        trainer = SFTTrainer(
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

    @pytest.mark.parametrize(
        "peft_type",
        [
            "prompt_tuning",
            "prefix_tuning",
            "prompt_encoder",
        ],
    )
    @require_peft
    def test_train_with_peft_config_prompt_tuning(self, peft_type):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer, p-tuning doesn't support gradient checkpointing
        training_args = SFTConfig(bf16=False, output_dir=self.tmp_dir, report_to="none", gradient_checkpointing=False)
        if peft_type == "prompt_tuning":
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=4,
                tokenizer_name_or_path="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            )
        elif peft_type == "prefix_tuning":
            if parse_version(peft.__version__) <= Version("0.17.1"):
                pytest.xfail(
                    "Prefix tuning with device_map='auto' is broken in peft 0.17.1 and below. See "
                    "https://github.com/huggingface/peft/issues/2821"
                )
            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=4,
            )
        elif peft_type == "prompt_encoder":
            peft_config = PromptEncoderConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=4,
                encoder_hidden_size=model.config.hidden_size,  # This will be overwritten below
            )
        trainer = SFTTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
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
            else:  # We expect the peft parameters to be different
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_peft
    def test_train_moe_with_peft_config(self):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-GptOssForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")

        trainer = SFTTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)

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

    # In practice, this test is the same as `test_train_dense_with_peft_config_lora`, since gradient checkpointing is
    # enabled by default in `SFTTrainer`. We keep it as a regression guard: if the default ever changes, we still
    # explicitly test PEFT + gradient checkpointing, which has caused issues in the past.
    @require_peft
    def test_train_with_peft_config_and_gradient_checkpointing(self):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, gradient_checkpointing=True, report_to="none")

        trainer = SFTTrainer(
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

    @pytest.mark.parametrize("use_reentrant", [True, False])
    @require_peft
    def test_train_with_peft_config_and_gradient_checkpointing_reentrant(self, use_reentrant):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
            report_to="none",
        )

        trainer = SFTTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, use_liger_kernel=True, report_to="none")
        trainer = SFTTrainer(
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

    @require_torch_accelerator
    @require_liger_kernel
    def test_compute_loss_skip_logits_on_eval_without_metrics_with_liger(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train[:1]")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            use_liger_kernel=False,
            report_to="none",
            max_length=8,
            bf16=False,
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            compute_metrics=None,
        )
        trainer.args.use_liger_kernel = True
        trainer.model.eval()

        captured = {}

        def mock_super_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
            captured["skip_logits"] = inputs.get("skip_logits")
            dummy_loss = torch.tensor(1.0, requires_grad=True)
            dummy_outputs = MagicMock()
            dummy_outputs.token_accuracy = None
            dummy_outputs.logits = torch.randn(1, 5, trainer.model.config.vocab_size)
            return (dummy_loss, dummy_outputs)

        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "labels": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

        with patch("trl.trainer.sft_trainer.BaseTrainer.compute_loss", side_effect=mock_super_compute_loss):
            trainer.compute_loss(trainer.model, inputs)

        assert captured["skip_logits"] is True

    @require_torch_accelerator
    @require_liger_kernel
    def test_predict_does_not_skip_logits_with_liger(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train[:1]")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            use_liger_kernel=False,
            report_to="none",
            max_length=8,
            bf16=False,
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            compute_metrics=None,
        )
        trainer.args.use_liger_kernel = True
        trainer.model.eval()

        captured = {}

        def mock_super_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
            captured["skip_logits"] = inputs.get("skip_logits")
            dummy_loss = torch.tensor(1.0, requires_grad=True)
            dummy_outputs = (dummy_loss, torch.randn(1, 5, trainer.model.config.vocab_size))
            return (dummy_loss, dummy_outputs)

        with patch("trl.trainer.sft_trainer.BaseTrainer.compute_loss", side_effect=mock_super_compute_loss):
            trainer.predict(trainer.train_dataset)

        assert captured["skip_logits"] is False

    def test_train_with_non_chatml_conversational_data(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        # Rename role/content to from/value to ensure SFT works with non-chatML conversational data
        def rename_fields(example: list[dict]):
            return {"conversations": [{"from": m["role"], "value": m["content"]} for m in example["messages"]]}

        dataset = dataset.map(rename_fields, remove_columns="messages")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
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

    def test_train_with_pretokenized_data(self):
        # Get the dataset
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        def tokenize_example(example):
            return tokenizer(example["text"])

        # Apply tokenization
        tokenized_dataset = dataset.map(tokenize_example, remove_columns=["text"])

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model_id, args=training_args, train_dataset=tokenized_dataset)

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
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train", streaming=True)

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, max_steps=3, report_to="none")
        trainer = SFTTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            padding_free=True,
            model_init_kwargs={"attn_implementation": "kernels-community/flash-attn2"},
            bf16=True,  # flash_attention_2 only supports bf16 and fp16
            report_to="none",
        )
        trainer = SFTTrainer(
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

    @pytest.mark.parametrize("packing_strategy", ["bfd", "wrapped"])
    @ignore_warnings(message="You are using packing, but the attention implementation is not.*", category=UserWarning)
    @ignore_warnings(message="Padding-free training is enabled, but the attention.*", category=UserWarning)
    def test_train_packing(self, packing_strategy):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir, packing=True, packing_strategy=packing_strategy, max_length=10, report_to="none"
        )
        trainer = SFTTrainer(
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

    @ignore_warnings(message="You are using packing, but the attention implementation is not.*", category=UserWarning)
    @ignore_warnings(message="Padding-free training is enabled, but the attention.*", category=UserWarning)
    def test_eval_packing(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            packing=True,
            max_length=64,
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        # Check the number of sequences in train and eval datasets
        num_train_seqs = sum(len(x) for x in trainer.train_dataset["seq_lengths"])
        num_eval_seqs = sum(len(x) for x in trainer.eval_dataset["seq_lengths"])
        assert num_train_seqs == 17  # we should still have 17 seqs
        assert num_eval_seqs == 2  # we should still have 2 seqs

        # Check that all sequences are shorter than the max length
        assert all(sum(x) <= 64 for x in trainer.train_dataset["seq_lengths"])
        assert all(sum(x) <= 64 for x in trainer.eval_dataset["seq_lengths"])

        # Check the number of sequences in train and eval datasets
        assert len(trainer.train_dataset["input_ids"]) == 3  # w/ this dataset, we end up with 46 seqs
        assert len(trainer.eval_dataset["input_ids"]) == 1  # w/ this dataset, we end up with 6 seqs

    @ignore_warnings(message="You are using packing, but the attention implementation is not.*", category=UserWarning)
    @ignore_warnings(message="Padding-free training is enabled, but the attention.*", category=UserWarning)
    def test_only_train_packing(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            packing=True,
            eval_packing=False,
            max_length=64,
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        # Check the number of sequences in train dataset
        num_train_seqs = sum(len(x) for x in trainer.train_dataset["seq_lengths"])
        assert num_train_seqs == 17  # we should still have 17 seqs

        # We expect eval dataset not having "seq_lengths" as eval_packing is False
        assert "seq_lengths" not in trainer.eval_dataset

        # Check that all sequences are shorter than the max length
        assert all(sum(x) <= 64 for x in trainer.train_dataset["seq_lengths"])

        # Check the number of sequences in train and eval datasets
        assert len(trainer.train_dataset["input_ids"]) == 3  # w/ this dataset, we end up with 46 seqs
        assert len(trainer.eval_dataset["input_ids"]) == 2  # w/ this dataset, we end up with 6 seqs

    def test_train_with_chat_template_kwargs(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")

        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        # The following template is a simplified version of the Qwen chat template, where an additional argument
        # `role_capital` is used to control the capitalization of roles.
        tokenizer.chat_template = '{%- if messages[0]["role"] == "system" -%}    {{ "<|im_start|>" + ("SYSTEM" if role_capital else "system") + "\\n" + messages[0]["content"] + "<|im_end|>\\n" }}{%- else -%}    {{ "<|im_start|>" + ("SYSTEM" if role_capital else "system") + "\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n" }}{%- endif -%}{%- for message in messages -%}    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) -%}        {{ "<|im_start|>" + (message.role.upper() if role_capital else message.role) + "\\n" + message.content + "<|im_end|>\\n" }}    {%- elif message.role == "assistant" -%}        {{ "<|im_start|>" + ("ASSISTANT" if role_capital else "assistant") }}        {%- if message.content -%}            {{ "\\n" + message.content }}        {%- endif -%}        {{ "<|im_end|>\\n" }}    {%- elif message.role == "tool" -%}        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") -%}            {{ "<|im_start|>" + ("USER" if role_capital else "user") }}        {%- endif -%}        {{ "\\n<tool_response>\\n" + message.content + "\\n</tool_response>" }}        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") -%}            {{ "<|im_end|>\\n" }}        {%- endif -%}    {%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}    {{ "<|im_start|>" + ("ASSISTANT" if role_capital else "assistant") + "\\n" }}{%- endif -%}'

        dataset = dataset.add_column(
            "chat_template_kwargs", [{"role_capital": bool(i % 2)} for i in range(len(dataset))]
        )
        assert "chat_template_kwargs" in dataset.features

        trainer = SFTTrainer(
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
            assert trainer.train_dataset[i]["input_ids"][: len(system_prompt_ids)] == system_prompt_ids

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

    def test_train_assistant_only(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, assistant_only_loss=True, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
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

    def test_train_completion_only(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, completion_only_loss=True, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
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

    def test_train_completion_only_harmony(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/harmony", "prompt_completion", split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, completion_only_loss=True, report_to="none")
        trainer = SFTTrainer(
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

    def test_train_assistant_only_and_completion_only(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        # To test this case, we need to add user messages in the completion (they'll be masked in the loss)
        def add_to_completion(example):
            example["completion"].append(example["prompt"][0])
            example["completion"].append(example["completion"][0])
            return example

        dataset = dataset.map(add_to_completion)

        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir, assistant_only_loss=True, completion_only_loss=True, report_to="none"
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
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

    def test_train_assistant_only_iterable_dataset(self):
        # Get the dataset
        dataset = load_dataset(
            "trl-internal-testing/zen", "conversational_language_modeling", split="train", streaming=True
        )

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, assistant_only_loss=True, max_steps=3, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
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

    def test_train_with_set_chat_template_from_model(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, chat_template_path="Qwen/Qwen3-4B", report_to="none")
        # trl-internal-testing/tiny-GPTNeoXForCausalLM doesn't have a chat template set by default
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-GPTNeoXForCausalLM", args=training_args, train_dataset=dataset
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

    def test_train_with_set_chat_template_from_path(self, lazy_shared_datadir):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            chat_template_path=str(lazy_shared_datadir / "template.jinja"),
            report_to="none",
        )
        # trl-internal-testing/tiny-GPTNeoXForCausalLM doesn't have a chat template set by default
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-GPTNeoXForCausalLM", args=training_args, train_dataset=dataset
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

        # Check that the template saved in the output directory is the same as the one used for training
        template_path = pathlib.Path(self.tmp_dir) / "checkpoint-9" / "chat_template.jinja"
        assert template_path.exists(), f"Chat template not found at {template_path}"

        with open(template_path) as f:
            template_content = f.read()
        with open(training_args.chat_template_path) as f:
            original_template_content = f.read()
        assert template_content == original_template_content, "Chat template content does not match the original"

    def test_train_toolcall_data(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/toolcall", "language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
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

    def test_train_toolcall_data_as_json(self):
        # Tabular backends (Arrow/Parquet) can insert `None` for missing keys in nested structures.
        # If `tools` is stored as a list of dicts and examples use different dict schemas, nulls may
        # be introduced and break tool processing. This test ensures we also support `tools` provided
        # as a list of dicts.
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/toolcall", "language_modeling", split="train")

        def convert_to_json(example):
            return {"tools": json.loads(example["tools"])}

        dataset = dataset.map(convert_to_json)

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = SFTTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = SFTTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        def dummy_compute_metrics(eval_pred):
            return {"my_metric": 0.123}

        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            eval_strategy="steps",
            eval_steps=3,
            report_to="none",
        )
        trainer = SFTTrainer(
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
    # `SFTTrainer`. We keep it as a regression guard: if the default ever changes, we still explicitly test gradient
    # checkpointing, which has caused issues in the past.
    def test_train_with_gradient_checkpointing(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, gradient_checkpointing=True, report_to="none")
        trainer = SFTTrainer(
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

    @pytest.mark.parametrize("use_reentrant", [True, False])
    def test_train_with_gradient_checkpointing_reentrant(self, use_reentrant):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
            report_to="none",
        )
        trainer = SFTTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
        )

        for tag in ["sft", "trl"]:
            assert tag in trainer.model.model_tags

    @require_peft
    def test_tag_added_peft(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        for tag in ["sft", "trl"]:
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
                        Version("5.0.0") <= Version(transformers.__version__) < Version("5.1.0"),
                        reason="Upstream transformers bug (transformers#43334) in 5.0.x; fixed in 5.1.0",
                    ),
                ],
            ),
        ],
    )
    @require_vision
    def test_train_vlm(self, model_id):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            report_to="none",
        )
        trainer = SFTTrainer(model=model_id, args=training_args, train_dataset=dataset)

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
        dataset = load_dataset(
            "trl-internal-testing/zen-multi-image", "conversational_prompt_completion", split="train"
        )

        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            report_to="none",
        )
        trainer = SFTTrainer(
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

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            # Special case for Gemma, as it uses token_type_ids, and we need to ensure they are properly in the collator:
            "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
        ],
    )
    @require_vision
    def test_train_vlm_prompt_completion(self, model_id):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_completion", split="train")

        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            report_to="none",
        )
        trainer = SFTTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_language_modeling", split="train")

        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            per_device_train_batch_size=1,
            model_init_kwargs={"dtype": "bfloat16"},
            report_to="none",
        )
        trainer = SFTTrainer(model="google/gemma-3n-E2B-it", args=training_args, train_dataset=dataset)

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
        ["conversational_language_modeling", "conversational_prompt_completion", "standard_prompt_completion"],
    )
    @require_vision
    def test_train_vlm_text_only_data(self, model_id, dataset_config):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", dataset_config, split="train")

        # Initialize the trainer
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
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
    def test_prompt_tuning(self):
        """Test that SFT works with Prompt Tuning."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=8),
        )

        # Save initial parameters to check they change during training
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        # Check that training completed successfully
        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-1]["mean_token_accuracy"] is not None

        # Check the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if "base_model" in n:  # We expect the base model parameters to be the same
                torch.testing.assert_close(param, new_param), f"Parameter {n} has changed"
            elif "prompt_encoder" in n:  # We expect the peft parameters to be different
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"
            else:
                raise ValueError(f"Unexpected parameter {n} in model: {trainer.model}")

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
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        # Initialize the trainer with the already configured PeftModel
        training_args = SFTConfig(output_dir=self.tmp_dir, learning_rate=0.1, report_to="none")
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset, peft_config=LoraConfig())

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

    @require_peft
    def test_prompt_tuning_peft_model(self):
        """Test that SFT works with Prompt Tuning and a pre-converted PeftModel"""
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32")
        model = get_peft_model(model, PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=8))

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)

        # Save initial parameters to check they change during training
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        # Check that training completed successfully
        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-1]["mean_token_accuracy"] is not None

        # Check the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if "base_model" in n:  # We expect the base model parameters to be the same
                torch.testing.assert_close(param, new_param), f"Parameter {n} has changed"
            elif "prompt_encoder" in n:  # We expect the peft parameters to be different
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"
            else:
                raise ValueError(f"Unexpected parameter {n} in model: {trainer.model}")


@pytest.mark.slow
@require_torch_accelerator
@require_peft
class TestSFTTrainerSlow(TrlTestCase):
    def setup_method(self):
        self.train_dataset = load_dataset("stanfordnlp/imdb", split="train[:10%]")
        self.eval_dataset = load_dataset("stanfordnlp/imdb", split="test[:10%]")
        self.max_length = 128
        self.peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    @pytest.mark.parametrize("packing", [True, False])
    @pytest.mark.parametrize(
        "model_name",
        [
            "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
            "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        ],
    )
    def test_sft_trainer_transformers_mp(self, model_name, packing):
        """
        Simply tests if passing a transformers model to `SFTTrainer` loads and runs the trainer as expected in mixed
        precision.
        """
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            logging_strategy="no",
            report_to="none",
            per_device_train_batch_size=2,
            max_steps=10,
            fp16=True,  # this is sufficient to enable amp
            packing=packing,
            max_length=self.max_length,
        )

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float32")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        trainer = SFTTrainer(
            model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )

        trainer.train()

        release_memory(model, trainer)

    @pytest.mark.parametrize("device_map", [{"": 0}, "auto"])
    @pytest.mark.parametrize(
        "gradient_checkpointing_kwargs", [None, {"use_reentrant": False}, {"use_reentrant": True}]
    )
    @pytest.mark.parametrize("packing", [True, False])
    @pytest.mark.parametrize(
        "model_name",
        [
            "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
            "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        ],
    )
    @require_torch_multi_accelerator
    def test_sft_trainer_transformers_mp_gc_device_map(
        self, model_name, packing, gradient_checkpointing_kwargs, device_map
    ):
        """
        Simply tests if passing a transformers model to `SFTTrainer` loads and runs the trainer as expected in mixed
        precision + different scenarios of gradient_checkpointing (single, multi-gpu, etc).
        """
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            logging_strategy="no",
            report_to="none",
            per_device_train_batch_size=2,
            max_steps=10,
            packing=packing,
            max_length=self.max_length,
            fp16=True,  # this is sufficient to enable amp
            gradient_checkpointing=True,  # default, here for clarity
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        )

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float32", device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        trainer = SFTTrainer(
            model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )

        trainer.train()

        release_memory(model, trainer)

    @pytest.mark.parametrize(
        "gradient_checkpointing_kwargs", [None, {"use_reentrant": False}, {"use_reentrant": True}]
    )
    @pytest.mark.parametrize("packing", [True, False])
    @pytest.mark.parametrize(
        "model_name",
        [
            "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
            "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        ],
    )
    @require_peft
    @require_bitsandbytes
    def test_sft_trainer_transformers_mp_gc_peft_qlora(self, model_name, packing, gradient_checkpointing_kwargs):
        """
        Simply tests if passing a transformers model + PEFT + bnb to `SFTTrainer` loads and runs the trainer as
        expected in mixed precision + different scenarios of gradient_checkpointing.
        """
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            logging_strategy="no",
            report_to="none",
            per_device_train_batch_size=2,
            max_steps=10,
            packing=packing,
            max_length=self.max_length,
            gradient_checkpointing=True,  # default, here for clarity
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        )

        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="float32", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        trainer = SFTTrainer(
            model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            peft_config=self.peft_config,
        )

        assert isinstance(trainer.model, PeftModel)

        trainer.train()

        release_memory(model, trainer)

    @pytest.mark.parametrize("packing", [True, False])
    @pytest.mark.parametrize(
        "model_name",
        [
            "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
            "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        ],
    )
    @require_peft
    @require_bitsandbytes
    def test_sft_trainer_with_chat_format_qlora(self, model_name, packing):
        """
        Simply tests if using setup_chat_format with a transformers model + peft + bnb config to `SFTTrainer` loads and
        runs the trainer as expected.
        """
        train_dataset = load_dataset("trl-internal-testing/dolly-chatml-sft", split="train")

        training_args = SFTConfig(
            packing=packing,
            max_length=self.max_length,
            output_dir=self.tmp_dir,
            logging_strategy="no",
            report_to="none",
            per_device_train_batch_size=2,
            max_steps=10,
        )

        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="float32", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        trainer = SFTTrainer(
            model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            peft_config=self.peft_config,
        )

        assert isinstance(trainer.model, PeftModel)

        trainer.train()

        release_memory(model, trainer)

    @pytest.mark.parametrize("packing", [True, False])
    @pytest.mark.parametrize(
        "model_name",
        [
            "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
            "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        ],
    )
    @require_liger_kernel
    def test_sft_trainer_with_liger(self, model_name, packing):
        """
        Tests if passing use_liger=True to SFTConfig loads and runs the trainer with AutoLigerKernelForCausalLM as
        expected.
        """
        import importlib

        def cleanup_liger_patches(trainer):
            """Clean up liger_kernel patches by reloading the model's specific module"""
            try:
                # Get the specific module that was used by the trainer's model
                module_path = trainer.model.__module__
                reload_module = importlib.import_module(module_path)
                importlib.reload(reload_module)
            except Exception:
                pass  # Continue if reload fails

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            logging_strategy="no",
            report_to="none",
            per_device_train_batch_size=2,
            max_steps=2,
            packing=packing,
            max_length=self.max_length,
            use_liger_kernel=True,
        )

        trainer = SFTTrainer(
            model_name,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )

        # Ensure cleanup of liger patches after the test
        try:
            trainer.train()
            release_memory(trainer.model, trainer)
        finally:
            cleanup_liger_patches(trainer)

    @pytest.mark.parametrize("packing", [True, False])
    @pytest.mark.parametrize(
        "model_name",
        [
            "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
            "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        ],
    )
    @require_torch_accelerator
    def test_train_offloading(self, model_name, packing):
        """Test that activation offloading works with SFTTrainer."""
        # Initialize the trainer
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            activation_offloading=True,
            report_to="none",
            per_device_train_batch_size=2,
            max_steps=2,
            packing=packing,
            max_length=self.max_length,
        )
        trainer = SFTTrainer(
            model=model_name, args=training_args, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset
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

        release_memory(trainer.model, trainer)
