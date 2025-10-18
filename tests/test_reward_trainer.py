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

import pathlib

import pytest
import torch
from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import is_peft_available

from trl import RewardConfig, RewardTrainer
from trl.trainer.reward_trainer import DataCollatorForPreference

from .testing_utils import TrlTestCase, require_peft


if is_peft_available():
    from peft import LoraConfig, PeftModel, get_peft_model


class TestDataCollatorForPreference(TrlTestCase):
    def test_basic_padding(self):
        """Test basic padding functionality without completion masks."""
        self.collator = DataCollatorForPreference(pad_token_id=0)
        examples = [
            {"chosen_input_ids": [1, 2, 3], "rejected_input_ids": [4, 5]},
            {"chosen_input_ids": [6, 7], "rejected_input_ids": [8]},
        ]

        result = self.collator(examples)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [6, 7, 0], [4, 5, 0], [8, 0, 0]]))
        torch.testing.assert_close(
            result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0]])
        )

    def test_pad_to_multiple_of(self):
        """Test padding to multiple of specified value."""
        collator = DataCollatorForPreference(pad_token_id=0, pad_to_multiple_of=4)
        examples = [
            {"chosen_input_ids": [1, 2, 3], "rejected_input_ids": [4, 5]},
            {"chosen_input_ids": [6, 7], "rejected_input_ids": [8]},
        ]

        result = collator(examples)

        torch.testing.assert_close(
            result["input_ids"], torch.tensor([[1, 2, 3, 0], [6, 7, 0, 0], [4, 5, 0, 0], [8, 0, 0, 0]])
        )
        torch.testing.assert_close(
            result["attention_mask"], torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
        )

    def test_single_example(self):
        """Test collator with a single example."""
        self.collator = DataCollatorForPreference(pad_token_id=0)
        examples = [{"chosen_input_ids": [1, 2, 3], "rejected_input_ids": [4, 5]}]

        result = self.collator(examples)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))

    def test_different_pad_token_id(self):
        """Test with different pad token ID."""
        collator = DataCollatorForPreference(pad_token_id=999)
        examples = [
            {"chosen_input_ids": [1, 2, 3], "rejected_input_ids": [4, 5]},
            {"chosen_input_ids": [6, 7], "rejected_input_ids": [8]},
        ]

        result = collator(examples)

        torch.testing.assert_close(
            result["input_ids"], torch.tensor([[1, 2, 3], [6, 7, 999], [4, 5, 999], [8, 999, 999]])
        )
        torch.testing.assert_close(
            result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0]])
        )

    def test_collate_with_margin(self):
        self.collator = DataCollatorForPreference(pad_token_id=0)
        examples = [
            {"chosen_input_ids": [1, 2, 3], "rejected_input_ids": [4, 5], "margin": 0.1},
            {"chosen_input_ids": [6, 7], "rejected_input_ids": [8], "margin": 0.2},
        ]

        result = self.collator(examples)

        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [6, 7, 0], [4, 5, 0], [8, 0, 0]]))
        torch.testing.assert_close(
            result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0]])
        )
        torch.testing.assert_close(result["margin"], torch.tensor([0.1, 0.2]))


class TestRewardTrainer(TrlTestCase):
    @parameterized.expand(
        [
            ("trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",),
            ("trl-internal-testing/tiny-Qwen3MoeForSequenceClassification",),
            ("trl-internal-testing/tiny-LlamaForSequenceClassification-3.2",),
        ]
    )
    def test_train(self, model_id):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RewardTrainer(model=model_id, args=training_args, train_dataset=dataset)

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

    @parameterized.expand(
        [
            ("standard_preference",),
            ("conversational_preference",),
            ("standard_implicit_prompt_preference",),
            ("conversational_implicit_prompt_preference",),
        ]
    )
    def test_train_dataset_types(self, config_name):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
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

    def test_train_model(self):
        # Instantiate the model
        model = AutoModelForSequenceClassification.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        )

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RewardTrainer(model=model, args=training_args, train_dataset=dataset)

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

    def test_train_from_causal_lm(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RewardTrainer(
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

    def test_train_model_dtype(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        training_args = RewardConfig(
            output_dir=self.tmp_dir,
            model_init_kwargs={"dtype": torch.float16},
            learning_rate=0.1,
            report_to="none",
        )
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
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
            # For some reasonn model.layers.0.input_layernorm.weight doesn't change in GitHub Actions but does
            # locally. We ignore this parameter for now
            if "layernorm" in n:
                continue
            new_param = trainer.model.get_parameter(n)
            # Check the torch dtype
            assert new_param.dtype == torch.float16
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_peft
    def test_train_dense_with_peft_config(self):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")

        trainer = RewardTrainer(
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
                assert torch.allclose(param, new_param), f"Parameter {n} has changed"
            elif "base_layer" not in n:  # We expect the peft parameters to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_peft
    def test_train_moe_with_peft_config(self):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-Qwen3MoeForSequenceClassification"
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")

        trainer = RewardTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(target_modules=["up_proj", "down_proj", "score"]),
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
                assert torch.allclose(param, new_param), f"Parameter {n} has changed"
            elif "base_layer" not in n:  # We expect the peft parameters to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_peft
    def test_train_peft_model(self):
        # Get the base model
        model_id = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        model = AutoModelForSequenceClassification.from_pretrained(model_id)

        # Get the base model parameter names
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Turn the model into a peft model
        lora_config = LoraConfig()
        model = get_peft_model(model, lora_config)

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RewardTrainer(model=model, args=training_args, train_dataset=dataset)

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
                assert torch.allclose(param, new_param), f"Parameter {n} has changed"
            elif "base_layer" not in n:  # We expect the peft parameters to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_peft
    def test_train_dense_with_peft_config_and_gradient_checkpointing(self):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, gradient_checkpointing=True, report_to="none")

        trainer = RewardTrainer(
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
                assert torch.allclose(param, new_param), f"Parameter {n} has changed"
            elif "base_layer" not in n:  # We expect the peft parameters to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_peft
    def test_train_moe_with_peft_config_and_gradient_checkpointing(self):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-Qwen3MoeForSequenceClassification"
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, gradient_checkpointing=True, report_to="none")

        trainer = RewardTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(target_modules=["up_proj", "down_proj", "score"]),
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
                assert torch.allclose(param, new_param), f"Parameter {n} has changed"
            elif "base_layer" not in n:  # We expect the peft parameters to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @require_peft
    def test_train_with_peft_model_and_gradient_checkpointing(self):
        # Get the base model parameter names
        model_id = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        model = get_peft_model(model, LoraConfig())

        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, gradient_checkpointing=True, report_to="none")

        trainer = RewardTrainer(model=model, args=training_args, train_dataset=dataset)

        # Verify model is a PeftModel
        assert isinstance(trainer.model, PeftModel)

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
                assert torch.allclose(param, new_param), f"Parameter {n} has changed"
            elif "base_layer" not in n:  # We expect the peft parameters to be different (except for the base layer)
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_train_with_pretokenized_data(self):
        # Get the dataset
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        def tokenize_example(example):
            return {
                "chosen_input_ids": tokenizer(example["chosen"]).input_ids,
                "rejected_input_ids": tokenizer(example["rejected"]).input_ids,
            }

        # Apply tokenization
        tokenized_dataset = dataset.map(tokenize_example, remove_columns=["chosen", "rejected"])

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RewardTrainer(model=model_id, args=training_args, train_dataset=tokenized_dataset)

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
        dataset = load_dataset(
            "trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train", streaming=True
        )

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, max_steps=3, report_to="none")
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
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

    def test_train_with_chat_template_kwargs(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")

        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5")
        # The following template is a simplified version of the Qwen chat template, where an additional argument
        # `role_capital` is used to control the capitalization of roles.
        tokenizer.chat_template = '{%- if messages[0]["role"] == "system" -%}    {{ "<|im_start|>" + ("SYSTEM" if role_capital else "system") + "\\n" + messages[0]["content"] + "<|im_end|>\\n" }}{%- else -%}    {{ "<|im_start|>" + ("SYSTEM" if role_capital else "system") + "\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n" }}{%- endif -%}{%- for message in messages -%}    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) -%}        {{ "<|im_start|>" + (message.role.upper() if role_capital else message.role) + "\\n" + message.content + "<|im_end|>\\n" }}    {%- elif message.role == "assistant" -%}        {{ "<|im_start|>" + ("ASSISTANT" if role_capital else "assistant") }}        {%- if message.content -%}            {{ "\\n" + message.content }}        {%- endif -%}        {{ "<|im_end|>\\n" }}    {%- elif message.role == "tool" -%}        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") -%}            {{ "<|im_start|>" + ("USER" if role_capital else "user") }}        {%- endif -%}        {{ "\\n<tool_response>\\n" + message.content + "\\n</tool_response>" }}        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") -%}            {{ "<|im_end|>\\n" }}        {%- endif -%}    {%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}    {{ "<|im_start|>" + ("ASSISTANT" if role_capital else "assistant") + "\\n" }}{%- endif -%}'

        dataset.add_column("chat_template_kwargs", [{"role_capital": bool(i % 2)} for i in range(len(dataset))])

        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
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

    def test_train_with_set_chat_template_from_model(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, chat_template_path="Qwen/Qwen3-4B", report_to="none")
        # trl-internal-testing/tiny-GPTNeoXForSequenceClassification doesn't have a chat template set by default
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-GPTNeoXForSequenceClassification",
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
            # RewardTrainer uses a mean-free loss that cancels uniform shifts in output scores. Since GPT-NeoX models
            # include a final LayerNorm, its bias consistently receives zero gradient and remains unchanged, so we skip
            # this parameter.
            if n == "gpt_neox.final_layer_norm.bias":
                continue
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_train_with_set_chat_template_from_path(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")

        # Initialize the trainer
        training_args = RewardConfig(
            output_dir=self.tmp_dir,
            chat_template_path=str(pathlib.Path(__file__).parent / "data" / "template.jinja"),
            report_to="none",
        )
        # trl-internal-testing/tiny-GPTNeoXForSequenceClassification doesn't have a chat template set by default
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-GPTNeoXForSequenceClassification",
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
            # RewardTrainer uses a mean-free loss that cancels uniform shifts in output scores. Since GPT-NeoX models
            # include a final LayerNorm, its bias consistently receives zero gradient and remains unchanged, so we skip
            # this parameter.
            if n == "gpt_neox.final_layer_norm.bias":
                continue
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

        # Check that the template saved in the output directory is the same as the one used for training
        template_path = pathlib.Path(self.tmp_dir) / "checkpoint-9" / "chat_template.jinja"
        assert template_path.exists(), f"Chat template not found at {template_path}"

        with open(template_path) as f:
            template_content = f.read()
        with open(training_args.chat_template_path) as f:
            original_template_content = f.read()
        assert template_content == original_template_content, "Chat template content does not match the original"

    @pytest.mark.skip(reason="Skipping until we have a dataset with tool calls")
    def test_train_toolcall_data(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/toolcall", split="train")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
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

    def test_train_with_eval(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset={"data1": dataset["test"], "data2": dataset["test"]},
        )
        # Train the model
        trainer.train()

        # Check that the eval losses are not None
        assert trainer.state.log_history[-3]["eval_data1_loss"] is not None
        assert trainer.state.log_history[-2]["eval_data2_loss"] is not None

    def test_train_with_gradient_checkpointing(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, gradient_checkpointing=True, report_to="none")
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
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

    def test_tag_added(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            train_dataset=dataset,
        )

        for tag in ["reward-trainer", "trl"]:
            assert tag in trainer.model.model_tags

    @require_peft
    def test_tag_added_peft(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        for tag in ["reward-trainer", "trl"]:
            assert tag in trainer.model.model_tags

    def test_train_with_margin(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        def add_margin(example):
            # dummy margin based on the length of the chosen summary
            return {"margin": len(example["chosen"])}

        dataset = dataset.map(add_margin)

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
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

    def test_train_with_center_rewards_coefficient(self):
        # Get the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Initialize the trainer
        training_args = RewardConfig(output_dir=self.tmp_dir, center_rewards_coefficient=0.01, report_to="none")
        trainer = RewardTrainer(
            model="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
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
