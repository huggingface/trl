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

import pytest
import torch
import transformers
from accelerate.utils.memory import release_memory
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from packaging.version import Version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers.testing_utils import backend_device_count, backend_empty_cache, torch_device
from transformers.utils import is_peft_available

from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import (
    DataCollatorForLanguageModeling,
)

from .testing_utils import (
    TrlTestCase,
    ignore_warnings,
    is_ampere_or_newer,
    require_bitsandbytes,
    require_kernels,
    require_liger_kernel,
    require_peft,
    require_peft_target_parameters,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    require_vision,
    xfail_data_parallel,
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


class TestDataCollatorForLanguageModeling(TrlTestCase):
    def test_basic_padding(self):
        """Test basic padding."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}, {"input_ids": [4, 5], "labels": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_without_labels(self):
        """When no labels are provided, they default to the input IDs (padding excluded)."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_provided_labels_not_reconstructed_from_masks(self):
        """Provided labels are used as is, never rebuilt from the mask columns. Here the labels match the input IDs
        (every token trainable); the masks would introduce -100 if they were consumed, so the labels staying equal to
        the input IDs proves the masks are ignored."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "completion_mask": [0, 0, 1], "assistant_masks": [0, 0, 1]},
            {"input_ids": [4, 5], "labels": [4, 5], "completion_mask": [0, 1], "assistant_masks": [0, 1]},
        ]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_return_position_ids(self):
        """Padded mode with return_position_ids: position IDs are returned alongside the attention mask."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, return_position_ids=True)
        examples = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}, {"input_ids": [4, 5], "labels": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "position_ids", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2], [0, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_return_position_ids_packed(self):
        """Padded mode with return_position_ids on packed examples: position IDs reset at document boundaries."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, return_position_ids=True)
        examples = [{"input_ids": [1, 2, 3, 4, 5], "seq_lengths": [3, 2]}, {"input_ids": [6, 7], "seq_lengths": [2]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "position_ids", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5], [6, 7, 0, 0, 0]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1], [0, 1, 0, 0, 0]]))

    def test_padding_free_mode(self):
        """Test padding-free mode where sequences are concatenated."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
        examples = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}, {"input_ids": [4, 5], "labels": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "position_ids", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, 2, 3, -100, 5]]))

    def test_padding_free_without_labels(self):
        """Padding-free mode without labels: labels default to the input IDs (document starts masked)."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "position_ids", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, 2, 3, -100, 5]]))

    def test_packing(self):
        """Test that when using packing with position_ids, attention_mask is dropped with fa2."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)

        # Simulate packed sequences with position_ids that restart (typical of BFD packing)
        examples = [
            {"input_ids": [1, 2, 3, 4, 5, 6], "seq_lengths": [3, 3], "labels": [1, 2, 3, 4, 5, 6]},
            {"input_ids": [7, 8, 9, 10, 11], "seq_lengths": [4, 1], "labels": [7, 8, 9, 10, 11]},
        ]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "position_ids", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, 2, 3, -100, 5, 6, -100, 8, 9, 10, -100]]))

    def test_pad_to_multiple_of(self):
        """Test padding to multiple of specified value."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, pad_to_multiple_of=4)
        examples = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}, {"input_ids": [4, 5], "labels": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3, -100], [4, 5, -100, -100]]))

    def test_pad_to_multiple_of_and_padding_free(self):
        """Test padding to multiple of specified value."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True, pad_to_multiple_of=4)
        examples = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}, {"input_ids": [4, 5], "labels": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "position_ids", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]]))
        torch.testing.assert_close(result["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 0, 0, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[-100, 2, 3, -100, 5, -100, -100, -100]]))

    def test_custom_position_ids_but_no_padding_free(self):
        """Test that custom position_ids are ignored if padding_free is False."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [
            {"input_ids": [1, 2, 3], "seq_lengths": [1, 2], "labels": [1, 2, 3]},
            {"input_ids": [4, 5], "seq_lengths": [2], "labels": [4, 5]},
        ]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 0]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

    def test_single_example(self):
        """Test collator with a single example."""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        examples = [{"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3, 4]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1, 1]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3, 4]]))

    def test_different_pad_token_id(self):
        """Test with different pad token ID."""
        collator = DataCollatorForLanguageModeling(pad_token_id=999)
        examples = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}, {"input_ids": [4, 5], "labels": [4, 5]}]

        result = collator(examples)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 999]]))
        torch.testing.assert_close(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
        torch.testing.assert_close(result["labels"], torch.tensor([[1, 2, 3], [4, 5, -100]]))

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

    def test_init_auto_processing_class_uses_model_revision(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=SFTConfig(
                output_dir=self.tmp_dir,
                report_to="none",
                model_init_kwargs={"revision": "8913f5819566"},
            ),
            train_dataset=dataset,
        )
        # This revision's chat template is 2558 chars; the one on `main` is 2507. Comparing the length is enough to
        # catch the tokenizer being loaded from the default branch instead of the pinned revision.
        assert len(trainer.processing_class.chat_template) == 2558

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Cohere2ForCausalLM",
            "trl-internal-testing/tiny-FalconMambaForCausalLM",
            pytest.param(
                "trl-internal-testing/tiny-Glm4MoeForCausalLM",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.0.0"),
                    reason="GLM4 tokenizer requires transformers>=5.0.0",
                ),
            ),
            "trl-internal-testing/tiny-GptOssForCausalLM",
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "trl-internal-testing/tiny-Qwen2ForCausalLM-R1-Distill",
            "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            pytest.param(
                "trl-internal-testing/tiny-NemotronHForCausalLM-nano",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.7.0"),
                    reason="Nemotron 3 gradient checkpointing requires transformers>=5.7.0 (see transformers#45625)",
                ),
            ),
            pytest.param(
                "trl-internal-testing/tiny-Olmo3ForCausalLM",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("4.57.0"),
                    reason="Olmo 3 requires transformers>=4.57.0",
                ),
            ),
            "trl-internal-testing/tiny-Lfm2ForCausalLM",
            pytest.param(
                "trl-internal-testing/tiny-Lfm2ForCausalLM-2.5",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.0.0"),
                    reason="LFM2.5 tokenizer requires transformers>=5.0.0",
                ),
            ),
        ],
    )
    def test_train(self, model_id):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model_id, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # MoE models log the load-balancing auxiliary loss (on by default)
        if trainer.aux_loss_enabled:
            assert trainer.state.log_history[-1]["aux_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_language_modeling",
            "conversational_language_modeling",
            "standard_prompt_completion",
            "conversational_prompt_completion",
        ],
    )
    def test_train_dataset_format(self, config_name):
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_trust_remote_code(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"

        with pytest.raises(ValueError, match="custom code"):
            SFTTrainer(
                model=model_id,
                args=SFTConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
            )

        trainer = SFTTrainer(
            model=model_id,
            args=SFTConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    # Special case for harmony
    def test_train_gpt_oss(self):
        dataset = load_dataset("trl-internal-testing/harmony", "language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-GptOssForCausalLM", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            dtype="float32",
        )

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_dft_loss(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

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

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_formatting_func(self):
        # Dummy formatting function
        def formatting_prompts_func(example):
            chosen, rejected = example["chosen"], example["rejected"]
            return f"### Chosen: {chosen}\n### Rejected: {rejected}"

        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            formatting_func=formatting_prompts_func,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_model_dtype(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            model_init_kwargs={"dtype": torch.float16},
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            # For some reasonn model.layers.0.input_layernorm.weight doesn't change in GitHub Actions but does
            # locally. We ignore this parameter for now
            if "layernorm" in n:
                continue
            new_param = trainer.model.get_parameter(n)
            # Check the torch dtype
            assert new_param.dtype == torch.float16
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    def test_peft_init_is_seeded(self):
        # Two trainers with the same seed start from the same adapter weights
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        adapters = []
        for global_seed in range(2):
            torch.manual_seed(global_seed)  # a different global RNG state, as in two separate runs
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=SFTConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
                peft_config=LoraConfig(),
            )
            adapters.append({n: p.clone() for n, p in trainer.model.named_parameters() if "lora_A" in n})

        assert adapters[0]
        for n, param in adapters[0].items():
            assert torch.equal(param, adapters[1][n]), f"Parameter {n} differs between the two trainers."

    @require_peft
    def test_train_dense_with_peft_config_lora(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")

        trainer = SFTTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.{n}" for n, _ in model.named_parameters()]

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
            if Version(peft.__version__) <= Version("0.17.1"):
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

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            else:  # We expect the peft params to be different
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft_target_parameters
    def test_train_moe_with_peft_config(self):
        model_id = "trl-internal-testing/tiny-GptOssForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")

        trainer = SFTTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(target_parameters=["mlp.experts.down_proj", "mlp.experts.gate_up_proj"]),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "model_id, expect_coef, expect_aux_loss",
        [
            # MoE whose forward returns an auxiliary loss: the architecture's own coefficient is applied
            ("trl-internal-testing/tiny-Qwen3MoeForCausalLM", 0.001, True),
            # MoE that balances its experts with a router bias: it declares no coefficient, so the term stays off
            ("trl-internal-testing/tiny-DeepseekV3ForCausalLM", 0.0, False),
            # Dense model: no coefficient to inherit
            ("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", 0.0, False),
        ],
    )
    def test_router_aux_loss_coef_defaults_to_the_architecture(self, model_id, expect_coef, expect_aux_loss):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train[:2]")
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model_id, args=training_args, train_dataset=dataset)

        assert trainer.router_aux_loss_coef == expect_coef
        assert trainer.aux_loss_enabled == expect_aux_loss

    def test_router_aux_loss_coef_fails_without_router_logits(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train[:2]")
        training_args = SFTConfig(output_dir=self.tmp_dir, router_aux_loss_coef=0.5, report_to="none")

        # Dense model: no `output_router_logits` on its config, so there is nothing to compute the term from
        with pytest.raises(ValueError, match="not a Mixture-of-Experts model"):
            SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
            )

    def test_router_aux_loss_coef_explicit_value_overrides_the_architecture(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train[:2]")
        training_args = SFTConfig(output_dir=self.tmp_dir, router_aux_loss_coef=0.5, report_to="none")

        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM", args=training_args, train_dataset=dataset
        )

        assert trainer.router_aux_loss_coef == 0.5
        assert trainer.aux_loss_enabled

    @require_peft
    def test_train_peft_model(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")

        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        lora_config = LoraConfig()
        model = get_peft_model(model, lora_config)

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    # In practice, this test is the same as `test_train_dense_with_peft_config_lora`, since gradient checkpointing is
    # enabled by default in `SFTTrainer`. We keep it as a regression guard: if the default ever changes, we still
    # explicitly test PEFT + gradient checkpointing, which has caused issues in the past.
    @require_peft
    def test_train_with_peft_config_and_gradient_checkpointing(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, gradient_checkpointing=True, report_to="none")

        trainer = SFTTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize("use_reentrant", [True, False])
    @require_peft
    def test_train_with_peft_config_and_gradient_checkpointing_reentrant(self, use_reentrant):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

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

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_liger_kernel
    def test_train_with_liger(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, use_liger_kernel=True, report_to="none")
        with pytest.warns(FutureWarning, match="`use_liger_kernel=True` is deprecated"):
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
            )
        # Liger's fused linear cross-entropy would replace the forward that carries the fused LM head
        assert trainer.args.liger_kernel_config["fused_linear_cross_entropy"] is False

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_non_chatml_conversational_data(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        # Rename role/content to from/value to ensure SFT works with non-chatML conversational data
        def rename_fields(example: list[dict]):
            return {"conversations": [{"from": m["role"], "value": m["content"]} for m in example["messages"]]}

        dataset = dataset.map(rename_fields, remove_columns="messages")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_pretokenized_data(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        def tokenize_example(example):
            return tokenizer(example["text"])

        # Apply tokenization
        tokenized_dataset = dataset.map(tokenize_example, remove_columns=["text"])

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model_id, args=training_args, train_dataset=tokenized_dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_dataset_with_transform_requires_skip_prepare_dataset(self):
        dataset = Dataset.from_dict({"text": ["hello world"]})

        def add_suffix(batch):
            batch["text"] = [text + " <AUG>" for text in batch["text"]]
            return batch

        dataset = dataset.with_transform(add_suffix)
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")

        with pytest.raises(
            ValueError,
            match=r"Dataset\.with_transform\(\).*skip_prepare_dataset.*trainer-ready",
        ):
            SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
            )

    def test_padding_free_without_packing_and_max_length_raises(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train[:2]")
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            max_length=16,
            padding_free=True,
            report_to="none",
        )

        with pytest.raises(ValueError, match="`max_length` is not enforced"):
            SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
            )

    @pytest.mark.parametrize(
        "attn_implementation, expect_warning",
        [
            ("kernels-community/flash-attn2", False),
            ("kernels-community/flash-attn2@v2", False),
            ("eager", True),
        ],
    )
    def test_padding_free_warns_only_for_unsupported_attention(self, attn_implementation, expect_warning, caplog):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train[:2]")
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        # Only the config value matters here: the warning is emitted at init, so no attention kernel is ever loaded.
        model.config._attn_implementation = attn_implementation
        training_args = SFTConfig(output_dir=self.tmp_dir, padding_free=True, max_length=None, report_to="none")

        with caplog.at_level("WARNING", logger="trl.trainer.sft_trainer"):
            SFTTrainer(model=model, args=training_args, train_dataset=dataset)

        assert ("supported Flash Attention variant" in caplog.text) == expect_warning

    def test_train_with_iterable_dataset(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train", streaming=True)

        training_args = SFTConfig(output_dir=self.tmp_dir, max_steps=3, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_kernels
    @pytest.mark.skipif(
        not is_ampere_or_newer() and torch_device != "xpu",
        reason="Flash Attention 2 requires Ampere or newer GPU, or XPU",
    )
    def test_train_padding_free(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            padding_free=True,
            max_length=None,  # padding-free without packing doesn't enforce max_length
            model_init_kwargs={"attn_implementation": "kernels-community/flash-attn2"},
            bf16=True,  # flash_attention_2 only supports bf16 and fp16
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize("packing_strategy", ["bfd", "wrapped"])
    @ignore_warnings(message="You are using packing, but the attention implementation is not.*", category=UserWarning)
    @ignore_warnings(message="Padding-free training is enabled, but the attention.*", category=UserWarning)
    def test_train_packing(self, packing_strategy):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir, packing=True, packing_strategy=packing_strategy, max_length=10, report_to="none"
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @ignore_warnings(message="You are using packing, but the attention implementation is not.*", category=UserWarning)
    @ignore_warnings(message="Padding-free training is enabled, but the attention.*", category=UserWarning)
    def test_eval_packing(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

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
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

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
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

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

        assert trainer.processing_class.chat_template == tokenizer.chat_template

        for i in range(2):
            role = "SYSTEM" if i else "system"
            system_prompt = (
                f"<|im_start|>{role}\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>"
            )
            system_prompt_ids = trainer.processing_class(system_prompt)["input_ids"]
            assert trainer.train_dataset[i]["input_ids"][: len(system_prompt_ids)] == system_prompt_ids

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_assistant_only(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, assistant_only_loss=True, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_dataset_preparation_builds_labels_for_assistant_only_loss(self):
        """Dataset preparation must bake the assistant masks into a labels column."""
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, assistant_only_loss=True, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
        )

        assert "labels" in trainer.train_dataset.column_names
        for example in trainer.train_dataset:
            labels, input_ids = example["labels"], example["input_ids"]
            assert len(labels) == len(input_ids)
            # Labels are input_ids with non-assistant tokens masked to -100.
            assert all(label == -100 or label == token_id for label, token_id in zip(labels, input_ids, strict=True))
            assert any(label != -100 for label in labels)  # assistant tokens contribute to the loss
            assert any(label == -100 for label in labels)  # non-assistant tokens are masked

    def test_fully_masked_examples_dropped_after_truncation(self):
        # Example 0's assistant tokens all lie beyond `max_length=3`, so keep_start truncation leaves it fully masked;
        # example 1 keeps a trainable token and survives.
        dataset = Dataset.from_list(
            [
                {"input_ids": [1, 2, 3, 4, 5], "assistant_masks": [0, 0, 0, 1, 1]},
                {"input_ids": [6, 7, 8], "assistant_masks": [1, 1, 1]},
            ]
        )

        training_args = SFTConfig(output_dir=self.tmp_dir, max_length=3, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        assert trainer.train_dataset[:]["labels"] == [[6, 7, 8]]

    def test_dataset_truncated_to_max_length(self):
        """Dataset preparation truncates every example to `max_length`."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, max_length=4, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        for example in trainer.train_dataset:
            assert len(example["input_ids"]) <= 4
            assert len(example["labels"]) <= 4

    def test_dataset_preparation_builds_labels_for_completion_only(self):
        """Dataset preparation must bake the completion mask into a labels column when completion_only_loss
        resolves to True (the default for prompt-completion datasets)."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_completion", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        assert "labels" in trainer.train_dataset.column_names
        for example in trainer.train_dataset:
            labels, input_ids = example["labels"], example["input_ids"]
            assert len(labels) == len(input_ids)
            # Labels are input_ids with prompt tokens masked to -100.
            assert all(label == -100 or label == token_id for label, token_id in zip(labels, input_ids, strict=True))
            assert any(label != -100 for label in labels)  # completion tokens contribute to the loss
            assert any(label == -100 for label in labels)  # prompt tokens are masked

    def test_dataset_preparation_respects_existing_labels(self):
        """A user-provided labels column must be taken as is, even when mask columns are also present."""
        dataset = Dataset.from_list(
            [
                {"input_ids": [1, 2, 3, 4], "labels": [1, -100, 3, -100], "assistant_masks": [1, 1, 1, 1]},
                {"input_ids": [5, 6], "labels": [-100, 6], "assistant_masks": [1, 1]},
            ]
        )

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        assert trainer.train_dataset[:]["labels"] == [[1, -100, 3, -100], [-100, 6]]

    def test_dataset_preparation_builds_labels_for_pretokenized_with_masks(self):
        """Pre-tokenized datasets that carry mask columns but no labels must get labels built at preparation."""
        dataset = Dataset.from_list(
            [
                {"input_ids": [1, 2, 3, 4], "assistant_masks": [0, 0, 1, 1]},
                {"input_ids": [5, 6, 7], "assistant_masks": [0, 1, 0]},
            ]
        )

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        assert trainer.train_dataset[:]["labels"] == [[-100, -100, 3, 4], [-100, 6, -100]]

    def test_packing_carries_labels(self):
        """With packing enabled, the labels column must be packed alongside input_ids (replacing the masks)."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_completion", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, packing=True, max_length=64, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        column_names = trainer.train_dataset.column_names
        assert "labels" in column_names
        assert "completion_mask" not in column_names
        for example in trainer.train_dataset:
            assert len(example["labels"]) == len(example["input_ids"])
        # The packed dataset must still contain trainable (non -100) labels
        assert any(label != -100 for example in trainer.train_dataset for label in example["labels"])

    def test_skip_prepare_dataset_with_masks_but_no_labels_raises(self):
        """With `skip_prepare_dataset=True`, labels are not built at preparation time and the collator doesn't
        consume the mask columns; such datasets must be rejected instead of silently training on the full sequence."""
        dataset = Dataset.from_list([{"input_ids": [1, 2, 3, 4], "assistant_masks": [0, 0, 1, 1]} for _ in range(2)])

        training_args = SFTConfig(
            output_dir=self.tmp_dir, dataset_kwargs={"skip_prepare_dataset": True}, report_to="none"
        )
        with pytest.raises(ValueError, match="Add a 'labels' column"):
            SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
            )

    def test_train_completion_only(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, completion_only_loss=True, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_completion_only_harmony(self):
        dataset = load_dataset("trl-internal-testing/harmony", "prompt_completion", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, completion_only_loss=True, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-GptOssForCausalLM", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_assistant_only_and_completion_only(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        # To test this case, we need to add user messages in the completion (they'll be masked in the loss)
        def add_to_completion(example):
            example["completion"].append(example["prompt"][0])
            example["completion"].append(example["completion"][0])
            return example

        dataset = dataset.map(add_to_completion)

        training_args = SFTConfig(
            output_dir=self.tmp_dir, assistant_only_loss=True, completion_only_loss=True, report_to="none"
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_assistant_only_iterable_dataset(self):
        dataset = load_dataset(
            "trl-internal-testing/zen", "conversational_language_modeling", split="train", streaming=True
        )

        training_args = SFTConfig(output_dir=self.tmp_dir, assistant_only_loss=True, max_steps=3, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_set_chat_template_from_model(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, chat_template_path="Qwen/Qwen3-4B", report_to="none")
        # trl-internal-testing/tiny-GPTNeoXForCausalLM doesn't have a chat template set by default
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-GPTNeoXForCausalLM", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_set_chat_template_from_path(self, lazy_shared_datadir):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            chat_template_path=str(lazy_shared_datadir / "template.jinja"),
            report_to="none",
        )
        # trl-internal-testing/tiny-GPTNeoXForCausalLM doesn't have a chat template set by default
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-GPTNeoXForCausalLM", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

        # Check that the template saved in the output directory is the same as the one used for training
        template_path = pathlib.Path(self.tmp_dir) / "checkpoint-9" / "chat_template.jinja"
        assert template_path.exists(), f"Chat template not found at {template_path}"

        with open(template_path) as f:
            template_content = f.read()
        with open(training_args.chat_template_path) as f:
            original_template_content = f.read()
        assert template_content == original_template_content, "Chat template content does not match the original"

    def test_train_toolcall_data(self):
        dataset = load_dataset("trl-internal-testing/toolcall", "language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,  # toolcall sequences are longer than standard data, reduce batch size to avoid OOM
            max_length=512,  # toolcall sequences are longer than standard data, limit length to avoid OOM
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_toolcall_data_as_json(self):
        # Tabular backends (Arrow/Parquet) can insert `None` for missing keys in nested structures.
        # If `tools` is stored as a list of dicts and examples use different dict schemas, nulls may
        # be introduced and break tool processing. This test ensures we also support `tools` provided
        # as a list of dicts.
        dataset = load_dataset("trl-internal-testing/toolcall", "language_modeling", split="train")

        def convert_to_json(example):
            return {"tools": json.loads(example["tools"])}

        dataset = dataset.map(convert_to_json)

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,  # toolcall sequences are longer than standard data, reduce batch size to avoid OOM
            max_length=512,  # toolcall sequences are longer than standard data, limit length to avoid OOM
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_eval(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        training_args = SFTConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()

        assert trainer.state.log_history[0]["eval_loss"] is not None

    @pytest.mark.parametrize(
        "eval_dataset_type",
        [
            "dataset",
            "iterable_dataset",
            "dataset_dict",
            "iterable_dataset_dict",
            "dict_of_dataset",
            "dict_of_iterable_dataset",
        ],
    )
    def test_evaluate_with_eval_dataset(self, eval_dataset_type):
        # `evaluate` accepts a raw (unprepared) dataset passed directly, not only a preprocessed `eval_dataset` set
        # at init. See https://github.com/huggingface/trl/issues/6115.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        streaming = "iterable" in eval_dataset_type
        eval_split = load_dataset(
            "trl-internal-testing/zen", "standard_language_modeling", split="test", streaming=streaming
        )
        if eval_dataset_type in ("dataset", "iterable_dataset"):
            eval_dataset = eval_split
        elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
            dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
            eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
        else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
            eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=train_dataset
        )

        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        if eval_dataset_type in ("dataset", "iterable_dataset"):
            assert metrics["eval_loss"] is not None
        else:
            assert metrics["eval_data1_loss"] is not None
            assert metrics["eval_data2_loss"] is not None

    def test_train_with_metric_for_best_model(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            eval_strategy="steps",
            eval_steps=3,
            # It's important to use a key that SFTTrainer adds itself (not one from the base Trainer), since the bug is
            # that trainer-specific metrics don't reach the dict returned by `evaluate()`.
            metric_for_best_model="eval_mean_token_accuracy",
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()

    def test_train_with_multiple_eval_dataset(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        training_args = SFTConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset={"data1": dataset["test"], "data2": dataset["test"]},
        )
        trainer.train()

        assert trainer.state.log_history[-3]["eval_data1_loss"] is not None
        assert trainer.state.log_history[-2]["eval_data2_loss"] is not None

    @pytest.mark.parametrize("train_dataset_type", ["dataset", "iterable_dataset", "none", "unsupported_dataset_dict"])
    def test_init_with_train_dataset(self, train_dataset_type):
        streaming = "iterable" in train_dataset_type
        if train_dataset_type == "none":
            train_dataset = None
        else:
            train_dataset = load_dataset(
                "trl-internal-testing/zen", "standard_language_modeling", split="train", streaming=streaming
            )
            if train_dataset_type == "unsupported_dataset_dict":
                # `DatasetDict` is representative of any unsupported type here; not exhaustive
                train_dataset = DatasetDict({"train": train_dataset})

        # Iterable (streaming) datasets have no length, so `max_steps` is required.
        training_args = SFTConfig(output_dir=self.tmp_dir, max_steps=3 if streaming else -1, report_to="none")

        if train_dataset_type == "none":
            with pytest.raises(ValueError, match="`train_dataset` is required"):
                SFTTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    args=training_args,
                    train_dataset=train_dataset,
                )
        elif train_dataset_type == "unsupported_dataset_dict":
            with pytest.raises(TypeError, match="`train_dataset` must be a `Dataset` or `IterableDataset`"):
                SFTTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    args=training_args,
                    train_dataset=train_dataset,
                )
        else:
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=train_dataset
            )
            assert "input_ids" in next(iter(trainer.train_dataset))

    @pytest.mark.parametrize(
        "eval_dataset_type",
        [
            "dataset",
            "iterable_dataset",
            "dataset_dict",
            "iterable_dataset_dict",
            "dict_of_dataset",
            "dict_of_iterable_dataset",
            "none",
        ],
    )
    def test_init_with_eval_dataset(self, eval_dataset_type):
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            streaming = "iterable" in eval_dataset_type
            eval_split = load_dataset(
                "trl-internal-testing/zen", "standard_language_modeling", split="test", streaming=streaming
            )
            if eval_dataset_type in ("dataset", "iterable_dataset"):
                eval_dataset = eval_split
            elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
                dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
                eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
            else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
                eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
            # Each split was tokenized independently.
            assert "input_ids" in next(iter(trainer.eval_dataset["data1"]))
            assert "input_ids" in next(iter(trainer.eval_dataset["data2"]))
        else:
            assert "input_ids" in next(iter(trainer.eval_dataset))

    def test_train_with_compute_metrics(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        def dummy_compute_metrics(eval_pred):
            # The logits are the full-vocabulary logits, as before the fused LM head
            assert eval_pred.predictions.shape[-1] == trainer.model.config.vocab_size
            return {"my_metric": 0.123}

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

        with pytest.warns(FutureWarning, match="`return_outputs=True`"):
            trainer.train()

        assert trainer.state.log_history[-2]["eval_my_metric"] == 0.123

    def test_train_with_compute_loss_func(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        vocab_sizes = []

        def compute_loss_func(outputs, labels, num_items_in_batch=None):
            # The outputs are the model's own, with the full logits, as before the fused LM head
            vocab_sizes.append(outputs.logits.shape[-1])
            return outputs.loss

        training_args = SFTConfig(output_dir=self.tmp_dir, max_steps=2, logging_steps=1, report_to="none")
        with pytest.warns(FutureWarning, match="`compute_loss_func` receives the model's own outputs"):
            trainer = SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
                compute_loss_func=compute_loss_func,
            )
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert vocab_sizes and all(v == trainer.model.config.vocab_size for v in vocab_sizes)
        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-2]["mean_token_accuracy"] is not None
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_predict_returns_logits(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="test")
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        with pytest.warns(FutureWarning, match="`return_outputs=True`"):
            predictions = trainer.predict(trainer.train_dataset)

        assert predictions.predictions.shape[-1] == trainer.model.config.vocab_size

    def test_chunked_nll_is_deprecated_alias_of_nll(self):
        with pytest.warns(FutureWarning, match="`loss_type='chunked_nll'` is deprecated"):
            training_args = SFTConfig(output_dir=self.tmp_dir, loss_type="chunked_nll", report_to="none")
        assert training_args.loss_type == "nll"

    # In practice, this test is the same as `test_train`, since gradient checkpointing is enabled by default in
    # `SFTTrainer`. We keep it as a regression guard: if the default ever changes, we still explicitly test gradient
    # checkpointing, which has caused issues in the past.
    def test_train_with_gradient_checkpointing(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, gradient_checkpointing=True, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize("use_reentrant", [True, False])
    def test_train_with_gradient_checkpointing_reentrant(self, use_reentrant):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_tag_added(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
        )

        for tag in ["sft", "trl"]:
            assert tag in trainer.model.model_tags

    @require_peft
    def test_tag_added_peft(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

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
            pytest.param(
                "trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.5.0"),
                    reason="Gemma4 models were introduced in transformers-5.5.0",
                ),
            ),
            # "trl-internal-testing/tiny-Idefics2ForConditionalGeneration",  high memory peak, skipped for now
            # "trl-internal-testing/tiny-Idefics3ForConditionalGeneration",  high memory peak, skipped for now
            pytest.param(
                "trl-internal-testing/tiny-Lfm2VlForConditionalGeneration-2.5",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.0.0"),
                    reason="LFM2.5-VL requires transformers>=5.0.0",
                ),
            ),
            "trl-internal-testing/tiny-LlavaForConditionalGeneration",
            "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
            pytest.param(
                "trl-internal-testing/tiny-MuseGlimmerForConditionalGeneration",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.15.0"),
                    reason="Muse Glimmer was introduced in transformers-5.15.0",
                ),
            ),
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
                ],
            ),
            pytest.param(
                "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.2.0"),
                    reason="Qwen3.5 models were introduced in transformers-5.2.0",
                ),
            ),
            pytest.param(
                "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.2.0"),
                    reason="Qwen3.5 models were introduced in transformers-5.2.0",
                ),
            ),
        ],
    )
    @require_vision
    def test_train_vlm(self, model_id):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,  # VLM training is memory intensive, reduce batch size to avoid OOM
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            report_to="none",
        )
        trainer = SFTTrainer(model=model_id, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            # LLaVA & LLaVA-Next: vision_feature_layer=-2 leaves the last encoder layer (layers.1) and
            # post_layernorm (pooler-only path) without gradient by design. Assert they stay frozen — if they
            # ever start training, the feature-selection plumbing has likely regressed.
            if model_id in (
                "trl-internal-testing/tiny-LlavaForConditionalGeneration",
                "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
            ) and ("encoder.layers.1" in n or "post_layernorm" in n):
                assert torch.equal(param, new_param), f"Param {n} expected frozen by LLaVA design, but changed"
            else:
                assert not torch.equal(param, new_param), f"Param {n} is not updated"

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        ],
    )
    @pytest.mark.xfail(
        Version(transformers.__version__) < Version("4.57.0"),
        reason="Mixing text-only and image+text examples is only supported in transformers >= 4.57.0",
        strict=False,
    )
    @require_vision
    def test_train_vlm_multi_image(self, model_id):
        dataset = load_dataset(
            "trl-internal-testing/zen-multi-image", "conversational_prompt_completion", split="train"
        )

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,  # VLM training is memory intensive, reduce batch size to avoid OOM
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Param {n} is not updated"

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            # Special case for Gemma, as it uses token_type_ids, and we need to ensure they are properly in the collator:
            "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
            pytest.param(
                "trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.5.0"),
                    reason="Gemma4 models were introduced in transformers-5.5.0",
                ),
            ),
        ],
    )
    @require_vision
    def test_train_vlm_prompt_completion(self, model_id):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_completion", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,  # VLM training is memory intensive, reduce batch size to avoid OOM
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Param {n} is not updated"

    # Gemma 3n uses a timm encoder, making it difficult to create a smaller variant for testing.
    # To ensure coverage, we run tests on the full model but mark them as slow to exclude from default runs.
    @pytest.mark.slow
    @require_vision
    @pytest.mark.skip(reason="Model google/gemma-3n-E2B-it is gated and requires HF token")
    def test_train_vlm_gemma_3n(self):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            per_device_train_batch_size=1,  # VLM training is memory intensive, reduce batch size to avoid OOM
            model_init_kwargs={"dtype": "bfloat16"},
            report_to="none",
        )
        trainer = SFTTrainer(model="google/gemma-3n-E2B-it", args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if "model.audio_tower" in n or "model.embed_audio" in n:
                # The audio embedding parameters are not updated because this dataset contains no audio data
                continue
            assert not torch.equal(param, new_param), f"Param {n} is not updated"

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        ],
    )
    @pytest.mark.parametrize(
        "dataset_config",
        [
            "conversational_language_modeling",
            "conversational_prompt_completion",
            "standard_language_modeling",  # Regression test for #5334
            "standard_prompt_completion",
        ],
    )
    @require_vision
    def test_train_vlm_text_only_data(self, model_id, dataset_config):
        dataset = load_dataset("trl-internal-testing/zen", dataset_config, split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n.startswith("model.visual"):
                torch.testing.assert_close(param, new_param, rtol=1e-12, atol=1e-12, msg=f"Param {n} is updated")
            else:
                assert not torch.equal(param, new_param), f"Param {n} is not updated"

    @ignore_warnings(message="You are using packing, but the attention implementation is not.*", category=UserWarning)
    @require_vision
    def test_train_vlm_text_only_data_packing(self):
        # Packing is incompatible with on-the-fly image processing, not with VLMs. A text-only dataset goes through
        # the regular text pipeline, so packing must be available on a VLM checkpoint too. Regression test for #6545.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            packing=True,
            report_to="none",
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n.startswith("model.visual"):
                torch.testing.assert_close(param, new_param, rtol=1e-12, atol=1e-12, msg=f"Param {n} is updated")
            else:
                assert not torch.equal(param, new_param), f"Param {n} is not updated"

    @require_vision
    def test_vision_dataset_with_text_model_raises(self):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_language_modeling", split="train")
        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        with pytest.raises(ValueError, match="vision-related.*vision-language model"):
            SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
            )

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

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-1]["mean_token_accuracy"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if "base_model" in n:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "prompt_encoder" in n:  # We expect the peft params to be different
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
            else:
                raise ValueError(f"Unexpected parameter {n} in model: {trainer.model}")

    @require_peft
    @require_bitsandbytes
    def test_train_peft_and_quantization(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, learning_rate=0.1, report_to="none")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",  # identifier, so that the trainer quantizes it
            args=training_args,
            train_dataset=dataset,
            quantization_config=quantization_config,
            peft_config=LoraConfig(),
        )

        # Check that the trainer applied the quantization config when loading the model
        assert trainer.model.base_model.model.is_loaded_in_4bit

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-1]["mean_token_accuracy"] is not None

        # Check that the peft params have changed, and that they are cast to bfloat16, as recommended by the QLoRA
        # paper. The base model params are not checked: bitsandbytes casts the biases of a Linear4bit in-place during
        # the forward pass, so some of them change in a way that is unrelated to training.
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if "lora" in n:  # We expect the peft params to be different
                assert param.dtype == torch.bfloat16, f"Parameter {n} is not in bfloat16."
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    def test_prompt_tuning_peft_model(self):
        """Test that SFT works with Prompt Tuning and a pre-converted PeftModel"""
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32")
        model = get_peft_model(model, PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=8))

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-1]["mean_token_accuracy"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if "base_model" in n:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "prompt_encoder" in n:  # We expect the peft params to be different
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
            else:
                raise ValueError(f"Unexpected parameter {n} in model: {trainer.model}")

    def test_pad_token_id_synced_with_model_config(self):
        # This model's tokenizer has no pad token, so the trainer falls back to the eos token. The model configs must
        # follow: otherwise `Trainer` realigns them at train time and reports it as a change the user did not make.
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-MistralForCausalLM-0.2", args=training_args, train_dataset=dataset
        )

        pad_token_id = trainer.processing_class.pad_token_id
        assert pad_token_id is not None
        assert trainer.model.config.pad_token_id == pad_token_id
        assert trainer.model.generation_config.pad_token_id == pad_token_id

    @require_vision
    def test_pad_token_id_synced_with_model_config_vision(self):
        # A vision dataset takes the other collator branch, which used to skip the pad token handling entirely.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, max_length=None, report_to="none")
        trainer = SFTTrainer(
            model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            args=training_args,
            train_dataset=dataset,
        )

        pad_token_id = trainer.processing_class.tokenizer.pad_token_id
        assert pad_token_id is not None
        assert trainer.model.config.get_text_config().pad_token_id == pad_token_id
        assert trainer.model.generation_config.pad_token_id == pad_token_id

    @pytest.mark.parametrize(
        "generation_eos_token_id, expected_eos_token_ids",
        [
            ([151645, 151643], [151644, 151645, 151643]),  # a list of ids, as Qwen and Llama ship
            (151645, [151644, 151645]),  # a single id, as Mistral and GPT-2 ship
            (None, [151644]),  # no id at all
            ([151644, 151645], [151644, 151645]),  # a list that already holds the requested token
        ],
    )
    def test_eos_token_id_synced_with_model_config(self, generation_eos_token_id, expected_eos_token_ids):
        # The trainer sets the requested eos token on the tokenizer. The model configs must follow: otherwise
        # `Trainer` realigns them at train time and reports it as a change the user did not make. The generation
        # config holds the eos token as a list, as a single id, or not at all, so cover the three shapes.
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32")
        model.generation_config.eos_token_id = generation_eos_token_id

        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")

        training_args = SFTConfig(output_dir=self.tmp_dir, eos_token="<|im_start|>", report_to="none")
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)

        eos_token_id = trainer.processing_class.eos_token_id
        assert eos_token_id == 151644  # <|im_start|>
        assert trainer.model.config.eos_token_id == eos_token_id
        # The model's own eos tokens are kept, since any of them halts generation
        assert trainer.model.generation_config.eos_token_id == expected_eos_token_ids


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

    @pytest.mark.parametrize("packing", [True, pytest.param(False, marks=xfail_data_parallel)])
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
    @pytest.mark.parametrize("packing", [True, pytest.param(False, marks=xfail_data_parallel)])
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

    @pytest.mark.parametrize("packing", [True, pytest.param(False, marks=xfail_data_parallel)])
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
    @pytest.mark.skipif(
        backend_device_count(torch_device) > 1,
        reason="segfaults in accelerate's get_max_memory when more than one accelerator is visible, taking the whole "
        "pytest process down; cause not yet diagnosed (https://github.com/huggingface/trl/issues/6836)",
    )
    def test_train_offloading(self, model_name, packing):
        """Test that activation offloading works with SFTTrainer."""
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

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

        release_memory(trainer.model, trainer)
