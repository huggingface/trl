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
import transformers
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_peft_available

from trl import KTOConfig, KTOTrainer
from trl.trainer.kto_trainer import DataCollatorForUnpairedPreference, DataCollatorForVisionUnpairedPreference

from .testing_utils import TrlTestCase, require_bitsandbytes, require_liger_kernel, require_peft, require_vision


if is_peft_available():
    import peft
    from peft import LoraConfig, PromptTuningConfig, get_peft_model
    from peft.utils import TaskType


@require_vision
class TestDataCollatorForVisionUnpairedPreference(TrlTestCase):
    @pytest.mark.skipif(
        Version(transformers.__version__) < Version("5.3.0"),
        reason="mm_token_type_ids are returned by default since transformers-5.3.0 (see transformers#43972)",
    )
    def test_mm_token_type_ids_shape(self):
        # Regression guard: when the processor returns mm_token_type_ids (Qwen2.5-VL after transformers#43972),
        # the collator must produce a KL_token_type_ids whose width matches KL_input_ids,
        # not the main completion's width (the two differ whenever their text lengths differ).
        from PIL import Image
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained("trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration")
        collator = DataCollatorForVisionUnpairedPreference(processor, calculate_kl=True)
        image = Image.new("RGB", (16, 16))
        examples = [
            {
                "images": [image],
                "prompt": [{"role": "user", "content": "What is this?"}],
                "completion": [{"role": "assistant", "content": "A red square."}],
                "label": True,
            },
            {
                "images": [image],
                "prompt": [{"role": "user", "content": "Describe it."}],
                "completion": [{"role": "assistant", "content": "An image."}],
                "label": False,
            },
        ]
        output = collator(examples)

        assert "mm_token_type_ids" in output
        assert output["mm_token_type_ids"].shape == output["input_ids"].shape, (
            f"mm_token_type_ids shape {output['mm_token_type_ids'].shape} != "
            f"input_ids shape {output['input_ids'].shape}"
        )
        assert "KL_mm_token_type_ids" in output
        assert output["KL_mm_token_type_ids"].shape == output["KL_input_ids"].shape, (
            f"KL_mm_token_type_ids shape {output['KL_mm_token_type_ids'].shape} != "
            f"KL_input_ids shape {output['KL_input_ids'].shape}"
        )

    def test_output_keys(self):
        from PIL import Image
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained("trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration")
        image = Image.new("RGB", (16, 16))

        def make_examples():
            return [
                {
                    "images": [image],
                    "prompt": [{"role": "user", "content": "What is this?"}],
                    "completion": [{"role": "assistant", "content": "A red square."}],
                    "label": True,
                },
                {
                    "images": [image],
                    "prompt": [{"role": "user", "content": "Describe it."}],
                    "completion": [{"role": "assistant", "content": "An image."}],
                    "label": False,
                },
            ]

        # With KL
        collator = DataCollatorForVisionUnpairedPreference(processor, calculate_kl=True)
        output = collator(make_examples())
        for key in ["input_ids", "attention_mask", "completion_mask", "pixel_values", "label"]:
            assert key in output, f"Missing key: {key}"
        for key in ["KL_input_ids", "KL_attention_mask", "KL_completion_mask"]:
            assert key in output, f"Missing KL key: {key}"

        # Without KL
        collator_no_kl = DataCollatorForVisionUnpairedPreference(processor, calculate_kl=False)
        output_no_kl = collator_no_kl(make_examples())
        assert "input_ids" in output_no_kl
        assert "KL_input_ids" not in output_no_kl

    def test_kl_cycling(self):
        # The KL completion for example i must be the completion from example i-1 (cycled by +1).
        from PIL import Image
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained("trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration")
        collator = DataCollatorForVisionUnpairedPreference(processor, calculate_kl=True)
        image = Image.new("RGB", (16, 16))
        # Two distinct completions so that cycling is detectable
        examples = [
            {
                "images": [image],
                "prompt": [{"role": "user", "content": "Q1"}],
                "completion": [{"role": "assistant", "content": "Answer one."}],
                "label": True,
            },
            {
                "images": [image],
                "prompt": [{"role": "user", "content": "Q2"}],
                "completion": [{"role": "assistant", "content": "Answer two."}],
                "label": False,
            },
        ]
        output = collator(examples)
        # KL completions are cycled: KL[0] = completion[-1], KL[1] = completion[0]
        # They must differ from the matching main completion (unless both are identical strings, which they aren't here)
        assert not torch.equal(output["input_ids"][0], output["KL_input_ids"][0])
        assert not torch.equal(output["input_ids"][1], output["KL_input_ids"][1])


class TestDataCollatorForUnpairedPreference(TrlTestCase):
    def test_padding_and_masks(self):
        collator = DataCollatorForUnpairedPreference(pad_token_id=0)
        examples = [
            {"prompt_ids": [1, 2, 3], "completion_ids": [4, 5], "KL_completion_ids": [6], "label": True},
            {"prompt_ids": [7, 8], "completion_ids": [9, 10], "KL_completion_ids": [11, 12, 13], "label": False},
        ]
        result = collator(examples)

        expected_input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],  # prompt + completion (example 1)
                [7, 8, 9, 10, 0],  # prompt + completion (example 2, padded)
            ]
        )
        expected_attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
            ]
        )
        expected_completion_mask = torch.tensor(
            [
                [0, 0, 0, 1, 1],  # completion (example 1)
                [0, 0, 1, 1, 0],  # completion (example 2, padded)
            ]
        )
        expected_kl_input_ids = torch.tensor(
            [
                [1, 2, 3, 6, 0],  # prompt + KL completion (example 1, padded)
                [7, 8, 11, 12, 13],  # prompt + KL completion (example 2)
            ]
        )
        expected_kl_attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        expected_kl_completion_mask = torch.tensor(
            [
                [0, 0, 0, 1, 0],  # KL completion (example 1, padded)
                [0, 0, 1, 1, 1],  # KL completion (example 2)
            ]
        )

        assert set(result.keys()) == {
            "input_ids",
            "attention_mask",
            "completion_mask",
            "KL_input_ids",
            "KL_attention_mask",
            "KL_completion_mask",
            "label",
        }
        torch.testing.assert_close(result["input_ids"], expected_input_ids)
        torch.testing.assert_close(result["attention_mask"], expected_attention_mask)
        torch.testing.assert_close(result["completion_mask"], expected_completion_mask)
        torch.testing.assert_close(result["KL_input_ids"], expected_kl_input_ids)
        torch.testing.assert_close(result["KL_attention_mask"], expected_kl_attention_mask)
        torch.testing.assert_close(result["KL_completion_mask"], expected_kl_completion_mask)
        torch.testing.assert_close(result["label"], torch.tensor([True, False]))

    def test_optional_reference_logps(self):
        collator = DataCollatorForUnpairedPreference(pad_token_id=0)
        examples = [
            {
                "prompt_ids": [1, 2],
                "completion_ids": [3],
                "KL_completion_ids": [4],
                "ref_logps": 0.1,
                "ref_KL_logps": 0.2,
                "label": True,
            },
            {
                "prompt_ids": [5],
                "completion_ids": [6, 7],
                "KL_completion_ids": [8, 9],
                "ref_logps": 0.3,
                "ref_KL_logps": 0.4,
                "label": False,
            },
        ]
        result = collator(examples)

        expected_ref_logps = torch.tensor([0.1, 0.3])
        expected_ref_kl_logps = torch.tensor([0.2, 0.4])

        assert set(result.keys()) == {
            "input_ids",
            "attention_mask",
            "completion_mask",
            "KL_input_ids",
            "KL_attention_mask",
            "KL_completion_mask",
            "ref_logps",
            "ref_KL_logps",
            "label",
        }
        torch.testing.assert_close(result["ref_logps"], expected_ref_logps)
        torch.testing.assert_close(result["ref_KL_logps"], expected_ref_kl_logps)

    def test_with_pad_to_multiple_of(self):
        collator = DataCollatorForUnpairedPreference(pad_token_id=0, pad_to_multiple_of=5)
        examples = [
            {"prompt_ids": [1], "completion_ids": [2], "KL_completion_ids": [3], "label": True},
            {"prompt_ids": [4, 5], "completion_ids": [6, 7], "KL_completion_ids": [8, 9], "label": False},
        ]
        result = collator(examples)

        expected_input_ids = torch.tensor(
            [
                [1, 2, 0, 0, 0],  # prompt + completion (example 1, padded to multiple of 5)
                [4, 5, 6, 7, 0],  # prompt + completion (example 2)
            ]
        )
        expected_kl_input_ids = torch.tensor(
            [
                [1, 3, 0, 0, 0],  # prompt + KL completion (example 1, padded to multiple of 5)
                [4, 5, 8, 9, 0],  # prompt + KL completion (example 2)
            ]
        )

        assert set(result.keys()) == {
            "input_ids",
            "attention_mask",
            "completion_mask",
            "KL_input_ids",
            "KL_attention_mask",
            "KL_completion_mask",
            "label",
        }
        torch.testing.assert_close(result["input_ids"], expected_input_ids)
        torch.testing.assert_close(result["KL_input_ids"], expected_kl_input_ids)


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
    def test_train(self, config_name, loss_type, pre_compute, eval_dataset):
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            gradient_accumulation_steps=1,
            learning_rate=9e-1,
            eval_strategy="steps" if eval_dataset else "no",
            beta=0.1,
            precompute_ref_log_probs=pre_compute,
            loss_type=loss_type,
            report_to="none",
        )

        dataset = load_dataset("trl-internal-testing/zen", config_name)

        trainer = KTOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"] if eval_dataset else None,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.equal(param, new_param)

    def test_fully_truncated_completion_examples_dropped(self):
        """The collator truncates with `keep_start`, so an example whose prompt alone fills `max_length` loses every
        completion token; it's dropped during dataset preparation."""
        dataset = Dataset.from_dict(
            {
                "prompt": ["Hi", "This is a very long prompt that fills up the whole max_length budget on its own"],
                "completion": [" there", " yes"],
                "label": [True, False],
            }
        )

        training_args = KTOConfig(output_dir=self.tmp_dir, max_length=6, report_to="none")
        trainer = KTOTrainer(model=self.model_id, args=training_args, train_dataset=dataset)

        # Only the short-prompt example survives.
        assert len(trainer.train_dataset) == 1
        assert trainer.train_dataset[0]["prompt_ids"] == trainer.processing_class("Hi")["input_ids"]

    @pytest.mark.parametrize("precompute_ref_log_probs", [False, True])
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
    def test_evaluate_with_eval_dataset(self, eval_dataset_type, precompute_ref_log_probs):
        # `evaluate` accepts a raw (unprepared) dataset passed directly, not only a preprocessed `eval_dataset` set
        # at init. See https://github.com/huggingface/trl/issues/6115. Also a regression test: Accelerate's dispatch
        # for `IterableDataset` requires every batch field to be a tensor, which previously broke streaming
        # evaluation. `apo_zero_unpaired` avoids KTO's KL term, which is incompatible with streaming datasets.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")
        streaming = "iterable" in eval_dataset_type
        eval_split = load_dataset(
            "trl-internal-testing/zen", "standard_unpaired_preference", split="test", streaming=streaming
        )
        if eval_dataset_type in ("dataset", "iterable_dataset"):
            eval_dataset = eval_split
        elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
            dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
            eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
        else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
            eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            loss_type="apo_zero_unpaired",
            precompute_ref_log_probs=precompute_ref_log_probs,
            report_to="none",
        )
        trainer = KTOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=train_dataset
        )

        if streaming and precompute_ref_log_probs:
            with pytest.raises(ValueError, match="precompute_ref_log_probs.*not supported with IterableDataset"):
                trainer.evaluate(eval_dataset=eval_dataset)
            return

        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        if eval_dataset_type in ("dataset", "iterable_dataset"):
            assert metrics["eval_loss"] is not None
        else:
            assert metrics["eval_data1_loss"] is not None
            assert metrics["eval_data2_loss"] is not None

    def test_evaluate_precompute_ref_log_probs_after_training_raises(self):
        # Full fine-tuning with `precompute_ref_log_probs=True` and no `ref_model` uses `self.model` as the reference.
        # That's valid only before training; a dataset passed to `evaluate()` afterwards can't get a correct reference.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")
        eval_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="test")
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            max_steps=1,
            precompute_ref_log_probs=True,
            report_to="none",
        )
        trainer = KTOTrainer(model=self.model_id, args=training_args, train_dataset=train_dataset)

        # Before training the reference is available, so evaluating a new dataset works.
        assert trainer.evaluate(eval_dataset=eval_dataset)["eval_loss"] is not None

        trainer.train()

        with pytest.raises(ValueError, match="Cannot compute reference log-probs for a dataset passed to"):
            trainer.evaluate(eval_dataset=eval_dataset)

    @pytest.mark.parametrize("eval_dataset_type", ["dataset", "dataset_dict", "dict_of_dataset"])
    def test_evaluate_precompute_ref_log_probs_at_init_after_training(self, eval_dataset_type):
        # Regression for the guard above: an `eval_dataset` set at init has its reference log-probs precomputed once
        # (against the untrained reference) and stored, so no-arg `evaluate()` reuses those stored values and must not
        # raise after training, even with full fine-tuning and `precompute_ref_log_probs=True`.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")
        eval_split = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="test")
        if eval_dataset_type == "dataset":
            eval_dataset = eval_split
        elif eval_dataset_type == "dataset_dict":
            eval_dataset = DatasetDict({"data1": eval_split, "data2": eval_split})
        else:  # "dict_of_dataset"
            eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            max_steps=1,
            precompute_ref_log_probs=True,
            report_to="none",
        )
        trainer = KTOTrainer(
            model=self.model_id, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
        )

        trainer.train()

        metrics = trainer.evaluate()
        if eval_dataset_type == "dataset":
            assert metrics["eval_loss"] is not None
        else:
            assert metrics["eval_data1_loss"] is not None
            assert metrics["eval_data2_loss"] is not None

    def test_trust_remote_code(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"

        with pytest.raises(ValueError, match="custom code"):
            KTOTrainer(
                model=model_id,
                args=KTOConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
            )

        trainer = KTOTrainer(
            model=model_id,
            args=KTOConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    # Special case for harmony
    def test_train_gpt_oss(self):
        dataset = load_dataset("trl-internal-testing/harmony", "preference", split="train")

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = KTOTrainer(
            model="trl-internal-testing/tiny-GptOssForCausalLM", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

        # MoE models log the load-balancing auxiliary loss (on by default)
        assert trainer.aux_loss_enabled
        assert trainer.state.log_history[-1]["aux_loss"] is not None

    def test_train_with_ref_model_is_model_raises(self):
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            report_to="none",
        )

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        with pytest.raises(ValueError):
            KTOTrainer(
                model=self.model,
                ref_model=self.model,  # ref_model can't be the same as model
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dataset,
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
            gradient_accumulation_steps=1,
            learning_rate=9e-1,
            eval_strategy="steps",
            beta=0.1,
            report_to="none",
        )

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")
        train_dataset = dataset["train"]

        trainer = KTOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=dataset["test"],
        )

        # Verify the tokenization step: dataset stores raw token IDs (aligned with DPO style).
        # prompt_ids must start with the tokenized prompt text.
        prompt_ids = self.tokenizer(train_dataset["prompt"][0])["input_ids"]
        assert trainer.train_dataset[0]["prompt_ids"][: len(prompt_ids)] == prompt_ids
        # completion_ids are the raw answer tokens (no prompt prefix, no BOS/EOS added yet).
        assert len(trainer.train_dataset[0]["completion_ids"]) > 0

        # Verify the collator output (assembly, BOS/EOS insertion, labels).
        example = trainer.train_dataset[0]
        batch = trainer.data_collator([example])
        # input_ids ends with EOS
        assert batch["input_ids"][0, -1].item() == self.tokenizer.eos_token_id
        # completion_mask: prompt tokens are 0, completion tokens are 1; at least the prompt is masked
        assert "completion_mask" in batch
        completion_mask = batch["completion_mask"][0].tolist()
        assert 0 in completion_mask and 1 in completion_mask
        first_completion = next(i for i, m in enumerate(completion_mask) if m == 1)
        assert first_completion > 0  # at least the prompt is masked
        assert all(m == 0 for m in completion_mask[:first_completion])

    def test_train_without_providing_ref_model(self):
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            gradient_accumulation_steps=4,
            learning_rate=9e-1,
            eval_strategy="steps",
            beta=0.1,
            report_to="none",
        )

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

        trainer = KTOTrainer(
            model=self.model,
            ref_model=None,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.equal(param, new_param)

    @require_peft
    def test_train_without_providing_ref_model_with_lora(self):
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
            gradient_accumulation_steps=4,
            learning_rate=9e-1,
            eval_strategy="steps",
            beta=0.1,
            report_to="none",
        )

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

        trainer = KTOTrainer(
            model=self.model,
            ref_model=None,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            peft_config=lora_config,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            if "lora" in n:
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
                    assert not torch.equal(param, new_param)

    def test_compute_metrics(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32")
        ref_model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

        def dummy_compute_metrics(*args, **kwargs):
            return {"test": 0.0}

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
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
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=dummy_compute_metrics,
        )

        trainer.train()

        assert trainer.state.log_history[-2]["eval_test"] == 0.0

    def test_train_with_explicit_ref_model(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        # When specifying a ref model, it's usually because we want it to be a different checkpoint, but for testing
        # purposes we will just use the same checkpoint
        ref_model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype="float32"
        )
        trainer = KTOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            ref_model=ref_model,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
            new_ref_param = trainer.ref_model.get_parameter(n)
            torch.testing.assert_close(param, new_ref_param, msg=f"Reference model parameter {n} has changed.")

    def test_train_with_sync_ref_model(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            sync_ref_model=True,
            ref_model_sync_steps=2,  # reduce sync steps to ensure a sync happens
            report_to="none",
        )
        trainer = KTOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
        assert trainer.ref_model is not None
        previous_ref_params = {n: param.clone() for n, param in trainer.ref_model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
            new_ref_param = trainer.ref_model.get_parameter(n)
            assert not torch.equal(previous_ref_params[n], new_ref_param), f"Ref Parameter {n} has not changed."

    def test_train_model_dtype(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            model_init_kwargs={"dtype": torch.float16},
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = KTOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            # For some reason model.layers.0.input_layernorm.weight doesn't change in GitHub Actions but does
            # locally. We ignore this parameter for now
            if "layernorm" in n:
                continue
            new_param = trainer.model.get_parameter(n)
            # Check the torch dtype
            assert new_param.dtype == torch.float16
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    def test_train_peft_model(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")

        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        lora_config = LoraConfig()
        model = get_peft_model(model, lora_config)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            learning_rate=1.0,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = KTOTrainer(model=model, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n and "ref" not in n:  # and the peft params to be different (except base and ref)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    # In practice, this test is the same as `test_kto_trainer_without_providing_ref_model_with_lora`, since gradient
    # checkpointing is enabled by default in `KTOTrainer`. We keep it as a regression guard: if the default ever
    # changes, we still explicitly test PEFT + gradient checkpointing, which has caused issues in the past.
    @require_peft
    def test_train_with_peft_config_and_gradient_checkpointing(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            gradient_checkpointing=True,
            report_to="none",
        )

        trainer = KTOTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            use_liger_kernel=True,
            report_to="none",
        )
        trainer = KTOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_liger_kernel
    @require_peft
    def test_train_with_liger_kernel_and_peft(self):
        # A LoRA adapter that does not target lm_head leaves the head as a plain Linear, so Liger reads the real
        # weight. Verify the full PEFT+Liger path actually trains (peft params change, base params stay frozen).
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            use_liger_kernel=True,
            report_to="none",
        )
        trainer = KTOTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(target_modules=["q_proj", "v_proj"]),
        )
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
        trainer.train()
        assert trainer.state.log_history[-1]["train_loss"] is not None
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_liger_kernel
    @require_peft
    def test_liger_kernel_with_peft_lm_head_raises(self):
        # The Liger fused KTO loss reads `lm_head.weight` directly, so a LoRA adapter on `lm_head` is silently
        # ignored and never trained. The trainer must fail fast instead of training a silently-frozen head.
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")
        training_args = KTOConfig(output_dir=self.tmp_dir, use_liger_kernel=True, report_to="none")
        # `ensure_weight_tying=True` silences PEFT's weight-tying warning fired when lm_head is in the adapter on a
        # model with untied embeddings; once tying respects `tie_word_embeddings`, the flag itself warns that no tied
        # modules were found, so it must only be set on the affected range.
        # - Introduced in PEFT 0.19.0 (peft#2879); fixed on main, unreleased as of 0.19.2.dev0 (peft#3171)
        needs_ensure_weight_tying = Version("0.19.0") <= Version(peft.__version__) < Version("0.19.2.dev0")
        lora_config = LoraConfig(
            target_modules=["q_proj", "v_proj", "lm_head"],
            **({"ensure_weight_tying": True} if needs_ensure_weight_tying else {}),
        )
        with pytest.raises(ValueError, match="lm_head"):
            KTOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
                peft_config=lora_config,
            )

    @require_liger_kernel
    @require_peft
    def test_liger_kernel_with_peft_prompt_learning_raises(self):
        # Prompt-learning methods inject virtual tokens via PeftModel.forward(), which the Liger KTO loss bypasses.
        # The trainer must fail fast to avoid computing the loss on the wrong (truncated) sequence.
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")
        training_args = KTOConfig(output_dir=self.tmp_dir, use_liger_kernel=True, report_to="none")
        with pytest.raises(ValueError, match="prompt-learning"):
            KTOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
                peft_config=PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=8),
            )

    @require_liger_kernel
    def test_init_fails_with_compute_metrics_and_liger(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            use_liger_kernel=True,
            report_to="none",
        )

        with pytest.raises(ValueError, match="compute_metrics is not supported with the Liger kernel"):
            KTOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
                compute_metrics=lambda _: {},
            )

    @require_liger_kernel
    def test_init_fails_with_moe_aux_loss_and_liger(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        # The MoE auxiliary loss is on by default; it is incompatible with the Liger fused loss.
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            use_liger_kernel=True,
            report_to="none",
        )

        with pytest.raises(ValueError, match="does not support the Mixture-of-Experts load-balancing auxiliary loss"):
            KTOTrainer(
                model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
                args=training_args,
                train_dataset=dataset,
            )

    @pytest.mark.parametrize("iterable_as", ["train", "eval", "eval_dict", "eval_iterable_dataset_dict"])
    def test_precompute_ref_log_probs_raises_for_iterable_dataset(self, iterable_as):
        # `precompute_ref_log_probs=True` caches reference log-probs by index, which requires random access and is
        # therefore incompatible with a streaming `IterableDataset` — whether passed directly or nested in a dict or
        # `IterableDatasetDict` as the eval dataset.
        map_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")
        iterable_dataset = load_dataset(
            "trl-internal-testing/zen", "standard_unpaired_preference", split="train", streaming=True
        )

        train_dataset, eval_dataset = map_dataset, None
        if iterable_as == "train":
            train_dataset = iterable_dataset
        elif iterable_as == "eval":
            eval_dataset = iterable_dataset
        elif iterable_as == "eval_dict":
            eval_dataset = {"eval": iterable_dataset}
        elif iterable_as == "eval_iterable_dataset_dict":
            eval_dataset = IterableDatasetDict({"eval": iterable_dataset})

        # `apo_zero_unpaired` keeps the trainer setup minimal (no KL-term constraints), so the precompute error is what
        # surfaces. `max_steps` lets the iterable-train case reach the precompute check instead of failing earlier on
        # the missing dataset length required by the learning-rate scheduler.
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            loss_type="apo_zero_unpaired",
            max_steps=1,
            report_to="none",
            precompute_ref_log_probs=True,
        )
        with pytest.raises(ValueError, match="precompute_ref_log_probs.*not supported with IterableDataset"):
            KTOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )

    @pytest.mark.parametrize("loss_type", ["kto", "apo_zero_unpaired"])
    def test_train_with_iterable_dataset(self, loss_type):
        # Streaming `IterableDataset` is supported for all loss types. Loss types that estimate the KL term (all except
        # `apo_zero_unpaired`) build the mismatched KL pairs from neighboring examples within each fixed-order batch,
        # which streaming preserves because it is consumed sequentially.
        dataset = load_dataset(
            "trl-internal-testing/zen", "standard_unpaired_preference", split="train", streaming=True
        )

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            loss_type=loss_type,
            per_device_train_batch_size=2,  # KL-term loss types require a per-device train batch size greater than 1
            max_steps=3,
            report_to="none",
        )
        trainer = KTOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_chat_template_kwargs(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")

        training_args = KTOConfig(
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

        trainer = KTOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        assert trainer.processing_class.chat_template == tokenizer.chat_template

        # `chat_template_kwargs` is applied per example: the prepared prompts must contain both the lowercase and the
        # uppercase system header, proving the argument was threaded through tokenization (the dataset alternates
        # `role_capital`). The exact row order is not asserted because preprocessing unpairs the dataset.
        decoded_prompts = [
            trainer.processing_class.decode(trainer.train_dataset[i]["prompt_ids"])
            for i in range(len(trainer.train_dataset))
        ]
        assert any("<|im_start|>SYSTEM" in prompt for prompt in decoded_prompts)
        assert any("<|im_start|>system" in prompt for prompt in decoded_prompts)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_toolcall_data(self):
        dataset = load_dataset("trl-internal-testing/toolcall", "preference", split="train")

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # toolcall sequences are longer than standard data, reduce batch size to avoid OOM
            max_length=512,  # toolcall sequences are longer than standard data, limit length to avoid OOM
            report_to="none",
        )
        trainer = KTOTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

        training_args = KTOConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = KTOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()

        assert trainer.state.log_history[0]["eval_loss"] is not None

    def test_train_with_multiple_eval_dataset(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

        training_args = KTOConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = KTOTrainer(
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
                "trl-internal-testing/zen", "standard_unpaired_preference", split="train", streaming=streaming
            )
            if train_dataset_type == "unsupported_dataset_dict":
                # `DatasetDict` is representative of any unsupported type here; not exhaustive
                train_dataset = DatasetDict({"train": train_dataset})

        # Iterable (streaming) datasets have no length, so `max_steps` is required.
        training_args = KTOConfig(output_dir=self.tmp_dir, max_steps=3 if streaming else -1, report_to="none")

        if train_dataset_type == "none":
            with pytest.raises(ValueError, match="`train_dataset` is required"):
                KTOTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    args=training_args,
                    train_dataset=train_dataset,
                )
        elif train_dataset_type == "unsupported_dataset_dict":
            with pytest.raises(TypeError, match="`train_dataset` must be a `Dataset` or `IterableDataset`"):
                KTOTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    args=training_args,
                    train_dataset=train_dataset,
                )
        else:
            trainer = KTOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=train_dataset
            )
            assert "prompt_ids" in next(iter(trainer.train_dataset))

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
        # `apo_zero_unpaired` avoids KTO's KL term, which is incompatible with streaming datasets.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            streaming = "iterable" in eval_dataset_type
            eval_split = load_dataset(
                "trl-internal-testing/zen", "standard_unpaired_preference", split="test", streaming=streaming
            )
            if eval_dataset_type in ("dataset", "iterable_dataset"):
                eval_dataset = eval_split
            elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
                dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
                eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
            else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
                eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = KTOConfig(output_dir=self.tmp_dir, loss_type="apo_zero_unpaired", report_to="none")
        trainer = KTOTrainer(
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
            assert "prompt_ids" in next(iter(trainer.eval_dataset["data1"]))
            assert "prompt_ids" in next(iter(trainer.eval_dataset["data2"]))
        else:
            assert "prompt_ids" in next(iter(trainer.eval_dataset))

    # In practice, this test is the same as `test_kto_trainer`, since gradient checkpointing is enabled by default in
    # `KTOTrainer`. We keep it as a regression guard: if the default ever changes, we still explicitly test gradient
    # checkpointing, which has caused issues in the past.
    def test_train_with_gradient_checkpointing(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            gradient_checkpointing=True,
            report_to="none",
        )
        trainer = KTOTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        trainer = KTOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
        )

        for tag in ["kto", "trl"]:
            assert tag in trainer.model.model_tags

    @require_peft
    def test_tag_added_peft(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        trainer = KTOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        for tag in ["kto", "trl"]:
            assert tag in trainer.model.model_tags

    @require_peft
    @require_bitsandbytes
    def test_peft_with_quantization(self):
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

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        # Initialize the trainer with the already configured PeftModel
        training_args = KTOConfig(output_dir=self.tmp_dir, learning_rate=0.1, report_to="none")
        trainer = KTOTrainer(model=model, args=training_args, train_dataset=dataset, peft_config=LoraConfig())

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
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

            if "lora" not in n:  # We expect the base model params to be the same
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "lora" in n:  # We expect the peft params to be different
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
            else:
                raise ValueError(f"Unexpected parameter {n} in model: {trainer.model}")


@require_vision
class TestKTOTrainerVLM(TrlTestCase):
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
    def test_train_vlm(self, model_id):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_unpaired_preference", split="train")
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            report_to="none",
        )
        trainer = KTOTrainer(model=model_id, args=training_args, train_dataset=dataset)

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
    def test_train_vlm_multi_image(self, model_id):
        dataset = load_dataset(
            "trl-internal-testing/zen-multi-image", "conversational_unpaired_preference", split="train"
        )

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            report_to="none",
        )
        trainer = KTOTrainer(
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

    def test_train_vlm_apo_zero_unpaired(self):
        # apo_zero_unpaired does not need the KL term: verify that calculate_kl=False path works end-to-end.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_unpaired_preference", split="train")
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            loss_type="apo_zero_unpaired",
            report_to="none",
        )
        trainer = KTOTrainer(
            model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
        ],
    )
    @pytest.mark.parametrize(
        "dataset_config",
        ["conversational_unpaired_preference", "standard_unpaired_preference"],
    )
    def test_train_vlm_text_only_data(self, model_id, dataset_config):
        dataset = load_dataset("trl-internal-testing/zen", dataset_config, split="train")
        training_args = KTOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = KTOTrainer(
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

    def test_train_vlm_with_max_length(self):
        # Regression test: mm_token_type_ids (and KL_mm_token_type_ids) must be truncated alongside
        # input_ids when max_length is set, otherwise a shape mismatch crashes the model forward pass.
        # max_length=37 truncates 1 completion token (total_len=38) while keeping all image tokens (prompt_len=34) safe.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_unpaired_preference", split="train")
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            max_length=37,  # total_len=38, prompt_len=34 — truncates completion, not image tokens
            per_device_train_batch_size=2,
            report_to="none",
        )
        trainer = KTOTrainer(
            model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_vision_dataset_with_text_model_raises(self):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_unpaired_preference", split="train")
        training_args = KTOConfig(output_dir=self.tmp_dir, report_to="none")
        with pytest.raises(ValueError, match="vision-related.*vision-language model"):
            KTOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
            )

    def test_precompute_ref_log_probs_raises_for_vision(self):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_unpaired_preference", split="train")
        training_args = KTOConfig(output_dir=self.tmp_dir, report_to="none", precompute_ref_log_probs=True)
        with pytest.raises(ValueError, match="precompute_ref_log_probs.*not supported for vision datasets"):
            KTOTrainer(
                model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
                args=training_args,
                train_dataset=dataset,
            )

    @require_liger_kernel
    def test_train_vlm_liger(self):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_unpaired_preference", split="train")
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            use_liger_kernel=True,
            report_to="none",
        )
        trainer = KTOTrainer(
            model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Param {n} is not updated"


@pytest.mark.slow
class TestKTOTrainerSlow(TrlTestCase):
    # Gemma 3n uses a timm encoder, making it difficult to create a smaller variant for testing.
    # To ensure coverage, we run tests on the full model but mark them as slow to exclude from default runs.
    @pytest.mark.skip(reason="Model google/gemma-3n-E2B-it is gated and requires HF token")
    @require_vision
    def test_train_vlm_gemma_3n(self):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_unpaired_preference", split="train")

        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            model_init_kwargs={"dtype": "bfloat16"},
            report_to="none",
        )
        trainer = KTOTrainer(model="google/gemma-3n-E2B-it", args=training_args, train_dataset=dataset)

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
