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
from datasets import Dataset, load_dataset
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.kto import KTOConfig, KTOTrainer
from trl.experimental.kto.kto_trainer import (
    DataCollatorForUnpairedPreference,
    DataCollatorForVisionUnpairedPreference,
    _get_kl_completion_ids,
)

from ..testing_utils import TrlTestCase, require_liger_kernel, require_peft, require_vision


@require_vision
class TestDataCollatorForVisionUnpairedPreference(TrlTestCase):
    @pytest.mark.skipif(
        Version(transformers.__version__) < Version("5.3.0"),
        reason="mm_token_type_ids are returned by default since transformers-5.3.0 (see transformers#43972)",
    )
    def test_mm_token_type_ids_shape(self):
        # Regression guard: when the processor returns mm_token_type_ids (Qwen2.5-VL after transformers#43972),
        # the collator must produce a KL_completion_token_type_ids whose width matches KL_completion_input_ids,
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
        assert output["mm_token_type_ids"].shape == output["completion_input_ids"].shape, (
            f"mm_token_type_ids shape {output['mm_token_type_ids'].shape} != "
            f"completion_input_ids shape {output['completion_input_ids'].shape}"
        )
        assert "KL_completion_mm_token_type_ids" in output
        assert output["KL_completion_mm_token_type_ids"].shape == output["KL_completion_input_ids"].shape, (
            f"KL_completion_mm_token_type_ids shape {output['KL_completion_mm_token_type_ids'].shape} != "
            f"KL_completion_input_ids shape {output['KL_completion_input_ids'].shape}"
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
        for key in ["completion_input_ids", "completion_attention_mask", "completion_mask", "pixel_values", "label"]:
            assert key in output, f"Missing key: {key}"
        for key in ["KL_completion_input_ids", "KL_completion_attention_mask", "KL_completion_mask"]:
            assert key in output, f"Missing KL key: {key}"

        # Without KL
        collator_no_kl = DataCollatorForVisionUnpairedPreference(processor, calculate_kl=False)
        output_no_kl = collator_no_kl(make_examples())
        assert "completion_input_ids" in output_no_kl
        assert "KL_completion_input_ids" not in output_no_kl

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
        assert not torch.equal(output["completion_input_ids"][0], output["KL_completion_input_ids"][0])
        assert not torch.equal(output["completion_input_ids"][1], output["KL_completion_input_ids"][1])


class TestDataCollatorForUnpairedPreference(TrlTestCase):
    def test_padding_and_masks(self):
        collator = DataCollatorForUnpairedPreference(pad_token_id=0)
        examples = [
            {"prompt_ids": [1, 2, 3], "completion_ids": [4, 5], "KL_completion_ids": [6], "label": True},
            {"prompt_ids": [7, 8], "completion_ids": [9, 10], "KL_completion_ids": [11, 12, 13], "label": False},
        ]
        result = collator(examples)

        expected_completion_input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],  # prompt + completion (example 1)
                [7, 8, 9, 10, 0],  # prompt + completion (example 2, padded)
            ]
        )
        expected_completion_attention_mask = torch.tensor(
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
        expected_kl_completion_input_ids = torch.tensor(
            [
                [1, 2, 3, 6, 0],  # prompt + KL completion (example 1, padded)
                [7, 8, 11, 12, 13],  # prompt + KL completion (example 2)
            ]
        )
        expected_kl_completion_attention_mask = torch.tensor(
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
            "completion_input_ids",
            "completion_attention_mask",
            "completion_mask",
            "KL_completion_input_ids",
            "KL_completion_attention_mask",
            "KL_completion_mask",
            "label",
        }
        torch.testing.assert_close(result["completion_input_ids"], expected_completion_input_ids)
        torch.testing.assert_close(result["completion_attention_mask"], expected_completion_attention_mask)
        torch.testing.assert_close(result["completion_mask"], expected_completion_mask)
        torch.testing.assert_close(result["KL_completion_input_ids"], expected_kl_completion_input_ids)
        torch.testing.assert_close(result["KL_completion_attention_mask"], expected_kl_completion_attention_mask)
        torch.testing.assert_close(result["KL_completion_mask"], expected_kl_completion_mask)
        assert result["label"] == [True, False]

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
            "completion_input_ids",
            "completion_attention_mask",
            "completion_mask",
            "KL_completion_input_ids",
            "KL_completion_attention_mask",
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

        expected_completion_input_ids = torch.tensor(
            [
                [1, 2, 0, 0, 0],  # prompt + completion (example 1, padded to multiple of 5)
                [4, 5, 6, 7, 0],  # prompt + completion (example 2)
            ]
        )
        expected_kl_completion_input_ids = torch.tensor(
            [
                [1, 3, 0, 0, 0],  # prompt + KL completion (example 1, padded to multiple of 5)
                [4, 5, 8, 9, 0],  # prompt + KL completion (example 2)
            ]
        )

        assert set(result.keys()) == {
            "completion_input_ids",
            "completion_attention_mask",
            "completion_mask",
            "KL_completion_input_ids",
            "KL_completion_attention_mask",
            "KL_completion_mask",
            "label",
        }
        torch.testing.assert_close(result["completion_input_ids"], expected_completion_input_ids)
        torch.testing.assert_close(result["KL_completion_input_ids"], expected_kl_completion_input_ids)


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

    @pytest.mark.parametrize("precompute_ref_log_probs", [False, True])
    def test_evaluate_with_raw_dataset(self, precompute_ref_log_probs):
        # `evaluate` should accept the same (unprocessed) dataset types as the trainer, e.g. a held-out test set
        # passed directly to `evaluate`. With `precompute_ref_log_probs=True`, the reference log-probs must also be
        # precomputed for the freshly-passed dataset. See https://github.com/huggingface/trl/issues/6115.
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = KTOConfig(
            output_dir=self.tmp_dir, precompute_ref_log_probs=precompute_ref_log_probs, report_to="none"
        )
        trainer = KTOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        metrics = trainer.evaluate(eval_dataset=dataset)
        assert metrics["eval_loss"] is not None

    def test_kto_trainer_with_ref_model_is_model(self):
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
        # completion_input_ids ends with EOS
        assert batch["completion_input_ids"][0, -1].item() == self.tokenizer.eos_token_id
        # completion_mask: prompt tokens are 0, completion tokens are 1; at least the prompt is masked
        assert "completion_mask" in batch
        completion_mask = batch["completion_mask"][0].tolist()
        assert 0 in completion_mask and 1 in completion_mask
        first_completion = next(i for i, m in enumerate(completion_mask) if m == 1)
        assert first_completion > 0  # at least the prompt is masked
        assert all(m == 0 for m in completion_mask[:first_completion])

        # Test corruption of (prompt, completion) pairs for KL dataset.
        # _get_kl_completion_ids shifts completion_ids by one within each batch; prompt_ids are unchanged.
        synthetic = Dataset.from_dict(
            {
                "prompt_ids": [[1, 2], [3, 4], [5, 6]],
                "completion_ids": [[10, 11], [20, 21], [30, 31]],
                "label": [True, False, True],
            }
        )
        for batch_size in [2, 3]:
            rotated = synthetic.map(_get_kl_completion_ids, batched=True, batch_size=batch_size)

            # Verify that completion_ids have been rotated (differ from original). When the dataset length
            # modulo batch_size equals 1, the last batch is unaltered: exclude it from the check.
            for i in range(len(rotated) - 1):
                assert synthetic["prompt_ids"][i] == rotated["prompt_ids"][i]
                assert synthetic["completion_ids"][i] != rotated["completion_ids"][i]

    def test_kto_trainer_without_providing_ref_model(self):
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

    @require_liger_kernel
    def test_kto_trainer_with_liger(self):
        """Test KTO trainer with Liger kernel enabled."""
        training_args = KTOConfig(
            output_dir=self.tmp_dir,
            report_to="none",
            use_liger_kernel=True,  # Enable Liger kernel
        )

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        trainer = KTOTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset,
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
        # Regression test: mm_token_type_ids (and KL_completion_mm_token_type_ids) must be truncated alongside
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
            per_device_train_batch_size=1,  # VLM training is memory intensive, reduce batch size to avoid OOM
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
