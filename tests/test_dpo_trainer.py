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
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.testing_utils import torch_device
from transformers.utils import is_peft_available

from trl import DPOConfig, DPOTrainer
from trl.trainer.dpo_trainer import DataCollatorForPreference, DataCollatorForVisionPreference

from .testing_utils import (
    TrlTestCase,
    is_ampere_or_newer,
    require_bitsandbytes,
    require_kernels,
    require_liger_kernel,
    require_peft,
    require_vision,
)


if is_peft_available():
    import peft
    from peft import LoraConfig, PromptTuningConfig, get_peft_model
    from peft.utils import TaskType


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


class TestDataCollatorForVisionPreference(TrlTestCase):
    @pytest.mark.skipif(
        Version(transformers.__version__) < Version("5.3.0"),
        reason="mm_token_type_ids are returned by default since transformers-5.3.0 (see transformers#43972)",
    )
    @require_vision
    def test_mm_token_type_ids_shape(self):
        # Regression test: when the processor returns mm_token_type_ids (e.g. Qwen2.5-VL after
        # transformers#43972), the collator must concatenate it with zeros for the completion part
        # so that its shape matches input_ids. Without the fix this raises an IndexError in the model.
        from PIL import Image
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained("trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration")
        collator = DataCollatorForVisionPreference(processor)
        image = Image.new("RGB", (16, 16))
        examples = [
            {
                "images": [image],
                "prompt": [{"role": "user", "content": "What is this?"}],
                "chosen": [{"role": "assistant", "content": "A red square."}],
                "rejected": [{"role": "assistant", "content": "A blue circle."}],
            }
        ]
        output = collator(examples)
        assert "mm_token_type_ids" in output
        assert output["mm_token_type_ids"].shape == output["input_ids"].shape, (
            f"mm_token_type_ids shape {output['mm_token_type_ids'].shape} != "
            f"input_ids shape {output['input_ids'].shape}"
        )


class TestDPOTrainer(TrlTestCase):
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(model=model_id, args=training_args, train_dataset=dataset)

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
        # at init. See https://github.com/huggingface/trl/issues/6115.
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        streaming = "iterable" in eval_dataset_type
        eval_split = load_dataset("trl-internal-testing/zen", "standard_preference", split="test", streaming=streaming)
        if eval_dataset_type in ("dataset", "iterable_dataset"):
            eval_dataset = eval_split
        elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
            dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
            eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
        else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
            eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = DPOConfig(
            output_dir=self.tmp_dir, precompute_ref_log_probs=precompute_ref_log_probs, report_to="none"
        )
        trainer = DPOTrainer(
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
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        eval_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="test")
        training_args = DPOConfig(
            output_dir=self.tmp_dir, max_steps=1, precompute_ref_log_probs=True, report_to="none"
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=train_dataset
        )

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
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        eval_split = load_dataset("trl-internal-testing/zen", "standard_preference", split="test")
        if eval_dataset_type == "dataset":
            eval_dataset = eval_split
        elif eval_dataset_type == "dataset_dict":
            eval_dataset = DatasetDict({"data1": eval_split, "data2": eval_split})
        else:  # "dict_of_dataset"
            eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = DPOConfig(
            output_dir=self.tmp_dir, max_steps=1, precompute_ref_log_probs=True, report_to="none"
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()

        metrics = trainer.evaluate()
        if eval_dataset_type == "dataset":
            assert metrics["eval_loss"] is not None
        else:
            assert metrics["eval_data1_loss"] is not None
            assert metrics["eval_data2_loss"] is not None

    def test_trust_remote_code(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"

        with pytest.raises(ValueError, match="custom code"):
            DPOTrainer(
                model=model_id,
                args=DPOConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
            )

        trainer = DPOTrainer(
            model=model_id,
            args=DPOConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_preference",
            "conversational_preference",
            "standard_implicit_prompt_preference",
            "conversational_implicit_prompt_preference",
        ],
    )
    def test_train_dataset_format(self, config_name):
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    # Special case for harmony
    def test_train_gpt_oss(self):
        dataset = load_dataset("trl-internal-testing/harmony", "preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(
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

        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(model=model, args=training_args, train_dataset=dataset)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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
            "sigmoid_norm",
        ],
    )
    def test_train_loss_types(self, loss_type):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

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

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_multi_loss_types(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

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

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_wpo(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

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

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_ld(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

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

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize(
        "f_divergence_type",
        ["reverse_kl", "forward_kl", "js_divergence", "alpha_divergence"],
    )
    def test_train_with_f_divergence(self, f_divergence_type):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

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

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_explicit_ref_model(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

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
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            model_init_kwargs={"dtype": torch.float16},
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(
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
    def test_train_dense_with_peft_config_lora(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

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

    @require_peft
    def test_train_moe_with_peft_config(self):
        model_id = "trl-internal-testing/tiny-GptOssForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

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

    @require_peft
    def test_train_peft_model(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")

        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        lora_config = LoraConfig()
        model = get_peft_model(model, lora_config)

        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=1.0,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(model=model, args=training_args, train_dataset=dataset)

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

    @require_peft
    def test_train_moe_peft_model(self):
        # Regression test for https://github.com/huggingface/trl/issues/5222. PEFT only supports one adapter per model
        # when the LoRA config uses `target_parameters` (see peft#3340), so no "ref" adapter can be created and the
        # reference log probs are computed with adapters disabled instead.
        model_id = "trl-internal-testing/tiny-GptOssForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        lora_config = LoraConfig(target_parameters=["mlp.experts.down_proj", "mlp.experts.gate_up_proj"])
        model = get_peft_model(model, lora_config)

        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = DPOTrainer(model=model, args=training_args, train_dataset=dataset)

        assert "ref" not in trainer.model.peft_config

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
    # enabled by default in `DPOTrainer`. We keep it as a regression guard: if the default ever changes, we still
    # explicitly test PEFT + gradient checkpointing, which has caused issues in the past.
    @require_peft
    def test_train_with_peft_config_and_gradient_checkpointing(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]

        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

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
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            use_liger_kernel=True,
            report_to="none",
        )
        trainer = DPOTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            use_liger_kernel=True,
            report_to="none",
        )
        trainer = DPOTrainer(
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
        # The Liger fused DPO loss reads `lm_head.weight` directly, so a LoRA adapter on `lm_head` is silently
        # ignored and never trained. The trainer must fail fast instead of training a silently-frozen head.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        training_args = DPOConfig(output_dir=self.tmp_dir, use_liger_kernel=True, report_to="none")
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
            DPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
                peft_config=lora_config,
            )

    @require_liger_kernel
    @require_peft
    def test_liger_kernel_with_peft_prompt_learning_raises(self):
        # Prompt-learning methods inject virtual tokens via PeftModel.forward(), which the Liger DPO loss bypasses.
        # The trainer must fail fast to avoid computing the loss on the wrong (truncated) sequence.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        training_args = DPOConfig(output_dir=self.tmp_dir, use_liger_kernel=True, report_to="none")
        with pytest.raises(ValueError, match="prompt-learning"):
            DPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
                peft_config=PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=8),
            )

    @require_liger_kernel
    def test_init_fails_with_moe_aux_loss_and_liger(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # The MoE auxiliary loss is on by default; it is incompatible with the Liger fused loss.
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            use_liger_kernel=True,
            report_to="none",
        )

        with pytest.raises(ValueError, match="does not support the Mixture-of-Experts load-balancing auxiliary loss"):
            DPOTrainer(
                model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
                args=training_args,
                train_dataset=dataset,
            )

    @require_liger_kernel
    def test_init_fails_with_compute_metrics_and_liger(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            use_liger_kernel=True,
            report_to="none",
        )

        with pytest.raises(ValueError, match="compute_metrics is not supported with the Liger kernel"):
            DPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
                compute_metrics=lambda _: {},
            )

    @pytest.mark.parametrize("iterable_as", ["train", "eval", "eval_dict", "eval_iterable_dataset_dict"])
    def test_precompute_ref_log_probs_raises_for_iterable_dataset(self, iterable_as):
        # `precompute_ref_log_probs=True` caches reference log-probs by index, which requires random access and is
        # therefore incompatible with a streaming `IterableDataset` — whether passed directly or nested in a dict or
        # `IterableDatasetDict` as the eval dataset.
        map_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        iterable_dataset = load_dataset(
            "trl-internal-testing/zen", "standard_preference", split="train", streaming=True
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

        # `max_steps` lets the iterable-train case reach the precompute check instead of failing earlier on the
        # missing dataset length required by the learning-rate scheduler.
        training_args = DPOConfig(
            output_dir=self.tmp_dir, max_steps=1, report_to="none", precompute_ref_log_probs=True
        )
        with pytest.raises(ValueError, match="precompute_ref_log_probs.*not supported with IterableDataset"):
            DPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )

    def test_train_with_iterable_dataset(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train", streaming=True)

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_steps=3,
            report_to="none",
        )
        trainer = DPOTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

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

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_with_chat_template_kwargs(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")

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

        assert trainer.processing_class.chat_template == tokenizer.chat_template

        for i in range(2):
            role = "SYSTEM" if i else "system"
            system_prompt = (
                f"<|im_start|>{role}\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>"
            )
            system_prompt_ids = trainer.processing_class(system_prompt)["input_ids"]
            assert trainer.train_dataset[i]["prompt_ids"][: len(system_prompt_ids)] == system_prompt_ids

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_fully_truncated_completion_examples_dropped(self):
        """With `keep_start` truncation, an example whose prompt alone fills `max_length` loses every completion token
        in the collator, so it's dropped during dataset preparation."""
        dataset = Dataset.from_dict(
            {
                "prompt": ["Hi", "This is a very long prompt that fills up the whole max_length budget on its own"],
                "chosen": [" there", " yes"],
                "rejected": [" bye", " no"],
            }
        )

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            max_length=6,
            truncation_mode="keep_start",
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        # Only the short-prompt example survives.
        assert len(trainer.train_dataset) == 1
        assert trainer.train_dataset[0]["prompt_ids"] == trainer.processing_class("Hi")["input_ids"]

    def test_train_toolcall_data(self):
        dataset = load_dataset("trl-internal-testing/toolcall", "preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # toolcall sequences are longer than standard data, reduce batch size to avoid OOM
            max_length=512,  # toolcall sequences are longer than standard data, limit length to avoid OOM
            report_to="none",
        )
        trainer = DPOTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        training_args = DPOConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()

        assert trainer.state.log_history[0]["eval_loss"] is not None

    def test_train_with_multiple_eval_dataset(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        training_args = DPOConfig(output_dir=self.tmp_dir, eval_strategy="steps", eval_steps=3, report_to="none")
        trainer = DPOTrainer(
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
                "trl-internal-testing/zen", "standard_preference", split="train", streaming=streaming
            )
            if train_dataset_type == "unsupported_dataset_dict":
                # `DatasetDict` is representative of any unsupported type here; not exhaustive
                train_dataset = DatasetDict({"train": train_dataset})

        # Iterable (streaming) datasets have no length, so `max_steps` is required.
        training_args = DPOConfig(output_dir=self.tmp_dir, max_steps=3 if streaming else -1, report_to="none")

        if train_dataset_type == "none":
            with pytest.raises(ValueError, match="`train_dataset` is required"):
                DPOTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    args=training_args,
                    train_dataset=train_dataset,
                )
        elif train_dataset_type == "unsupported_dataset_dict":
            with pytest.raises(TypeError, match="`train_dataset` must be a `Dataset` or `IterableDataset`"):
                DPOTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    args=training_args,
                    train_dataset=train_dataset,
                )
        else:
            trainer = DPOTrainer(
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
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            streaming = "iterable" in eval_dataset_type
            eval_split = load_dataset(
                "trl-internal-testing/zen", "standard_preference", split="test", streaming=streaming
            )
            if eval_dataset_type in ("dataset", "iterable_dataset"):
                eval_dataset = eval_split
            elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
                dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
                eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
            else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
                eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = DPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = DPOTrainer(
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

    def test_train_with_compute_metrics(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        def dummy_compute_metrics(eval_pred):
            return {"my_metric": 0.123}

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

        trainer.train()

        assert trainer.state.log_history[-2]["eval_my_metric"] == 0.123

    # In practice, this test is the same as `test_train`, since gradient checkpointing is enabled by default in
    # `DPOTrainer`. We keep it as a regression guard: if the default ever changes, we still explicitly test gradient
    # checkpointing, which has caused issues in the past.
    def test_train_with_gradient_checkpointing(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            gradient_checkpointing=True,
            report_to="none",
        )
        trainer = DPOTrainer(
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
        )

        for tag in ["dpo", "trl"]:
            assert tag in trainer.model.model_tags

    @require_peft
    def test_tag_added_peft(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        for tag in ["dpo", "trl"]:
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

        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        # Initialize the trainer with the already configured PeftModel
        training_args = DPOConfig(output_dir=self.tmp_dir, learning_rate=0.1, report_to="none")
        trainer = DPOTrainer(model=model, args=training_args, train_dataset=dataset, peft_config=LoraConfig())

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-1]["mean_token_accuracy"] is not None

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
class TestDPOTrainerVLM(TrlTestCase):
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
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            report_to="none",
        )
        trainer = DPOTrainer(model=model_id, args=training_args, train_dataset=dataset)

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
        dataset = load_dataset("trl-internal-testing/zen-multi-image", "conversational_preference", split="train")

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
        ],
    )
    @pytest.mark.parametrize(
        "dataset_config",
        ["conversational_preference", "standard_preference"],
    )
    def test_train_vlm_text_only_data(self, model_id, dataset_config):
        dataset = load_dataset("trl-internal-testing/zen", dataset_config, split="train")

        training_args = DPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = DPOTrainer(
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
        # Regression test for #5283: mm_token_type_ids must be truncated alongside input_ids when max_length is set,
        # otherwise a shape mismatch crashes the model forward pass.
        # max_length=37 truncates 1 completion token (total_len=38) while keeping all image tokens (prompt_len=34) safe.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_preference", split="train")
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            max_length=37,  # total_len=38, prompt_len=34 — truncates completion, not image tokens
            per_device_train_batch_size=2,
            report_to="none",
        )
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_train_vlm_keep_end_raises(self):
        # Regression test for #5285: keep_end with a VLM must raise at init time, not silently corrupt training.
        # Image tokens live at the start of the sequence (in the prompt); keep_end would drop them.
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_preference", split="train")
        with pytest.warns(FutureWarning, match="keep_end.*deprecated"):
            training_args = DPOConfig(
                output_dir=self.tmp_dir,
                max_length=32,
                truncation_mode="keep_end",
                report_to="none",
            )
        with pytest.raises(ValueError, match="truncation_mode='keep_end' is not supported for vision-language models"):
            DPOTrainer(
                model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
                args=training_args,
                train_dataset=dataset,
            )

    def test_vision_dataset_with_text_model_raises(self):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_preference", split="train")
        training_args = DPOConfig(output_dir=self.tmp_dir, report_to="none")
        with pytest.raises(ValueError, match="vision-related.*vision-language model"):
            DPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
            )

    def test_precompute_ref_log_probs_raises_for_vision(self):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_preference", split="train")
        training_args = DPOConfig(output_dir=self.tmp_dir, report_to="none", precompute_ref_log_probs=True)
        with pytest.raises(ValueError, match="precompute_ref_log_probs.*not supported for vision datasets"):
            DPOTrainer(
                model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
                args=training_args,
                train_dataset=dataset,
            )

    @require_liger_kernel
    def test_train_vlm_liger(self):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_preference", split="train")
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            per_device_train_batch_size=2,  # VLM training is memory intensive, reduce batch size to avoid OOM
            use_liger_kernel=True,
            report_to="none",
        )
        trainer = DPOTrainer(
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
class TestDPOTrainerSlow(TrlTestCase):
    # Gemma 3n uses a timm encoder, making it difficult to create a smaller variant for testing.
    # To ensure coverage, we run tests on the full model but mark them as slow to exclude from default runs.
    @pytest.mark.skip(reason="Model google/gemma-3n-E2B-it is gated and requires HF token")
    @require_vision
    def test_train_vlm_gemma_3n(self):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_preference", split="train")

        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            max_length=None,  # for VLMs, truncating can remove image tokens, leading to errors
            per_device_train_batch_size=1,  # VLM training is memory intensive, reduce batch size to avoid OOM
            model_init_kwargs={"dtype": "bfloat16"},
            report_to="none",
        )
        trainer = DPOTrainer(model="google/gemma-3n-E2B-it", args=training_args, train_dataset=dataset)

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
