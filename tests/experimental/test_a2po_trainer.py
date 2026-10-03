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
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.a2po import A2POConfig, A2POTrainer

from ..testing_utils import TrlTestCase


def completion_parity_reward(completions, **kwargs):
    """Completion-dependent binary reward, so samples within a prompt vary (nonzero advantages)."""
    return [float(len(completion) % 2 == 0) for completion in completions]


class TestA2POTrainer(TrlTestCase):
    def test_trust_remote_code(self):
        dataset = Dataset.from_dict({"prompt": ["The capital of France is", "Two plus two equals"]})
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"

        with pytest.raises(ValueError, match="custom code"):
            A2POTrainer(
                model=model_id,
                reward_funcs=completion_parity_reward,
                args=A2POConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
            )

        trainer = A2POTrainer(
            model=model_id,
            reward_funcs=completion_parity_reward,
            args=A2POConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    def test_train(self):
        # Main two-stage smoke test: Stage 1 estimates V*, Stage 2 regresses on a single on-policy generation.
        dataset = Dataset.from_dict(
            {"prompt": ["The capital of France is", "Two plus two equals", "Water is made of", "The sky is"]}
        )
        training_args = A2POConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_value_samples=2,  # reduce Stage 1 sampling to reduce memory usage
            filter_all_incorrect=False,  # keep all training prompts (the dummy reward may score a prompt all-zero)
            report_to="none",
        )
        trainer = A2POTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=completion_parity_reward,
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
        previous_reference_params = {n: param.clone() for n, param in trainer.ref_model.named_parameters()}
        assert not trainer.ref_model.training
        assert all(not param.requires_grad for param in trainer.ref_model.parameters())

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer._optimal_values is not None and len(trainer._optimal_values) == len(dataset)

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
        for n, param in previous_reference_params.items():
            torch.testing.assert_close(trainer.ref_model.get_parameter(n), param, rtol=0, atol=0)

    @pytest.mark.parametrize("eval_dataset_type", ["dataset", "dataset_dict", "dict_of_dataset", "none"])
    def test_init_with_eval_dataset(self, eval_dataset_type):
        train_dataset = Dataset.from_dict(
            {"prompt": ["The capital of France is", "Two plus two equals", "Water is made of", "The sky is"]}
        )
        eval_split = Dataset.from_dict({"prompt": ["The capital of Italy is", "Three plus three equals"]})

        if eval_dataset_type == "none":
            eval_dataset = None
        elif eval_dataset_type == "dataset":
            eval_dataset = eval_split
        elif eval_dataset_type == "dataset_dict":
            eval_dataset = DatasetDict({"data1": eval_split, "data2": eval_split})
        else:  # "dict_of_dataset"
            eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = A2POConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = A2POTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=completion_parity_reward,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
        else:
            assert trainer.eval_dataset is eval_dataset

    def test_reward_kwargs_are_forwarded(self):
        # Regression test: extra dataset columns must reach the reward function (e.g. a verifier needs `solution`).
        received_subjects = []

        def reward_using_extra_column(prompts, completions, subject, **kwargs):
            received_subjects.extend(subject)
            return [1.0] * len(prompts)

        dataset = Dataset.from_dict({"prompt": ["q1", "q2", "q3", "q4"], "subject": ["math", "geo", "chem", "phys"]})
        training_args = A2POConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_value_samples=2,  # reduce Stage 1 sampling to reduce memory usage
            filter_all_incorrect=False,  # keep all training prompts (the dummy reward may score a prompt all-zero)
            report_to="none",
        )
        trainer = A2POTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_using_extra_column,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()

        # Stage 1 repeats each prompt's column value `num_value_samples` times; the column must have been forwarded.
        assert received_subjects, "reward function never received the `subject` column"
        assert set(received_subjects) <= {"math", "geo", "chem", "phys"}

    def test_filter_all_incorrect_drops_prompts(self):
        # Regression test: prompts whose reference samples all score zero must be dropped, not crash Stage 2.
        def reward_from_target(prompts, completions, target, **kwargs):
            return [float(t) for t in target]

        dataset = Dataset.from_dict({"prompt": ["solvable a", "unsolvable", "solvable b"], "target": [1, 0, 1]})
        training_args = A2POConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_value_samples=2,  # reduce Stage 1 sampling to reduce memory usage
            filter_all_incorrect=True,
            report_to="none",
        )
        trainer = A2POTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_from_target,
            args=training_args,
            train_dataset=dataset,
        )

        # Should run without raising a KeyError for the dropped prompt.
        trainer.train()

        assert len(trainer.train_dataset) == 2
        assert "unsolvable" not in list(trainer.train_dataset["prompt"])

    def test_evaluate(self):
        # Regression test: eval prompts (never seen in training) must also get a cached V*, and evaluating
        # without a preceding train() must not raise.
        train_dataset = Dataset.from_dict({"prompt": ["alpha", "beta"]})
        eval_dataset = Dataset.from_dict({"prompt": ["gamma", "delta"]})
        training_args = A2POConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=2,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_value_samples=2,  # reduce Stage 1 sampling to reduce memory usage
            filter_all_incorrect=False,  # keep all training prompts (the dummy reward may score a prompt all-zero)
            report_to="none",
        )
        trainer = A2POTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=completion_parity_reward,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        metrics = trainer.evaluate()

        assert "eval_loss" in metrics
        assert "gamma" in trainer._optimal_values and "delta" in trainer._optimal_values

    @pytest.mark.parametrize("model_input", ["path", "model"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_reference_model_from_checkpoint(self, model_input, dtype):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model_init_kwargs = {"dtype": dtype, "device_map": None, "trust_remote_code": False}
        expected_model_init_kwargs = dict(model_init_kwargs)
        model = (
            model_id if model_input == "path" else AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        )
        training_args = A2POConfig(
            output_dir=self.tmp_dir,
            model_init_kwargs=model_init_kwargs,
            report_to="none",
        )
        trainer = A2POTrainer(
            model=model,
            reward_funcs=completion_parity_reward,
            args=training_args,
            train_dataset=Dataset.from_dict({"prompt": ["The capital of France is"]}),
        )

        assert trainer.ref_model is not trainer.model
        assert not trainer.ref_model.training
        assert all(not param.requires_grad for param in trainer.ref_model.parameters())
        assert training_args.model_init_kwargs == expected_model_init_kwargs
        for name, reference_param in trainer.ref_model.named_parameters():
            policy_param = trainer.model.get_parameter(name)
            assert reference_param.dtype == policy_param.dtype == dtype
            assert reference_param.data_ptr() != policy_param.data_ptr()
            torch.testing.assert_close(reference_param, policy_param, rtol=0, atol=0)

    def test_reference_model_from_local_checkpoint(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        with torch.no_grad():
            model.get_input_embeddings().weight.add_(0.1)
        model.save_pretrained(self.tmp_dir)
        tokenizer.save_pretrained(self.tmp_dir)
        trainer = A2POTrainer(
            model=self.tmp_dir,
            reward_funcs=completion_parity_reward,
            args=A2POConfig(output_dir=self.tmp_dir, report_to="none"),
            train_dataset=Dataset.from_dict({"prompt": ["The capital of France is"]}),
        )

        assert trainer.ref_model is not trainer.model
        for name, reference_param in trainer.ref_model.named_parameters():
            torch.testing.assert_close(reference_param, model.get_parameter(name), rtol=0, atol=0)
