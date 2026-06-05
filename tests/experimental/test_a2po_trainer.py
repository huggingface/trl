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

import torch
from datasets import Dataset

from trl.experimental.a2po import A2POConfig, A2POTrainer

from ..testing_utils import TrlTestCase


def completion_parity_reward(completions, **kwargs):
    """Completion-dependent binary reward, so samples within a prompt vary (nonzero advantages)."""
    return [float(len(completion) % 2 == 0) for completion in completions]


class TestA2POTrainer(TrlTestCase):
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

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer._optimal_values is not None and len(trainer._optimal_values) == len(dataset)

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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
