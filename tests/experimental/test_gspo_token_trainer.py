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
from datasets import DatasetDict, load_dataset

from trl import GRPOConfig
from trl.experimental.gspo_token import GRPOTrainer as GSPOTokenTrainer

from ..testing_utils import TrlTestCase


class TestGSPOTokenTrainer(TrlTestCase):
    def test_train(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_iterations=2,  # the importance sampling weights won't be 0 in this case
            importance_sampling_level="sequence_token",
            report_to="none",
        )
        trainer = GSPOTokenTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
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

    def test_train_with_vllm_importance_sampling_correction(self):
        # GSPO-token overrides `_compute_loss`, which is where `GRPOTrainer` computes the vLLM importance sampling
        # ratio on the on-policy non-liger path (it is no longer pre-computed in `_generate_and_score_completions`).
        # This checks that the override carries that computation as well, so that `inputs["importance_sampling_ratio"]`
        # exists when the loss is scaled, and that the sampling diagnostics are logged once per generation batch.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        steps_per_generation = 2
        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            num_generations=3,
            gradient_accumulation_steps=steps_per_generation,  # on-policy
            steps_per_generation=steps_per_generation,
            max_completion_length=8,
            max_steps=1,
            beta=0.0,
            logging_steps=1000,  # never log during the run, so that `_metrics` is not cleared
            importance_sampling_level="sequence_token",
            report_to="none",
        )
        trainer = GSPOTokenTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        # Enable the correction after construction so that no vLLM server is required, and inject sampling logprobs
        # into the trainer's own generation output.
        trainer.use_vllm = True
        trainer.vllm_importance_sampling_correction = True
        original_generate = trainer._generate

        def generate_with_sampling_logps(prompts):
            trainer.use_vllm = False
            try:
                outputs = list(original_generate(prompts))
            finally:
                trainer.use_vllm = True
            outputs[4] = [[-0.5] * len(ids) for ids in outputs[1]]
            return tuple(outputs)

        trainer._generate = generate_with_sampling_logps

        # Snapshot after every micro-batch, since `log` clears `_metrics` at the end of training.
        keys = [
            "sampling/sampling_logp_difference/mean",
            "sampling/sampling_logp_difference/max",
            "sampling/importance_sampling_ratio/min",
            "sampling/importance_sampling_ratio/mean",
            "sampling/importance_sampling_ratio/max",
        ]
        original_compute_loss = trainer._compute_loss
        num_logged_after_each_micro_batch = []
        logged = {}

        def record_loss(model, inputs):
            loss = original_compute_loss(model, inputs)
            num_logged_after_each_micro_batch.append(len(trainer._metrics["train"][keys[0]]))
            logged.update({key: list(trainer._metrics["train"][key]) for key in keys})
            return loss

        trainer._compute_loss = record_loss

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        # 2 micro-batches, 1 iteration: logged once, at the second micro-batch
        assert num_logged_after_each_micro_batch == [0, 1]
        for key in keys:
            assert len(logged[key]) == 1, f"{key} must be logged exactly once per generation batch"
            assert torch.isfinite(torch.tensor(logged[key][0])), f"{key} must be finite"

    @pytest.mark.parametrize("eval_dataset_type", ["dataset", "dataset_dict", "dict_of_dataset", "none"])
    def test_init_with_eval_dataset(self, eval_dataset_type):
        # Streaming datasets are not yet supported in GSPO-token
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        if eval_dataset_type == "none":
            eval_dataset = None
        elif eval_dataset_type == "dataset":
            eval_dataset = dataset["test"]
        elif eval_dataset_type == "dataset_dict":
            eval_dataset = DatasetDict({"data1": dataset["test"], "data2": dataset["test"]})
        else:  # "dict_of_dataset"
            eval_dataset = {"data1": dataset["test"], "data2": dataset["test"]}

        training_args = GRPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = GSPOTokenTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=eval_dataset,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
        else:
            assert trainer.eval_dataset is eval_dataset
