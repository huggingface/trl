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


from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from datasets import DatasetDict, load_dataset

from trl import GRPOConfig
from trl.experimental.gspo_token import GRPOTrainer as GSPOTokenTrainer

from ..testing_utils import TrlTestCase


class TestGSPOTokenTrainer(TrlTestCase):
    @pytest.mark.parametrize(
        ("tool_mask", "num_items_in_batch", "expected_loss", "expected_entropy"),
        [
            ([1, 0, 1], 2, 1.0, 2.0),
            ([0, 0, 0], 0, 0.0, 0.0),
        ],
    )
    def test_dapo_uses_tool_mask_for_loss_and_metrics(
        self, tool_mask, num_items_in_batch, expected_loss, expected_entropy
    ):
        trainer = object.__new__(GSPOTokenTrainer)
        trainer.accelerator = SimpleNamespace(num_processes=1, gather=lambda tensor: tensor)
        trainer.top_entropy_quantile = 1.0
        trainer.beta = 0.0
        trainer.importance_sampling_level = "sequence_token"
        trainer.epsilon_low = 0.2
        trainer.epsilon_high = 0.2
        trainer.args = SimpleNamespace(delta=None)
        trainer.use_vllm = False
        trainer.loss_type = "dapo"
        trainer.model = SimpleNamespace(training=True)
        trainer._metrics = {"train": defaultdict(list)}

        per_token_logps = torch.zeros((1, 3))
        entropies = torch.tensor([[1.0, 100.0, 3.0]])
        trainer._get_per_token_logps_and_entropies = lambda *args, **kwargs: (
            per_token_logps,
            entropies,
            None,
        )
        inputs = {
            "prompt_ids": torch.tensor([[1]]),
            "prompt_mask": torch.tensor([[1]]),
            "completion_ids": torch.tensor([[2, 3, 4]]),
            "completion_mask": torch.tensor([[1, 1, 1]]),
            "tool_mask": torch.tensor([tool_mask]),
            "advantages": torch.tensor([-1.0]),
            # A large policy ratio on the masked tool-result token must not affect GSPO's sequence weight.
            "old_per_token_logps": torch.tensor([[0.0, -6.0, 0.0]]),
            "num_items_in_batch": torch.tensor(num_items_in_batch),
        }

        loss = trainer._compute_loss(trainer.model, inputs)

        assert loss.item() == pytest.approx(expected_loss)
        assert torch.isfinite(loss)
        assert trainer._metrics["train"]["entropy"][-1] == pytest.approx(expected_entropy)

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
