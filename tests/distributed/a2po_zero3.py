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

import math

import torch
from datasets import Dataset
from deepspeed import zero
from transformers import HfArgumentParser

from trl import ModelConfig
from trl.experimental.a2po import A2POConfig, A2POTrainer


def binary_reward(completions, **kwargs):
    """Give Stage 1 both reward values so Stage 2 has a nonzero learning signal."""
    return [float(index % 2) for index in range(len(completions))]


def snapshot_parameters(model):
    with zero.GatheredParameters(list(model.parameters())):
        return {name: param.detach().cpu().clone() for name, param in model.named_parameters()}


def main():
    training_args, model_args = HfArgumentParser((A2POConfig, ModelConfig)).parse_args_into_dataclasses()
    dataset = Dataset.from_dict(
        {"prompt": ["The capital of France is", "Two plus two equals", "Water is made of", "The sky is"]}
    )
    trainer = A2POTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=binary_reward,
        args=training_args,
        train_dataset=dataset,
    )
    stage = trainer.accelerator.state.deepspeed_plugin.zero_stage
    assert stage in (2, 3)
    assert trainer.ref_model.zero_optimization_stage() == (3 if stage == 3 else 0)
    assert all(not module.training for module in trainer.ref_model.module.modules())
    assert all(not param.requires_grad for param in trainer.ref_model.parameters())
    policy_before = snapshot_parameters(trainer.model)
    reference_before = snapshot_parameters(trainer.ref_model.module)
    assert policy_before.keys() == reference_before.keys()
    for name, param in reference_before.items():
        torch.testing.assert_close(param, policy_before[name], rtol=0, atol=0)

    result = trainer.train()

    assert trainer.state.global_step == 1
    assert math.isfinite(result.training_loss)
    assert trainer._optimal_values is not None and len(trainer._optimal_values) == len(dataset)
    assert trainer.model_wrapped.zero_optimization_stage() == stage
    policy_after = snapshot_parameters(trainer.model)
    reference_after = snapshot_parameters(trainer.ref_model.module)
    assert any(not torch.equal(param, policy_after[name]) for name, param in policy_before.items())
    for name, param in reference_before.items():
        torch.testing.assert_close(reference_after[name], param, rtol=0, atol=0)
    assert all(not module.training for module in trainer.ref_model.module.modules())
    assert all(not param.requires_grad for param in trainer.ref_model.parameters())


if __name__ == "__main__":
    main()
