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

from datasets import Dataset
from torch.distributed.fsdp import FSDPModule

from trl import TrlParser
from trl.experimental.a2po import A2POConfig, A2POTrainer


def reward(completions, **kwargs):
    return [float(len(completion) % 2 == 0) for completion in completions]


if __name__ == "__main__":
    args = TrlParser(A2POConfig).parse_args_and_config()[0]
    trainer = A2POTrainer(
        model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        reward_funcs=reward,
        args=args,
        train_dataset=Dataset.from_dict({"prompt": ["Hello", "World", "One", "Two"]}),
    )
    assert trainer.is_fsdp_enabled
    assert isinstance(trainer.ref_model, FSDPModule)
    assert not trainer.ref_model.training
    assert all(not parameter.requires_grad for parameter in trainer.ref_model.parameters())
    result = trainer.train()
    assert trainer.state.global_step == 1
    assert math.isfinite(result.training_loss)
    assert trainer._optimal_values is not None
    assert all(not parameter.requires_grad for parameter in trainer.ref_model.parameters())
