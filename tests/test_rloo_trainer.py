# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import platform
import subprocess

import torch


def test():
    command = """\
python examples/scripts/rloo/rloo.py \
    --learning_rate 3e-6 \
    --output_dir models/minimal/rloo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10 \
    --model_name_or_path EleutherAI/pythia-14m \
    --non_eos_penalty \
    --stop_token eos \
"""
    if platform.system() == "Windows":
        # windows CI does not work with subprocesses for some reason
        # e.g., https://github.com/huggingface/trl/actions/runs/9600036224/job/26475286210?pr=1743
        return
    subprocess.run(
        command,
        shell=True,
        check=True,
    )


def test_rloo_reward():
    local_batch_size = 3
    rloo_k = 4
    # fmt: off
    rlhf_reward = torch.tensor([
        1, 2, 3, # first rlhf reward for three prompts
        2, 3, 4, # second rlhf reward for three prompts
        5, 6, 7, # third rlhf reward for three prompts
        8, 9, 10, # fourth rlhf reward for three prompts
    ]).float()
    # fmt: on

    baseline = (rlhf_reward.sum(0) - rlhf_reward) / (rloo_k - 1)
    advantages = torch.zeros_like(rlhf_reward)
    for i in range(0, len(advantages), local_batch_size):
        other_response_rlhf_rewards = []
        for j in range(0, len(advantages), local_batch_size):
            if i != j:
                other_response_rlhf_rewards.append(rlhf_reward[j : j + local_batch_size])
        advantages[i : i + local_batch_size] = rlhf_reward[i : i + local_batch_size] - torch.stack(
            other_response_rlhf_rewards
        ).mean(0)
    assert (1 - (2 + 5 + 8) / 3 - advantages[0].item()) < 1e-6
    assert (6 - (3 + 2 + 9) / 3 - advantages[7].item()) < 1e-6

    # vectorized impl
    rlhf_reward = rlhf_reward.reshape(rloo_k, local_batch_size)
    baseline = (rlhf_reward.sum(0) - rlhf_reward) / (rloo_k - 1)
    vec_advantages = rlhf_reward - baseline
    torch.testing.assert_close(vec_advantages.flatten(), advantages)
