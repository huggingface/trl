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


def test():
    command = """\
python examples/scripts/ppo/ppo.py \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
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


def test_num_train_epochs():
    command = """\
python examples/scripts/ppo/ppo.py \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 0.003 \
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
