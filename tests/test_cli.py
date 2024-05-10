# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import subprocess
import sys
import unittest


@unittest.skipIf(sys.platform.startswith("win"), "Skipping on Windows")
def test_sft_cli():
    try:
        subprocess.run(
            "trl sft --max_steps 1 --output_dir tmp-sft --model_name_or_path trl-internal-testing/tiny-random-LlamaForCausalLM --dataset_name imdb --learning_rate 1e-4 --lr_scheduler_type cosine --dataset_text_field text",
            shell=True,
            check=True,
        )
    except BaseException as exc:
        raise AssertionError("An error occured while running the CLI, please double check") from exc


@unittest.skipIf(sys.platform.startswith("win"), "Skipping on Windows")
def test_dpo_cli():
    try:
        subprocess.run(
            "trl dpo --max_steps 1 --output_dir tmp-dpo --model_name_or_path trl-internal-testing/tiny-random-LlamaForCausalLM --dataset_name trl-internal-testing/hh-rlhf-helpful-base-trl-style --learning_rate 1e-4 --lr_scheduler_type cosine --sanity_check",
            shell=True,
            check=True,
        )
    except BaseException as exc:
        raise AssertionError("An error occured while running the CLI, please double check") from exc
