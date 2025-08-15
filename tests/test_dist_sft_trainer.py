# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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


from unittest import TestCase

from accelerate.test_utils import require_multi_gpu
from datasets import load_dataset

from trl import SFTTrainer

from .testing_utils import distributed


class DistributedSFTTester(TestCase):
    @require_multi_gpu
    @distributed
    def test_sft_basic_distributed(self):
        trainer = SFTTrainer(
            model="Qwen/Qwen2.5-0.5B",
            train_dataset=load_dataset("trl-lib/Capybara", split="train").select(range(10)),
        )
        trainer.train()
