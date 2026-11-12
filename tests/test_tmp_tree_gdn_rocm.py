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

# TEMPORARY, to remove before merging: the AMD runners only run `tests/`, so this reruns the Gated DeltaNet cases of
# the tree packing parity test from `tests/experimental/` there.
import pytest

from .experimental import test_async_grpo_trainer as async_grpo_tests
from .testing_utils import TrlTestCase


stub_weight_transfer = async_grpo_tests.stub_weight_transfer


@pytest.mark.usefixtures("stub_weight_transfer")
class TestTreeGatedDeltaNetParity(TrlTestCase):
    _samples = async_grpo_tests.TestTreePackingParity._samples

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-Think",
            "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-3.8",
            "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6",
        ],
    )
    @pytest.mark.parametrize("truncated", [False, True], ids=["full-vocab", "truncated-sampling"])
    def test_loss_and_grads_match_each_sample_alone(self, model_id, truncated):
        async_grpo_tests.TestTreePackingParity.test_loss_and_grads_match_each_sample_alone(
            self, model_id, {}, truncated
        )
