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

import os

import pytest
import transformers
from packaging.version import Version

from trl import SFTConfig


class TestBaseConfig:
    @pytest.mark.skipif(
        Version(transformers.__version__) >= Version("5.0.0"),
        reason="the mixed precision is no longer published process-wide since transformers-5.0.0",
    )
    def test_explicit_fp32_clears_the_published_mixed_precision(self, tmp_path):
        # The first config publishes `bf16` process-wide and the second one reads it back as its own default, so
        # without clearing it the second config would train in bf16.
        SFTConfig(output_dir=tmp_path, bf16=True)
        SFTConfig(output_dir=tmp_path, bf16=False)

        assert os.environ["ACCELERATE_MIXED_PRECISION"] == "no"
