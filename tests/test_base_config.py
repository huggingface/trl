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

import warnings

import pytest
import transformers
from packaging.version import Version

from trl import SFTConfig


@pytest.mark.parametrize("gradient_checkpointing", [None, False])
def test_gradient_checkpointing_default_does_not_warn(tmp_path, gradient_checkpointing):
    # TRL defaults `gradient_checkpointing` to True. transformers 5.23 deprecated that name in favor of
    # `activation_checkpointing`, and the default must not trip the deprecation warning on every run.
    kwargs = {} if gradient_checkpointing is None else {"gradient_checkpointing": gradient_checkpointing}
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        args = SFTConfig(output_dir=tmp_path, report_to="none", **kwargs)
    expected = gradient_checkpointing is None
    assert args.gradient_checkpointing is expected
    if Version(transformers.__version__) >= Version("5.23.0.dev0"):
        assert args.activation_checkpointing is expected
