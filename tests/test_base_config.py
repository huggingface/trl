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


def test_checkpointing_default_does_not_warn(tmp_path):
    # TRL enables checkpointing by default. transformers 5.23 deprecated `gradient_checkpointing` in favor of
    # `activation_checkpointing`, and the default must not trip the deprecation warning on every run.
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        args = SFTConfig(output_dir=tmp_path, report_to="none")
    assert args.gradient_checkpointing is True
    if Version(transformers.__version__) >= Version("5.23.0"):
        assert args.activation_checkpointing is True


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.23.0"), reason="`gradient_checkpointing` is deprecated from 5.23"
)
def test_gradient_checkpointing_warns(tmp_path):
    with pytest.warns(FutureWarning, match="`gradient_checkpointing` is deprecated"):
        args = SFTConfig(output_dir=tmp_path, report_to="none", gradient_checkpointing=False)
    assert args.gradient_checkpointing is False
    assert args.activation_checkpointing is False


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.23.0"), reason="`activation_checkpointing` is added in 5.23"
)
def test_activation_checkpointing_overrides_the_default(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        args = SFTConfig(output_dir=tmp_path, report_to="none", activation_checkpointing=False)
    assert args.activation_checkpointing is False
    assert args.gradient_checkpointing is False
