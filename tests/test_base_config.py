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
import warnings

import pytest
import transformers
from packaging.version import Version

from trl import DPOConfig, GRPOConfig, SFTConfig


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.23.0"), reason="`gradient_checkpointing` is deprecated from 5.23"
)
def test_checkpointing_default_does_not_warn(tmp_path):
    # TRL enables checkpointing by default. transformers 5.23 deprecated `gradient_checkpointing` in favor of
    # `activation_checkpointing`, and the default must not trip the deprecation warning on every run.
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        args = SFTConfig(output_dir=tmp_path, report_to="none")
    assert args.gradient_checkpointing is True
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


def test_fp8_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("ACCELERATE_MIXED_PRECISION", raising=False)
    monkeypatch.delenv("ACCELERATE_FP8_BACKEND", raising=False)
    args = SFTConfig(output_dir=tmp_path, report_to="none")
    assert args.fp8 is False
    assert args.fp8_backend is None
    assert args.mixed_precision == "bf16"


@pytest.mark.parametrize("config_cls", [SFTConfig, DPOConfig, GRPOConfig])
def test_fp8_sets_mixed_precision(tmp_path, monkeypatch, config_cls):
    # FP8 is wired in the shared base config, so every trainer gets it.
    monkeypatch.delenv("ACCELERATE_MIXED_PRECISION", raising=False)
    monkeypatch.delenv("ACCELERATE_FP8_BACKEND", raising=False)
    args = config_cls(output_dir=tmp_path, report_to="none", fp8=True)
    assert args.mixed_precision == "fp8"
    # transformers 4.56 reads mixed precision from the env var, which `TrainingArguments` sets from bf16/fp16.
    assert os.environ["ACCELERATE_MIXED_PRECISION"] == "fp8"


def test_fp8_backend(tmp_path, monkeypatch):
    monkeypatch.delenv("ACCELERATE_MIXED_PRECISION", raising=False)
    monkeypatch.delenv("ACCELERATE_FP8_BACKEND", raising=False)
    args = SFTConfig(output_dir=tmp_path, report_to="none", fp8=True, fp8_backend="te")
    assert args.mixed_precision == "fp8"
    # accelerate picks the FP8 backend from this env var.
    assert os.environ["ACCELERATE_FP8_BACKEND"] == "TE"


def test_fp8_with_fp16_raises(tmp_path):
    with pytest.raises(ValueError, match="fp8.*fp16"):
        SFTConfig(output_dir=tmp_path, report_to="none", fp8=True, fp16=True)


def test_fp8_backend_without_fp8_raises(tmp_path):
    with pytest.raises(ValueError, match="fp8_backend.*fp8=True"):
        SFTConfig(output_dir=tmp_path, report_to="none", fp8_backend="te")


def test_fp8_backend_invalid_raises(tmp_path):
    with pytest.raises(ValueError, match="fp8_backend"):
        SFTConfig(output_dir=tmp_path, report_to="none", fp8=True, fp8_backend="msamp")
