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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from trl.generation.vllm_generation import VLLMGeneration


class FakeLinear4bit:
    pass


def make_server_generation(model):
    generation = VLLMGeneration.__new__(VLLMGeneration)
    generation.model = model
    generation.accelerator = MagicMock(is_main_process=True, device="cuda")
    generation.mode = "server"
    generation.server_base_url = "http://localhost:1"
    generation.group_port = 51216
    generation.server_timeout = 1
    return generation


def test_server_mode_rejects_bnb_4bit_peft_model():
    model = MagicMock()
    model.named_modules.return_value = [("", FakeLinear4bit())]
    generation = make_server_generation(model)

    with (
        patch("trl.generation.vllm_generation.is_vllm_available", return_value=True),
        patch("trl.generation.vllm_generation.is_peft_model", return_value=True),
        patch("trl.generation.vllm_generation.is_bitsandbytes_available", return_value=True),
        patch(
            "trl.generation.vllm_generation.bnb",
            SimpleNamespace(nn=SimpleNamespace(Linear4bit=FakeLinear4bit)),
            create=True,
        ),
        pytest.raises(ValueError, match="QLoRA is not supported"),
    ):
        generation._init_vllm()


def test_server_mode_allows_non_quantized_peft_model():
    model = MagicMock()
    model.named_modules.return_value = []
    generation = make_server_generation(model)
    client = MagicMock()

    with (
        patch("trl.generation.vllm_generation.is_vllm_available", return_value=True),
        patch("trl.generation.vllm_generation.is_peft_model", return_value=True),
        patch("trl.generation.vllm_generation.is_bitsandbytes_available", return_value=True),
        patch(
            "trl.generation.vllm_generation.bnb",
            SimpleNamespace(nn=SimpleNamespace(Linear4bit=FakeLinear4bit)),
            create=True,
        ),
        patch("trl.generation.vllm_generation.VLLMClient", return_value=client),
    ):
        generation._init_vllm()

    client.init_communicator.assert_called_once_with(device="cuda")
