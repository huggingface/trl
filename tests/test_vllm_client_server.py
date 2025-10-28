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

import os
import subprocess

import pytest
from transformers import AutoModelForCausalLM
from transformers.testing_utils import torch_device

from trl.extras.vllm_client import VLLMClient
from trl.scripts.vllm_serve import chunk_list

from .testing_utils import (
    TrlTestCase,
    kill_process,
    require_3_accelerators,
    require_torch_multi_accelerator,
    require_vllm,
)


class TestChunkList(TrlTestCase):
    def test_even_split(self):
        assert chunk_list([1, 2, 3, 4, 5, 6], 2) == [[1, 2, 3], [4, 5, 6]]

    def test_uneven_split(self):
        assert chunk_list([1, 2, 3, 4, 5, 6], 4) == [[1, 2], [3, 4], [5], [6]]

    def test_more_chunks_than_elements(self):
        assert chunk_list([1, 2, 3, 4, 5, 6], 8) == [[1], [2], [3], [4], [5], [6], [], []]

    def test_n_equals_len(self):
        assert chunk_list([1, 2, 3], 3) == [[1], [2], [3]]

    def test_n_is_1(self):
        assert chunk_list([1, 2, 3], 1) == [[1, 2, 3]]

    def test_single_element_list(self):
        assert chunk_list([42], 2) == [[42], []]

    def test_any_dtype(self):
        assert chunk_list([1, "two", 3.0, {"four": 4}, ["f", "i", "v", "e"]], 2) == [
            [1, "two", 3.0],
            [{"four": 4}, ["f", "i", "v", "e"]],
        ]


@pytest.mark.slow
@require_torch_multi_accelerator
@require_vllm
class TestVLLMClientServer(TrlTestCase):
    model_id = "Qwen/Qwen2.5-1.5B"

    @classmethod
    def setup_class(cls):
        # We want the server to run on accelerator 1, so we set VISIBLE_DEVICES to "1"
        env = os.environ.copy()
        VISIBLE_DEVICES = "ZE_AFFINITY_MASK" if torch_device == "xpu" else "CUDA_VISIBLE_DEVICES"
        env[VISIBLE_DEVICES] = "1"  # Restrict to accelerator 1

        # Start the server process
        cls.server_process = subprocess.Popen(
            ["trl", "vllm-serve", "--model", cls.model_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

        # Initialize the client
        cls.client = VLLMClient(connection_timeout=240, host="localhost")
        cls.client.init_communicator()

    def test_generate(self):
        prompts = ["Hello, AI!", "Tell me a joke"]
        outputs = self.client.generate(prompts)
        prompt_ids = outputs["prompt_ids"]
        completion_ids = outputs["completion_ids"]

        # Check that the outputs are lists
        assert isinstance(prompt_ids, list)
        assert isinstance(completion_ids, list)

        # Check that the number of sequences are equal to the number of prompts
        assert len(prompt_ids) == len(prompts)
        assert len(completion_ids) == len(prompts)

        # Check that the sequences are lists of integers
        for seq in prompt_ids:
            assert all(isinstance(tok, int) for tok in seq)
        for seq in completion_ids:
            assert all(isinstance(tok, int) for tok in seq)

    def test_generate_with_params(self):
        prompts = ["Hello, AI!", "Tell me a joke"]
        completion_ids = self.client.generate(prompts, n=2, repetition_penalty=0.9, temperature=0.8, max_tokens=32)[
            "completion_ids"
        ]

        # Check that the output is a list
        assert isinstance(completion_ids, list)

        # Check that the number of generated sequences is 2 times the number of prompts
        assert len(completion_ids) == 2 * len(prompts)

        # Check that the generated sequences are lists of integers
        for seq in completion_ids:
            assert all(isinstance(tok, int) for tok in seq)

        # Check that the length of the generated sequences is less than or equal to 32
        for seq in completion_ids:
            assert len(seq) <= 32

    def test_update_model_params(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map=torch_device)
        self.client.update_model_params(model)

    def test_reset_prefix_cache(self):
        # Test resetting the prefix cache
        self.client.reset_prefix_cache()

    @classmethod
    def teardown_class(cls):
        # Close the client
        cls.client.close_communicator()

        # vLLM x pytest (or Popen) seems not to handle process termination well. To avoid zombie processes, we need to
        # kill the server process and its children explicitly.
        kill_process(cls.server_process)


# Same as above but using base_url to instantiate the client.
@pytest.mark.slow
@require_torch_multi_accelerator
@require_vllm
class TestVLLMClientServerBaseURL(TrlTestCase):
    model_id = "Qwen/Qwen2.5-1.5B"

    @classmethod
    def setup_class(cls):
        # We want the server to run on accelerator 1, so we set VISIBLE_DEVICES to "1"
        env = os.environ.copy()
        VISIBLE_DEVICES = "ZE_AFFINITY_MASK" if torch_device == "xpu" else "CUDA_VISIBLE_DEVICES"
        env[VISIBLE_DEVICES] = "1"  # Restrict to accelerator 1

        # Start the server process
        cls.server_process = subprocess.Popen(
            ["trl", "vllm-serve", "--model", cls.model_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

        # Initialize the client
        cls.client = VLLMClient(base_url="http://localhost:8000", connection_timeout=240)
        cls.client.init_communicator()

    def test_generate(self):
        prompts = ["Hello, AI!", "Tell me a joke"]
        outputs = self.client.generate(prompts)
        prompt_ids = outputs["prompt_ids"]
        completion_ids = outputs["completion_ids"]

        # Check that the outputs are lists
        assert isinstance(prompt_ids, list)
        assert isinstance(completion_ids, list)

        # Check that the number of sequences are equal to the number of prompts
        assert len(prompt_ids) == len(prompts)
        assert len(completion_ids) == len(prompts)

        # Check that the sequences are lists of integers
        for seq in prompt_ids:
            assert all(isinstance(tok, int) for tok in seq)
        for seq in completion_ids:
            assert all(isinstance(tok, int) for tok in seq)

    def test_generate_with_params(self):
        prompts = ["Hello, AI!", "Tell me a joke"]
        completion_ids = self.client.generate(prompts, n=2, repetition_penalty=0.9, temperature=0.8, max_tokens=32)[
            "completion_ids"
        ]

        # Check that the output is a list
        assert isinstance(completion_ids, list)

        # Check that the number of generated sequences is 2 times the number of prompts
        assert len(completion_ids) == 2 * len(prompts)

        # Check that the generated sequences are lists of integers
        for seq in completion_ids:
            assert all(isinstance(tok, int) for tok in seq)

        # Check that the length of the generated sequences is less than or equal to 32
        for seq in completion_ids:
            assert len(seq) <= 32

    def test_update_model_params(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map=torch_device)
        self.client.update_model_params(model)

    def test_reset_prefix_cache(self):
        # Test resetting the prefix cache
        self.client.reset_prefix_cache()

    @classmethod
    def teardown_class(cls):
        # Close the client
        cls.client.close_communicator()

        # vLLM x pytest (or Popen) seems not to handle process termination well. To avoid zombie processes, we need to
        # kill the server process and its children explicitly.
        kill_process(cls.server_process)


@pytest.mark.slow
@require_3_accelerators
@require_vllm
class TestVLLMClientServerTP(TrlTestCase):
    model_id = "Qwen/Qwen2.5-1.5B"

    @classmethod
    def setup_class(cls):
        # We want the server to run on accelerator 1 and 2, so we set VISIBLE_DEVICES to "1,2"
        env = os.environ.copy()
        VISIBLE_DEVICES = "ZE_AFFINITY_MASK" if torch_device == "xpu" else "CUDA_VISIBLE_DEVICES"
        env[VISIBLE_DEVICES] = "1,2"  # Restrict to accelerator 1 and 2

        # Start the server process
        cls.server_process = subprocess.Popen(
            ["trl", "vllm-serve", "--model", cls.model_id, "--tensor_parallel_size", "2"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Initialize the client
        cls.client = VLLMClient(connection_timeout=240, host="localhost")
        cls.client.init_communicator()

    def test_generate(self):
        prompts = ["Hello, AI!", "Tell me a joke"]
        outputs = self.client.generate(prompts)
        prompt_ids = outputs["prompt_ids"]
        completion_ids = outputs["completion_ids"]

        # Check that the outputs are lists
        assert isinstance(prompt_ids, list)
        assert isinstance(completion_ids, list)

        # Check that the number of sequences are equal to the number of prompts
        assert len(prompt_ids) == len(prompts)
        assert len(completion_ids) == len(prompts)

        # Check that the sequences are lists of integers
        for seq in prompt_ids:
            assert all(isinstance(tok, int) for tok in seq)
        for seq in completion_ids:
            assert all(isinstance(tok, int) for tok in seq)

    def test_update_model_params(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map=torch_device)
        self.client.update_model_params(model)

    def test_reset_prefix_cache(self):
        # Test resetting the prefix cache
        self.client.reset_prefix_cache()

    @classmethod
    def teardown_class(cls):
        # Close the client
        cls.client.close_communicator()

        # vLLM x pytest (or Popen) seems not to handle process termination well. To avoid zombie processes, we need to
        # kill the server process and its children explicitly.
        kill_process(cls.server_process)


@pytest.mark.slow
@require_3_accelerators
@require_vllm
class TestVLLMClientServerDP(TrlTestCase):
    model_id = "Qwen/Qwen2.5-1.5B"

    @classmethod
    def setup_class(cls):
        # We want the server to run on accelerator 1 and 2, so we set VISIBLE_DEVICES to "1,2"
        env = os.environ.copy()
        VISIBLE_DEVICES = "ZE_AFFINITY_MASK" if torch_device == "xpu" else "CUDA_VISIBLE_DEVICES"
        env[VISIBLE_DEVICES] = "1,2"  # Restrict to accelerator 1 and 2

        # Start the server process
        cls.server_process = subprocess.Popen(
            ["trl", "vllm-serve", "--model", cls.model_id, "--data_parallel_size", "2"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Initialize the client
        cls.client = VLLMClient(connection_timeout=240, host="localhost")
        cls.client.init_communicator()

    def test_generate(self):
        prompts = ["Hello, AI!", "Tell me a joke"]
        outputs = self.client.generate(prompts)
        prompt_ids = outputs["prompt_ids"]
        completion_ids = outputs["completion_ids"]

        # Check that the outputs are lists
        assert isinstance(prompt_ids, list)
        assert isinstance(completion_ids, list)

        # Check that the number of sequences are equal to the number of prompts
        assert len(prompt_ids) == len(prompts)
        assert len(completion_ids) == len(prompts)

        # Check that the sequences are lists of integers
        for seq in prompt_ids:
            assert all(isinstance(tok, int) for tok in seq)
        for seq in completion_ids:
            assert all(isinstance(tok, int) for tok in seq)

    def test_update_model_params(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map=torch_device)
        self.client.update_model_params(model)

    def test_reset_prefix_cache(self):
        # Test resetting the prefix cache
        self.client.reset_prefix_cache()

    @classmethod
    def teardown_class(cls):
        # Close the client
        cls.client.close_communicator()

        # vLLM x pytest (or Popen) seems not to handle process termination well. To avoid zombie processes, we need to
        # kill the server process and its children explicitly.
        kill_process(cls.server_process)


@pytest.mark.slow
@require_torch_multi_accelerator
@require_vllm
class TestVLLMClientServerDeviceParameter(TrlTestCase):
    """Test the device parameter functionality in init_communicator."""

    model_id = "Qwen/Qwen2.5-1.5B"

    @classmethod
    def setup_class(cls):
        # We want the server to run on accelerator 1, so we set VISIBLE_DEVICES to "1"
        env = os.environ.copy()
        VISIBLE_DEVICES = "ZE_AFFINITY_MASK" if torch_device == "xpu" else "CUDA_VISIBLE_DEVICES"
        env[VISIBLE_DEVICES] = "1"  # Restrict to accelerator 1

        # Start the server process
        cls.server_process = subprocess.Popen(
            ["trl", "vllm-serve", "--model", cls.model_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

    def test_init_communicator_with_device_int(self):
        """Test init_communicator with integer device parameter."""
        client = VLLMClient(connection_timeout=240, host="localhost")
        client.init_communicator(device=0)  # Explicitly specify device 0

        # Test basic functionality
        prompts = ["Hello, AI!"]
        outputs = client.generate(prompts)
        prompt_ids = outputs["prompt_ids"]
        completion_ids = outputs["completion_ids"]
        assert isinstance(prompt_ids, list)
        assert len(prompt_ids) == len(prompts)
        assert isinstance(completion_ids, list)
        assert len(completion_ids) == len(prompts)

        client.close_communicator()

    def test_init_communicator_with_device_string(self):
        """Test init_communicator with string device parameter."""
        client = VLLMClient(connection_timeout=240, host="localhost")
        client.init_communicator(device=0)  # Explicitly specify device as string

        # Test basic functionality
        prompts = ["Hello, AI!"]
        outputs = client.generate(prompts)["completion_ids"]
        assert isinstance(outputs, list)
        assert len(outputs) == len(prompts)

        client.close_communicator()

    def test_init_communicator_with_torch_device(self):
        """Test init_communicator with torch.device object."""
        import torch

        client = VLLMClient(connection_timeout=240, host="localhost")
        device = torch.device(0)
        client.init_communicator(device=device)  # Explicitly specify torch.device object

        # Test basic functionality
        prompts = ["Hello, AI!"]
        outputs = client.generate(prompts)["completion_ids"]
        assert isinstance(outputs, list)
        assert len(outputs) == len(prompts)

        client.close_communicator()

    @classmethod
    def teardown_class(cls):
        # vLLM x pytest (or Popen) seems not to handle process termination well. To avoid zombie processes, we need to
        # kill the server process and its children explicitly.
        kill_process(cls.server_process)
