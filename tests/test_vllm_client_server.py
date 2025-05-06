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
import signal
import subprocess
import unittest

import psutil
import pytest
from transformers import AutoModelForCausalLM
from transformers.testing_utils import require_torch_multi_gpu

from trl.extras.vllm_client import VLLMClient
from trl.scripts.vllm_serve import chunk_list

from .testing_utils import require_3_gpus


class TestChunkList(unittest.TestCase):
    def test_even_split(self):
        self.assertEqual(chunk_list([1, 2, 3, 4, 5, 6], 2), [[1, 2, 3], [4, 5, 6]])

    def test_uneven_split(self):
        self.assertEqual(chunk_list([1, 2, 3, 4, 5, 6], 4), [[1, 2], [3, 4], [5], [6]])

    def test_more_chunks_than_elements(self):
        self.assertEqual(chunk_list([1, 2, 3, 4, 5, 6], 8), [[1], [2], [3], [4], [5], [6], [], []])

    def test_n_equals_len(self):
        self.assertEqual(chunk_list([1, 2, 3], 3), [[1], [2], [3]])

    def test_n_is_1(self):
        self.assertEqual(chunk_list([1, 2, 3], 1), [[1, 2, 3]])

    def test_single_element_list(self):
        self.assertEqual(chunk_list([42], 2), [[42], []])

    def test_any_dtype(self):
        self.assertEqual(
            chunk_list([1, "two", 3.0, {"four": 4}, ["f", "i", "v", "e"]], 2),
            [[1, "two", 3.0], [{"four": 4}, ["f", "i", "v", "e"]]],
        )


@pytest.mark.slow
@require_torch_multi_gpu
class TestVLLMClientServer(unittest.TestCase):
    model_id = "Qwen/Qwen2.5-1.5B"

    @classmethod
    def setUpClass(cls):
        # We want the server to run on GPU 1, so we set CUDA_VISIBLE_DEVICES to "1"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1"  # Restrict to GPU 1

        # Start the server process
        cls.server_process = subprocess.Popen(
            ["trl", "vllm-serve", "--model", cls.model_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

        # Initialize the client
        cls.client = VLLMClient(connection_timeout=240)
        cls.client.init_communicator()

    def test_generate(self):
        prompts = ["Hello, AI!", "Tell me a joke"]
        outputs = self.client.generate(prompts)

        # Check that the output is a list
        self.assertIsInstance(outputs, list)

        # Check that the number of generated sequences is equal to the number of prompts
        self.assertEqual(len(outputs), len(prompts))

        # Check that the generated sequences are lists of integers
        for seq in outputs:
            self.assertTrue(all(isinstance(tok, int) for tok in seq))

    def test_generate_with_params(self):
        prompts = ["Hello, AI!", "Tell me a joke"]
        outputs = self.client.generate(prompts, n=2, repetition_penalty=0.9, temperature=0.8, max_tokens=32)

        # Check that the output is a list
        self.assertIsInstance(outputs, list)

        # Check that the number of generated sequences is 2 times the number of prompts
        self.assertEqual(len(outputs), 2 * len(prompts))

        # Check that the generated sequences are lists of integers
        for seq in outputs:
            self.assertTrue(all(isinstance(tok, int) for tok in seq))

        # Check that the length of the generated sequences is less than or equal to 32
        for seq in outputs:
            self.assertLessEqual(len(seq), 32)

    def test_update_model_params(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="cuda")
        self.client.update_model_params(model)

    def test_reset_prefix_cache(self):
        # Test resetting the prefix cache
        self.client.reset_prefix_cache()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        # Close the client
        cls.client.close_communicator()

        # vLLM x pytest (or Popen) seems not to handle process termination well. To avoid zombie processes, we need to
        # kill the server process and its children explicitly.
        parent = psutil.Process(cls.server_process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.send_signal(signal.SIGTERM)
        cls.server_process.terminate()
        cls.server_process.wait()


@pytest.mark.slow
@require_3_gpus
class TestVLLMClientServerTP(unittest.TestCase):
    model_id = "Qwen/Qwen2.5-1.5B"

    @classmethod
    def setUpClass(cls):
        # We want the server to run on GPU 1 and 2, so we set CUDA_VISIBLE_DEVICES to "1,2"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1,2"  # Restrict to GPU 1 and 2

        # Start the server process
        cls.server_process = subprocess.Popen(
            ["trl", "vllm-serve", "--model", cls.model_id, "--tensor_parallel_size", "2"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Initialize the client
        cls.client = VLLMClient(connection_timeout=240)
        cls.client.init_communicator()

    def test_generate(self):
        prompts = ["Hello, AI!", "Tell me a joke"]
        outputs = self.client.generate(prompts)

        # Check that the output is a list
        self.assertIsInstance(outputs, list)

        # Check that the number of generated sequences is equal to the number of prompts
        self.assertEqual(len(outputs), len(prompts))

        # Check that the generated sequences are lists of integers
        for seq in outputs:
            self.assertTrue(all(isinstance(tok, int) for tok in seq))

    def test_update_model_params(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="cuda")
        self.client.update_model_params(model)

    def test_reset_prefix_cache(self):
        # Test resetting the prefix cache
        self.client.reset_prefix_cache()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        # Close the client
        cls.client.close_communicator()

        # vLLM x pytest (or Popen) seems not to handle process termination well. To avoid zombie processes, we need to
        # kill the server process and its children explicitly.
        parent = psutil.Process(cls.server_process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.send_signal(signal.SIGTERM)
        cls.server_process.terminate()
        cls.server_process.wait()


@pytest.mark.slow
@require_3_gpus
class TestVLLMClientServerDP(unittest.TestCase):
    model_id = "Qwen/Qwen2.5-1.5B"

    @classmethod
    def setUpClass(cls):
        # We want the server to run on GPU 1 and 2, so we set CUDA_VISIBLE_DEVICES to "1,2"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1,2"  # Restrict to GPU 1 and 2

        # Start the server process
        cls.server_process = subprocess.Popen(
            ["trl", "vllm-serve", "--model", cls.model_id, "--data_parallel_size", "2"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Initialize the client
        cls.client = VLLMClient(connection_timeout=240)

    def test_generate(self):
        prompts = ["Hello, AI!", "Tell me a joke"]
        outputs = self.client.generate(prompts)

        # Check that the output is a list
        self.assertIsInstance(outputs, list)

        # Check that the number of generated sequences is equal to the number of prompts
        self.assertEqual(len(outputs), len(prompts))

        # Check that the generated sequences are lists of integers
        for seq in outputs:
            self.assertTrue(all(isinstance(tok, int) for tok in seq))

    def test_update_model_params(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="cuda")
        self.client.update_model_params(model)

    def test_reset_prefix_cache(self):
        # Test resetting the prefix cache
        self.client.reset_prefix_cache()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        # Close the client
        cls.client.close_communicator()

        # vLLM x pytest (or Popen) seems not to handle process termination well. To avoid zombie processes, we need to
        # kill the server process and its children explicitly.
        parent = psutil.Process(cls.server_process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.send_signal(signal.SIGTERM)
        cls.server_process.terminate()
        cls.server_process.wait()
