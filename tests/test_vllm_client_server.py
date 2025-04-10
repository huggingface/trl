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

import pytest
from transformers import AutoModelForCausalLM
from transformers.testing_utils import require_torch_multi_gpu

from trl.extras.vllm_client import VLLMClient

from .testing_utils import require_3_gpus


@pytest.mark.slow
@require_torch_multi_gpu
class TestVLLMClientServer(unittest.TestCase):
    model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"

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
        cls.client = VLLMClient(connection_timeout=120)

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

        # Check if the process is still running
        if cls.server_process.poll() is None:
            try:
                os.kill(cls.server_process.pid, signal.SIGTERM)
                cls.server_process.wait()
            except ProcessLookupError:
                pass  # Process already gone — safe to ignore


@pytest.mark.slow
@require_3_gpus
class TestVLLMClientServerTP(unittest.TestCase):
    model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"

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
        cls.client = VLLMClient(connection_timeout=120)

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

        # Check if the process is still running
        if cls.server_process.poll() is None:
            try:
                os.kill(cls.server_process.pid, signal.SIGTERM)
                cls.server_process.wait()
            except ProcessLookupError:
                pass  # Process already gone — safe to ignore
