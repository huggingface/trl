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
import subprocess
from types import SimpleNamespace

import pytest
from transformers import AutoModelForCausalLM
from transformers.testing_utils import torch_device

from trl.generation.vllm_client import VLLMClient
from trl.generation.vllm_generation import extract_logprobs
from trl.import_utils import is_vllm_available
from trl.scripts.vllm_serve import chunk_list

from .testing_utils import (
    TrlTestCase,
    kill_process,
    require_3_accelerators,
    require_torch_multi_accelerator,
    require_vllm,
)


if is_vllm_available():
    from vllm import LLM, SamplingParams


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


class TestExtractLogprobs(TrlTestCase):
    def test_extract_logprobs_sorts_by_rank_and_replaces_nan(self):
        all_outputs = [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        logprobs=[
                            {
                                11: SimpleNamespace(rank=1, logprob=-0.2),
                                99: SimpleNamespace(rank=0, logprob=-0.1),
                                42: SimpleNamespace(rank=2, logprob=float("nan")),
                            },
                            {
                                5: SimpleNamespace(rank=0, logprob=-1.1),
                            },
                        ]
                    )
                ]
            ),
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        logprobs=[
                            {
                                3: SimpleNamespace(rank=1, logprob=-0.5),
                                7: SimpleNamespace(rank=0, logprob=-0.4),
                            }
                        ]
                    )
                ]
            ),
        ]

        all_logprobs, all_token_ids = extract_logprobs(all_outputs)

        assert all_token_ids == [
            [[99, 11, 42], [5]],
            [[7, 3]],
        ]
        assert all_logprobs == [
            [[-0.1, -0.2, None], [-1.1]],
            [[-0.4, -0.5]],
        ]

    def test_extract_logprobs_returns_none_token_ids_when_logprobs_missing(self):
        all_outputs = [SimpleNamespace(outputs=[SimpleNamespace(logprobs=None)])]

        all_logprobs, all_token_ids = extract_logprobs(all_outputs)

        assert all_logprobs is None
        assert all_token_ids is None


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

    def test_chat(self):
        messages = [[{"role": "user", "content": "Hello, AI!"}], [{"role": "user", "content": "Tell me a joke"}]]
        outputs = self.client.chat(messages)
        prompt_ids = outputs["prompt_ids"]
        completion_ids = outputs["completion_ids"]

        # Check that the outputs are lists
        assert isinstance(prompt_ids, list)
        assert isinstance(completion_ids, list)

        # Check that the number of sequences are equal to the number of messages
        assert len(prompt_ids) == len(messages)
        assert len(completion_ids) == len(messages)

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

    @pytest.mark.xfail(reason="Importing `bitsandbytes` causes issues, see vllm-project/vllm#32793")
    def test_logprobs_match_with_non_default_sampling(self):
        prompts = ["Hello, AI!", "Tell me a joke"]
        # Use non-default sampling parameters (especially temperature) to ensure vLLM applies logprob processing. With
        # default sampling, raw and processed logprobs are identical, so mismatches would not be detected.
        temperature = 0.7
        repetition_penalty = 1.05
        top_p = 0.9
        max_tokens = 8
        seed = 1234
        num_logprobs = 5

        server_outputs = self.client.generate(
            prompts,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
            generation_kwargs={"seed": seed},
        )
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        llm = LLM(
            model=self.model_id,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.2,
            max_model_len=128,
            logprobs_mode="processed_logprobs",
        )

        sampling_params = SamplingParams(
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
            seed=seed,
        )
        colocate_outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        colocate_prompt_ids = [output.prompt_token_ids for output in colocate_outputs]
        colocate_completion_ids = [
            list(output.token_ids) for outputs in colocate_outputs for output in outputs.outputs
        ]
        colocate_logprobs, colocate_logprob_token_ids = extract_logprobs(colocate_outputs)

        # Generation correctness: prompt and completion IDs match between server and colocate
        assert server_outputs["prompt_ids"] == colocate_prompt_ids
        assert server_outputs["completion_ids"] == colocate_completion_ids

        server_logprobs = server_outputs["logprobs"]
        server_logprob_token_ids = server_outputs["logprob_token_ids"]

        # Shape: both should be (num_sequences, seq_len, num_logprobs) with multiple logprobs per token
        assert len(server_logprobs) == len(prompts)
        assert len(server_logprob_token_ids) == len(prompts)
        for seq_lps in server_logprobs:
            for token_lps in seq_lps:
                assert len(token_lps) > 1, "Expected multiple logprobs per token when logprobs > 0"

        # Value correctness: server extraction matches colocate extraction via extract_logprobs
        assert server_logprob_token_ids == colocate_logprob_token_ids
        for server_seq, colocate_seq in zip(server_logprobs, colocate_logprobs, strict=True):
            assert len(server_seq) == len(colocate_seq)
            for server_token_lps, colocate_token_lps in zip(server_seq, colocate_seq, strict=True):
                assert server_token_lps == pytest.approx(colocate_token_lps, rel=1e-6, abs=1e-6)

        # Ordering: logprobs at each position should be sorted descending
        for seq_lps in server_logprobs:
            for token_lps in seq_lps:
                assert token_lps == sorted(token_lps, reverse=True), "Logprobs should be sorted descending"

        # Sampled token presence: the actual completion token should appear in the logprob token IDs
        for seq_idx, (completion_seq, token_ids_seq) in enumerate(
            zip(server_outputs["completion_ids"], server_logprob_token_ids, strict=True)
        ):
            for pos, (sampled_id, lp_ids) in enumerate(zip(completion_seq, token_ids_seq, strict=True)):
                assert sampled_id in lp_ids, (
                    f"Sampled token {sampled_id} not found in logprob token IDs {lp_ids} "
                    f"at sequence {seq_idx}, position {pos}"
                )

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

    def test_chat(self):
        messages = [[{"role": "user", "content": "Hello, AI!"}], [{"role": "user", "content": "Tell me a joke"}]]
        outputs = self.client.chat(messages)
        prompt_ids = outputs["prompt_ids"]
        completion_ids = outputs["completion_ids"]

        # Check that the outputs are lists
        assert isinstance(prompt_ids, list)
        assert isinstance(completion_ids, list)

        # Check that the number of sequences are equal to the number of messages
        assert len(prompt_ids) == len(messages)
        assert len(completion_ids) == len(messages)

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

    def test_chat(self):
        messages = [[{"role": "user", "content": "Hello, AI!"}], [{"role": "user", "content": "Tell me a joke"}]]
        outputs = self.client.chat(messages)
        prompt_ids = outputs["prompt_ids"]
        completion_ids = outputs["completion_ids"]

        # Check that the outputs are lists
        assert isinstance(prompt_ids, list)
        assert isinstance(completion_ids, list)

        # Check that the number of sequences are equal to the number of messages
        assert len(prompt_ids) == len(messages)
        assert len(completion_ids) == len(messages)

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

    def test_chat(self):
        messages = [[{"role": "user", "content": "Hello, AI!"}], [{"role": "user", "content": "Tell me a joke"}]]
        outputs = self.client.chat(messages)
        prompt_ids = outputs["prompt_ids"]
        completion_ids = outputs["completion_ids"]

        # Check that the outputs are lists
        assert isinstance(prompt_ids, list)
        assert isinstance(completion_ids, list)

        # Check that the number of sequences are equal to the number of messages
        assert len(prompt_ids) == len(messages)
        assert len(completion_ids) == len(messages)

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
