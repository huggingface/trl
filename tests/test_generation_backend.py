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

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from trl.generation.backend import (
    RolloutCompletion,
    TransformersBackendAdapter,
    TransformersPagedBackendAdapter,
    VLLMBackendAdapter,
    create_generation_backend,
)


@contextmanager
def _yield_unwrapped_model(model):
    yield model


class TestGenerationBackendAdapters:
    def test_vllm_adapter_output_normalization(self):
        vllm_generation = MagicMock()
        vllm_generation.generate.return_value = (
            [[1, 2]],
            [[10, 11]],
            [[[-0.1, -1.1], [-0.2, -1.2]]],
            [[[]]],
            {"env_mask": [[1, 0]]},
        )
        profiler_factory = MagicMock(return_value=SimpleNamespace())

        adapter = VLLMBackendAdapter(
            vllm_generation=vllm_generation,
            profiler_factory=profiler_factory,
            vllm_mode="server",
            processing_class=MagicMock(),
            temperature=1.0,
            top_k=50,
            min_p=None,
            max_completion_length=32,
            repetition_penalty=None,
            top_p=None,
            generation_kwargs=None,
            chat_template_kwargs={},
            tools=None,
            chat_template=None,
            vllm_tensor_parallel_size=1,
            vllm_enable_sleep_mode=False,
        )
        result = adapter.generate(
            prompts=["hello"],
            num_generations=1,
            processing_class=MagicMock(),
            generation_config=None,
        )

        assert result.prompt_ids == [[1, 2]]
        assert result.completion_ids == [[10, 11]]
        assert result.logprobs == [[-0.1, -0.2]]
        assert result.extra_fields == {"env_mask": [[1, 0]]}
        profiler_factory.assert_called_once_with("vLLM.generate")

    def test_transformers_adapter_chat_template_and_output_shape(self):
        processing_class = MagicMock()
        generate_inputs = {
            "input_ids": torch.tensor([[101, 102, 103]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
        processing_class.apply_chat_template.return_value = generate_inputs

        unwrapped_model = MagicMock()
        unwrapped_model.generate.return_value = torch.tensor([[101, 102, 103, 201, 202, 0, 999]], dtype=torch.long)

        adapter = TransformersBackendAdapter(
            model_wrapped=MagicMock(),
            accelerator=SimpleNamespace(device=torch.device("cpu")),
            is_fsdp_enabled=False,
            ds3_gather_for_generation=False,
            generation_kwargs={"foo": "bar"},
            eos_token_id=0,
            chat_template_kwargs={"temperature_hint": "test"},
            tools=[{"name": "tool"}],
            chat_template="template",
            include_tools_in_chat_template=True,
            prepare_inputs=lambda x: x,
        )

        with patch(
            "trl.generation.backend.unwrap_model_for_generation",
            return_value=_yield_unwrapped_model(unwrapped_model),
        ):
            result = adapter.generate(
                prompts=[[{"role": "user", "content": "hi"}]],
                num_generations=1,
                processing_class=processing_class,
                generation_config=SimpleNamespace(),
            )

        processing_class.apply_chat_template.assert_called_once()
        assert result.prompt_ids == [[101, 102, 103]]
        assert result.completion_ids == [[201, 202, 0]]
        assert result.logprobs is None
        assert result.extra_fields == {}

    def test_transformers_paged_adapter_output_shaping(self):
        processing_class = MagicMock()
        processing_class.apply_chat_template.return_value = {"input_ids": [[1, 2], [3, 4]]}

        output_0 = SimpleNamespace(generated_tokens=[11, 12])
        output_1 = SimpleNamespace(generated_tokens=[21, 22, 23])
        unwrapped_model = MagicMock()
        unwrapped_model.generate_batch.return_value = {0: output_0, 1: output_1}

        adapter = TransformersPagedBackendAdapter(
            model_wrapped=MagicMock(),
            accelerator=SimpleNamespace(device=torch.device("cpu")),
            is_fsdp_enabled=False,
            ds3_gather_for_generation=False,
            chat_template_kwargs={"foo": "bar"},
            tools=[{"name": "tool"}],
            chat_template="template",
            include_tools_in_chat_template=True,
        )

        with patch(
            "trl.generation.backend.unwrap_model_for_generation",
            return_value=_yield_unwrapped_model(unwrapped_model),
        ):
            result = adapter.generate(
                prompts=[[{"role": "user", "content": "hello"}], [{"role": "user", "content": "world"}]],
                num_generations=2,
                processing_class=processing_class,
                generation_config=SimpleNamespace(),
            )

        assert result.prompt_ids == [[1, 2], [3, 4]]
        assert result.completion_ids == [[11, 12], [21, 22, 23]]
        assert result.logprobs is None
        assert result.extra_fields == {}

    def test_sync_behavior(self):
        vllm_generation = MagicMock()
        vllm_adapter = VLLMBackendAdapter(
            vllm_generation=vllm_generation,
            vllm_mode="server",
            processing_class=MagicMock(),
            temperature=1.0,
            top_k=50,
            min_p=None,
            max_completion_length=32,
            repetition_penalty=None,
            top_p=None,
            generation_kwargs=None,
            chat_template_kwargs={},
            tools=None,
            chat_template=None,
            vllm_tensor_parallel_size=1,
            vllm_enable_sleep_mode=False,
        )
        vllm_adapter.sync_weights()
        vllm_generation.sync_weights.assert_called_once()

        transformers_adapter = TransformersBackendAdapter(
            model_wrapped=MagicMock(),
            accelerator=SimpleNamespace(device=torch.device("cpu")),
            is_fsdp_enabled=False,
            ds3_gather_for_generation=False,
            generation_kwargs=None,
            eos_token_id=0,
        )
        transformers_paged_adapter = TransformersPagedBackendAdapter(
            model_wrapped=MagicMock(),
            accelerator=SimpleNamespace(device=torch.device("cpu")),
            is_fsdp_enabled=False,
            ds3_gather_for_generation=False,
        )

        transformers_adapter.sync_weights()
        transformers_paged_adapter.sync_weights()

    def test_vllm_rollout_completions_server_chat(self):
        vllm_generation = MagicMock()
        vllm_generation.vllm_client.chat.return_value = {
            "prompt_ids": [[1, 2]],
            "completion_ids": [[10, 11]],
            "logprobs": [[-0.1, -0.2]],
        }
        processing_class = MagicMock()
        processing_class.decode.return_value = "hello"

        adapter = VLLMBackendAdapter(
            vllm_generation=vllm_generation,
            vllm_mode="server",
            processing_class=processing_class,
            temperature=1.0,
            top_k=50,
            min_p=None,
            max_completion_length=32,
            repetition_penalty=None,
            top_p=None,
            generation_kwargs=None,
            chat_template_kwargs={"foo": "bar"},
            tools=[{"name": "tool"}],
            chat_template="template",
            vllm_tensor_parallel_size=1,
            vllm_enable_sleep_mode=False,
        )

        prompts = [[{"role": "user", "content": "hi"}]]
        results = adapter.generate_rollout_completions(prompts=prompts, processing_class=MagicMock(), as_chat=True)

        assert results == [
            RolloutCompletion(prompt_ids=[1, 2], completion_ids=[10, 11], logprobs=[-0.1, -0.2], text="hello")
        ]
        vllm_generation.vllm_client.chat.assert_called_once()

    def test_vllm_rollout_completions_server_text(self):
        vllm_generation = MagicMock()
        vllm_generation.vllm_client.generate.return_value = {
            "prompt_ids": [[3, 4]],
            "completion_ids": [[20]],
            "logprobs": [[-0.9]],
        }
        processing_class = MagicMock()
        processing_class.decode.return_value = "ok"

        adapter = VLLMBackendAdapter(
            vllm_generation=vllm_generation,
            vllm_mode="server",
            processing_class=processing_class,
            temperature=1.0,
            top_k=50,
            min_p=None,
            max_completion_length=32,
            repetition_penalty=None,
            top_p=None,
            generation_kwargs=None,
            chat_template_kwargs={},
            tools=None,
            chat_template=None,
            vllm_tensor_parallel_size=1,
            vllm_enable_sleep_mode=False,
        )

        results = adapter.generate_rollout_completions(prompts=["hello"], processing_class=MagicMock(), as_chat=False)

        assert results == [RolloutCompletion(prompt_ids=[3, 4], completion_ids=[20], logprobs=[-0.9], text="ok")]
        vllm_generation.vllm_client.generate.assert_called_once()

    def test_vllm_rollout_completions_server_auto_chat_inference(self):
        vllm_generation = MagicMock()
        vllm_generation.vllm_client.chat.return_value = {
            "prompt_ids": [[1]],
            "completion_ids": [[2]],
            "logprobs": [[-0.3]],
        }
        processing_class = MagicMock()
        processing_class.decode.return_value = "x"

        adapter = VLLMBackendAdapter(
            vllm_generation=vllm_generation,
            vllm_mode="server",
            processing_class=processing_class,
            temperature=1.0,
            top_k=50,
            min_p=None,
            max_completion_length=32,
            repetition_penalty=None,
            top_p=None,
            generation_kwargs=None,
            chat_template_kwargs={},
            tools=None,
            chat_template=None,
            vllm_tensor_parallel_size=1,
            vllm_enable_sleep_mode=False,
        )

        prompts = [[{"role": "user", "content": "hi"}]]
        results = adapter.generate_rollout_completions(prompts=prompts, processing_class=MagicMock(), as_chat=None)

        assert results == [RolloutCompletion(prompt_ids=[1], completion_ids=[2], logprobs=[-0.3], text="x")]
        vllm_generation.vllm_client.chat.assert_called_once()
        vllm_generation.vllm_client.generate.assert_not_called()

    def test_vllm_rollout_completions_rejects_n_not_equal_to_one(self):
        adapter = VLLMBackendAdapter(
            vllm_generation=MagicMock(),
            vllm_mode="server",
            processing_class=MagicMock(),
            temperature=1.0,
            top_k=50,
            min_p=None,
            max_completion_length=32,
            repetition_penalty=None,
            top_p=None,
            generation_kwargs=None,
            chat_template_kwargs={},
            tools=None,
            chat_template=None,
            vllm_tensor_parallel_size=1,
            vllm_enable_sleep_mode=False,
        )

        with pytest.raises(ValueError, match="expects n=1"):
            adapter.generate_rollout_completions(
                prompts=["hello"],
                processing_class=MagicMock(),
                generation_overrides={"n": 2},
            )

    def test_vllm_rollout_completions_colocate_mapping_and_empty_outputs(self):
        fake_sampling_params_module = SimpleNamespace(StructuredOutputsParams=MagicMock())
        fake_vllm_module = SimpleNamespace(SamplingParams=MagicMock())

        def _token_logprob(value):
            return {0: SimpleNamespace(logprob=value)}

        seq = SimpleNamespace(
            token_ids=[7, 8],
            logprobs=[_token_logprob(-0.4), _token_logprob(-0.5)],
            text="done",
        )
        request_with_output = SimpleNamespace(prompt_token_ids=[1, 2], outputs=[seq])
        request_empty = SimpleNamespace(prompt_token_ids=[3], outputs=[])

        vllm_generation = MagicMock()
        vllm_generation.structured_outputs_regex = None
        vllm_generation.llm.generate.return_value = [request_with_output, request_empty]

        adapter = VLLMBackendAdapter(
            vllm_generation=vllm_generation,
            vllm_mode="colocate",
            processing_class=MagicMock(),
            temperature=1.0,
            top_k=50,
            min_p=None,
            max_completion_length=32,
            repetition_penalty=None,
            top_p=None,
            generation_kwargs=None,
            chat_template_kwargs={},
            tools=None,
            chat_template=None,
            vllm_tensor_parallel_size=1,
            vllm_enable_sleep_mode=False,
        )

        with patch.dict(
            sys.modules,
            {
                "vllm": fake_vllm_module,
                "vllm.sampling_params": fake_sampling_params_module,
            },
        ):
            results = adapter.generate_rollout_completions(
                prompts=["a", "b"], processing_class=MagicMock(), as_chat=False
            )

        assert results == [
            RolloutCompletion(prompt_ids=[1, 2], completion_ids=[7, 8], logprobs=[-0.4, -0.5], text="done"),
            RolloutCompletion(prompt_ids=[3], completion_ids=[], logprobs=[], text=""),
        ]
        vllm_generation.llm.generate.assert_called_once()

    def test_vllm_rollout_completions_colocate_sleep_mode_hooks(self):
        fake_sampling_params_module = SimpleNamespace(StructuredOutputsParams=MagicMock())
        fake_vllm_module = SimpleNamespace(SamplingParams=MagicMock())

        request = SimpleNamespace(
            prompt_token_ids=[1],
            outputs=[
                SimpleNamespace(
                    token_ids=[9],
                    logprobs=[{0: SimpleNamespace(logprob=-0.2)}],
                    text="z",
                )
            ],
        )

        vllm_generation = MagicMock()
        vllm_generation.structured_outputs_regex = None
        vllm_generation.llm.generate.return_value = [request]

        adapter = VLLMBackendAdapter(
            vllm_generation=vllm_generation,
            vllm_mode="colocate",
            processing_class=MagicMock(),
            temperature=1.0,
            top_k=50,
            min_p=None,
            max_completion_length=32,
            repetition_penalty=None,
            top_p=None,
            generation_kwargs=None,
            chat_template_kwargs={},
            tools=None,
            chat_template=None,
            vllm_tensor_parallel_size=1,
            vllm_enable_sleep_mode=True,
        )

        with patch.dict(
            sys.modules,
            {
                "vllm": fake_vllm_module,
                "vllm.sampling_params": fake_sampling_params_module,
            },
        ):
            results = adapter.generate_rollout_completions(prompts=["a"], processing_class=MagicMock(), as_chat=False)

        assert results == [RolloutCompletion(prompt_ids=[1], completion_ids=[9], logprobs=[-0.2], text="z")]
        vllm_generation.llm.wake_up.assert_called_once_with(tags=["kv_cache"])
        vllm_generation.llm.collective_rpc.assert_called_once_with("reload_weights")
        vllm_generation.llm.sleep.assert_called_once_with(level=2)

    def test_rollout_capability_unsupported_for_non_vllm_backends(self):
        transformers_adapter = TransformersBackendAdapter(
            model_wrapped=MagicMock(),
            accelerator=SimpleNamespace(device=torch.device("cpu")),
            is_fsdp_enabled=False,
            ds3_gather_for_generation=False,
            generation_kwargs=None,
            eos_token_id=0,
        )
        transformers_paged_adapter = TransformersPagedBackendAdapter(
            model_wrapped=MagicMock(),
            accelerator=SimpleNamespace(device=torch.device("cpu")),
            is_fsdp_enabled=False,
            ds3_gather_for_generation=False,
        )

        with pytest.raises(RuntimeError, match="does not support rollout completions"):
            transformers_adapter.generate_rollout_completions(prompts=["x"], processing_class=MagicMock())
        with pytest.raises(RuntimeError, match="does not support rollout completions"):
            transformers_paged_adapter.generate_rollout_completions(prompts=["x"], processing_class=MagicMock())


class TestGenerationBackendFactory:
    def _base_trainer(self):
        class _BaseTrainer:
            def _prepare_inputs(self, inputs):
                return inputs

        class _Trainer(_BaseTrainer):
            pass

        trainer = _Trainer()
        trainer.use_vllm = False
        trainer.use_transformers_paged = False
        trainer.model_wrapped = MagicMock()
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        trainer.is_fsdp_enabled = False
        trainer.args = SimpleNamespace(
            ds3_gather_for_generation=False,
            bf16=False,
            fp16=False,
            cast_lm_head_to_fp32=False,
            generation_kwargs={},
            vllm_enable_sleep_mode=False,
        )
        trainer.generation_kwargs = {"foo": "bar"}
        trainer.eos_token_id = 0
        trainer.chat_template_kwargs = {}
        trainer.processing_class = MagicMock()
        trainer.temperature = 1.0
        trainer.top_k = 50
        trainer.min_p = None
        trainer.max_completion_length = 32
        trainer.repetition_penalty = None
        trainer.top_p = None
        trainer.vllm_tensor_parallel_size = 1
        trainer.vllm_mode = "server"
        return trainer

    def test_factory_selects_vllm_backend(self):
        trainer = self._base_trainer()
        trainer.use_vllm = True
        trainer.vllm_generation = MagicMock()

        backend = create_generation_backend(trainer)
        assert isinstance(backend, VLLMBackendAdapter)

    def test_factory_selects_transformers_paged_backend(self):
        trainer = self._base_trainer()
        trainer.use_transformers_paged = True
        trainer.tools = []
        trainer.chat_template = None

        backend = create_generation_backend(trainer)
        assert isinstance(backend, TransformersPagedBackendAdapter)

    def test_factory_selects_transformers_backend(self):
        trainer = self._base_trainer()

        backend = create_generation_backend(trainer)
        assert isinstance(backend, TransformersBackendAdapter)
