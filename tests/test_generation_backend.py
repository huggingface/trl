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

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from trl.generation.backend import (
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

        adapter = VLLMBackendAdapter(vllm_generation=vllm_generation, profiler_factory=profiler_factory)
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
        vllm_adapter = VLLMBackendAdapter(vllm_generation=vllm_generation)
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


class TestGenerationBackendFactory:
    def _base_trainer(self):
        return SimpleNamespace(
            use_vllm=False,
            use_transformers_paged=False,
            model_wrapped=MagicMock(),
            accelerator=SimpleNamespace(device=torch.device("cpu")),
            is_fsdp_enabled=False,
            args=SimpleNamespace(
                ds3_gather_for_generation=False,
                bf16=False,
                fp16=False,
                cast_lm_head_to_fp32=False,
            ),
            generation_kwargs={"foo": "bar"},
            eos_token_id=0,
            chat_template_kwargs={},
            _prepare_inputs=lambda x: x,
        )

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
