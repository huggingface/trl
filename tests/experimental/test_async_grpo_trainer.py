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

import asyncio
import itertools
import queue

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo import async_grpo_trainer as async_grpo_trainer_module
from trl.experimental.async_grpo import async_rollout_worker as async_rollout_worker_module
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample

from ..testing_utils import TrlTestCase


def dummy_reward_func(completions, **kwargs):
    return [float(hash(c[0]["content"]) % 100) / 100.0 for c in completions]


class _StubRolloutWorker:
    """Minimal rollout worker stub for testing the trainer in isolation."""

    def __init__(self, tokenizer, dataset, num_generations: int = 8, samples_per_weight_sync: int = 10):
        self.rollout_buffer = queue.Queue()
        self._samples_per_weight_sync = samples_per_weight_sync
        self._model_version = 0
        self._sample_iter = self._make_sample_iter(tokenizer, dataset, num_generations)

    def _make_sample_iter(self, tokenizer, dataset, num_generations):
        for row in itertools.cycle(dataset):
            completions = [
                [{"role": "assistant", "content": f"{row['completion'][0]['content']} {idx}"}]
                for idx in range(num_generations)
            ]
            prompt_completions = [row["prompt"] + completion for completion in completions]
            prompt_ids = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            prompt_completion_ids = tokenizer.apply_chat_template(
                prompt_completions, tokenize=True, add_generation_prompt=False, return_dict=False
            )
            rewards = np.array(dummy_reward_func(completions))
            advantages = (rewards - rewards.mean()) / rewards.std()
            for idx in range(num_generations):
                completion_ids = prompt_completion_ids[idx][len(prompt_ids) :]
                yield RolloutSample(
                    prompt=row["prompt"],
                    completion=completions[idx],
                    input_ids=prompt_ids + completion_ids,
                    completion_mask=[0] * len(prompt_ids) + [1] * len(completion_ids),
                    old_log_probs=[0.0] * len(prompt_ids) + [-0.5] * len(completion_ids),
                    advantage=float(advantages[idx]),
                    model_version=self._model_version,
                    metrics={"reward": float(rewards[idx]), "reward_std": float(rewards.std())},
                )

    def _fill_queue(self):
        for _ in range(self._samples_per_weight_sync):
            self.rollout_buffer.put(next(self._sample_iter))

    def start(self):
        self._fill_queue()

    def update_model_version(self, version):
        self._model_version = version
        self._fill_queue()

    def stop(self):
        pass

    def pause(self):
        pass

    def resume(self):
        pass

    def send_weights(self, iterator):
        pass


class _CapturingRolloutWorker:
    last_init_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = kwargs
        self.rollout_buffer = queue.Queue()

    def start(self):
        pass

    def update_model_version(self, version):
        pass

    def stop(self):
        pass

    def pause(self):
        pass

    def resume(self):
        pass

    def send_weights(self, iterator):
        pass


class TestAsyncGRPOTrainer(TrlTestCase):
    def test_init_passes_sampling_config_to_rollout_worker(self, monkeypatch):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        training_args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            num_generations=3,
            max_completion_length=8,
            temperature=0.7,
            top_p=0.9,
            top_k=10,
            min_p=0.01,
            repetition_penalty=1.1,
            generation_kwargs={"top_k": 50, "seed": 7},
            report_to="none",
        )

        monkeypatch.setattr(async_grpo_trainer_module, "AsyncRolloutWorker", _CapturingRolloutWorker)

        AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        assert _CapturingRolloutWorker.last_init_kwargs["max_tokens"] == 8
        assert _CapturingRolloutWorker.last_init_kwargs["temperature"] == 0.7
        assert _CapturingRolloutWorker.last_init_kwargs["top_p"] == 0.9
        assert _CapturingRolloutWorker.last_init_kwargs["top_k"] == 10
        assert _CapturingRolloutWorker.last_init_kwargs["min_p"] == 0.01
        assert _CapturingRolloutWorker.last_init_kwargs["repetition_penalty"] == 1.1
        assert _CapturingRolloutWorker.last_init_kwargs["generation_kwargs"] == {"top_k": 50, "seed": 7}

    def test_rollout_worker_generation_kwargs_override_named_sampling_params(self):
        worker = async_rollout_worker_module.AsyncRolloutWorker.__new__(async_rollout_worker_module.AsyncRolloutWorker)
        worker.model_name = "Qwen/Qwen3-4B"
        worker.max_tokens = 32
        worker.temperature = 0.9
        worker.top_p = 0.95
        worker.top_k = 10
        worker.min_p = None
        worker.repetition_penalty = 1.1
        worker.generation_kwargs = {"top_k": 50, "temperature": 0.2, "seed": 123}
        worker.request_timeout = 17
        captured = {}

        async def fake_post(path, payload, timeout):
            captured["path"] = path
            captured["payload"] = payload
            captured["timeout"] = timeout
            return {"choices": [{"token_ids": [42], "logprobs": {"token_logprobs": [-0.5]}}]}

        worker._post = fake_post

        completion_ids, completion_logprobs = asyncio.run(worker._generate_one_turn([1, 2, 3]))

        assert completion_ids == [42]
        assert completion_logprobs == [-0.5]
        assert captured["path"] == "/v1/completions"
        assert captured["timeout"] == 17
        assert captured["payload"] == {
            "model": "Qwen/Qwen3-4B",
            "prompt": [1, 2, 3],
            "max_tokens": 32,
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 50,
            "min_p": 0.0,
            "repetition_penalty": 1.1,
            "n": 1,
            "return_token_ids": True,
            "logprobs": 0,
            "seed": 123,
        }

    def test_init_minimal(self):
        # Test that AsyncGRPOTrainer can be instantiated with only model, reward_model and train_dataset
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
        )

    def test_training(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        training_args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            vllm_server_timeout=5.0,  # short timeout so test fails fast if queue runs dry
            report_to="none",
        )
        trainer = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,  # unused: the stub pre-computes rewards, but the trainer requires this argument
            args=training_args,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
