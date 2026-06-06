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

import itertools
import queue
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample

from ..testing_utils import TrlTestCase, require_peft


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

    def check_health(self, stale_after_s):
        pass


class TestAsyncGRPOTrainer(TrlTestCase):
    def test_model_init_kwargs(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        training_args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            model_init_kwargs={"dtype": "bfloat16"},
            report_to="none",
        )
        trainer = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            args=training_args,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=8),
        )

        # Verify model was loaded in bfloat16
        for param in trainer.model.parameters():
            assert param.dtype == torch.bfloat16, f"Expected bfloat16 but got {param.dtype}"

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

    def test_train(self):
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

    @require_peft
    def test_train_peft_config(self):
        from peft import LoraConfig

        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        training_args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
        )
        trainer = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
        )

        # Verify the model is a PEFT model
        from accelerate.utils import is_peft_model

        assert is_peft_model(trainer.model)

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the peft params have changed and the base model params have not changed
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        base_param_names = [f"base_model.model.{n}" for n, _ in base_model.named_parameters()]
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if n in base_param_names:
                torch.testing.assert_close(param, new_param, msg=f"Parameter {n} has changed.")
            elif "base_layer" not in n:
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_peft
    def test_peft_quantized_trainable_params_cast_to_bfloat16(self):
        from peft import LoraConfig

        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        @dataclass
        class DummyQuantizationConfig:
            load_in_4bit: bool = True
            load_in_8bit: bool = False

        training_args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            model_init_kwargs={"quantization_config": DummyQuantizationConfig()},
            report_to="none",
        )

        import trl.experimental.async_grpo.async_grpo_trainer as async_grpo_trainer_module

        original_create_model_from_path = async_grpo_trainer_module.create_model_from_path

        def create_quantized_model(*args, **kwargs):
            kwargs.pop("quantization_config")
            return original_create_model_from_path(*args, **kwargs)

        from unittest.mock import patch

        with patch.object(async_grpo_trainer_module, "create_model_from_path", create_quantized_model):
            trainer = AsyncGRPOTrainer(
                model=model_id,
                reward_funcs=dummy_reward_func,
                args=training_args,
                train_dataset=dataset,
                peft_config=LoraConfig(),
                rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=8),
            )

        for param in trainer.model.parameters():
            if param.requires_grad:
                assert param.dtype == torch.bfloat16

    @require_peft
    def test_peft_weight_metadata_excludes_adapters_and_modules_to_save(self):
        from peft import LoraConfig

        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        training_args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            report_to="none",
        )

        # Use a capturing worker to inspect weight metadata
        import trl.experimental.async_grpo.async_grpo_trainer as async_grpo_trainer_module

        captured_kwargs = {}

        class _CapturingRolloutWorker:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)
                self.rollout_buffer = queue.Queue()

            def start(self):
                pass

            def stop(self):
                pass

            def pause(self):
                pass

            def resume(self):
                pass

            def send_weights(self, iterator):
                pass

            def update_model_version(self, version):
                pass

            def check_health(self, stale_after_s):
                pass

        from unittest.mock import patch

        class _CapturingWeightTransferClient:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        with (
            patch.object(async_grpo_trainer_module, "AsyncRolloutWorker", _CapturingRolloutWorker),
            patch.object(async_grpo_trainer_module, "WeightTransferClient", _CapturingWeightTransferClient),
        ):
            AsyncGRPOTrainer(
                model=model_id,
                reward_funcs=dummy_reward_func,
                args=training_args,
                train_dataset=dataset,
                peft_config=LoraConfig(target_modules="all-linear", modules_to_save=["embed_tokens", "lm_head"]),
                processing_class=AutoTokenizer.from_pretrained(model_id),
            )

        # Weight names should not contain PEFT-specific names
        weight_names = captured_kwargs["weight_update_info"]["names"]
        for name in weight_names:
            assert "lora" not in name, f"Adapter weight {name} leaked into weight metadata"
            assert "original_module" not in name, f"Original module weight {name} leaked into weight metadata"
            assert "modules_to_save" not in name, f"Module-to-save prefix {name} leaked into weight metadata"
        assert "model.embed_tokens.weight" in weight_names
        assert "lm_head.weight" in weight_names

    @require_peft
    def test_peft_streaming_iter_excludes_adapters_and_modules_to_save(self):
        from peft import LoraConfig

        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        training_args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            report_to="none",
        )
        trainer = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(target_modules="all-linear", modules_to_save=["embed_tokens", "lm_head"]),
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=8),
        )

        weight_names = [name for name, _ in trainer._streaming_iter()]
        for name in weight_names:
            assert "lora" not in name, f"Adapter weight {name} leaked into streaming weights"
            assert "original_module" not in name, f"Original module weight {name} leaked into streaming weights"
            assert "modules_to_save" not in name, f"Module-to-save prefix {name} leaked into streaming weights"
        assert "model.embed_tokens.weight" in weight_names
        assert "lm_head.weight" in weight_names
