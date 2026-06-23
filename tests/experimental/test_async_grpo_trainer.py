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
import json
import os
import queue
import subprocess
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample

from ..testing_utils import TrlTestCase, require_torch_multi_accelerator


ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).parent
_FSDP2_WORKER = _HERE / "_async_grpo_fsdp2_worker.py"
_FSDP2_CONFIG = _HERE / "data" / "accelerate_configs" / "fsdp2_reshard.yaml"
_FSDP2_RESULT_PREFIX = "ASYNC_GRPO_FSDP2_RESULT"


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

    @require_torch_multi_accelerator
    def test_train_fsdp2(self):
        # Functional smoke: AsyncGRPOTrainer trains under a 2-process FSDP2 group. This exercises the
        # `patch_chunked_lm_head` chunked-logprob path on FSDP2-sharded parameters end-to-end and confirms
        # the optimizer actually updates them. The worker uses an in-process stub rollout worker (no vLLM
        # server / NCCL weight transfer), so the only distributed surface is the FSDP2 parameter lifecycle.
        #
        # (This is NOT a #6077 all-gather microbenchmark: under FSDP2 the per-parameter gathers are driven
        # by autograd unshard hooks, not by `DTensor.full_tensor`, and the trainer's weight-sync path calls
        # `full_tensor` on every parameter every step — so counting `full_tensor` cannot isolate the chunk
        # path. The #6077 question is settled by static analysis instead: `patch_chunked_lm_head` has no
        # `torch.utils.checkpoint` recompute, so the per-chunk re-gather that PR #6077 fixed for SFT's
        # `chunked_nll` is structurally absent here.)
        #
        # Pin the repo root onto PYTHONPATH for the child: `accelerate launch` re-execs each rank via
        # torch.distributed.elastic, which sets sys.path[0] to the launched script's directory
        # (tests/experimental/), not cwd. Without this, a non-editable `trl` already in site-packages
        # would shadow the working tree and the test would exercise the wrong code.
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        result = subprocess.run(
            ["accelerate", "launch", "--config_file", str(_FSDP2_CONFIG), str(_FSDP2_WORKER)],
            env=env,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"FSDP2 worker failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

        result_lines = [ln for ln in result.stdout.splitlines() if ln.startswith(_FSDP2_RESULT_PREFIX)]
        assert len(result_lines) == 1, f"expected exactly one result line, got {result_lines}\n{result.stdout}"
        measured = json.loads(result_lines[0][len(_FSDP2_RESULT_PREFIX) :].strip())

        # Training actually ran under FSDP2, produced a finite loss, and updated the parameters.
        assert measured["steps"] >= 1, f"no training steps ran: {measured}"
        assert measured["train_loss_finite"], f"train loss not finite under FSDP2: {measured}"
        assert measured["params_changed"], f"parameters did not change under FSDP2: {measured}"
