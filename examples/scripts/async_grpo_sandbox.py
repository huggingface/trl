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

# /// script
# dependencies = [
#     "trl",
#     "huggingface_hub>=1.22.0",
#     "trackio",
# ]
# ///

"""
Async GRPO on GSM8K where the model solves each problem with a code sandbox.

Each rollout gets a `SandboxEnvironment`, which exposes a `run` tool: an isolated Hugging Face Sandbox the model uses
to execute Python and compute the answer (instead of doing arithmetic in its head). Reward is exact-match on the final
boxed number — the sandbox is not needed at reward time, so this stays a pure completion check.

Running the sandbox requires a Hugging Face token with Jobs access (`hf auth login`); each rollout spins up a small
cloud VM.

Give the vLLM server GPU headroom for the NCCL weight-transfer buffers: at the default `--gpu-memory-utilization 0.9`
the KV cache leaves no room for them and the first weight sync OOMs. `--gpu-memory-utilization 0.7` plus
`expandable_segments:True` (reduces fragmentation) fixes it. Restart the server between runs — an interrupted trainer
leaves the server holding fragmented transfer buffers.

Give `--max-model-len` room for multi-turn tool traces (prompt grows with every tool call + its output). Too small
and per-turn completion requests overflow the context and vLLM returns 400; the KV cache is tiny here, so this is
nearly free.

CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    vllm serve Qwen/Qwen3-4B \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.7 \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}'

CUDA_VISIBLE_DEVICES=0 accelerate launch examples/scripts/async_grpo_sandbox.py
"""

import re

from datasets import load_dataset

from trl.environments import SandboxEnvironment
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer


SYSTEM_PROMPT = (
    "Solve the math problem. You have a `run` tool that executes shell commands in a Python sandbox — use it to run "
    'Python for any calculation (e.g. `python3 -c "print(17 * 23)"`) rather than computing in your head. When you are '
    "done, give the final answer on its own line as \\boxed{...}."
)


def format_sample(sample):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["question"]},
        ],
        "solution": sample["answer"].split("####")[-1].strip().replace(",", ""),
    }


def boxed_match_reward(completions, solution, **kwargs):
    """1.0 when the final \\boxed{...} number matches the gold answer, else 0.0. No external dependency."""
    rewards = []
    for completion, gold in zip(completions, solution, strict=True):
        # A tool-use completion is a list of messages (assistant / tool turns); scan them all for the last box.
        text = "\n".join(m["content"] for m in completion if isinstance(m.get("content"), str))
        boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
        pred = boxes[-1].strip().replace(",", "") if boxes else None
        try:
            rewards.append(float(pred is not None and float(pred) == float(gold)))
        except ValueError:  # non-numeric answer -> no match
            rewards.append(0.0)
    return rewards


def main() -> None:
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    config = AsyncGRPOConfig(
        output_dir="async_grpo_sandbox_gsm8k",
        save_strategy="no",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        max_completion_length=1024,  # per-turn budget; keep it well under max-model-len so tool turns don't overflow
        max_tool_calling_iterations=6,  # allow a few sandbox round-trips per rollout
        max_inflight_tasks=16,  # each in-flight rollout holds its own sandbox VM; cap concurrent VMs (and cost)
        chat_template_kwargs={"enable_thinking": False},
        max_steps=200,
        learning_rate=1e-5,
        report_to="trackio",
        trackio_space_id="async-grpo-sandbox-gsm8k",
        project="async-grpo-sandbox-gsm8k",
        log_completions=True,
        num_completions_to_print=1,
    )
    trainer = AsyncGRPOTrainer(
        model="Qwen/Qwen3-4B",
        args=config,
        train_dataset=dataset,
        reward_funcs=boxed_match_reward,
        environment_factory=SandboxEnvironment,  # gives each rollout a `run` tool backed by a cloud sandbox
    )
    trainer.train()


if __name__ == "__main__":
    main()
