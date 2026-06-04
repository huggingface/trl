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

"""
AsyncGRPO with delta weight sync (Transport B: HF Storage Bucket + in-place sparse apply).

Only changed bf16 weights are encoded as a sparse safetensors patch, uploaded to a bucket, and
applied in place on vLLM via PR #40096 — no full-model broadcast, no vLLM-side snapshot.

Start the vLLM server with the `delta` backend + worker extension (registers the engine) and the
`transformers` model impl (so vLLM's runtime param names match the trainer's HF names — every
param is then addressable by the in-place sparse apply, no fuse/unfuse remap needed):

# VLLM_USE_V2_MODEL_RUNNER=0 is required: the in-place sparse apply (apply_sparse_weight_patches,
# vLLM #40096) exists only on the V1 model runner. Without it the server picks V2 and every sparse
# delta update fails (the dense anchors still work, so it silently degrades to anchor-only sync).
CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 VLLM_USE_V2_MODEL_RUNNER=0 vllm serve Qwen/Qwen3-1.7B \
    --model-impl transformers \
    --worker-extension-cls trl.experimental.async_grpo.delta_engine.DeltaWorkerExtension \
    --weight-transfer-config '{"backend":"delta"}' \
    --max-model-len 2560

CUDA_VISIBLE_DEVICES=0 accelerate launch examples/scripts/async_grpo_delta.py
"""

import logging
import os

from datasets import load_dataset

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.rewards import accuracy_reward


logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("trl").setLevel(logging.INFO)


def format_sample(sample):
    return {
        "prompt": [{"role": "user", "content": sample["question"]}],
        "solution": sample["answer"].split("####")[-1].strip(),
    }


def main() -> None:
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    config = AsyncGRPOConfig(
        output_dir="./results/async_grpo_delta",
        per_device_train_batch_size=1,
        num_generations=8,
        max_completion_length=512,
        max_steps=60,
        learning_rate=1e-5,
        logging_steps=1,
        bf16=True,
        report_to="none",
        project="async_grpo_delta",
        log_completions=True,
        # Qwen3 thinking traces blow past the completion cap on GSM8K (truncated -> no answer ->
        # zero reward); disable thinking so completions are short and accuracy_reward gets signal.
        chat_template_kwargs={"enable_thinking": False},
        # --- delta weight sync (Transport B) ---
        delta_sync_enabled=True,
        delta_sync_repo_id="aminediroHF/async-grpo-delta-demo",
        delta_sync_anchor_interval=20,  # full anchor every N syncs; sparse deltas in between
        delta_sync_encoding="gap_delta",  # raw | gap_delta | nvcomp_cascaded
    )
    trainer = AsyncGRPOTrainer(
        model="Qwen/Qwen3-1.7B",
        args=config,
        train_dataset=dataset,
        reward_funcs=accuracy_reward,
    )
    trainer.train()


if __name__ == "__main__":
    main()
