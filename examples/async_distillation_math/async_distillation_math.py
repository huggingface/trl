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
#     "trackio",
# ]
# ///

"""
Async on-policy distillation: a background rollout worker generates the student's own completions and scores them
against a teacher served over HTTP, decoupling generation from the gradient-update loop. Unlike
[`~trl.experimental.distillation.DistillationTrainer`], the teacher is never loaded locally — only a vLLM server URL
is needed, so the teacher can live on entirely different hardware from the student/trainer.

The teacher is a static server that is only ever scored (no weight sync); the student needs its own vLLM server that
receives live weight updates from the trainer (`--weight-transfer-config`).

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --port 8001 \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1

CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --port 8000 \
    --weight-transfer-config '{"backend":"nccl"}'

CUDA_VISIBLE_DEVICES=2 accelerate launch examples/async_distillation_math/async_distillation_math.py
"""

from datasets import load_dataset

from trl.experimental.async_distillation import AsyncDistillationConfig, AsyncDistillationTrainer


def format_sample(sample):
    return {"prompt": [{"role": "user", "content": sample["question"]}]}


def main() -> None:
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    config = AsyncDistillationConfig(
        output_dir="async_distillation_gsm8k",
        save_strategy="no",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        max_completion_length=256,
        teacher_server_urls={"default": "http://localhost:8001"},
        vllm_server_base_url="http://localhost:8000",
        max_steps=100,
        learning_rate=1e-6,
        report_to="trackio",
        trackio_space_id="async-distillation-gsm8k",
        project="async-distillation-gsm8k",
        log_completions=True,
    )
    trainer = AsyncDistillationTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        args=config,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
