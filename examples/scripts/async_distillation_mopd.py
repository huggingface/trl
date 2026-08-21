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
MOPD (multi-teacher on-policy distillation): [`~trl.experimental.async_distillation.AsyncDistillationTrainer`] with
more than one entry in `teacher_server_urls`. Each training sample carries a `teacher_id` column selecting which
teacher scores it — here, math prompts (GSM8K) route to a math teacher and code prompts
(`iamtarun/python_code_instructions_18k_alpaca`) route to a code teacher, each served independently.

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --port 8001 \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1

CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --port 8002 \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1

CUDA_VISIBLE_DEVICES=2 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --port 8000 \
    --weight-transfer-config '{"backend":"nccl"}'

CUDA_VISIBLE_DEVICES=3 accelerate launch examples/scripts/async_distillation_mopd.py
"""

from datasets import concatenate_datasets, load_dataset

from trl.experimental.async_distillation import AsyncDistillationConfig, AsyncDistillationTrainer


def format_math(sample):
    return {"prompt": [{"role": "user", "content": sample["question"]}], "teacher_id": "math"}


def format_code(sample):
    content = sample["instruction"]
    if sample["input"].strip():
        content = f"{content}\n\n{sample['input']}"
    return {"prompt": [{"role": "user", "content": content}], "teacher_id": "code"}


def main() -> None:
    math_dataset = load_dataset("openai/gsm8k", "main", split="train")
    math_dataset = math_dataset.map(format_math, remove_columns=math_dataset.column_names)

    code_dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    code_dataset = code_dataset.map(format_code, remove_columns=code_dataset.column_names)

    dataset = concatenate_datasets([math_dataset, code_dataset]).shuffle(seed=42)

    config = AsyncDistillationConfig(
        output_dir="async_distillation_mopd",
        save_strategy="no",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        max_completion_length=256,
        beta=1.0,  # the paper's Stage 3 minimizes reverse KL against the routed teacher
        teacher_server_urls={
            "math": "http://localhost:8001",
            "code": "http://localhost:8002",
        },
        vllm_server_base_url="http://localhost:8000",
        max_steps=100,
        learning_rate=1e-6,
        report_to="trackio",
        trackio_space_id="async-distillation-mopd",
        project="async-distillation-mopd",
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
