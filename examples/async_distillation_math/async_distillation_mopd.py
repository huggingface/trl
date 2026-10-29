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
#     "trl[kernels]",
#     "trackio",
# ]
# ///

"""
MOPD (multi-teacher on-policy distillation): [`~trl.experimental.async_distillation.AsyncDistillationTrainer`] with
more than one entry in `teacher_server_urls`. Each training sample carries a `teacher_id` column selecting which
teacher scores it.

The experts are the ones released with [Open-MOPD](https://huggingface.co/papers/2608.19098): three SmolLM3-3B models
(math, code and instruction following), each trained with GRPO from the same mixed-domain SFT checkpoint. The student
starts from that SFT checkpoint, so this is the fusion stage of the paper's recipe. Every prompt of the released RL
prompt mix has a `domain` label, which becomes the `teacher_id` that routes it to its expert. All checkpoints share one
tokenizer and chat template, as the trainer requires.

Runs on 4 GPUs: the student's vLLM server, two teachers sharing a GPU, the third teacher, and training. Each teacher
gets a larger `--max-model-len` than the student, because scoring sends the student's whole sequence plus one token.
Start the teachers that share GPU 1 one after the other, since vLLM sizes its cache from the memory free at startup.

CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT \
    --port 8000 \
    --max-model-len 18432 \
    --weight-transfer-config '{"backend":"nccl"}'

CUDA_VISIBLE_DEVICES=1 vllm serve BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Math \
    --port 8001 \
    --max-model-len 18496 \
    --gpu-memory-utilization 0.42 \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1

CUDA_VISIBLE_DEVICES=1 vllm serve BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-IF \
    --port 8003 \
    --max-model-len 18496 \
    --gpu-memory-utilization 0.42 \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1

CUDA_VISIBLE_DEVICES=2 vllm serve BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Code \
    --port 8002 \
    --max-model-len 18496 \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1

CUDA_VISIBLE_DEVICES=3 accelerate launch examples/async_distillation_math/async_distillation_mopd.py
"""

from datasets import load_dataset

from trl.experimental.async_distillation import AsyncDistillationConfig, AsyncDistillationTrainer


def main() -> None:
    dataset = load_dataset("BytedTsinghua-SIA/Open-MOPD-Data", data_files="rl_prompt_mix/train.parquet", split="train")
    # The prompts already carry each domain's system prompt, so only the domain label needs renaming
    dataset = dataset.select_columns(["prompt", "domain"]).rename_column("domain", "teacher_id").shuffle(seed=42)

    config = AsyncDistillationConfig(
        output_dir="async_distillation_mopd",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        max_completion_length=16384,
        beta=1.0,  # the paper's fusion stage minimizes reverse KL against the routed teacher
        teacher_top_k=16,
        teacher_server_urls={
            "math": "http://localhost:8001",
            "code": "http://localhost:8002",
            "if": "http://localhost:8003",
        },
        vllm_server_base_url="http://localhost:8000",
        # 16K-token completions can take longer than the default 600 s to generate under load
        request_timeout=3600,
        max_staleness=8,
        max_steps=90,
        learning_rate=1.5e-6,
        lr_scheduler_type="constant",
        report_to="trackio",
        trackio_space_id="async-distillation-mopd",
        project="async-distillation-mopd",
        log_completions=True,
    )
    trainer = AsyncDistillationTrainer(
        model="BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT",
        args=config,
        train_dataset=dataset,
    )
    trainer.train()

    trainer.end()


if __name__ == "__main__":
    main()
