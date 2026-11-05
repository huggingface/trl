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
    --max-model-len 19456 \
    --weight-transfer-config '{"backend":"nccl"}'

CUDA_VISIBLE_DEVICES=1 vllm serve BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Math \
    --port 8001 \
    --max-model-len 19520 \
    --gpu-memory-utilization 0.42 \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1

CUDA_VISIBLE_DEVICES=1 vllm serve BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-IF \
    --port 8003 \
    --max-model-len 19520 \
    --gpu-memory-utilization 0.42 \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1

CUDA_VISIBLE_DEVICES=2 vllm serve BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Code \
    --port 8002 \
    --max-model-len 19520 \
    --gpu-memory-utilization 0.85 \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1

CUDA_VISIBLE_DEVICES=3 accelerate launch examples/async_distillation_mopd/async_distillation_mopd.py
"""

import argparse

from datasets import interleave_datasets, load_dataset

from trl.experimental.async_distillation import AsyncDistillationConfig, AsyncDistillationTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", default="BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT")
    parser.add_argument("--teacher-model-prefix", default="BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B")
    parser.add_argument("--distillation-method", choices=("token", "csd"), default="token")
    parser.add_argument("--teacher-top-k", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--max-completion-length", type=int, default=16384)
    parser.add_argument("--output-dir", default="async_distillation_mopd")
    parser.add_argument("--max-concurrent-rollouts", type=int)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--report-to", default="trackio")
    parser.add_argument("--run-name")
    parser.add_argument("--trackio-space-id")
    args = parser.parse_args()
    csd = args.distillation_method == "csd"
    dataset = load_dataset("BytedTsinghua-SIA/Open-MOPD-Data", data_files="rl_prompt_mix/train.parquet", split="train")
    # The prompts already carry each domain's system prompt, so only the domain label needs renaming
    dataset = dataset.select_columns(["prompt", "domain"]).rename_column("domain", "teacher_id").shuffle(seed=42)
    # The raw mix is about half instruction following, so sample math, code and instruction following 40/40/20
    dataset = interleave_datasets(
        [dataset.filter(lambda row, d=domain: row["teacher_id"] == d) for domain in ("math", "code", "if")],
        probabilities=[0.4, 0.4, 0.2],
        seed=42,
    )

    config = AsyncDistillationConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        max_completion_length=args.max_completion_length,
        distillation_method=args.distillation_method,
        # CSD defaults to NeMo's symmetric JSD; --beta 1 selects reverse KL on common events plus REST.
        beta=args.beta if args.beta is not None else (0.5 if csd else 1.0),
        teacher_top_k=args.teacher_top_k if args.teacher_top_k is not None else (64 if csd else 1),
        teacher_tokenizers={
            "math": f"{args.teacher_model_prefix}-RL-Math",
            "code": f"{args.teacher_model_prefix}-RL-Code",
            "if": f"{args.teacher_model_prefix}-RL-IF",
        }
        if csd
        else None,
        teacher_server_urls={
            "math": "http://localhost:8001",
            "code": "http://localhost:8002",
            "if": "http://localhost:8003",
        },
        vllm_server_base_url="http://localhost:8000",
        # 16K-token completions can take longer than the default 1800 s to generate under load
        vllm_server_timeout=3600,
        max_staleness=8,
        max_concurrent_rollouts=args.max_concurrent_rollouts
        if args.max_concurrent_rollouts is not None
        else (16 if csd else None),
        max_steps=args.max_steps,
        save_strategy="steps" if csd else "no",
        save_steps=args.save_steps,
        save_total_limit=2,
        learning_rate=1.5e-6,
        lr_scheduler_type="constant",
        report_to=args.report_to,
        trackio_space_id=args.trackio_space_id
        or ("async-distillation-mopd-csd" if csd else "async-distillation-mopd"),
        project="async-distillation-mopd-csd" if csd else "async-distillation-mopd",
        run_name=args.run_name or (f"csd-{args.student_model.split('/')[-1]}" if csd else None),
        log_completions=True,
    )
    trainer = AsyncDistillationTrainer(
        model=args.student_model,
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model()

    trainer.end()


if __name__ == "__main__":
    main()
