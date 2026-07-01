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
# dependencies = ["trl[vllm,openreward]"]
# ///

"""GRPO training against the SETA ORS environment.

Defaults target ``Eigent/SETA`` on the openreward.ai catalog (requires
``OPENREWARD_API_KEY``). Pass ``--target https://...hf.space`` to point
at a self-hosted Space.

Usage (colocate vLLM, single-node):

```sh
accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 4 \
    examples/scripts/openreward/seta.py \
    --vllm-mode colocate
```

Usage (server vLLM, single-node 2+2 GPU split):

```sh
# Terminal 1 — vLLM
CUDA_VISIBLE_DEVICES=2,3 trl vllm-serve --model Qwen/Qwen3-4B \
    --tensor-parallel-size 2 --port 8000

# Terminal 2 — training
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 2 \
    examples/scripts/openreward/seta.py \
    --vllm-mode server --vllm-server-base-url http://localhost:8000
```
"""

import argparse

from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openreward import OpenRewardSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training against the SETA ORS environment.")

    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--target",
        type=str,
        default="Eigent/SETA",
        help="ORS env target — either a catalog name (e.g. 'Eigent/SETA') or a URL "
        "(e.g. 'https://you-seta.hf.space').",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-tasks", type=int, default=64)

    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=20)

    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate")
    parser.add_argument("--vllm-server-base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--report-to", type=str, default="none")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # One spec object — fans out into TRL's three slots.
    spec = OpenRewardSpec(args.target, num_tasks=args.num_tasks, split=args.split)

    config_kwargs: dict = dict(
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        report_to=[s.strip() for s in args.report_to.split(",") if s.strip() and s.strip() != "none"] or "none",
    )
    if args.output_dir:
        config_kwargs["output_dir"] = args.output_dir
    if args.vllm_mode == "colocate":
        config_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
    else:
        config_kwargs["vllm_server_base_url"] = args.vllm_server_base_url

    trainer = GRPOTrainer(
        model=args.model,
        args=GRPOConfig(**config_kwargs),
        train_dataset=spec.train_dataset,
        environment_factory=spec.environment_factory,
        reward_funcs=spec.reward_funcs,
    )
    trainer.train()


if __name__ == "__main__":
    main()
