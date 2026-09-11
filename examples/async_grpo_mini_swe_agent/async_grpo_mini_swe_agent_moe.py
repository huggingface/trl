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
#     "peft",
#     "trackio",
#     "datasets",
#     "openai",
#     "mini-swe-agent",
#     "huggingface_hub>=1.22",
#     "openenv @ git+https://github.com/huggingface/OpenEnv.git",
#     "swegym @ git+https://github.com/SWE-Gym/SWE-Bench-Package.git",
# ]
# ///

"""The mini-swe-agent example on a mixture-of-experts policy.

Same agent, sandboxes, verifier and rollout worker as `async_grpo_mini_swe_agent.py` (imported from it). What
changes for a MoE checkpoint the size of `Qwen/Qwen3-Coder-30B-A3B-Instruct`: the base weights are loaded in
bfloat16 (under the trainer's float32 default every rank materialized 122 GB on the CPU before FSDP sharded it, and
the Job was killed on host memory); the experts are fused 3D tensors, so the LoRA targets the attention projections
only; and the tokenizer gets a plain ChatML response template, since the rollout worker does not recognize
Qwen3-Coder's chat template.

```sh
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --host 0.0.0.0 --port 8000 --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --logprobs-mode processed_logprobs --generation-config vllm \
    --weight-transfer-config '{"backend":"nccl"}' \
    --enable-lora --max-lora-rank 16 --max-loras 6

CUDA_VISIBLE_DEVICES=1,2,3,4 HF_TOKEN=... accelerate launch --config_file examples/async_grpo_mini_swe_agent/fsdp2.yaml --num_processes 4 \
    examples/async_grpo_mini_swe_agent/async_grpo_mini_swe_agent_moe.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct
```
"""

from __future__ import annotations

import argparse
import json
import logging

from async_grpo_mini_swe_agent import MiniSWEAgentSessionFactory, build_dataset
from peft import LoraConfig
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import HarnessRolloutWorker


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument("--output-dir", default="async_grpo_mini_swe_agent")
    p.add_argument("--n-prompts", type=int, default=256)
    p.add_argument("--instances-file", default=None)  # JSON list of SWE-Gym instance ids to train on
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=32)  # concurrent rollouts, one sandbox each
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-turn-tokens", type=int, default=4096)  # per model call; the context is bounded by vLLM
    p.add_argument(
        "--token-budget", type=int, default=None
    )  # tokens per training micro-batch row; defaults to vLLM's max_model_len
    p.add_argument("--enable-thinking", action="store_true")  # Qwen3 hybrid models think before every command
    p.add_argument("--max-staleness", type=int, default=4)
    p.add_argument("--weight-sync-steps", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sandbox-flavor", default="cpu-basic")
    p.add_argument("--step-limit", type=int, default=250)  # NeMo Gym's mini_swe_agent step_limit
    p.add_argument("--step-timeout", type=int, default=300)  # seconds per command
    p.add_argument("--agent-timeout", type=int, default=3600)  # wall clock per rollout
    p.add_argument("--eval-timeout", type=int, default=900)  # seconds for the held-out tests
    p.add_argument("--project", default="async-grpo-mini-swe-agent-moe")
    p.add_argument("--run-name", default=None)
    p.add_argument("--trackio-space-id", default=None)  # defaults to the project: every run lands in a Space
    args = p.parse_args()

    # trl logs through the stdlib root logger, which a bare script leaves without a handler.
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.getLogger("trl").addHandler(handler)
    logging.getLogger("trl").setLevel(logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # The rollout worker wants a response template on the tokenizer and does not recognize Qwen3-Coder's chat
    # template; the agent owns the loop here, so a plain ChatML assistant turn is all it needs.
    tokenizer.response_template = {
        "defaults": {"role": "assistant"},
        "start_anchor": "<|im_start|>assistant\n",
        "fields": {"content": {"close_pattern": r"<\|im_end\|>\s*|$", "content": "text"}},
    }
    instance_ids = json.load(open(args.instances_file)) if args.instances_file else None
    dataset, instances = build_dataset(args.n_prompts, args.seed, instance_ids)

    factory = MiniSWEAgentSessionFactory(
        instances=instances,
        vllm_url=args.vllm_url,
        model=args.model,
        flavor=args.sandbox_flavor,
        chat_template_kwargs={} if args.enable_thinking else {"enable_thinking": False},
        temperature=args.temperature,
        max_turn_tokens=args.max_turn_tokens,
        step_limit=args.step_limit,
        step_timeout=args.step_timeout,
        agent_timeout=args.agent_timeout,
        eval_timeout=args.eval_timeout,
    )

    config = AsyncGRPOConfig(
        output_dir=args.output_dir,
        bf16=True,
        dtype="bfloat16",
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_turn_tokens,
        token_budget=args.token_budget,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_staleness=args.max_staleness,
        weight_sync_steps=args.weight_sync_steps,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        vllm_server_base_url=args.vllm_url,
        report_to="trackio",
        project=args.project,
        run_name=args.run_name,
        trackio_space_id=args.trackio_space_id or args.project,
    )
    worker = HarnessRolloutWorker(
        harness_session_factory=factory,
        harness_adapter=None,  # loop-owning: mini-swe-agent runs its own loop; TRL reads the recorded turns
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[],  # the reward is the verifier's, through the session
        processing_class=tokenizer,
        num_generations=args.num_generations,
        max_inflight_tasks=args.max_inflight,
        vllm_server_url=args.vllm_url,
        max_tokens=args.max_turn_tokens,
        temperature=args.temperature,
        log_completions=True,
        num_completions_to_print=1,
    )
    trainer = AsyncGRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_worker=worker,
        # Plain LoRA on the linear layers: anything else (`modules_to_save`, DoRA, trained biases) cannot be served
        # as a vLLM adapter and would fall back to syncing the merged weights.
        peft_config=LoraConfig(
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            target_modules=(
                args.lora_target_modules
                if args.lora_target_modules == "all-linear"
                else args.lora_target_modules.split(",")
            ),
        ),
    )

    trainer.train()


if __name__ == "__main__":
    main()
