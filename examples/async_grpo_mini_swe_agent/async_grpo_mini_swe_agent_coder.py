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
#     "datasets",
#     "openai",
#     "mini-swe-agent",
#     "huggingface_hub>=1.22",
#     "openenv @ git+https://github.com/huggingface/OpenEnv.git",
#     "swegym @ git+https://github.com/SWE-Gym/SWE-Bench-Package.git",
#     "transformers @ git+https://github.com/huggingface/transformers.git@ep-fsdp-2d-mesh",
# ]
# ///

"""The mini-swe-agent example on a mixture-of-experts policy, trained in full over an expert-parallel mesh.

Same agent, sandboxes, verifier and rollout worker as `async_grpo_mini_swe_agent.py` (imported from it). Three
things differ from both the dense example and the LoRA MoE one (`async_grpo_mini_swe_agent_moe.py`):

- **Every weight trains.** A 30B MoE holds 95% of its parameters in the experts, which are fused 3D tensors that a
  LoRA cannot target, so a LoRA run trains the attention projections of a model that is almost entirely experts.
- **The experts are sharded across ranks, not replicated.** `DistributedConfig(tp_size=N, enable_expert_parallel=
  True)` gives each rank `128 / N` experts, which is what makes the full fine-tune fit at all, and
  `experts_dispatch="all-to-all"` routes each token to the rank that owns its expert instead of having every rank
  recompute the whole batch. Needs `huggingface/transformers#48204` (branch `ep-fsdp-2d-mesh`).
- **The updated weights reach vLLM over NCCL.** There is no adapter to ship, so the two Jobs join one network group
  and the merged weights stream between the pods. vLLM's Qwen3-MoE loader takes the fused expert tensors as they
  are, so nothing is renamed or split on the way out.

A model sharded at load time cannot checkpoint its optimizer state, so only the weights are saved.

```sh
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --logprobs-mode processed_logprobs --generation-config vllm \
    --weight-transfer-config '{"backend":"nccl"}'

CUDA_VISIBLE_DEVICES=1,2,3,4 HF_TOKEN=... torchrun --nproc_per_node 4 \
    examples/async_grpo_mini_swe_agent/async_grpo_mini_swe_agent_coder.py --tp-size 4
```
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from async_grpo_mini_swe_agent import MiniSWEAgentSessionFactory, build_dataset
from transformers import AutoTokenizer
from transformers.distributed import DistributedConfig

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import HarnessRolloutWorker


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument("--output-dir", default="async_grpo_mini_swe_agent")
    p.add_argument("--n-prompts", type=int, default=512)
    p.add_argument("--instances-file", default=None)  # JSON list of SWE-Gym instance ids to train on
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=32)  # concurrent rollouts, one sandbox each
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-6)  # full fine-tuning, not LoRA
    p.add_argument("--tp-size", type=int, default=0)  # expert-parallel ranks; 0 keeps the plain FSDP2 path
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-turn-tokens", type=int, default=4096)  # per model call; the context is bounded by vLLM
    p.add_argument(
        "--token-budget", type=int, default=None
    )  # tokens per training micro-batch row; defaults to vLLM's max_model_len
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
    p.add_argument("--project", default="async-grpo-mini-swe-agent-coder")
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
        chat_template_kwargs={"enable_thinking": False},
        temperature=args.temperature,
        max_turn_tokens=args.max_turn_tokens,
        step_limit=args.step_limit,
        step_timeout=args.step_timeout,
        agent_timeout=args.agent_timeout,
        eval_timeout=args.eval_timeout,
    )

    # A 2-D (fsdp, tp) mesh: the experts are expert-parallel across `tp` and route each token to the rank that owns
    # its expert, FSDP2 shards the rest across the whole mesh. `tp_size` is the expert-parallel size and
    # `tp_size * fsdp_size` must be the world size.
    #
    # The dispatch is what keeps the trainer's one-row-per-rank batches correct: every rank routes its own tokens, so
    # the whole world stays data-parallel and a rank's row is its own, as on the plain FSDP2 path. Under the
    # `"all-reduce"` default the ranks of an expert-parallel group instead recompute one batch together and would
    # have to be handed the same row, which this trainer does not do.
    distributed_config = (
        DistributedConfig(
            tp_size=args.tp_size,
            fsdp_size=int(os.environ["WORLD_SIZE"]) // args.tp_size,
            enable_expert_parallel=True,
            experts_dispatch="all-to-all",
        )
        if args.tp_size
        else None
    )

    config = AsyncGRPOConfig(
        output_dir=args.output_dir,
        bf16=True,
        dtype="bfloat16",
        model_init_kwargs={"distributed_config": distributed_config} if distributed_config else None,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_turn_tokens,
        token_budget=args.token_budget,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Adam moments in the model's own dtype: fp32 moments are 16 bytes per parameter, which a 30B MoE cannot
        # fit on four GPUs, and the 8-bit optimizers take a raw pointer that a sharded parameter cannot give.
        optim="adamw_torch",
        max_staleness=args.max_staleness,
        weight_sync_steps=args.weight_sync_steps,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        # A model sharded at load time reloads its optimizer state per expert, which the sharded checkpoint cannot
        # express, so the Trainer saves the weights alone and a restart begins a fresh run.
        save_only_model=True,
        vllm_server_base_url=args.vllm_url,
        report_to="trackio",
        project=args.project,
        run_name=args.run_name,
        trackio_space_id=args.trackio_space_id or args.project,
    )
    # `AsyncGRPOConfig` pins accelerate's `dispatch_batches` on, and that explicit flag is the one thing the Trainer
    # reads to refuse expert-parallel token dispatch. Unsetting it changes no behaviour: accelerate then resolves it
    # from the dataset, an `IterableDataset` on every rank here, so rank 0 still drives the rollout queue and
    # broadcasts one row per rank, which is the layout token dispatch wants.
    if args.tp_size:
        config.accelerator_config.dispatch_batches = None

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
    )

    trainer.train()


if __name__ == "__main__":
    main()
