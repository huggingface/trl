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
#     "peft",
#     "kernels",
#     "openenv @git+https://github.com/huggingface/OpenEnv.git@86a180ede21e044f7929b9a7783ad83aa67d83a3",
# ]
# ///

"""AsyncGRPO on Harbor tasks, with an off-the-shelf coding agent, served through OpenEnv.

A Harbor task contains a container image, instruction, and held-out verifier. This example uses
`mini-swe-agent` in an E2B sandbox:

    mini-swe-agent solves a Harbor task in a sandbox
      -> every model call it makes goes through the OpenEnv server's capture proxy to your vLLM
      -> the proxy records exact token ids and the sampling distribution's logprobs
      -> AsyncGRPO trains on them and syncs new weights back into that same vLLM

The agent owns the tool loop; TRL consumes OpenEnv's captured token IDs, logprobs, and masks.
Rewritten histories fork into separate training rows to preserve eligible tokens. See README.md
for the contract, dependency pin, and memory/weighting limitations.

Select another qualified agent with `--harness`. Rollouts and weight updates must use the same vLLM
instance and full-vocabulary sampling policy. Weight updates pause active inference requests without waiting for them to finish.

The reward adds `0.3 * tool_efficiency` only when correctness is at least 1.0. Use `--reward-key` to
select a component when the verifier returns a reward dictionary.

Requirements:
  - A running OpenEnv Harbor server, which owns the dataset and the sandbox templates:
        MAX_CONCURRENT_ENVS=9 openenv harbor serve --dataset <hf-dataset> --port 8200 --capture-port 8300 --expose gradio
  - A sandbox backend credential for the server's environment, e.g. `E2B_API_KEY`.
  - An OpenAI-compatible vLLM server (below) reachable at `--vllm-url`.
  - The pinned OpenEnv checkout and `PYTHONPATH` setup in this example's README.md.

Run (2 GPUs: vLLM on one, trainer on the other):

```sh
# Terminal 1: serve the policy with token capture and weight updates.
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3.5-2B \
    --host 0.0.0.0 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --logprobs-mode processed_logprobs \
    --return-tokens-as-token-ids \
    --weight-transfer-config '{"backend":"nccl"}'

# Terminal 2 - train.
CUDA_VISIBLE_DEVICES=1 python examples/async_grpo_harbor/async_grpo_harbor.py \
    --server http://localhost:8200 \
    --vllm-url http://localhost:8000 \
    --model Qwen/Qwen3.5-2B \
    --split <hf-dataset> \
    --max-steps 20
```
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
import uuid
from functools import partial

from datasets import Dataset
from harbor_env.harness import HarborSessionFactory
from peft import LoraConfig
from transformers import AutoTokenizer

from kernels import get_kernel
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import HarnessRolloutOutcome, HarnessRolloutWorker
from trl.experimental.async_grpo.vllm_client import VLLMClient


logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

W_TOOL_EFFICIENCY = 0.3
TOOL_BUDGET = 15.0


def tool_efficiency(n_tool_calls: int | None) -> float | None:
    """`clip(1 - n/TOOL_BUDGET, 0, 1)`, or `None` when the tool count is unknown."""
    if n_tool_calls is None or TOOL_BUDGET <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - n_tool_calls / TOOL_BUDGET))


def harbor_reward(outcome: HarnessRolloutOutcome) -> float | None:
    """Reward for one rollout, or `None` when it is unscorable.

    Args:
        outcome (`HarnessRolloutOutcome`):
            The verifier's reward, transcript, tool-call count, and
            whether the agent ran out of wall clock.

    Returns:
        `float` or `None`: Unscored rollouts are excluded from the group baseline.
    """
    correctness = outcome.env_reward
    if correctness is None:
        logger.warning("verifier did not run (tool_calls=%d); rollout unscorable", outcome.tool_call_count)
        return None

    if outcome.timed_out:
        logger.warning("agent timed out; keeping the verifier's score of %.3f on the partial work", correctness)

    correctness = float(correctness)
    reward = correctness

    eff = tool_efficiency(outcome.tool_call_count)
    if eff is not None and correctness >= 1.0:
        reward += W_TOOL_EFFICIENCY * eff

    return reward


class IPCWeightTransfer:
    """Push weights to a vLLM server on the trainer's own GPU through CUDA IPC (NCCL refuses two ranks per device).

    Implements the `weight_transfer` protocol of `AsyncGRPOTrainer`, which still merges a LoRA adapter before sending.
    The server needs `--weight-transfer-config '{"backend":"ipc"}'` and `VLLM_ALLOW_INSECURE_SERIALIZATION=1`: the
    IPC handles travel pickled over HTTP.
    """

    def __init__(self, vllm_url: str):
        self.vllm_url = vllm_url
        self.vllm = VLLMClient(vllm_url)

    def init_weight_transfer(self) -> None:
        self.vllm.wait_for_server_ready()
        self.vllm.init_weight_transfer_engine({})

    def pause(self) -> None:
        self.vllm.pause()

    def send_weights(self, iterator) -> None:
        from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerSendWeightsArgs, IPCWeightTransferEngine

        self.vllm.start_weight_update()
        try:
            # Unpacked: vLLM reads the parameters' own storage, so a sync allocates nothing. The packed mode copies
            # into fresh buffers every sync, and CUDA IPC keeps them alive until the server releases them; a packed
            # run ran out of memory after ~80 steps with the trainer holding ~90 GB.
            IPCWeightTransferEngine.trainer_send_weights(
                iterator, IPCTrainerSendWeightsArgs(send_mode="http", url=self.vllm_url, packed=False)
            )
        finally:
            self.vllm.finish_weight_update()

    def resume(self) -> None:
        self.vllm.resume()

    def destroy(self) -> None:
        pass


def task_indices(spec: str) -> list[int] | None:
    """Read comma-separated task indices, directly or from `@path` (safe for Slurm exports)."""
    spec = (spec or "").strip()
    if not spec:
        return None
    if spec.startswith("@"):
        spec = pathlib.Path(spec[1:]).read_text(encoding="utf-8")
    return [int(x) for x in spec.replace("\n", ",").split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="http://127.0.0.1:8200", help="a running `openenv harbor serve`")
    p.add_argument("--vllm-url", required=True, help="the engine AsyncGRPO generates from and syncs weights into")
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--split", required=True, help="the Harbor task dataset the server was started with")
    p.add_argument("--harness", default="mini-swe-agent", help="any harness the server reports; see the docstring")
    p.add_argument("--sandbox", default="e2b")
    p.add_argument("--reward-key", default="")
    p.add_argument("--n-tasks", type=int, default=32)
    p.add_argument("--task-indices", default="", help="comma-separated indices, or @path to a file of them")
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=8)
    p.add_argument("--max-completion-length", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-staleness", type=int, default=4)
    # Sandbox setup has its own timeouts; this bounds the agent run.
    p.add_argument("--agent-timeout", type=float, default=300.0)
    # Bound model calls; rewritten histories can still produce multiple training rows.
    p.add_argument("--agent-step-limit", type=int, default=12)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    # LoRA keeps vLLM on the merged-weight NCCL sync: the agents request the base model name, never an adapter name.
    p.add_argument("--lora-r", type=int, default=0, help="train a LoRA adapter of this rank; 0 trains all weights")
    p.add_argument("--lora-alpha", type=int, default=None, help="defaults to 2 * --lora-r")
    # `ipc` lets vLLM and the trainer share one GPU (see IPCWeightTransfer); `nccl` needs one GPU each.
    p.add_argument("--weight-transfer", choices=("nccl", "ipc"), default="nccl")
    p.add_argument("--token-budget", type=int, default=None, help="max tokens per packed row; bounds trainer memory")
    # flash-attn3 (the default) needs Hopper; pass kernels-community/flash-attn2 on A10G, L40S or A100.
    p.add_argument("--attn-implementation", default=None)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=3, help="0 keeps every checkpoint")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--project", default="async-grpo-harbor")
    p.add_argument("--trackio-space-id", default=None, help="host the trackio dashboard on a HF Space")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if "Qwen3.5" in args.model:
        # Qwen3.5's linear-attention layers keep packed samples apart only with the `causal_conv1d` package, for their
        # short convolutions. PyPI ships no wheel for it, so serve the prebuilt Hub kernel under that name before
        # transformers imports it with the model.
        sys.modules["causal_conv1d"] = get_kernel("kernels-community/causal-conv1d", version=1)
    stamp = os.environ.get("SLURM_JOB_ID") or os.environ.get("HF_JOB_ID") or os.environ.get("JOB_ID") or "local"
    stamp = f"{stamp}-{uuid.uuid4().hex[:8]}"
    run_name = args.run_name or f"{args.model.split('/')[-1]}-{args.harness}-{args.max_steps}steps-{stamp}"
    output_dir = args.output_dir or f"runs/async_grpo_harbor/{run_name}"

    factory = partial(
        HarborSessionFactory,
        args.server,
        split=args.split,
        harness=args.harness,
        sandbox=args.sandbox,
        llm_url=args.vllm_url,
        model=args.model,
        agent_timeout_sec=args.agent_timeout,
        agent_step_limit=args.agent_step_limit,
        reward_key=args.reward_key,
        num_tasks=args.n_tasks,
        indices=task_indices(args.task_indices),
    )
    metadata_factory = factory()
    try:
        dataset = Dataset.from_list(metadata_factory.prompt_rows())
    finally:
        # The pinned OpenEnv factory has no public close method.
        if metadata_factory._env is not None:
            metadata_factory._env.close()

    print(f"server    {args.server}")
    print(f"vllm      {args.vllm_url}   model {args.model}")
    print(f"rollouts  {args.harness} on {args.sandbox}, {args.num_generations}x{args.max_inflight}")
    print(f"tasks     {len(dataset)} from {args.split}")
    print(f"output    {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    config = AsyncGRPOConfig(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit or None,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        top_p=1.0,
        top_k=-1,
        max_staleness=args.max_staleness,
        token_budget=args.token_budget,
        model_init_kwargs={"attn_implementation": args.attn_implementation} if args.attn_implementation else None,
        vllm_server_base_url=args.vllm_url,
        optim="adamw_torch",
        dtype="bfloat16",
        bf16=True,
        gradient_checkpointing=True,
        # Support layers whose inputs are passed as keyword arguments.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="trackio",
        project=args.project,
        trackio_space_id=args.trackio_space_id,
        run_name=run_name,
        log_completions=True,
        logging_steps=1,
        seed=0,
    )

    eos_token_ids = tokenizer.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    worker = HarnessRolloutWorker(
        harness_session_factory=factory,
        harness_adapter=None,
        rollout_reward_fn=harbor_reward,
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[],  # the reward is the task's own verifier, via `rollout_reward_fn`
        processing_class=tokenizer,
        chat_template_kwargs={"enable_thinking": False},
        num_generations=args.num_generations,
        max_inflight_tasks=args.max_inflight,
        vllm_server_url=args.vllm_url,
        max_tokens=args.max_completion_length,
        temperature=args.temperature,
        top_p=1.0,
        top_k=-1,
        log_completions=True,
        num_completions_to_print=2,
        eos_token_ids=eos_token_ids,
    )

    peft_config = None
    if args.lora_r:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha or 2 * args.lora_r,
            target_modules="all-linear",
            # The agents send text only, so adapters on a vision tower would never train.
            exclude_modules=r".*visual.*",
            task_type="CAUSAL_LM",
        )

    trainer = AsyncGRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_worker=worker,
        peft_config=peft_config,
        weight_transfer=IPCWeightTransfer(args.vllm_url) if args.weight_transfer == "ipc" else None,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
