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
#     "openenv-harbor-env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/harbor_env",
# ]
# ///

"""AsyncGRPO on Harbor tasks, with an off-the-shelf coding agent, served through OpenEnv.

A Harbor task is a container image, an instruction, and a held-out verifier. This example trains against
one through a *real* coding agent — `mini-swe-agent` — running in an E2B sandbox:

    mini-swe-agent solves a Harbor task in a sandbox
      -> every model call it makes goes through the OpenEnv server's capture proxy to your vLLM
      -> the proxy records exact token ids and the sampling distribution's logprobs
      -> AsyncGRPO trains on them and syncs new weights back into that same vLLM

The agent owns its own loop. TRL never calls `step()`; it stands up an endpoint, lets the agent drive, and
reads back what happened. That is what makes any installed harness trainable without reimplementing it.

Everything Harbor-specific lives in `harbor_env.harness` (OpenEnv). Nothing is added to TRL, so the file
below is the whole integration, and every training-facing object is module-level (picklable) so the
rollout worker can pickle the factory and reward into its spawned child.

WHY `mini-swe-agent`. Measured, not chosen by taste. Across a 15-harness sweep on the same 50 tasks
(`Qwen3.5-2B`, k=4) it was both the most accurate and the most turn-efficient — and, decisively for
training, its prompt re-render is byte-exact against the engine's own `prompt_token_ids`. TRL re-renders
each prompt locally because `TraceEntry` carries no prompt ids, and for three of the twelve harnesses
measured that re-render drifts (`claude-code` +2 tokens, `gemini-cli` +2, `kimi-cli` -10 per tool call).
A two-token drift is invisible for eval and forks the trajectory *every turn* when training. It is also
the only harness that can express a step limit, which matters below.

WHAT MAKES THE ROLLOUTS ON-POLICY. The agent's calls and the trainer's weight updates go to the SAME
vLLM. The server is pointed at that engine per rollout, so changing engines needs no server restart, and
the tier is decided by probing the endpoint: token ids plus processed logprobs mean `train`; anything
less means `eval`, and the session yields no trainable turns rather than rows of zeros.

THE REWARD. `harbor_reward` below is `correctness + 0.3 * tool_efficiency`, with efficiency gated on
correctness — ungated, the cheapest way to look efficient is to do nothing. Suites that emit a reward
dict rather than a single scalar can name a component with `--reward-key` and shape it from there.

Requirements:
  - A running OpenEnv Harbor server, which owns the dataset and the sandbox templates:
        openenv harbor serve --dataset <hf-dataset> --port 8200 --capture-port 8300 --expose gradio
  - A sandbox backend credential for the server's environment, e.g. `E2B_API_KEY`.
  - An OpenAI-compatible vLLM server (below) reachable at `--vllm-url`.
  - `pip install git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/harbor_env`

Run (2 GPUs: vLLM on one, trainer on the other):

```sh
# Terminal 1 - serve the policy. Tool calling, token ids, processed logprobs and NCCL weight sync are
# all required: without the token ids and logprobs the proxy grades every rollout `eval` and nothing is
# trainable.
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

from datasets import Dataset
from harbor_env.harness import HarborSessionFactory
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import HarnessRolloutOutcome, HarnessRolloutWorker, has_tool_call


logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Weight on the efficiency term, and the tool-call budget it is measured against. The budget is a
# property of the task family, not of the model: on data-analysis tasks a competent rollout inspects the
# data in well under 15 calls.
W_TOOL_EFFICIENCY = float(os.environ.get("REWARD_W_TOOL_EFFICIENCY", "0.3"))
TOOL_BUDGET = float(os.environ.get("TOOL_BUDGET", "15"))


def tool_efficiency(n_tool_calls: int | None) -> float | None:
    """`clip(1 - n/TOOL_BUDGET, 0, 1)`, or `None` when the tool count is unknown."""
    if n_tool_calls is None or TOOL_BUDGET <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - n_tool_calls / TOOL_BUDGET))


def harbor_reward(outcome: HarnessRolloutOutcome) -> float | None:
    """Reward for one rollout, or `None` when it is unscorable.

    Args:
        outcome (`HarnessRolloutOutcome`):
            What the rollout produced — the verifier's reward, the transcript, the tool-call count, and
            whether the agent ran out of wall clock.

    Returns:
        `float` or `None`: `None` drops the rollout from its group baseline instead of scoring it `0`.
            That distinction matters. Scoring an unmeasured rollout `0` teaches the policy that a crashed
            sandbox is as good as a wrong answer, and poisons the baseline with a value nothing produced.
    """
    correctness = outcome.env_reward
    if correctness is None:
        logger.warning("verifier did not run (tool_calls=%d); rollout unscorable", outcome.tool_call_count)
        return None

    # A timeout is a real outcome, not a broken measurement: the agent had the wall clock and did not
    # finish. Whatever the verifier scored on the partial workspace stands.
    if outcome.timed_out:
        logger.warning("agent timed out; keeping the verifier's score of %.3f on the partial work", correctness)

    correctness = float(correctness)
    reward = correctness

    # Gated: efficiency pays only when the answer is right.
    eff = tool_efficiency(outcome.tool_call_count)
    if eff is not None and correctness >= 1.0:
        reward += W_TOOL_EFFICIENCY * eff

    return reward


def task_indices(spec: str) -> list[int] | None:
    """Task indices from a literal list, or from `@path` to a file holding them.

    The file form exists because `sbatch --export=ALL,VAR=a,b,c` splits on commas, so a comma-separated
    list passed that way arrives truncated at the first comma — silently.
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    if spec.startswith("@"):
        spec = pathlib.Path(spec[1:]).read_text()
    return [int(x) for x in spec.replace("\n", ",").split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="http://127.0.0.1:8200", help="a running `openenv harbor serve`")
    p.add_argument("--vllm-url", required=True, help="the engine AsyncGRPO generates from and syncs weights into")
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--split", required=True, help="the Harbor task dataset the server was started with")
    p.add_argument("--harness", default="mini-swe-agent", help="any harness the server reports; see the docstring")
    p.add_argument("--sandbox", default="e2b")
    # "" takes the verifier's single scalar. Name a component (e.g. `correctness`) when the suite emits
    # a reward dict, rather than depending on whichever one the default picks.
    p.add_argument("--reward-key", default="")
    p.add_argument("--n-tasks", type=int, default=32)
    # Prefer tasks whose outcome actually SPLITS for your model. A group whose generations all score the
    # same has `reward_std == 0` and teaches nothing, however healthy the loss looks — and a suite's
    # inherited difficulty labels are usually measured with a different harness, so re-measure rather
    # than trust them.
    p.add_argument("--task-indices", default="", help="comma-separated indices, or @path to a file of them")
    # >1 is what creates the within-group spread the advantage is computed against.
    p.add_argument("--num-generations", type=int, default=8)
    # Each in-flight rollout is one sandbox AND one env session on the server, so keep this under both
    # your sandbox budget and the server's concurrency ceiling.
    p.add_argument("--max-inflight", type=int, default=8)
    p.add_argument("--max-completion-length", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--temperature", type=float, default=1.0)
    # A Harbor rollout is a sandbox boot plus a full agent loop, so staleness accumulates fast.
    p.add_argument("--max-staleness", type=int, default=4)
    # The only bound on a wedged rollout: it holds a generation slot for the whole call, and the task
    # file's own timeout covers the agent run but not sandbox setup.
    p.add_argument("--agent-timeout", type=float, default=300.0)
    # Bounds the packed training row, not just the bill. Every turn re-sends the whole conversation, so a
    # rollout's packed length grows with the SQUARE of its turn count; unbounded 58-turn rollouts were
    # enough to OOM the loss step on an 80 GiB card. Only some harnesses can express this — the rest log
    # a warning and run unbounded.
    p.add_argument("--agent-step-limit", type=int, default=12)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--optim", default="adamw_torch")
    p.add_argument("--no-bf16", dest="bf16", action="store_false", default=True)
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="on by default; rollout sequences here are long enough that activations dominate",
    )
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--save-steps", type=int, default=0, help="0 disables checkpointing; set it for long runs")
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--project", default="async-grpo-harbor")
    p.add_argument("--trackio-space-id", default=None, help="host the trackio dashboard on a HF Space")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Trackio keys a run by name inside a project, so two relaunches of the same config land on top of
    # each other and the earlier metrics read as part of the later run's history — worst exactly when
    # relaunching after a crash. Stamping the name keeps them apart.
    stamp = os.environ.get("SLURM_JOB_ID", "local")
    run_name = args.run_name or f"{args.model.split('/')[-1]}-{args.harness}-{args.max_steps}steps-{stamp}"
    output_dir = args.output_dir or f"runs/async_grpo_harbor/{run_name}"

    factory = HarborSessionFactory(
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
    # Built from the factory so the instruction the trainer sends is the one the server can resolve back
    # to a task. All `num_generations` of a group share a row, so they all get the same task and the group
    # baseline is well formed without any seed plumbing.
    dataset = Dataset.from_list(factory.prompt_rows())

    print(f"server    {args.server}")
    print(f"vllm      {args.vllm_url}   model {args.model}")
    print(f"rollouts  {args.harness} on {args.sandbox}, {args.num_generations}x{args.max_inflight}")
    print(f"tasks     {len(dataset)} from {args.split}")
    print(f"output    {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    config = AsyncGRPOConfig(
        output_dir=output_dir,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=args.save_total_limit,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        max_staleness=args.max_staleness,
        vllm_server_base_url=args.vllm_url,
        optim=args.optim,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        # `use_reentrant=False` is required: the reentrant checkpointer does not see inputs that reach a
        # block through anything but positional args.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="trackio",
        project=args.project,
        trackio_space_id=args.trackio_space_id,
        run_name=run_name,
        log_completions=True,
        # Every rollout costs a sandbox and minutes, so nothing is logged in arrears: flush each step.
        logging_steps=1,
        seed=args.seed,
    )

    worker = HarnessRolloutWorker(
        harness_session_factory=factory,
        # Loop-owning: the agent runs its own loop in the sandbox and we read what it did.
        harness_adapter=None,
        rollout_reward_fn=harbor_reward,
        # Reinforce turns that took an ACTION, not prose — correct for a coding agent. It only works
        # because `to_trace_entries` hands TRL tool calls in the nested OpenAI shape; flattened,
        # `has_tool_call` is False for every turn and the whole rollout is silently discarded.
        train_turn_fn=has_tool_call,
        # No `agent_turn_fn`: the capture layer already dropped auxiliary calls and de-duplicated forked
        # paths structurally, which a flat trace cannot do.
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[],  # the reward is the task's own verifier, via `rollout_reward_fn`
        processing_class=tokenizer,
        # Must match how the engine was served, or every prompt is re-rendered under a different template
        # than the rollout was generated with — silent skew, not an error.
        chat_template_kwargs={"enable_thinking": False},
        num_generations=args.num_generations,
        max_inflight_tasks=args.max_inflight,
        vllm_server_url=args.vllm_url,
        max_tokens=args.max_completion_length,
        temperature=args.temperature,
        log_completions=True,
        num_completions_to_print=2,
    )

    AsyncGRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_worker=worker,
    ).train()


if __name__ == "__main__":
    main()
