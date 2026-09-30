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
#     "pandas",
#     "pyarrow",
#     "huggingface_hub>=1.31",
#     "mimoagent @ git+https://github.com/XiaomiMiMo/MiMo-Agent.git",
#     "openenv @ git+https://github.com/huggingface/OpenEnv.git",
# ]
# ///

"""AsyncGRPO training of MiMo's CC agent on the General domain of `XiaomiMiMo/MiMo-V2.6-RL-oss`, one Hugging Face
sandbox per rollout.

The rollout is the one `probe_general.py` runs: `mimoagent`'s CC agent (Bash/Read/Write/Edit/Grep/Glob plus the
task's MCP tools) working in a sandbox that holds the task's workspace and business systems, graded by the task's
own rubric verifier with an LLM judge. Here the agent's model calls are recorded the way OpenEnv's interception
proxy records them (the request as sent, the generated token ids, their logprobs), and `HarnessRolloutWorker`
rebuilds one training row per turn from those records and trains with GRPO on the outcome reward. Upstream trains
on the binarized reward, 1 only when every rubric item passes, which is the default here; `--reward score` keeps
the weighted score. A rollout the judge could not grade is unscorable and leaves the group baseline.

`--packing tree` trains on a packed prefix forest instead of rows laid end to end. These rollouts are the case it is
built for: a turn re-sends the whole conversation, so the rows of one rollout are nested prefixes of each other and
every row of a group repeats the same prompt. `run_slurm.sh` runs the same job both ways to compare.

Single node, 2 GPUs (vLLM on GPU 0, trainer on GPU 1):

```sh
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-8B \
    --host 0.0.0.0 --port 8000 --max-model-len 40960 \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
    --logprobs-mode processed_logprobs --generation-config vllm \
    --weight-transfer-config '{"backend":"nccl"}'

CUDA_VISIBLE_DEVICES=1 HF_TOKEN=... accelerate launch --num_processes 1 \
    examples/async_grpo_mimo_rl_oss/async_grpo_mimo_general.py --model Qwen/Qwen3-8B
```
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from datasets import Dataset
from mimoagent.models.openai_chat import (  # noqa: E402
    _CLIENT_KWARGS,
    OpenAIChatModel,
    _build_assistant_payload,
    _extract_token_stats,
    _route_unknown_kwargs,
)
from openai import OpenAI
from openenv.core.harness import ResourceSession, ResourceSessionFactory, ToolResult, VerifyResult
from peft import LoraConfig
from probe_general import GeneralTaskSession, GeneralTaskSessionFactory, judge_env, load_tasks  # noqa: E402
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

import trl.experimental.async_grpo.async_grpo_trainer as async_grpo_trainer
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import HarnessRolloutWorker, TraceEntry


class TracingModel(OpenAIChatModel):
    """mimoagent's chat model, recording what OpenEnv's interception proxy records: the request as sent, the response,
    and the generated token ids with their logprobs. `HarnessRolloutWorker` rebuilds the training rows from these
    entries."""

    def __init__(self, *, chat_template_kwargs: dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.chat_template_kwargs = chat_template_kwargs
        self.trace: list[TraceEntry] = []

    def _query(self, messages: list[dict], **kwargs):
        call_kwargs = {k: v for k, v in (self.config.model_kwargs | kwargs).items() if k not in _CLIENT_KWARGS}
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            logprobs=True,
            extra_body={"return_token_ids": True, "chat_template_kwargs": self.chat_template_kwargs},
            **_route_unknown_kwargs(call_kwargs),
        )
        choice = response.choices[0]
        self.trace.append(
            {
                "request": {
                    "messages": list(messages),
                    "tools": kwargs.get("tools"),
                    "chat_template_kwargs": self.chat_template_kwargs,
                },
                "response": response.model_dump(mode="json"),
                "completion_token_ids": choice.model_extra["token_ids"],
                "per_token_logps": [token.logprob for token in choice.logprobs.content],
            }
        )
        payload = _build_assistant_payload(choice.message)
        if not (payload.get("content") or payload.get("reasoning_content") or payload.get("tool_calls")):
            if choice.finish_reason != "length":
                raise ValueError("Empty assistant response: content, reasoning_content and tool_calls are all empty.")
        return _extract_token_stats(response.usage), payload


def policy_model_name(client: OpenAI, base_model: str) -> str:
    """In vLLM's API a LoRA adapter is a model name: a request naming the base model is served by the base model even
    while an adapter is loaded. The trainer publishes the adapter as `trl-policy-v<N>`, so the newest one served is
    the current policy; before the first sync there is none and the base model is the policy."""
    versions = [int(m.id.rsplit("-v", 1)[1]) for m in client.models.list().data if m.id.startswith("trl-policy-v")]
    return f"trl-policy-v{max(versions)}" if versions else base_model


class TrainingSession(GeneralTaskSession, ResourceSession):
    """The probe's session as an OpenEnv loop-owning session: `wait_for_completion` runs the agent to its end,
    `fetch_proxy_trace` returns the model's recorded turns, and `verify` grades the sandbox with the task's verifier."""

    def __init__(self, *, binary_reward: bool, **kwargs):
        super().__init__(**kwargs)
        self.binary_reward = binary_reward
        self.exit_status = None
        self.exit_message = ""

    def initial_messages(self) -> list[dict]:
        return [{"role": "user", "content": self.task["problem_statement"]}]

    def list_tools(self) -> list:
        return []  # the agent owns its tool loop; nothing is exposed to the harness

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(error="mimoagent owns its own tool loop.")

    def wait_for_completion(self, timeout_s: float | None = None) -> int:
        # The agent's own limits (`step_limit`, the deadline) end the run; a rollout the agent could not finish (context
        # overflow, a sandbox that died) is a failed rollout, not a crashed worker.
        try:
            self.exit_status, self.exit_message = self.run()
        except Exception as e:
            self.exit_status = type(e).__name__
        return 0 if self.exit_status == "Idle" else 1

    def fetch_proxy_trace(self) -> list[TraceEntry]:
        return self.agent.model.trace

    def verify(self, transcript: list[dict], final_state: Any | None = None) -> VerifyResult:
        graded = self.grade(self.exit_message)
        metrics = {
            "exit_status": self.exit_status,
            "steps": self.agent._steps_taken,
            "score": graded["reward"],
            "reward_error": graded["reward_error"],
            "items": graded["items"],
        }
        print(
            f"[rollout] {self.task['instance_id']} {self.exit_status} steps={metrics['steps']} score={graded['reward']} {graded['reward_error'] or ''}",
            flush=True,
        )
        if graded["reward"] is None:
            return VerifyResult(env_reward=None, done=True, metrics=metrics)
        reward = float(graded["reward"] >= 1 - 1e-6) if self.binary_reward else graded["reward"]
        return VerifyResult(env_reward=reward, done=True, metrics=metrics)


class TrainingSessionFactory(GeneralTaskSessionFactory, ResourceSessionFactory):
    def __init__(self, *, tasks: list[dict], binary_reward: bool, **kwargs):
        super().__init__(**kwargs)
        self.tasks = {task["problem_statement"]: task for task in tasks}
        self.binary_reward = binary_reward

    def create(self, task: Any, seed: int | None = None, episode_id: str | None = None) -> TrainingSession:
        return super().create(self.tasks[task[-1]["content"]], episode_id[:8])

    def make_model(self) -> TracingModel:
        return TracingModel(
            model_name=policy_model_name(OpenAI(base_url=self.base_url, api_key=self.api_key), self.model),
            chat_template_kwargs=self.chat_template_kwargs,
            model_kwargs={
                "base_url": self.base_url,
                "api_key": self.api_key,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_turn_tokens,
                "parallel_tool_calls": True,
            },
        )

    def make_session(self, **kwargs) -> TrainingSession:
        return TrainingSession(
            binary_reward=self.binary_reward, judge_env=self.judge_env, verify_timeout=self.verify_timeout, **kwargs
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument("--output-dir", default="async_grpo_mimo_general")
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--instances-file", default=None)  # JSON list of instance ids to train on
    p.add_argument("--language", default="en", choices=["en", "zh", "all"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=32)  # concurrent rollouts, one sandbox each
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--lora-rank", type=int, default=0)  # 0: full fine-tuning
    p.add_argument("--optim", default="adamw_torch")
    p.add_argument("--temperature", type=float, default=1.0)
    # top_p 1.0 rather than upstream's 0.95: the PPO denominator is vLLM's processed logprobs, and nucleus sampling
    # would put mass the trainer cannot reproduce into them.
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-turn-tokens", type=int, default=4096)  # per model call; the context is bounded by vLLM
    p.add_argument("--max-observation-length", type=int, default=8000)  # characters per tool result, see the probe
    p.add_argument(
        "--token-budget", type=int, default=None
    )  # tokens per training micro-batch row; defaults to vLLM's max_model_len
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--packing", default="sequence", choices=["sequence", "tree"])
    p.add_argument("--fork-threshold-tokens", type=int, default=1024)
    p.add_argument("--reward", default="binary", choices=["binary", "score"])
    p.add_argument("--max-staleness", type=int, default=4)
    p.add_argument("--weight-sync-steps", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--sandbox-flavor", default="cpu-basic")
    p.add_argument("--step-limit", type=int, default=500)
    p.add_argument("--agent-timeout", type=int, default=1200)
    p.add_argument("--verify-timeout", type=int, default=900)
    p.add_argument("--judge-url", default="https://router.huggingface.co/v1")
    p.add_argument("--judge-model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    p.add_argument("--judge-key", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--project", default="async-grpo-mimo-general")
    p.add_argument("--run-name", default=None)
    p.add_argument("--trackio-space-id", default=None)  # defaults to the project: every run lands in a Space
    args = p.parse_args()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.getLogger("trl").addHandler(handler)
    logging.getLogger("trl").setLevel(logging.INFO)
    logging.getLogger("mimoagent").setLevel(logging.WARNING)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    instance_ids = json.load(open(args.instances_file)) if args.instances_file else None
    tasks = load_tasks(
        args.n_prompts,
        args.seed,
        language=None if args.language == "all" else args.language,
        instance_ids=instance_ids,
    )
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": task["problem_statement"]}]} for task in tasks]
    )
    transcripts_dir = Path(args.output_dir) / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    factory = TrainingSessionFactory(
        tasks=tasks,
        binary_reward=args.reward == "binary",
        base_url=f"{args.vllm_url}/v1",
        api_key="trl",
        model=args.model,
        chat_template_kwargs={} if args.enable_thinking else {"enable_thinking": False},
        temperature=args.temperature,
        top_p=args.top_p,
        max_turn_tokens=args.max_turn_tokens,
        max_observation_length=args.max_observation_length,
        step_limit=args.step_limit,
        agent_timeout=args.agent_timeout,
        verify_timeout=args.verify_timeout,
        judge_env=judge_env(args.judge_url, args.judge_key, args.judge_model),
        flavor=args.sandbox_flavor,
        transcripts_dir=transcripts_dir,
    )

    config = AsyncGRPOConfig(
        output_dir=args.output_dir,
        bf16=True,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_turn_tokens,
        gradient_checkpointing=args.gradient_checkpointing,
        packing=args.packing,
        token_budget=args.token_budget,
        learning_rate=args.learning_rate,
        optim=args.optim,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # The worker refreshes its heartbeat from its async loops, so a synchronous sandbox call blocks them both
        # and looks exactly like a hang. The longest one is the verifier's `--verify-timeout`.
        heartbeat_stale_after_s=3 * max(300, args.verify_timeout),
        # Explicit rather than the trainer's `max_staleness x rows per step`: a tree row holds tens of samples, so a
        # step is one row per rank and the derived limit would starve the rollout loop.
        max_inflight_tasks=args.max_inflight,
        max_staleness=args.max_staleness,
        weight_sync_steps=args.weight_sync_steps,
        max_steps=args.max_steps,
        logging_steps=1,
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
        harness_adapter=None,  # loop-owning: mimoagent runs its own loop; TRL reads the recorded turns
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[],  # the reward is the verifier's, through the session
        processing_class=tokenizer,
        num_generations=args.num_generations,
        fork_threshold_tokens=args.fork_threshold_tokens,
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
        peft_config=LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, target_modules="all-linear")
        if args.lora_rank
        else None,
    )

    last_checkpoint = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if not args.lora_rank:
        return
    async_grpo_trainer.save_lora_adapter(
        trainer.accelerator.unwrap_model(trainer.model),
        trainer.accelerator,
        trainer._adapter_name,
        os.path.join(args.output_dir, "final-adapter"),
    )


if __name__ == "__main__":
    main()
