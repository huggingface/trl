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

"""LoRA AsyncGRPO training of `mini-swe-agent` on SWE-Gym, one Hugging Face sandbox per rollout.

`mini-swe-agent` is a real coding agent (bash is its only tool) that owns its own loop: it renders SWE-bench's
prompts, calls the model, runs each command, formats the observation, and stops when the model submits a patch or
hits a limit. TRL does not sample its turns. The agent runs to completion, and the `HarnessRolloutWorker` reads back
the trace of the model calls it made (the exact request, the generated token ids, their logprobs), rebuilds one
training row per turn, and trains with GRPO on the outcome reward.

Task: SWE-Gym (`SWE-Gym/SWE-Gym`), 2438 GitHub issues from 11 Python repositories with a prebuilt Docker image per
instance. Each rollout gets its own Hugging Face sandbox started from the instance image, the agent works in
`/testbed`, and the verifier applies SWE-bench's held-out test patch in that same sandbox, runs the tests and grades
them with the SWE-Gym harness: reward 1.0 if every FAIL_TO_PASS and PASS_TO_PASS test passes, 0.0 otherwise.

Ported from NeMo Gym's `responses_api_agents/mini_swe_agent`: same agent config (`mini-swe-agent`'s `swebench.yaml`
prompts and observation templates), same step limit, same binary resolved reward, same dataset. The sandbox is a
Hugging Face sandbox instead of a local Docker/Singularity container, and the eval runs inside it.

The policy is a LoRA adapter. Every weight sync publishes the adapter to `<output_dir>/.vllm_lora/` and tells vLLM
to load it, so the vLLM server must be started with `--enable-lora`. It can run on another machine, as long as it
reads the same `output_dir` (see the `run_*.sh` scripts for the Hugging Face Jobs + Storage Bucket setup).

Single node, 2 GPUs (vLLM on GPU 0, trainer on GPU 1):

```sh
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 vllm serve Qwen/Qwen3-32B \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
    --logprobs-mode processed_logprobs --generation-config vllm \
    --weight-transfer-config '{"backend":"nccl"}' \
    --enable-lora --max-lora-rank 32 --max-loras 6

CUDA_VISIBLE_DEVICES=1 HF_TOKEN=... accelerate launch --num_processes 1 \
    examples/async_grpo_mini_swe_agent/async_grpo_mini_swe_agent.py --model Qwen/Qwen3-32B
```
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, load_dataset
from huggingface_hub import Sandbox
from openai import BadRequestError, OpenAI
from openenv.core.harness import ResourceSession, ResourceSessionFactory, ToolResult, VerifyResult
from peft import LoraConfig
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

import trl.experimental.async_grpo.async_grpo_trainer as async_grpo_trainer
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import HarnessRolloutWorker, TraceEntry


os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")  # mini-swe-agent prints a banner on import otherwise
from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.config import builtin_config_dir  # noqa: E402
from minisweagent.exceptions import Submitted  # noqa: E402
from minisweagent.models.litellm_model import LitellmModel  # noqa: E402
from minisweagent.models.utils.actions_toolcall import BASH_TOOL  # noqa: E402
from swegym.harness.constants import ResolvedStatus  # noqa: E402
from swegym.harness.grading import get_eval_tests_report, get_logs_eval, get_resolution_status  # noqa: E402
from swegym.harness.test_spec import make_test_spec  # noqa: E402


# mini-swe-agent's own SWE-bench config: system/instance prompts, observation and format-error templates, and the
# environment block (cwd `/testbed`, `BASH_ENV` so the image's `conda activate testbed` runs in every command).
SWEBENCH_CONFIG = yaml.safe_load((builtin_config_dir / "benchmarks" / "swebench.yaml").read_text())


# ============================================================================================================
# Dataset
# ============================================================================================================


def build_dataset(n_prompts: int, seed: int) -> tuple[Dataset, dict[str, dict]]:
    """Return `(dataset, instances)`: prompt rows holding the issue text only, and the full SWE-Gym instances keyed
    by that text, for the factory (image, base commit) and the verifier (test patch, FAIL_TO_PASS, PASS_TO_PASS)."""
    rows = load_dataset("SWE-Gym/SWE-Gym", split="train").shuffle(seed=seed).select(range(n_prompts))
    instances = {row["problem_statement"]: row for row in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": text}]} for text in instances])
    return dataset, instances


def instance_image(instance: dict) -> str:
    # SWE-Gym publishes one image per instance; Docker forbids `__` and capitals in repository names.
    return f"xingyaoww/sweb.eval.x86_64.{instance['instance_id'].replace('__', '_s_')}".lower()


# ============================================================================================================
# mini-swe-agent environment and model
# ============================================================================================================


class HFSandboxEnvironment:
    """mini-swe-agent `Environment` running each command in a Hugging Face sandbox. Same contract as its
    `DockerEnvironment`: a fresh `bash -c` per action, stderr merged into stdout, `Submitted` raised when the
    agent echoes the submission marker."""

    def __init__(self, sandbox: Sandbox, *, cwd: str, env: dict[str, str], timeout: int):
        self.sandbox = sandbox
        self.config = {"cwd": cwd, "env": env, "timeout": timeout}

    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]:
        command = action.get("command", "")
        try:
            result = self.sandbox.run(
                ["bash", "-c", "exec 2>&1\n" + command],
                cwd=cwd or self.config["cwd"],
                env=self.config["env"],
                timeout=self.config["timeout"],
                check=False,
            )
        except Exception as e:
            return {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
            }
        if result.timed_out:
            return {
                "output": result.stdout or "",
                "returncode": -1,
                "exception_info": f"Command timed out after {self.config['timeout']} seconds",
            }
        output = {"output": result.stdout or "", "returncode": int(result.exit_code or 0), "exception_info": ""}
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict) -> None:
        lines = output["output"].lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {**self.config, **platform.uname()._asdict(), **kwargs}

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": self.config, "environment_type": "HFSandboxEnvironment"}}}


class TracingModel(LitellmModel):
    """mini-swe-agent's tool-calling model, talking to vLLM through the OpenAI client and recording what OpenEnv's
    interception proxy records: the request as sent, the response, and the generated token ids with their
    logprobs. `HarnessRolloutWorker` rebuilds the training rows from these entries."""

    # A 400 is final: the usual one is the conversation outgrowing vLLM's `--max-model-len`, which ends the episode.
    abort_exceptions = [*LitellmModel.abort_exceptions, BadRequestError]

    def __init__(self, *, base_url: str, chat_template_kwargs: dict[str, Any], **kwargs):
        super().__init__(cost_tracking="ignore_errors", **kwargs)
        self.client = OpenAI(base_url=f"{base_url}/v1", api_key="trl", timeout=600, max_retries=0)
        self.chat_template_kwargs = chat_template_kwargs
        self.trace: list[TraceEntry] = []

    def _query(self, messages: list[dict], **kwargs):
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            tools=[BASH_TOOL],
            logprobs=True,
            extra_body={"return_token_ids": True, "chat_template_kwargs": self.chat_template_kwargs},
            **(self.config.model_kwargs | kwargs),
        )
        choice = response.choices[0]
        self.trace.append(
            {
                "request": {
                    "messages": messages,
                    "tools": [BASH_TOOL],
                    "chat_template_kwargs": self.chat_template_kwargs,
                },
                "response": response.model_dump(mode="json"),
                "completion_token_ids": choice.model_extra["token_ids"],
                "per_token_logps": [token.logprob for token in choice.logprobs.content],
            }
        )
        return response


def policy_model_name(client: OpenAI, base_model: str) -> str:
    """In vLLM's API a LoRA adapter is a model name: a request naming the base model is served by the base model even
    while an adapter is loaded. The trainer publishes the adapter as `trl-policy-v<N>`, so the newest one served is
    the current policy; before the first sync there is none and the base model is the policy."""
    versions = [int(m.id.rsplit("-v", 1)[1]) for m in client.models.list().data if m.id.startswith("trl-policy-v")]
    return f"trl-policy-v{max(versions)}" if versions else base_model


# ============================================================================================================
# OpenEnv session and factory
# ============================================================================================================


class MiniSWEAgentSession(ResourceSession):
    """One rollout: a sandbox started from the instance image, and a `mini-swe-agent` that works in it. Loop-owning:
    `wait_for_completion` runs the agent to its end, `fetch_proxy_trace` returns the model's recorded turns, and
    `verify` grades the sandbox's `/testbed` with the SWE-Gym evaluation script."""

    def __init__(self, *, sandbox: Sandbox, agent: DefaultAgent, instance: dict, eval_timeout: int):
        self.sandbox = sandbox
        self.agent = agent
        self.instance = instance
        self.eval_timeout = eval_timeout
        self.exit_status = None

    def initial_messages(self) -> list[dict]:
        return [{"role": "user", "content": self.instance["problem_statement"]}]

    def list_tools(self) -> list:
        return []  # the agent owns its tool loop; nothing is exposed to the harness

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(error="mini-swe-agent owns its own tool loop.")

    def wait_for_completion(self, timeout_s: float | None = None) -> int:
        # The agent's own limits (`step_limit`, `wall_time_limit_seconds`) end the run; a rollout the agent could not
        # finish (context overflow, a sandbox that died) is a failed rollout, not a crashed worker.
        try:
            info = self.agent.run(self.instance["problem_statement"])
        except Exception as e:
            info = {"exit_status": type(e).__name__}
        self.exit_status = info.get("exit_status")
        return 0 if self.exit_status == "Submitted" else 1

    def fetch_proxy_trace(self) -> list[TraceEntry]:
        return self.agent.model.trace

    def verify(self, transcript: list[dict], final_state: Any | None = None) -> VerifyResult:
        metrics = {"exit_status": self.exit_status, "steps": self.agent.n_calls}
        # Nothing changed under version control: nothing to test.
        if self.sandbox.run(["bash", "-c", "cd /testbed && git diff --quiet"], check=False).exit_code == 0:
            print(f"[rollout] {self.instance['instance_id']} {metrics} no changes", flush=True)
            return VerifyResult(env_reward=0.0, done=True, metrics=metrics)
        # SWE-bench's evaluation script: check the test files out at the base commit, apply the held-out test patch,
        # run the FAIL_TO_PASS and PASS_TO_PASS tests. The agent's changes to the source files stay in place.
        self.sandbox.files.write("/eval.sh", make_test_spec(self.instance).eval_script)
        result = self.sandbox.run(["bash", "-c", "bash /eval.sh 2>&1"], timeout=self.eval_timeout, check=False)
        if result.timed_out:
            return VerifyResult(env_reward=0.0, done=True, metrics={**metrics, "eval_timed_out": True})
        # The SWE-Gym grader reads the repo off the log's parent directory name (the instance id, lowercased: its
        # parser table is keyed by lowercase repo names).
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / self.instance["instance_id"].lower() / "test_output.txt"
            log_path.parent.mkdir()
            log_path.write_text(result.stdout or "")
            status_map, tests_ran = get_logs_eval(str(log_path))
        report = get_eval_tests_report(status_map, self.instance)
        resolved = tests_ran and get_resolution_status(report) == ResolvedStatus.FULL.value
        print(f"[rollout] {self.instance['instance_id']} {metrics} resolved={resolved}", flush=True)
        return VerifyResult(env_reward=float(resolved), done=True, metrics={**metrics, "report": report})

    def close(self) -> None:
        self.sandbox.kill()


class MiniSWEAgentSessionFactory(ResourceSessionFactory):
    def __init__(
        self,
        *,
        instances: dict[str, dict],
        vllm_url: str,
        model: str,
        flavor: str,
        chat_template_kwargs: dict[str, Any],
        temperature: float,
        max_turn_tokens: int,
        step_limit: int,
        step_timeout: int,
        agent_timeout: int,
        eval_timeout: int,
    ):
        self.instances = instances
        self.vllm_url = vllm_url
        self.model = model
        self.flavor = flavor
        self.chat_template_kwargs = chat_template_kwargs
        self.temperature = temperature
        self.max_turn_tokens = max_turn_tokens
        self.step_limit = step_limit
        self.step_timeout = step_timeout
        self.agent_timeout = agent_timeout
        self.eval_timeout = eval_timeout

    def create(self, task: Any, seed: int | None = None, episode_id: str | None = None) -> MiniSWEAgentSession:
        instance = self.instances[task[-1]["content"]]
        sandbox = Sandbox.create(
            image=instance_image(instance),
            flavor=self.flavor,
            idle_timeout=self.agent_timeout + self.eval_timeout,
            labels={"episode_id": episode_id or ""},
        )
        try:
            model = TracingModel(
                base_url=self.vllm_url,
                chat_template_kwargs=self.chat_template_kwargs,
                model_name=policy_model_name(OpenAI(base_url=f"{self.vllm_url}/v1", api_key="trl"), self.model),
                model_kwargs={
                    "temperature": self.temperature,
                    "top_p": 1.0,
                    "max_tokens": self.max_turn_tokens,
                    "parallel_tool_calls": True,
                },
                observation_template=SWEBENCH_CONFIG["model"]["observation_template"],
                format_error_template=SWEBENCH_CONFIG["model"]["format_error_template"],
            )
            env = HFSandboxEnvironment(
                sandbox,
                cwd=SWEBENCH_CONFIG["environment"]["cwd"],
                env=SWEBENCH_CONFIG["environment"]["env"],
                timeout=self.step_timeout,
            )
            agent = DefaultAgent(
                model,
                env,
                system_template=SWEBENCH_CONFIG["agent"]["system_template"],
                instance_template=SWEBENCH_CONFIG["agent"]["instance_template"],
                step_limit=self.step_limit,
                cost_limit=0,  # no cost tracking against a local server
                wall_time_limit_seconds=self.agent_timeout,
            )
        except Exception:
            sandbox.kill()
            raise
        return MiniSWEAgentSession(sandbox=sandbox, agent=agent, instance=instance, eval_timeout=self.eval_timeout)


# ============================================================================================================
# Training
# ============================================================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument("--output-dir", default="async_grpo_mini_swe_agent")
    p.add_argument("--n-prompts", type=int, default=256)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=32)  # concurrent rollouts, one sandbox each
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-turn-tokens", type=int, default=4096)  # per model call; the context is bounded by vLLM
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
    p.add_argument("--project", default="async-grpo-mini-swe-agent")
    p.add_argument("--run-name", default=None)
    p.add_argument("--trackio-space-id", default=None)  # defaults to the project: every run lands in a Space
    args = p.parse_args()

    # trl logs through the stdlib root logger, which a bare script leaves without a handler.
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.getLogger("trl").addHandler(handler)
    logging.getLogger("trl").setLevel(logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset, instances = build_dataset(args.n_prompts, args.seed)

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
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_turn_tokens,
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
        peft_config=LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, target_modules="all-linear"),
    )

    # Checkpoints live in `output_dir`, so a restarted job continues instead of starting over.
    last_checkpoint = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # The final adapter, outside `.vllm_lora/` where old versions are deleted as they leave the staleness window.
    # Every rank calls this: materializing a sharded adapter parameter all-gathers, and only rank 0 writes.
    async_grpo_trainer.save_lora_adapter(
        trainer.accelerator.unwrap_model(trainer.model),
        trainer.accelerator,
        trainer._adapter_name,
        os.path.join(args.output_dir, "final-adapter"),
    )


if __name__ == "__main__":
    main()
