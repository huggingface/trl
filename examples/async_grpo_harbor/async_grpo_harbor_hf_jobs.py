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
#     "huggingface_hub>=1.22",
#     "openenv @ git+https://github.com/huggingface/OpenEnv.git",
#     "openenv-harbor-env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/harbor_env",
# ]
# ///

"""AsyncGRPO on Harbor tasks in a single Hugging Face Job: any harness, any sandbox, any Harbor dataset.

Same training path as `async_grpo_harbor.py`, packaged so one `hf jobs uv run` brings the whole stack up
inside one container. `hf jobs uv run` uploads a single script, so this file is self-contained rather
than importing its sibling.

Three processes, one job:

    openenv harbor serve   (CPU)     the dataset + the capture proxy, published over a tunnel
    vllm serve             (GPU 0)   the policy the agent calls and the trainer syncs weights into
    this script            (GPU 1)   AsyncGRPO

WHY EVERYTHING IS IN ONE JOB, and not split across a Job and a Space. AsyncGRPO syncs weights into vLLM
over NCCL, which needs both on the same host's GPUs — so the trainer and the engine are not separable.
That leaves the OpenEnv server, and putting it here keeps the proxy's hop to vLLM on `localhost`. Hosting
the server on a Space instead (`openenv harbor push`) works, but the proxy then reaches vLLM across the
internet — a second public hop on *every* model call, where this layout has one.

WHY A TUNNEL AND NOT `expose`. Jobs can publish a container port at
`https://<job_id>--<port>.hf.jobs`, but access requires an HF token. The agent's `Authorization` header
already carries its rollout session key — that key IS how the proxy routes concurrent rollouts — so it
cannot also present an HF token. `--expose gradio` opens an unauthenticated outbound tunnel instead,
which needs no ingress from the jobs proxy. (`expose` is still the right tool for the other direction:
`run_rollout` takes `api_key`/`auth_header` precisely so a caller can authenticate to a published engine.)

WHAT THE MOUNTED BUCKET IS FOR. Jobs are ephemeral, so `/data` holds the two things worth keeping:
checkpoints (`--output-dir`) and the HF cache (`HF_HOME`), the latter so a cold start does not re-download
the model every run. Notably NOT the sandbox templates: those live on the sandbox provider's side keyed by
an image hash, so they already survive across jobs — which is the expensive warm step, and it is free.

Requirements:
  - A Hugging Face account with a positive credit balance; Jobs is pay-as-you-go.
  - `HF_TOKEN` and a sandbox credential (e.g. `E2B_API_KEY`) passed as job secrets.
  - A Harbor task dataset on the Hub.

Run:

```sh
# Runs as written. Every argument has a working default; the bucket is optional (without it the job
# still trains, it just loses its checkpoints when the container goes away).
hf jobs uv run \
    --flavor h200x2 \
    --image huggingface/trl \
    --secrets HF_TOKEN --secrets E2B_API_KEY \
    https://raw.githubusercontent.com/huggingface/trl/main/examples/async_grpo_harbor/async_grpo_harbor_hf_jobs.py

# ...and with a bucket, so checkpoints and the model cache survive the job:
hf jobs uv run \
    --flavor h200x2 --image huggingface/trl \
    --secrets HF_TOKEN --secrets E2B_API_KEY \
    --volume type=bucket,source=<user>/<bucket>,mount_path=/data \
    examples/async_grpo_harbor/async_grpo_harbor_hf_jobs.py \
    -- --max-steps 20 --save-steps 10
```

Or from Python, which is easier to script and gives you the job id back:

```python
from huggingface_hub import Volume, run_uv_job

job = run_uv_job(
    "examples/async_grpo_harbor/async_grpo_harbor_hf_jobs.py",
    script_args=["--max-steps", "20", "--save-steps", "10"],
    flavor="h200x2",
    image="huggingface/trl",
    secrets={"HF_TOKEN": "...", "E2B_API_KEY": "..."},
    volumes=[Volume(type="bucket", source="<user>/<bucket>", mount_path="/data")],
    timeout="4h",
)
print(job.id, job.url)
```
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

W_TOOL_EFFICIENCY = float(os.environ.get("REWARD_W_TOOL_EFFICIENCY", "0.3"))
TOOL_BUDGET = float(os.environ.get("TOOL_BUDGET", "15"))

# Where the mounted bucket lands. Everything worth surviving the container goes under here.
DATA_ROOT = pathlib.Path(os.environ.get("DATA_ROOT", "/data"))

_children: list[subprocess.Popen] = []


# --------------------------------------------------------------------------------------------------
# reward
# --------------------------------------------------------------------------------------------------
def tool_efficiency(n_tool_calls: int | None) -> float | None:
    """`clip(1 - n/TOOL_BUDGET, 0, 1)`, or `None` when the tool count is unknown."""
    if n_tool_calls is None or TOOL_BUDGET <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - n_tool_calls / TOOL_BUDGET))


def harbor_reward(outcome):
    """Reward for one rollout, or `None` when it is unscorable.

    `None` drops the rollout from its group baseline instead of scoring it `0`. Scoring an unmeasured
    rollout `0` teaches the policy that a crashed sandbox is as good as a wrong answer.
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


# --------------------------------------------------------------------------------------------------
# process orchestration
# --------------------------------------------------------------------------------------------------
def _spawn(cmd: list[str], log_path: pathlib.Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    """Start a long-lived child with its own process group, logging to a file.

    Its own group so cleanup can signal the whole tree: vLLM spawns engine workers that outlive a plain
    `kill` of the parent and then hold the GPU.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env={**os.environ, **(env or {})},
        start_new_session=True,
    )
    _children.append(proc)
    print(f"[jobs] started pid={proc.pid}: {' '.join(cmd[:4])}...  -> {log_path}", flush=True)
    return proc


def _cleanup(*_: object) -> None:
    for proc in _children:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass


def _get_json(url: str, timeout: float = 10.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            import json

            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def wait_for_public_proxy(server_log: pathlib.Path, capture_port: int, deadline_s: float = 420.0) -> str:
    """Return the proxy's public URL, only once it is *serving our proxy*.

    Two separate things have to be true and only one of them is about the port. The server prints
    `capture   :<port> -> https://...` once the tunnel is up, but a tunnel can exist and still not reach
    us: when the forwarding process dies the domain keeps answering, with the tunnel provider's own error
    page. Agents then receive HTML where an OpenAI endpoint should be, make zero model calls, and every
    rollout comes back unscorable while the server's own `/health` still says it is fine. So the URL is
    only accepted after a request THROUGH it returns our health document.
    """
    pattern = re.compile(rf"capture\s+:{capture_port}\s+->\s+(https://\S+)")
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    started = time.monotonic()
    url: str | None = None
    while time.monotonic() - started < deadline_s:
        if url is None and server_log.exists():
            match = pattern.search(ansi.sub("", server_log.read_text(errors="replace")))
            if match:
                url = match.group(1).rstrip(".,")
                print(f"[jobs] server published {url}; verifying it reaches the proxy", flush=True)
        if url is not None:
            health = _get_json(f"{url}/health")
            if health and "status" in health:
                print(f"[jobs] proxy reachable at {url} (capture_level={health.get('capture_level')})", flush=True)
                return url
        time.sleep(5)
    tail = server_log.read_text(errors="replace")[-1500:] if server_log.exists() else "(no log)"
    raise RuntimeError(f"the capture proxy never became publicly reachable within {deadline_s:.0f}s\n{tail}")


def wait_for_vllm(url: str, proc: subprocess.Popen, log_path: pathlib.Path, deadline_s: float = 900.0) -> None:
    """Block until vLLM answers `/health`, failing fast if it died instead."""
    started = time.monotonic()
    while time.monotonic() - started < deadline_s:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-2000:]
            raise RuntimeError(f"vLLM exited with code {proc.returncode} before serving\n{tail}")
        if _get_json(f"{url}/health") is not None or _probe_ok(f"{url}/health"):
            print(f"[jobs] vLLM ready at {url}", flush=True)
            return
        time.sleep(5)
    raise RuntimeError(f"vLLM did not become ready within {deadline_s:.0f}s")


def _probe_ok(url: str) -> bool:
    """vLLM's `/health` returns an empty 200 body, which is not JSON."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError):
        return False


def start_openenv_server(args: argparse.Namespace, logs: pathlib.Path) -> str:
    cmd = [
        "openenv",
        "harbor",
        "serve",
        "--dataset",
        args.split,
        "--port",
        str(args.server_port),
        "--capture-port",
        str(args.capture_port),
        # Unauthenticated outbound tunnel: see WHY A TUNNEL AND NOT `expose` above.
        "--expose",
        "gradio",
    ]
    log = logs / "openenv-server.log"
    _spawn(cmd, log)
    return wait_for_public_proxy(log, args.capture_port)


def start_vllm(args: argparse.Namespace, logs: pathlib.Path) -> tuple[str, subprocess.Popen]:
    url = f"http://127.0.0.1:{args.vllm_port}"
    cmd = [
        "vllm",
        "serve",
        args.model,
        "--host",
        "0.0.0.0",
        "--port",
        str(args.vllm_port),
        "--trust-remote-code",
        "--max-model-len",
        str(args.max_model_len),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        args.tool_call_parser,
        # The three flags below are load-bearing, not tuning. Without token ids and processed logprobs
        # the proxy grades every rollout `eval` and the run produces nothing trainable; without the
        # weight-transfer backend AsyncGRPO cannot push new weights back into the engine.
        "--return-tokens-as-token-ids",
        "--logprobs-mode",
        "processed_logprobs",
        "--weight-transfer-config",
        '{"backend":"nccl"}',
    ]
    if args.reasoning_parser:
        cmd += ["--reasoning-parser", args.reasoning_parser]
    if args.enable_thinking is False:
        cmd += ["--default-chat-template-kwargs", '{"enable_thinking": false}']
    log = logs / "vllm.log"
    proc = _spawn(cmd, log, env={"CUDA_VISIBLE_DEVICES": args.vllm_device, "VLLM_SERVER_DEV_MODE": "1"})
    wait_for_vllm(url, proc, log)
    return url, proc


# --------------------------------------------------------------------------------------------------
# args
# --------------------------------------------------------------------------------------------------
def task_indices(spec: str) -> list[int] | None:
    spec = (spec or "").strip()
    if not spec:
        return None
    if spec.startswith("@"):
        spec = pathlib.Path(spec[1:]).read_text()
    return [int(x) for x in spec.replace("\n", ",").split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--split",
        default="AdithyaSK/data_agent_rl_environment_train",
        help="any Harbor task dataset on the Hub; the default is a public data-analysis suite",
    )
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--harness", default="mini-swe-agent")
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
    p.add_argument("--agent-timeout", type=float, default=300.0)
    p.add_argument("--agent-step-limit", type=int, default=12)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--optim", default="adamw_torch")
    p.add_argument("--no-bf16", dest="bf16", action="store_false", default=True)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--save-steps", type=int, default=0)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--project", default="async-grpo-harbor")
    p.add_argument("--trackio-space-id", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=0)
    # infra
    p.add_argument("--server-port", type=int, default=8200)
    p.add_argument("--capture-port", type=int, default=8300)
    p.add_argument("--vllm-port", type=int, default=8000)
    p.add_argument("--vllm-device", default="0", help="CUDA_VISIBLE_DEVICES for the engine")
    p.add_argument("--train-device", default="1", help="CUDA_VISIBLE_DEVICES for the trainer")
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--tool-call-parser", default="qwen3_xml")
    p.add_argument("--reasoning-parser", default="qwen3")
    # Thinking is off by default and the trainer's chat_template_kwargs must agree with how the engine
    # was served, or every prompt is re-rendered under a different template than it was generated with.
    p.add_argument("--enable-thinking", action="store_true", default=False)
    p.add_argument("--data-root", default=str(DATA_ROOT), help="the mounted bucket")
    return p.parse_args()


# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    data_root = pathlib.Path(args.data_root)
    mounted = data_root.is_dir()
    if not mounted:
        # Not fatal: without the mount the job still trains, it just loses its checkpoints when the
        # container goes away. Say so once, loudly, rather than discovering it afterwards.
        print(f"[jobs] WARNING {data_root} is not a directory — no bucket mounted. Checkpoints and the", flush=True)
        print("[jobs] WARNING model cache will NOT survive this job. Pass --volume type=bucket,...", flush=True)
        data_root = pathlib.Path("./out")
    logs = data_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    # A persistent HF cache is the difference between re-downloading the model every cold start and not.
    if mounted:
        os.environ.setdefault("HF_HOME", str(data_root / "hf-cache"))
    print(f"[jobs] data root {data_root} (mounted={mounted})   HF_HOME={os.environ.get('HF_HOME')}", flush=True)

    for tool in ("openenv", "vllm"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"`{tool}` is not on PATH; the PEP 723 dependencies did not install")
    if not (os.environ.get("E2B_API_KEY") or args.sandbox != "e2b"):
        raise RuntimeError("sandbox is e2b but E2B_API_KEY is unset; pass it as a job secret")

    public_proxy = start_openenv_server(args, logs)
    vllm_url, _ = start_vllm(args, logs)

    # Imported only now: the heavy imports cost a minute, and there is no point paying it before the
    # servers are known good.
    from datasets import Dataset
    from harbor_env.harness import HarborSessionFactory
    from transformers import AutoTokenizer

    from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
    from trl.experimental.async_grpo.openenv_harness import HarnessRolloutWorker, has_tool_call

    os.environ["CUDA_VISIBLE_DEVICES"] = args.train_device

    job_id = os.environ.get("HF_JOB_ID", "local")
    run_name = args.run_name or f"{args.model.split('/')[-1]}-{args.harness}-{args.max_steps}steps-{job_id}"
    output_dir = str(data_root / "runs" / run_name)

    factory = HarborSessionFactory(
        f"http://127.0.0.1:{args.server_port}",
        split=args.split,
        harness=args.harness,
        sandbox=args.sandbox,
        llm_url=vllm_url,
        model=args.model,
        agent_timeout_sec=args.agent_timeout,
        agent_step_limit=args.agent_step_limit,
        reward_key=args.reward_key,
        num_tasks=args.n_tasks,
        indices=task_indices(args.task_indices),
    )
    dataset = Dataset.from_list(factory.prompt_rows())

    print(f"[jobs] proxy     {public_proxy}   (what the sandboxed agent calls)", flush=True)
    print(f"[jobs] vllm      {vllm_url}   model {args.model}", flush=True)
    print(f"[jobs] rollouts  {args.harness} on {args.sandbox}, {args.num_generations}x{args.max_inflight}", flush=True)
    print(f"[jobs] tasks     {len(dataset)} from {args.split}", flush=True)
    print(f"[jobs] output    {output_dir}", flush=True)

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
        vllm_server_base_url=vllm_url,
        optim=args.optim,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="trackio",
        project=args.project,
        trackio_space_id=args.trackio_space_id,
        run_name=run_name,
        log_completions=True,
        logging_steps=1,
        seed=args.seed,
    )

    worker = HarnessRolloutWorker(
        harness_session_factory=factory,
        harness_adapter=None,  # loop-owning: the harness runs its own loop; we read what it did
        rollout_reward_fn=harbor_reward,
        train_turn_fn=has_tool_call,
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[],
        processing_class=tokenizer,
        chat_template_kwargs={"enable_thinking": False},
        num_generations=args.num_generations,
        max_inflight_tasks=args.max_inflight,
        vllm_server_url=vllm_url,
        max_tokens=args.max_completion_length,
        temperature=args.temperature,
        log_completions=True,
        num_completions_to_print=2,
    )

    try:
        AsyncGRPOTrainer(
            model=args.model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            rollout_worker=worker,
        ).train()
    finally:
        _cleanup()


if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup()
        sys.stdout.flush()
