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
# requires-python = ">=3.12"
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "openenv @ git+https://github.com/huggingface/OpenEnv.git",
#     "openenv-harbor-env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/harbor_env",
#     "vllm>=0.22,<0.26",
#     "datasets>=3.2",
#     "trackio",
#     "transformers>=5.2",
#     "huggingface_hub>=1.22",
# ]
# ///

"""Run `async_grpo_harbor.py` on Hugging Face Jobs: one job, one command.

A Job gives you one container and one command, and this recipe needs three processes. The launcher
wraps them:

    openenv harbor serve   (CPU)     the Harbor dataset + the capture proxy, published over a tunnel
    vllm serve             (GPU 0)   the policy the agent calls and the trainer syncs weights into
    async_grpo_harbor.py   (GPU 1)   the training script, downloaded and run unmodified

The training script is *downloaded*, not duplicated here, so what runs on Jobs is byte-identical to what
runs locally and the two cannot drift.

Everything is in one job because AsyncGRPO syncs weights into vLLM over NCCL, which needs both on the
same host's GPUs. That leaves only the OpenEnv server placeable, and keeping it here means the capture
proxy reaches vLLM over `localhost`; hosting it on a Space instead adds a public hop to every model call.

WHICH HOP THE TUNNEL IS FOR. The sandboxed agent is the only participant outside this container, and what
it needs to reach is the capture proxy — not vLLM, which is what the opencode recipe tunnels. So
`openenv harbor serve --expose gradio` publishes the proxy and the engine stays entirely private.
Jobs can also publish a port at `https://<job_id>--<port>.hf.jobs`, but that requires an HF token, and
the agent's `Authorization` header already carries its rollout session key — the key the proxy routes on
— so it cannot carry a second credential.

Requirements:
  - A Hugging Face account with a positive credit balance; Jobs is pay-as-you-go.
  - `HF_TOKEN` and a sandbox credential (`E2B_API_KEY` for the default backend) as job secrets.

Run:

```sh
hf jobs uv run \
    --flavor h200x2 \
    --image huggingface/trl \
    --secrets HF_TOKEN --secrets E2B_API_KEY \
    --timeout 7200s \
    https://raw.githubusercontent.com/huggingface/trl/main/examples/async_grpo_harbor/launcher.py
```

Add a bucket so checkpoints and the model cache outlive the container, which is otherwise ephemeral:

```sh
hf jobs uv run --flavor h200x2 --image huggingface/trl \
    --secrets HF_TOKEN --secrets E2B_API_KEY --timeout 7200s \
    --volume type=bucket,source=<user>/<bucket>,mount_path=/data \
    https://raw.githubusercontent.com/huggingface/trl/main/examples/async_grpo_harbor/launcher.py \
    -- --save-steps 10
```

Anything after `--` is forwarded to the training script, so its full argument surface is available:
`--model`, `--split`, `--harness`, `--sandbox`, `--max-steps`, and so on.
"""

from __future__ import annotations

import argparse
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


# The canonical training script. Overridable with --train-script-url so a branch can be tested before it
# is merged -- worth having: the equivalent opencode launcher still points at a pre-reorg path that now
# 404s, and the failure surfaces as a download error minutes into a paid job.
TRAIN_SCRIPT_URL = (
    "https://raw.githubusercontent.com/huggingface/trl/main/examples/async_grpo_harbor/async_grpo_harbor.py"
)
DATA_ROOT = pathlib.Path(os.environ.get("DATA_ROOT", "/data"))

_children: list[subprocess.Popen] = []


def _spawn(cmd: list[str], log_path: pathlib.Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    """Start a long-lived child in its own process group, logging to a file.

    Its own group so cleanup can signal the whole tree: vLLM spawns engine workers that outlive a plain
    kill of the parent and then keep holding the GPU.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        env={**os.environ, **(env or {})},
        start_new_session=True,
    )
    _children.append(proc)
    print(f"[launcher] pid={proc.pid} {' '.join(cmd[:3])} ... -> {log_path}", flush=True)
    return proc


def _cleanup(*_: object) -> None:
    for proc in _children:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass


def _http_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError):
        return False


def _http_json_has(url: str, key: str, timeout: float = 15.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            import json

            return key in json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_for_public_proxy(log: pathlib.Path, capture_port: int, deadline_s: float = 420.0) -> str:
    """Return the proxy's public URL, but only once a request THROUGH it reaches the proxy.

    The published URL appearing in the log is not enough. When a tunnel's forwarding process dies the
    domain keeps resolving and answers with the tunnel provider's own error page, so the sandboxed agent
    receives HTML where an OpenAI endpoint should be, makes zero model calls, and every rollout comes
    back unscorable — while the server's own /health still reports healthy, because it checks itself and
    not its tunnel. A check that does not traverse the path the workload uses proves nothing.
    """
    published = re.compile(rf"capture\s+:{capture_port}\s+->\s+(https://\S+)")
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    started = time.monotonic()
    url: str | None = None
    while time.monotonic() - started < deadline_s:
        if url is None and log.exists():
            found = published.search(ansi.sub("", log.read_text(errors="replace")))
            if found:
                url = found.group(1).rstrip(".,")
                print(f"[launcher] server published {url}; verifying it reaches the proxy", flush=True)
        if url and _http_json_has(f"{url}/health", "status"):
            print(f"[launcher] capture proxy reachable at {url}", flush=True)
            return url
        time.sleep(5)
    tail = log.read_text(errors="replace")[-1500:] if log.exists() else "(no log)"
    raise RuntimeError(f"the capture proxy never became publicly reachable in {deadline_s:.0f}s\n{tail}")


def wait_for_vllm(url: str, proc: subprocess.Popen, log: pathlib.Path, deadline_s: float = 1200.0) -> None:
    """Block until vLLM answers /health, failing fast with its log if it died instead."""
    started = time.monotonic()
    while time.monotonic() - started < deadline_s:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM exited {proc.returncode} before serving\n{log.read_text(errors='replace')[-2000:]}"
            )
        if _http_ok(f"{url}/health"):
            print(f"[launcher] vLLM ready at {url}", flush=True)
            return
        time.sleep(5)
    raise RuntimeError(f"vLLM did not become ready in {deadline_s:.0f}s")


def warm_sandbox_template(args: argparse.Namespace, vllm_url: str, logs: pathlib.Path) -> None:
    """Build the sandbox image once, serially, before any group runs concurrently.

    Not an optimisation. Harbor decides whether to build from `alias_exists()`, which flips true when a
    build STARTS rather than when it finishes, so N generations racing their first visit to a task all
    see "exists" and then fail against a half-built image with `404: tag 'default' does not exist`. With
    `num_generations` rollouts launched together on a cold template that is the common case, and the
    failures read as the harness misbehaving rather than a build race.
    """
    cmd = [
        "openenv", "harbor", "rollout",
        "--llm-url", vllm_url,
        "--model", args.model,
        "--dataset", args.split,
        "--harness", args.harness,
        "--sandbox", args.sandbox,
        "--task-index", "0",
        "-n", "1",
    ]  # fmt: skip
    log = logs / "warm-template.log"
    print("[launcher] warming the sandbox template (one serial rollout) ...", flush=True)
    started = time.monotonic()
    with open(log, "w") as handle:
        rc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode
    took = time.monotonic() - started
    if rc != 0:
        # Not fatal: the warm rollout can fail for reasons that say nothing about training (an
        # ungradeable task 0, a flaky verifier) and the build it triggered still happened.
        print(f"[launcher] WARNING warm rollout exited {rc} after {took:.0f}s; continuing", flush=True)
        print(f"[launcher] WARNING tail:\n{log.read_text(errors='replace')[-800:]}", flush=True)
    else:
        print(f"[launcher] template warm after {took:.0f}s", flush=True)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")  # also forwarded to the trainer
    p.add_argument("--split", default="AdithyaSK/data_agent_rl_environment_train")  # a public Harbor suite
    p.add_argument("--harness", default="mini-swe-agent")  # see the training script for why this default
    p.add_argument("--sandbox", default="e2b")  # needs the matching credential as a job secret
    p.add_argument("--server-port", type=int, default=8200)
    p.add_argument("--capture-port", type=int, default=8300)
    p.add_argument("--vllm-port", type=int, default=8000)
    p.add_argument("--vllm-device", default="0")  # the engine
    p.add_argument("--train-device", default="1")  # the trainer; NCCL weight sync needs the same host
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--tool-call-parser", default="qwen3_xml")  # Qwen3.5; use `hermes` for most others
    p.add_argument("--reasoning-parser", default="qwen3")
    p.add_argument("--train-script-url", default=TRAIN_SCRIPT_URL)
    p.add_argument("--data-root", default=str(DATA_ROOT))  # a mounted bucket, if any
    p.add_argument("--skip-warm", action="store_true")
    return p.parse_known_args()


def main() -> None:
    args, forwarded = parse_args()
    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    data_root = pathlib.Path(args.data_root)
    mounted = data_root.is_dir()
    if not mounted:
        print(f"[launcher] WARNING {data_root} is not a directory, so no bucket is mounted: checkpoints", flush=True)
        print("[launcher] WARNING and the model cache will NOT survive this job.", flush=True)
        data_root = pathlib.Path("./out")
    logs = data_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if mounted:
        os.environ.setdefault("HF_HOME", str(data_root / "hf-cache"))

    for tool in ("openenv", "vllm"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"`{tool}` is not on PATH; the PEP 723 dependencies did not install")
    if args.sandbox == "e2b" and not os.environ.get("E2B_API_KEY"):
        raise RuntimeError("sandbox is e2b but E2B_API_KEY is unset; pass it with --secrets E2B_API_KEY")
    n_gpu = len(
        [
            line
            for line in subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.splitlines()
            if line.strip()
        ]
    )
    if n_gpu < 2:
        raise RuntimeError(
            f"needs 2 GPUs (engine + trainer on one host for NCCL) but sees {n_gpu}; try --flavor h200x2"
        )

    # 1. the Harbor dataset + the capture proxy, published for the sandboxed agent
    server_log = logs / "openenv-server.log"
    _spawn(
        ["openenv", "harbor", "serve",
         "--dataset", args.split,
         "--port", str(args.server_port),
         "--capture-port", str(args.capture_port),
         "--expose", "gradio"],
        server_log,
    )  # fmt: skip
    public_proxy = wait_for_public_proxy(server_log, args.capture_port)

    # 2. the policy. The token-id and logprob flags are load-bearing: without them the proxy grades
    #    every rollout `eval` and the run produces nothing trainable.
    vllm_url = f"http://127.0.0.1:{args.vllm_port}"
    vllm_log = logs / "vllm.log"
    vllm_cmd = [
        "vllm", "serve", args.model,
        "--host", "0.0.0.0",
        "--port", str(args.vllm_port),
        "--trust-remote-code",
        "--max-model-len", str(args.max_model_len),
        "--enable-auto-tool-choice",
        "--tool-call-parser", args.tool_call_parser,
        "--reasoning-parser", args.reasoning_parser,
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--return-tokens-as-token-ids",
        "--logprobs-mode", "processed_logprobs",
        "--weight-transfer-config", '{"backend":"nccl"}',
    ]  # fmt: skip
    proc = _spawn(vllm_cmd, vllm_log, env={"CUDA_VISIBLE_DEVICES": args.vllm_device, "VLLM_SERVER_DEV_MODE": "1"})
    wait_for_vllm(vllm_url, proc, vllm_log)

    if not args.skip_warm:
        warm_sandbox_template(args, vllm_url, logs)

    # 3. the training script, downloaded rather than duplicated so Jobs and local runs cannot drift
    script = pathlib.Path("async_grpo_harbor.py")
    print(f"[launcher] fetching {args.train_script_url}", flush=True)
    try:
        urllib.request.urlretrieve(args.train_script_url, script)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"could not download the training script from {args.train_script_url}: {exc}") from exc

    stamp = os.environ.get("HF_JOB_ID") or os.environ.get("JOB_ID") or time.strftime("%m%d-%H%M%S")
    cmd = [
        sys.executable, str(script),
        "--server", f"http://127.0.0.1:{args.server_port}",
        "--vllm-url", vllm_url,
        "--model", args.model,
        "--split", args.split,
        "--harness", args.harness,
        "--sandbox", args.sandbox,
        "--output-dir", str(data_root / "runs" / f"{args.model.split('/')[-1]}-{args.harness}-{stamp}"),
        *forwarded,
    ]  # fmt: skip
    print(f"[launcher] proxy {public_proxy}  (what the sandboxed agent calls)", flush=True)
    print(f"[launcher] train {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": args.train_device}, check=False).returncode
    print(f"[launcher] training exited {rc}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup()
        sys.stdout.flush()
