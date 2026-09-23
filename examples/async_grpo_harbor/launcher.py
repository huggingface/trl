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
#     "openenv[harbor] @ git+https://github.com/huggingface/OpenEnv.git@34825a772ae54760fb6bf7a8b2073a4a85714004",
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

The trainer and vLLM share a host for NCCL weight sync. The capture proxy is tunneled to the remote
sandbox. The training script is downloaded from `--train-script-url`; OpenEnv uses the revision below.

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
    -- --max-steps 20
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
import threading
import time
import urllib.error
import urllib.request
import uuid


# Use --train-script-url to test an unmerged revision.
TRAIN_SCRIPT_URL = (
    "https://raw.githubusercontent.com/huggingface/trl/main/examples/async_grpo_harbor/async_grpo_harbor.py"
)
DATA_ROOT = pathlib.Path(os.environ.get("DATA_ROOT", "/data"))
OPENENV_REVISION = "34825a772ae54760fb6bf7a8b2073a4a85714004"

_children: list[subprocess.Popen] = []
_stopping = threading.Event()


def _spawn(cmd: list[str], log_path: pathlib.Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    """Start a child in a process group so cleanup also stops its workers."""
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


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def _cleanup(*_: object) -> None:
    _stopping.set()
    for proc in list(_children):
        _stop_process(proc)


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
    """Return the public URL after its health endpoint reaches the capture proxy."""
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


def _serve_cmd(args: argparse.Namespace) -> list[str]:
    return [
        "openenv", "harbor", "serve",
        "--dataset", args.split,
        "--port", str(args.server_port),
        "--capture-port", str(args.capture_port),
        "--expose", "gradio",
    ]  # fmt: skip


def start_harbor_server(args: argparse.Namespace, log: pathlib.Path) -> str:
    """Start the server and return its verified public proxy URL."""
    _spawn(_serve_cmd(args), log)
    return wait_for_public_proxy(log, args.capture_port)


def supervise_tunnel(args: argparse.Namespace, log: pathlib.Path, interval_s: float) -> None:
    """Restart after two failed public health checks. Existing sessions may be lost."""
    if interval_s <= 0:
        return
    consecutive = 0
    while not _stopping.wait(interval_s):
        url = _published_url(log, args.capture_port)
        if url and _http_json_has(f"{url}/health", "status"):
            consecutive = 0
            continue
        consecutive += 1
        print(f"[launcher] tunnel probe failed ({consecutive}/2) for {url or '<no url yet>'}", flush=True)
        if consecutive < 2:
            continue
        print("[launcher] the published proxy is not reachable; restarting the Harbor server", flush=True)
        for proc in list(_children):
            if "harbor" in proc.args:
                _stop_process(proc)
                _children.remove(proc)
        if _stopping.is_set():
            return
        try:
            new_url = start_harbor_server(args, log)
            print(f"[launcher] Harbor server back up; proxy now {new_url}", flush=True)
            consecutive = 0
        except RuntimeError as exc:
            print(f"[launcher] WARNING could not republish the proxy: {exc}", flush=True)


def _published_url(log: pathlib.Path, capture_port: int) -> str | None:
    """The most recent published URL in the server log, or None."""
    if not log.exists():
        return None
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
    found = re.findall(rf"capture\s+:{capture_port}\s+->\s+(https://\S+)", text)
    return found[-1].rstrip(".,") if found else None


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
    """Warm task 0's sandbox image before concurrent rollouts can race its first build."""
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
        # A verifier failure does not imply that the image build failed.
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
    p.add_argument("--tunnel-check-s", type=float, default=60.0)  # 0 disables the supervisor
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

    for tool in ("openenv", "vllm", "git"):
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

    # The pinned wheel omits harbor_env.harness; import it from the matching source checkout.
    source = data_root / f"openenv-{uuid.uuid4().hex}"
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "fetch",
            "--depth=1",
            "https://github.com/huggingface/OpenEnv.git",
            OPENENV_REVISION,
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)
    env_path = str(source.resolve() / "envs")
    if os.environ.get("PYTHONPATH"):
        env_path += os.pathsep + os.environ["PYTHONPATH"]

    server_log = logs / "openenv-server.log"
    public_proxy = start_harbor_server(args, server_log)
    if args.tunnel_check_s > 0:
        threading.Thread(target=supervise_tunnel, args=(args, server_log, args.tunnel_check_s), daemon=True).start()

    # Serve processed logprobs and engine token IDs for training.
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

    script = pathlib.Path("async_grpo_harbor.py")
    print(f"[launcher] fetching {args.train_script_url}", flush=True)
    try:
        urllib.request.urlretrieve(args.train_script_url, script)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"could not download the training script from {args.train_script_url}: {exc}") from exc

    stamp = os.environ.get("HF_JOB_ID") or os.environ.get("JOB_ID") or "local"
    run_name = f"{args.model.split('/')[-1]}-{args.harness}-{stamp}-{uuid.uuid4().hex[:8]}"
    cmd = [
        sys.executable, str(script),
        "--server", f"http://127.0.0.1:{args.server_port}",
        "--vllm-url", vllm_url,
        "--model", args.model,
        "--split", args.split,
        "--harness", args.harness,
        "--sandbox", args.sandbox,
        "--run-name", run_name,
        "--output-dir", str(data_root / "runs" / run_name),
        *forwarded,
    ]  # fmt: skip
    print(f"[launcher] proxy {public_proxy}  (what the sandboxed agent calls)", flush=True)
    print(f"[launcher] train {' '.join(cmd)}", flush=True)
    rc = subprocess.run(
        cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": args.train_device, "PYTHONPATH": env_path}, check=False
    ).returncode
    print(f"[launcher] training exited {rc}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup()
        sys.stdout.flush()
