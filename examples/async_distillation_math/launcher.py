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
#     "trl[vllm,kernels] @ git+https://github.com/huggingface/trl.git",
#     "trackio",
# ]
# ///

"""Run `async_distillation_mopd.py` on Hugging Face Jobs: one job, one command.

A Job gives you one container and one command, and MOPD needs five processes. The launcher wraps them on one
4-GPU host:

    vllm serve MixSFT    (GPU 0)   the student, which the trainer syncs weights into over NCCL
    vllm serve RL-Math   (GPU 1)   the math teacher, only scored
    vllm serve RL-IF     (GPU 1)   the instruction-following teacher, sharing GPU 1 with the math one
    vllm serve RL-Code   (GPU 2)   the code teacher, alone since code prompts are the longest
    async_distillation_mopd.py (GPU 3)   the training script, downloaded and run unmodified

Every teacher gets a larger `--max-model-len` than the student: scoring sends the student's whole sequence plus one
token. The two teachers on GPU 1 start one after the other, since vLLM sizes its cache from the memory free at
startup.

Requirements:
  - A Hugging Face account with a positive credit balance; Jobs is pay-as-you-go.
  - `HF_TOKEN` as a job secret, for the Trackio dashboard the training script reports to.

Run:

```sh
hf jobs uv run \
    --flavor h200x4 \
    --image huggingface/trl \
    --secrets HF_TOKEN \
    --timeout 6h \
    https://raw.githubusercontent.com/huggingface/trl/main/examples/async_distillation_math/launcher.py
```
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


# Use --train-script-url to test an unmerged revision.
TRAIN_SCRIPT_URL = "https://raw.githubusercontent.com/huggingface/trl/main/examples/async_distillation_math/async_distillation_mopd.py"
CHECKPOINTS = "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B"

_children: list[subprocess.Popen] = []


def _spawn(cmd: list[str], log_path: pathlib.Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    """Start a child in a process group so cleanup also stops its workers."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        stdout=open(log_path, "w", encoding="utf-8"),
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


def _cleanup() -> None:
    for proc in reversed(list(_children)):
        _stop_process(proc)


def _handle_signal(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def _http_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError):
        return False


def wait_for_vllm(url: str, proc: subprocess.Popen, log: pathlib.Path, deadline_s: float = 1200.0) -> None:
    """Block until vLLM answers /health, failing fast with its log if it died instead."""
    started = time.monotonic()
    while time.monotonic() - started < deadline_s:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM exited {proc.returncode} before serving\n{log.read_text(encoding='utf-8', errors='replace')[-2000:]}"
            )
        if _http_ok(f"{url}/health"):
            print(f"[launcher] vLLM ready at {url}", flush=True)
            return
        time.sleep(5)
    raise RuntimeError(f"vLLM did not become ready in {deadline_s:.0f}s")


def start_teacher(name: str, port: int, device: str, memory: float, args: argparse.Namespace, logs: pathlib.Path):
    cmd = [
        "vllm", "serve", f"{CHECKPOINTS}-{name}",
        "--port", str(port),
        "--max-model-len", str(args.max_model_len + 64),
        "--gpu-memory-utilization", str(memory),
        "--logprobs-mode", "processed_logprobs",
        "--max-logprobs", "-1",
    ]  # fmt: skip
    log = logs / f"vllm-{name}.log"
    return _spawn(cmd, log, env={"CUDA_VISIBLE_DEVICES": device}), f"http://127.0.0.1:{port}", log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Prompt (the longest in the prompt mix is under 2048 tokens) + the script's max_completion_length (16384)
    p.add_argument("--max-model-len", type=int, default=18432)
    p.add_argument("--train-script-url", default=TRAIN_SCRIPT_URL)
    p.add_argument("--logs", default="./logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    logs = pathlib.Path(args.logs)

    if shutil.which("vllm") is None:
        raise RuntimeError("`vllm` is not on PATH; the PEP 723 dependencies did not install")
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
    if n_gpu < 4:
        raise RuntimeError(
            f"needs 4 GPUs (student, teachers, trainer on one host) but sees {n_gpu}; try --flavor h200x4"
        )

    student_url = "http://127.0.0.1:8000"
    student_log = logs / "vllm-student.log"
    student_cmd = [
        "vllm", "serve", f"{CHECKPOINTS}-MixSFT",
        "--port", "8000",
        "--max-model-len", str(args.max_model_len),
        "--weight-transfer-config", '{"backend":"nccl"}',
    ]  # fmt: skip
    student = _spawn(student_cmd, student_log, env={"CUDA_VISIBLE_DEVICES": "0", "VLLM_SERVER_DEV_MODE": "1"})

    code, code_url, code_log = start_teacher("RL-Code", 8002, "2", 0.85, args, logs)
    # The two teachers sharing GPU 1 start one after the other
    for name, port in (("RL-Math", 8001), ("RL-IF", 8003)):
        proc, url, log = start_teacher(name, port, "1", 0.42, args, logs)
        wait_for_vllm(url, proc, log)
    wait_for_vllm(code_url, code, code_log)
    wait_for_vllm(student_url, student, student_log)

    script = pathlib.Path("async_distillation_mopd.py")
    print(f"[launcher] fetching {args.train_script_url}", flush=True)
    try:
        urllib.request.urlretrieve(args.train_script_url, script)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"could not download the training script from {args.train_script_url}: {exc}") from exc

    print("[launcher] starting training on GPU 3", flush=True)
    trainer = subprocess.Popen([sys.executable, str(script)], env={**os.environ, "CUDA_VISIBLE_DEVICES": "3"})
    _children.append(trainer)
    rc = trainer.wait()
    print(f"[launcher] training exited {rc}", flush=True)
    raise SystemExit(rc)


if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup()
