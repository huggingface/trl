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
#     "trl @ git+https://github.com/younesbelkada/trl.git@async-grpo-harbor-slurm",
#     "openenv[harbor] @ git+https://github.com/huggingface/OpenEnv.git@86a180ede21e044f7929b9a7783ad83aa67d83a3",
#     "vllm==0.25.1",
#     "flashinfer-jit-cache==0.6.13",
#     "kernels>=0.17.0,<0.18.0",
#     "flash-linear-attention",
#     "ray",
#     "datasets>=3.2",
#     "peft",
#     "trackio",
#     "transformers>=5.2",
#     "huggingface_hub>=1.22",
# ]
#
# [[tool.uv.index]]
# name = "flashinfer"
# url = "https://flashinfer.ai/whl/cu130"
# explicit = true
#
# [tool.uv.sources]
# flashinfer-jit-cache = { index = "flashinfer" }
# ///
# vllm 0.25.1 pins flashinfer 0.6.13, and the JIT cache must match it. vllm>=0.30 needs mcp>=2, which breaks the
# fastmcp<4 the pinned OpenEnv imports. kernels 0.17.x is what transformers accepts for the flash-attn3 hub kernel,
# flash-linear-attention gives Qwen3.5 its fast linear-attention path, and ray is imported by vLLM's IPC weight transfer.

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

Add a bucket so checkpoints, logs and the trackio database outlive the container, which is otherwise ephemeral:

```sh
hf jobs uv run --flavor h200x2 --image huggingface/trl \
    --secrets HF_TOKEN --secrets E2B_API_KEY --timeout 7200s \
    --volume hf://buckets/<user>/<bucket>:/outputs \
    https://raw.githubusercontent.com/huggingface/trl/main/examples/async_grpo_harbor/launcher.py \
    -- --data-root /outputs --max-steps 20
```

(`/data` is reserved for Jobs artifacts when the script is a local file, hence `/outputs` and `--data-root`.)

Anything after `--` is forwarded to the training script, so its full argument surface is available:
`--model`, `--split`, `--harness`, `--sandbox`, `--max-steps`, and so on.

One GPU, LoRA, HF sandboxes: vLLM and the trainer share the GPU (`--weight-transfer ipc`), each rollout runs in an HF
sandbox job, and `--n-tasks`/`--difficulty` select the tasks (downloaded and patched at start-up). `HF_TOKEN` is the
only secret: the tasks download their data with it and HF sandboxes authenticate with it. Hopper only (flash-attn3).

```sh
hf jobs uv run --flavor h200 --image huggingface/trl --secrets HF_TOKEN --timeout 12h \
    --volume hf://buckets/<user>/<bucket>:/outputs \
    https://raw.githubusercontent.com/huggingface/trl/main/examples/async_grpo_harbor/launcher.py -- \
    --data-root /outputs \
    --train-script-url https://raw.githubusercontent.com/huggingface/trl/main/examples/async_grpo_harbor/async_grpo_harbor.py \
    --model Qwen/Qwen3.5-0.8B --harness pi --split FineEnvs/SmolDataEnvs-harbor-train --n-tasks 1000 --difficulty easy \
    --sandbox hf-sandbox --docker-image docker.io/savatar101/env-data-agent-train:base \
    --vllm-device 0 --train-device 0 --weight-transfer ipc --gpu-memory-utilization 0.3 --token-budget 65536 \
    --max-inflight 32 --max-steps 200 --save-steps 25 --save-total-limit 0 --lora-r 32 --learning-rate 2e-5 \
    --attn-implementation kernels-community/flash-attn3 \
    --num-generations 8 --temperature 0.8 --max-completion-length 16384 --agent-step-limit 17 --agent-timeout 600 \
    --per-device-train-batch-size 4 --gradient-accumulation-steps 4
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tarfile
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
OPENENV_REVISION = "86a180ede21e044f7929b9a7783ad83aa67d83a3"

_children: list[subprocess.Popen] = []
_stopping = threading.Event()


def _spawn(cmd: list[str], log_path: pathlib.Path | None, env: dict[str, str] | None = None) -> subprocess.Popen:
    """Start a child in a process group so cleanup also stops its workers."""
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        stdout=open(log_path, "w") if log_path is not None else None,
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
    _stopping.set()
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


def _http_json_has(url: str, key: str, timeout: float = 15.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            import json

            return key in json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_for_public_proxy(log: pathlib.Path, capture_port: int, deadline_s: float = 420.0) -> str:
    """Return the public URL after its health endpoint reaches the capture proxy."""
    started = time.monotonic()
    url: str | None = None
    while time.monotonic() - started < deadline_s:
        latest = _published_url(log, capture_port)
        if latest and latest != url:
            url = latest
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
    # The factory keeps one metadata connection in addition to its rollout sessions.
    _spawn(_serve_cmd(args), log, env={"MAX_CONCURRENT_ENVS": str(args.max_inflight + 1)})
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
            if proc.args[:3] == ["openenv", "harbor", "serve"]:
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
        # Its own capture proxy; the default 8100 collides when several launchers share a host.
        "--port", str(args.capture_port + 1),
    ]  # fmt: skip
    log = logs / "warm-template.log"
    print("[launcher] warming the sandbox template (one serial rollout) ...", flush=True)
    started = time.monotonic()
    rc = _spawn(cmd, log).wait()
    took = time.monotonic() - started
    if rc != 0:
        # A verifier failure does not imply that the image build failed.
        print(f"[launcher] WARNING warm rollout exited {rc} after {took:.0f}s; continuing", flush=True)
        print(f"[launcher] WARNING tail:\n{log.read_text(errors='replace')[-800:]}", flush=True)
    else:
        print(f"[launcher] template warm after {took:.0f}s", flush=True)


def merge_lora(adapter: str, out: pathlib.Path) -> str:
    """Merge a LoRA checkpoint into its base model (the class TRL trained) so vLLM can serve it."""
    import torch
    import transformers
    from peft import PeftModel

    base = json.loads((pathlib.Path(adapter) / "adapter_config.json").read_text())["base_model_name_or_path"]
    architecture = getattr(transformers, transformers.AutoConfig.from_pretrained(base).architectures[0])
    model = PeftModel.from_pretrained(architecture.from_pretrained(base, dtype=torch.bfloat16), adapter)
    model.merge_and_unload().save_pretrained(out)
    transformers.AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
    return str(out)


def run_eval(args: argparse.Namespace, data_root: pathlib.Path, env_path: str) -> int:
    """Evaluate each of `--eval-checkpoints` in turn on the prepared test tasks; results go to <data root>/eval/."""
    script = pathlib.Path("evaluate_harbor.py")
    urllib.request.urlretrieve(args.eval_script_url, script)
    # A run folder stands for all of its checkpoints, in step order.
    checkpoints = []
    for path in args.eval_checkpoints:
        found = sorted(pathlib.Path(path).glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
        checkpoints += [str(p) for p in found] if found else [path]
    rc = 0
    for checkpoint in checkpoints:
        # /outputs/runs/<run>/checkpoint-N -> runs_<run>_checkpoint-N; a Hub id keeps its name.
        label = re.sub(r"[^\w.-]+", "_", str(pathlib.Path(checkpoint).resolve().relative_to(data_root.resolve()))
                       if checkpoint.startswith(str(data_root)) else checkpoint)
        out = data_root / "eval" / label
        out.mkdir(parents=True, exist_ok=True)
        summary = out / "summary.json"
        if summary.exists():
            done = json.loads(summary.read_text())
            same_harnesses = sorted(done["harnesses"]) == sorted(args.eval_harnesses.split(","))
            if same_harnesses and done["overall"]["coverage"] >= 1:
                print(f"[launcher] {checkpoint} already evaluated, skipping ({summary})", flush=True)
                continue
        trials = pathlib.Path("trials") / label  # local disk: thousands of small files are slow on the bucket
        print(f"[launcher] evaluating {checkpoint} -> {out}", flush=True)

        served = checkpoint
        if (pathlib.Path(checkpoint) / "adapter_config.json").exists():
            served = merge_lora(checkpoint, pathlib.Path("merged") / label)
        # One server per policy, so its trial folders stay apart; the factories keep one connection per harness.
        server_log = out / "openenv-server.log"
        server = _spawn(_serve_cmd(args), server_log, env={
            "MAX_CONCURRENT_ENVS": str(args.max_inflight + 8), "OPENENV_HARBOR_TRIALS_DIR": str(trials.resolve())})
        vllm_url = f"http://127.0.0.1:{args.vllm_port}"
        vllm_cmd = [
            "vllm", "serve", served,
            "--served-model-name", args.model,  # the name the harnesses ask for
            "--language-model-only",  # checkpoints carry no image processor, and the agents send text only
            "--host", "0.0.0.0", "--port", str(args.vllm_port), "--trust-remote-code",
            "--max-model-len", str(args.max_model_len),
            "--enable-auto-tool-choice", "--tool-call-parser", args.tool_call_parser,
            "--reasoning-parser", args.reasoning_parser,
            "--default-chat-template-kwargs", '{"enable_thinking": false}',
            "--return-tokens-as-token-ids", "--logprobs-mode", "processed_logprobs",
        ]  # fmt: skip
        if args.gpu_memory_utilization is not None:
            vllm_cmd += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
        vllm = _spawn(vllm_cmd, out / "vllm.log", env={"CUDA_VISIBLE_DEVICES": args.vllm_device})
        try:
            wait_for_public_proxy(server_log, args.capture_port)
            wait_for_vllm(vllm_url, vllm, out / "vllm.log")
            cmd = [
                sys.executable, str(script),
                "--server", f"http://127.0.0.1:{args.server_port}", "--vllm-url", vllm_url,
                "--model", args.model, "--checkpoint", checkpoint, "--split", args.split,
                "--harnesses", args.eval_harnesses, "--sandbox", args.sandbox,
                "--n-tasks", str(args.n_tasks), "--concurrency", str(args.max_inflight), "--output-dir", str(out),
            ]  # fmt: skip
            with open(out / "eval.log", "w") as log:
                env = {**os.environ, "PYTHONPATH": env_path}
                rc |= subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            print((out / "eval.log").read_text().rsplit("\n\n", 1)[-1], flush=True)  # the summary table
        finally:
            for proc in (vllm, server):
                _stop_process(proc)
                _children.remove(proc)
            if trials.exists():  # one archive per policy, for analyze_trajectories.py / compare_trajectories.py
                with tarfile.open(out / "trials.tar.gz", "w:gz") as tar:
                    tar.add(trials, arcname="trials")
    return rc


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--split", default="AdithyaSK/data_agent_rl_environment_train")
    p.add_argument("--harness", default="mini-swe-agent")
    p.add_argument("--sandbox", default="e2b")
    p.add_argument("--max-inflight", type=int, default=8)
    p.add_argument("--server-port", type=int, default=8200)
    p.add_argument("--capture-port", type=int, default=8300)
    p.add_argument("--vllm-port", type=int, default=8000)
    p.add_argument("--vllm-device", default="0")
    p.add_argument("--train-device", default="1")  # the trainer; NCCL weight sync needs the same host
    # `ipc` shares one GPU between vLLM and the trainer (same --vllm-device and --train-device); `nccl` needs two.
    p.add_argument("--weight-transfer", choices=("nccl", "ipc"), default="nccl")
    p.add_argument("--gpu-memory-utilization", type=float, default=None, help="vLLM's share; lower it when colocated")
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--tool-call-parser", default="qwen3_xml")  # Qwen3.5; use `hermes` for most others
    p.add_argument("--reasoning-parser", default="qwen3")
    p.add_argument("--train-script-url", default=TRAIN_SCRIPT_URL)
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--skip-warm", action="store_true")
    p.add_argument("--tunnel-check-s", type=float, default=60.0)  # 0 disables the supervisor
    # With --n-tasks, a Hub --split is downloaded once into the data root (only the selected tasks), patched below, and
    # served from local disk. --n-tasks is also forwarded to the training script.
    p.add_argument("--n-tasks", type=int, default=None)
    p.add_argument("--difficulty", default="", help="e.g. `easy`, `easy,medium` or `easy:33,medium:17`")
    p.add_argument("--docker-image", default="", help="prebuilt task image, required by --sandbox hf-sandbox")
    p.add_argument("--sandbox-memory-mb", type=int, default=4096)  # tasks ship 1024, too little for agents + pandas
    # Evaluation mode: no training. Each policy (Hub id or checkpoint folder) is served in turn and evaluated with
    # evaluate_harbor.py on the --n-tasks prepared tasks of --split, under every --eval-harnesses.
    p.add_argument("--eval-checkpoints", nargs="+", default=None)
    p.add_argument("--eval-harnesses", default="opencode,mini-swe-agent,pi,openhands-sdk")
    p.add_argument("--eval-script-url", default=TRAIN_SCRIPT_URL.replace("async_grpo_harbor.py", "evaluate_harbor.py"))
    return p.parse_known_args()


def prepare_tasks(args: argparse.Namespace, target: pathlib.Path) -> str:
    """Download the first `--n-tasks` tasks of `--split` (filtered by `--difficulty`) and patch their task.toml."""
    from huggingface_hub import HfApi, snapshot_download

    names = sorted(e.path for e in HfApi().list_repo_tree(args.split, path_in_repo="tasks", repo_type="dataset"))
    # {difficulty: how many to keep, None for all}
    quotas = {d: int(n) if n else None for d, _, n in (s.partition(":") for s in args.difficulty.split(",") if s)}
    if quotas:
        # Read every task.toml from a cache outside `target`: a folder holding only task.toml would count as a task.
        tomls = snapshot_download(args.split, repo_type="dataset", allow_patterns=[f"{p}/task.toml" for p in names],
                                  cache_dir=f"{target}-toml-cache")
        found = {p: re.search(r'^difficulty = "(\w+)"', pathlib.Path(tomls, p, "task.toml").read_text(), re.M) for p in names}
        picked = {d: [p for p in names if found[p] and found[p][1] == d][:n] for d, n in quotas.items()}
        names = sorted(p for selected in picked.values() for p in selected)
        print("[launcher] selected " + ", ".join(f"{len(s)} {d}" for d, s in picked.items()), flush=True)
    names = names[: args.n_tasks]
    # local_dir writes real files: Harbor tars tests/ into the sandbox and HF cache symlinks would dangle.
    snapshot_download(args.split, repo_type="dataset", allow_patterns=[f"{p}/**" for p in names], local_dir=target)
    for p in names:
        toml = target / p / "task.toml"
        # The grader's LLM judge key is optional, but `${VAR}` makes Harbor require it on the host.
        text = toml.read_text().replace('"${OPENAI_API_KEY}"', '"${OPENAI_API_KEY:-}"')
        # OpenEnv passes no memory override to Harbor, so the sandbox size comes from task.toml.
        text = re.sub(r"^memory_mb = .*$", f"memory_mb = {args.sandbox_memory_mb}", text, flags=re.M)
        if args.docker_image and "docker_image" not in text:
            text = re.sub(r"^\[environment\]$", f'[environment]\ndocker_image = "{args.docker_image}"', text, count=1, flags=re.M)
        toml.write_text(text)
    print(f"[launcher] prepared {len(names)} tasks in {target}", flush=True)
    return str(target)


def patch_harbor_untar() -> None:
    """Make Harbor's remote untar skip restoring file owners, which HF sandboxes refuse (`Cannot change ownership`).

    Harbor packs uploads with the host's uid and untars them as root, so every verifier upload fails there. Remove
    once `remote_unpack_command` in harbor/environments/tar_transfer.py passes `--no-same-owner` upstream.
    """
    import harbor.environments.tar_transfer as tar_transfer

    path = pathlib.Path(tar_transfer.__file__)
    text = path.read_text()
    if "--no-same-owner" not in text:
        path.write_text(text.replace("tar -xzf ", "tar --no-same-owner -xzf "))


def main() -> None:
    args, forwarded = parse_args()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    data_root = pathlib.Path(args.data_root)
    mounted = data_root.is_dir()
    if not mounted:
        print(f"[launcher] WARNING {data_root} is not a directory, so no bucket is mounted: checkpoints", flush=True)
        print("[launcher] WARNING and logs will NOT survive this job.", flush=True)
        data_root = pathlib.Path("./out")
    # One folder per job: jobs sharing a bucket would otherwise truncate each other's logs, including the server log
    # each launcher reads its public proxy URL from.
    logs = data_root / "logs" / (os.environ.get("JOB_ID") or os.environ.get("HF_JOB_ID") or "local")
    logs.mkdir(parents=True, exist_ok=True)
    if mounted:
        # HF_HOME stays on local disk: gradio downloads its tunnel binary there and must execute it, which a mounted
        # bucket does not allow. Only the trackio database goes to the bucket, one per job like the logs: jobs writing
        # one SQLite file over the mount at the same time lose rows.
        os.environ.setdefault("TRACKIO_DIR", str(data_root / "trackio" / logs.name))

    for tool in ("openenv", "vllm", "git"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"`{tool}` is not on PATH; the PEP 723 dependencies did not install")
    if args.sandbox == "e2b" and not os.environ.get("E2B_API_KEY"):
        raise RuntimeError("sandbox is e2b but E2B_API_KEY is unset; pass it with --secrets E2B_API_KEY")
    if args.sandbox == "hf-sandbox":
        if not args.docker_image:
            raise RuntimeError("hf-sandbox cannot build a task's Dockerfile: pass --docker-image with a prebuilt image")
        patch_harbor_untar()
    if args.n_tasks and not os.path.isdir(args.split):
        # Served from local disk: thousands of small files are slow to write to a bucket and slow for the server to read
        # there. The bucket keeps one archive per task selection, so a relaunch unpacks it instead of preparing again.
        target = pathlib.Path("tasks").resolve()
        selection = [args.split, args.n_tasks, args.difficulty, args.docker_image, args.sandbox_memory_mb]
        archive = data_root / f"tasks-{hashlib.sha1(json.dumps(selection).encode()).hexdigest()[:12]}.tar.gz"
        if archive.exists():
            with tarfile.open(archive) as tar:
                tar.extractall(target, filter="data")
            print(f"[launcher] unpacked the tasks prepared earlier from {archive}", flush=True)
        else:
            prepare_tasks(args, target)
            partial = archive.with_name(archive.name + ".partial")
            with tarfile.open(partial, "w:gz") as tar:
                tar.add(target, arcname=".")
            partial.rename(archive)  # a crash mid-write never leaves a truncated archive under the final name
            print(f"[launcher] saved the prepared tasks to {archive}", flush=True)
        args.split = str(target)
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
    # Evaluation serves the policy only; training also runs the trainer, on its own GPU unless colocated.
    needed = 1 if args.eval_checkpoints else len({args.vllm_device, args.train_device})
    if n_gpu < needed:
        raise RuntimeError(f"needs {needed} GPU(s) (engine + trainer on one host) but sees {n_gpu}; try --flavor h200x2")
    if not args.eval_checkpoints and needed == 1 and args.weight_transfer != "ipc":
        raise RuntimeError("vLLM and the trainer share a GPU, which NCCL cannot do: pass --weight-transfer ipc")

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

    if args.eval_checkpoints:
        sys.exit(run_eval(args, data_root, env_path))

    server_log = logs / "openenv-server.log"
    public_proxy = start_harbor_server(args, server_log)
    if args.tunnel_check_s > 0:
        threading.Thread(target=supervise_tunnel, args=(args, server_log, args.tunnel_check_s), daemon=True).start()

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
        "--weight-transfer-config", f'{{"backend":"{args.weight_transfer}"}}',
    ]  # fmt: skip
    if args.gpu_memory_utilization is not None:
        vllm_cmd += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    vllm_env = {"CUDA_VISIBLE_DEVICES": args.vllm_device, "VLLM_SERVER_DEV_MODE": "1"}
    if args.weight_transfer == "ipc":
        vllm_env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # the IPC handles arrive pickled over HTTP
    proc = _spawn(vllm_cmd, vllm_log, env=vllm_env)
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
        "--max-inflight", str(args.max_inflight),
        "--weight-transfer", args.weight_transfer,
        *(["--n-tasks", str(args.n_tasks)] if args.n_tasks else []),
        "--run-name", run_name,
        "--output-dir", str(data_root / "runs" / run_name),
        *forwarded,
    ]  # fmt: skip
    print(f"[launcher] proxy {public_proxy}  (what the sandboxed agent calls)", flush=True)
    print(f"[launcher] train {' '.join(cmd)}", flush=True)
    rc = _spawn(cmd, None, env={"CUDA_VISIBLE_DEVICES": args.train_device, "PYTHONPATH": env_path}).wait()
    print(f"[launcher] training exited {rc}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup()
        sys.stdout.flush()
