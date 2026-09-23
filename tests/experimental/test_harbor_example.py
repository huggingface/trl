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

import importlib.util
import signal
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def load_example(name):
    path = Path(__file__).resolve().parents[2] / "examples" / "async_grpo_harbor" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def launcher():
    return load_example("launcher")


def test_zero_interval_disables_supervision(launcher, monkeypatch):
    probe = MagicMock()
    monkeypatch.setattr(launcher, "_published_url", probe)
    launcher.supervise_tunnel(SimpleNamespace(), Path("unused"), 0)
    probe.assert_not_called()


@pytest.mark.parametrize("max_inflight", [2, 8, 16])
def test_server_reserves_metadata_connection(launcher, monkeypatch, max_inflight):
    spawn = MagicMock()
    monkeypatch.setattr(launcher, "_spawn", spawn)
    monkeypatch.setattr(launcher, "wait_for_public_proxy", lambda *args: "https://proxy.example")
    args = SimpleNamespace(split="tasks", server_port=8200, capture_port=8300, max_inflight=max_inflight)
    launcher.start_harbor_server(args, Path("unused"))
    assert int(spawn.call_args.kwargs["env"]["MAX_CONCURRENT_ENVS"]) > max_inflight


def test_restart_waits_for_server_exit(launcher, monkeypatch):
    proc = MagicMock(args=["openenv", "harbor", "serve"], pid=12345)
    proc.poll.return_value = None
    launcher._children.append(proc)
    launcher._stopping = MagicMock()
    launcher._stopping.wait.side_effect = [False, False, True]
    launcher._stopping.is_set.return_value = False
    monkeypatch.setattr(launcher, "_published_url", lambda *args: None)
    kill = MagicMock()
    monkeypatch.setattr(launcher.os, "killpg", kill)

    def restart(*args):
        proc.wait.assert_called_once_with(timeout=30)
        kill.assert_called_once_with(12345, signal.SIGTERM)
        assert proc not in launcher._children
        return "https://proxy.example"

    start = MagicMock(side_effect=restart)
    monkeypatch.setattr(launcher, "start_harbor_server", start)
    launcher.supervise_tunnel(SimpleNamespace(capture_port=8300), Path("unused"), 1)
    start.assert_called_once()


def test_slow_process_is_killed_and_reaped(launcher, monkeypatch):
    proc = MagicMock(pid=12345)
    proc.poll.return_value = None
    proc.wait.side_effect = [subprocess.TimeoutExpired("server", 30), 0]
    kill = MagicMock()
    monkeypatch.setattr(launcher.os, "killpg", kill)
    launcher._stop_process(proc)
    assert [call.args[1] for call in kill.call_args_list] == [signal.SIGTERM, signal.SIGKILL]
    assert [call.kwargs["timeout"] for call in proc.wait.call_args_list] == [30, 5]


def test_stop_releases_real_listening_port(launcher):
    script = (
        "import signal, socket, time\n"
        "server = socket.socket()\n"
        "server.bind(('127.0.0.1', 0))\n"
        "server.listen()\n"
        "def stop(*args):\n"
        "    time.sleep(0.1)\n"
        "    server.close()\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "print(server.getsockname()[1], flush=True)\n"
        "while True:\n"
        "    time.sleep(1)\n"
    )
    proc = subprocess.Popen([sys.executable, "-u", "-c", script], stdout=subprocess.PIPE, start_new_session=True)
    try:
        port = int(proc.stdout.readline())
        launcher._stop_process(proc)
        assert proc.returncode == 0
        with socket.socket() as replacement:
            replacement.bind(("127.0.0.1", port))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()


def test_jobs_launcher_uses_pinned_source_and_unique_names(launcher, monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        ["launcher", "--sandbox", "daytona", "--data-root", str(tmp_path), "--skip-warm", "--tunnel-check-s", "0"],
    )
    monkeypatch.setenv("HF_JOB_ID", "job42")
    monkeypatch.setattr(launcher.signal, "signal", MagicMock())
    monkeypatch.setattr(launcher.shutil, "which", lambda tool: f"/bin/{tool}")
    run = MagicMock(return_value=SimpleNamespace(stdout="0\n1\n", returncode=0))
    monkeypatch.setattr(launcher.subprocess, "run", run)
    monkeypatch.setattr(launcher, "_spawn", MagicMock())
    monkeypatch.setattr(launcher, "start_harbor_server", lambda *args: "https://proxy.example")
    monkeypatch.setattr(launcher, "wait_for_vllm", MagicMock())
    monkeypatch.setattr(launcher.urllib.request, "urlretrieve", MagicMock())
    thread = MagicMock()
    monkeypatch.setattr(launcher.threading, "Thread", thread)
    for _ in range(2):
        with pytest.raises(SystemExit) as stopped:
            launcher.main()
        assert stopped.value.code == 0
    thread.assert_not_called()
    commands = [call for call in run.call_args_list if call.args[0][0] == sys.executable]
    names = [call.args[0][call.args[0].index("--run-name") + 1] for call in commands]
    assert names[0] != names[1] and all("job42" in name for name in names)
    for call, name in zip(commands, names, strict=True):
        command = call.args[0]
        assert command[command.index("--max-inflight") + 1] == "8"
        assert Path(command[command.index("--output-dir") + 1]).name == name
        assert call.kwargs["env"]["PYTHONPATH"].split(launcher.os.pathsep)[0].endswith("/envs")
    fetches = [call.args[0] for call in run.call_args_list if "fetch" in call.args[0]]
    assert len(fetches) == 2 and all(command[-1] == launcher.OPENENV_REVISION for command in fetches)


@pytest.mark.parametrize("job_key", [None, "SLURM_JOB_ID", "HF_JOB_ID", "JOB_ID"])
def test_training_run_names_are_unique_and_can_be_overridden(monkeypatch, job_key):
    pytest.importorskip("harbor_env.harness")
    example = load_example("async_grpo_harbor")
    for key in ("SLURM_JOB_ID", "HF_JOB_ID", "JOB_ID"):
        monkeypatch.delenv(key, raising=False)
    if job_key:
        monkeypatch.setenv(job_key, "job42")
    monkeypatch.setattr(sys, "argv", ["example", "--vllm-url", "http://engine", "--split", "tasks"])
    factory = MagicMock()
    factory.return_value.prompt_rows.return_value = [{"prompt": [{"role": "user", "content": "task"}]}]
    monkeypatch.setattr(example, "HarborSessionFactory", factory)
    monkeypatch.setattr(example, "AutoTokenizer", MagicMock())
    monkeypatch.setattr(example, "HarnessRolloutWorker", MagicMock())
    config = MagicMock()
    monkeypatch.setattr(example, "AsyncGRPOConfig", config)
    trainer = MagicMock()
    monkeypatch.setattr(example, "AsyncGRPOTrainer", trainer)
    example.main()
    example.main()
    first, second = [call.kwargs for call in config.call_args_list]
    assert first["run_name"] != second["run_name"]
    assert first["output_dir"] != second["output_dir"]
    assert ("job42" if job_key else "local") in first["run_name"]
    assert factory.call_args.kwargs["sampling"] == {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
    monkeypatch.setattr(sys, "argv", sys.argv + ["--run-name", "explicit", "--output-dir", "runs/custom"])
    example.main()
    assert config.call_args.kwargs["run_name"] == "explicit"
    assert config.call_args.kwargs["output_dir"] == "runs/custom"
    assert trainer.return_value.train.call_count == 3
