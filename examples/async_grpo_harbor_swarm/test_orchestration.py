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

"""Lifecycle regressions with real subprocesses and an external-engine test double."""

import asyncio
import json
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import podman_healthchecks
import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from podman_runtime import SwarmPodmanEnvironment
from runtime import Settings, TeamRunner


def environment(tmp_path):
    directory = tmp_path / "environment"
    directory.mkdir(exist_ok=True)
    (directory / "Dockerfile").write_text("FROM scratch\n")
    return SwarmPodmanEnvironment(
        environment_dir=directory,
        environment_name="lifecycle-test",
        session_id="lifecycle-test",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(),
        auto_mounts=[],
    )


def test_cancelled_exec_terminates_compose_descendants(tmp_path, monkeypatch):
    async def run():
        pid_file = tmp_path / "child.pid"
        script = (
            "import subprocess,time,pathlib; "
            "p=subprocess.Popen(['sleep','120']); "
            f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(120)"
        )
        wrapper = Path(__file__).with_name("podman_compose_compat.py")
        command = (
            "import sys; sys.path.insert(0," + repr(str(wrapper.parent)) + "); "
            "from podman_compose_compat import run; "
            "raise SystemExit(run([sys.executable,'-c'," + repr(script) + "]))"
        )
        engine = tmp_path / "podman"
        engine.write_text(
            "#!"
            + sys.executable
            + "\nimport sys\nif '--help' in sys.argv:\n print('usage: podman-compose'); sys.exit(0)\n"
            + command
            + "\n"
        )
        engine.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ["PATH"])
        collector = asyncio.create_task(environment(tmp_path).exec("sleep 120"))
        for _ in range(100):
            if pid_file.exists():
                break
            await asyncio.sleep(0.05)
        assert pid_file.exists()
        child = int(pid_file.read_text())
        collector.cancel()
        with pytest.raises(asyncio.CancelledError):
            await collector
        for _ in range(100):
            stat = Path(f"/proc/{child}/stat")
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            await asyncio.sleep(0.05)
        else:
            os.kill(child, signal.SIGKILL)
            pytest.fail("Compose child survived cancellation")

    asyncio.run(run())


def test_failed_team_setup_releases_admission(tmp_path):
    class UnavailableInference(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"error":"unavailable"}')

        do_GET = do_POST

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), UnavailableInference)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('[metadata]\nscenario_sha256 = "setup-failure"\n')

    async def run():
        runner = TeamRunner(
            Settings(
                vllm_url=f"http://127.0.0.1:{server.server_port}/v1",
                capture_url="http://127.0.0.1:1",
                output_dir=str(tmp_path / "run"),
            )
        )
        # Use real boards and upstream discovery. More teams than admission slots
        # must all finish when the external inference service refuses setup.
        results = await asyncio.wait_for(
            asyncio.gather(*(runner.run(task, runner.settings.model) for _ in range(6))), 45
        )
        assert all(state["stop_reason"] == "unscorable" and traces == [] for state, traces in results)
        assert runner.active_teams == 0

    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_health_scheduler_recovers_inspection_timeout_and_caches_config(monkeypatch):
    inspections = 0
    checks = 0

    async def engine(*args):
        nonlocal inspections, checks
        if args[0] == "ps":
            return 0, b"123456789abc\n", b""
        if args[0] == "inspect":
            inspections += 1
            if inspections == 1:
                raise TimeoutError("engine busy")
            return (
                0,
                json.dumps(
                    [{"Id": "123456789abc", "Config": {"Healthcheck": {"Test": ["CMD", "true"], "Interval": 1}}}]
                ).encode(),
                b"",
            )
        assert args[0] == "healthcheck"
        checks += 1
        return 0, b"", b""

    # Only the external Podman command is replaced; run the actual scheduler.
    monkeypatch.setattr(podman_healthchecks, "podman", engine)

    async def run():
        task = asyncio.create_task(podman_healthchecks.main())
        try:
            async with asyncio.timeout(8):
                while checks < 2:
                    await asyncio.sleep(0.05)
            assert not task.done()
            assert inspections == 2
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())


def test_daemon_probe_does_not_block_event_loop(tmp_path, monkeypatch):
    # A real slow engine executable models the blocking podman info probe.
    engine = tmp_path / "podman"
    engine.write_text(
        "#!/bin/sh\ncase \"$*\" in *--help*) echo 'usage: podman-compose';; *info*) sleep 0.3; echo linux;; *inspect*) echo linux;; esac\n"
    )
    engine.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ["PATH"])
    SwarmPodmanEnvironment._detect_daemon_os.cache_clear()

    async def run():
        ticks = 0

        async def beat():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.02)

        task = asyncio.create_task(beat())
        try:
            env = environment(tmp_path)
            await env.start(force_build=False)
            assert ticks >= 5
            await env.stop(delete=False)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            SwarmPodmanEnvironment._detect_daemon_os.cache_clear()

    asyncio.run(run())


def test_new_container_readiness_precedes_periodic_healthy_checks(monkeypatch):
    existing = [f"{i:012x}" for i in range(16)]
    newcomer = "ffffffffffff"
    checks = []

    async def engine(*args):
        if args[0] == "ps":
            ids = existing + ([newcomer] if len(checks) >= 16 else [])
            return 0, "\n".join(ids).encode(), b""
        if args[0] == "inspect":
            return (
                0,
                json.dumps(
                    [
                        {"Id": key, "Config": {"Healthcheck": {"Test": ["CMD", "true"], "Interval": 1}}}
                        for key in args[1:]
                    ]
                ).encode(),
                b"",
            )
        assert args[0] == "healthcheck"
        checks.append(args[-1])
        return 0, b"", b""

    monkeypatch.setattr(podman_healthchecks, "podman", engine)

    async def run():
        task = asyncio.create_task(podman_healthchecks.main())
        try:
            async with asyncio.timeout(8):
                while len(checks) < 17:
                    await asyncio.sleep(0.05)
            assert checks[16] == newcomer
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
