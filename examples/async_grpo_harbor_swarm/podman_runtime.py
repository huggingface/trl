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

"""Bounded Harbor Podman lifecycle with cancellation-safe Compose commands."""

import asyncio
import functools
import os
from pathlib import Path

import yaml
from harbor.environments.podman import PodmanEnvironment


class SwarmPodmanEnvironment(PodmanEnvironment):
    build_slots = asyncio.Semaphore(1)
    start_slots = asyncio.Semaphore(8)
    stop_slots = asyncio.Semaphore(8)

    @classmethod
    @functools.cache
    def _detect_daemon_os(cls):
        # Harbor calls this synchronous 10-second probe for every environment.
        # The engine is fixed for the lifetime of this rollout worker.
        return super()._detect_daemon_os()

    async def start(self, force_build):
        async with self.start_slots:
            await asyncio.to_thread(type(self)._detect_daemon_os)
            await super().start(force_build)

    async def stop(self, delete):
        async with self.stop_slots:
            # Keep images shared by other teams; remove containers and volumes.
            async with asyncio.timeout(120):
                await super().stop(delete=False)

    async def _run_docker_compose_command(self, command, *args, **kwargs):
        if command[0] == "build":
            overlay = os.environ.get("SWARM_AGENT_OVERLAY")
            if overlay and Path(overlay) in self.extra_docker_compose_paths:
                # The agent image was loaded before rollout startup. Build only
                # the task-authored puzzle/worker services, preserving their code.
                services = yaml.safe_load(self._environment_docker_compose_path.read_text())["services"]
                task_services = [name for name, config in services.items() if name != "main" and "build" in config]
                if not task_services:
                    return None
                command = [*command, *task_services]
            # AgentEscape tasks share worker/agent image tags across task names.
            # Harbor's per-task build lock does not protect those shared images.
            async with self.build_slots:
                return await super()._run_docker_compose_command(command, *args, **kwargs)
        return await super()._run_docker_compose_command(command, *args, **kwargs)

    @staticmethod
    async def _collect_buffered_output(process, *, timeout_sec, stdin_data=None):
        try:
            return await PodmanEnvironment._collect_buffered_output(
                process, timeout_sec=timeout_sec, stdin_data=stdin_data
            )
        except BaseException:
            # Harbor's buffered collector handles timeout, but not cancellation.
            # The Compose wrapper propagates termination to its process group.
            await asyncio.shield(PodmanEnvironment._terminate_process(process))
            raise
