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

"""Schedule native Podman health checks without a user systemd session.

Adapted from Flock's native healthcheck scheduler.

Podman normally schedules periodic health checks through user systemd timers,
but Slurm jobs on clusters without user systemd have no working session. The
launcher starts this helper to run each container's existing health check at
its configured interval; it does not replace or bypass the checks.

Compose dependencies can require healthy services before startup (e.g.
worker -> puzzle -> agent). Without scheduled checks, services can remain marked
"starting" even when their servers are ready, leaving Compose waiting. Removing
the health-based dependency conditions instead would allow startup races and
connection errors unless another readiness mechanism replaced them.
"""

import asyncio
import json
import time


async def podman(*args):
    process = await asyncio.create_subprocess_exec(
        "podman", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    return process.returncode, stdout, stderr


def interval(config):
    # Podman stores Docker healthcheck intervals as nanoseconds; zero means 30s.
    return max(config.get("Interval", 0) / 1_000_000_000, 0) or 30


async def main():
    # Cache immutable healthcheck configuration; inspecting every container every
    # second blocked the scheduler under load. Bound healthcheck subprocesses too.
    deadlines, configs, running = {}, {}, {}
    awaiting_first_success = set()
    while True:
        for key, task in list(running.items()):
            if task.done():
                try:
                    code, _, _ = task.result()
                    if code == 0:
                        awaiting_first_success.discard(key)
                except TimeoutError:
                    print(f"Healthcheck timed out for {key}; will retry", flush=True)
                deadlines[key] = time.monotonic() + interval(configs.get(key, {}))
                del running[key]
        try:
            code, stdout, stderr = await podman("ps", "--format", "{{.ID}}")
            if code:
                print(f"Podman discovery failed: {stderr.decode(errors='replace')}", flush=True)
                await asyncio.sleep(2)
                continue
            ids = stdout.decode().split()
            active = set(ids)
            awaiting_first_success.intersection_update(active)
            configs = {key: value for key, value in configs.items() if key in active}
            deadlines = {key: value for key, value in deadlines.items() if key in active}
            # Discover only a small batch per tick, so one slow inspection cannot
            # prevent already-discovered dependencies from becoming healthy.
            unknown = [key for key in ids if key not in configs][:16]
            if unknown:
                code, stdout, _ = await podman("inspect", *unknown)
                if not code:
                    for container in json.loads(stdout):
                        key = container["Id"][:12]
                        configs[key] = container.get("Config", {}).get("Healthcheck") or {}
                        awaiting_first_success.add(key)
                        deadlines[key] = time.monotonic()
        except TimeoutError:
            print("Podman discovery timed out; continuing health checks", flush=True)
        now = time.monotonic()
        for key in sorted(configs, key=lambda key: (key not in awaiting_first_success, deadlines.get(key, 0))):
            health = configs[key]
            if not health.get("Test") or health["Test"] == ["NONE"]:
                continue
            if len(running) >= 16:
                break
            if key not in running and now >= deadlines.get(key, 0):
                running[key] = asyncio.create_task(podman("healthcheck", "run", key))
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
