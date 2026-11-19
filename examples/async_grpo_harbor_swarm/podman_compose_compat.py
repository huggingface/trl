#!/usr/bin/env python
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

"""Podman Compose 1.x CLI compatibility for Harbor's Docker Compose commands."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    args = sys.argv[1:]
    commands = {"build", "up", "down", "exec", "logs", "ps", "config", "stop", "start", "pull", "rm", "kill", "cp"}
    index = next((i for i, arg in enumerate(args) if arg in commands), None)
    if index is None:
        return subprocess.call([str(Path(sys.executable).parent / "podman-compose"), *args])
    command = args[index]
    if command == "cp":
        flags, paths = args[:index], args[index + 1 :]
        project_flag = "--project-name" if "--project-name" in flags else "-p"
        project = flags[flags.index(project_flag) + 1]
        if len(paths) != 2:
            raise ValueError("Expected compose cp SOURCE DESTINATION")
        mapped = []
        for path in paths:
            if ":" in path and not path.startswith("/"):
                service, location = path.split(":", 1)
                container = f"{project}_{service}_1"
                subprocess.run(["podman", "container", "exists", container], check=True)
                mapped.append(f"{container}:{location}")
            else:
                mapped.append(path)
        return subprocess.call(["podman", "cp", *mapped])
    if command == "exec":
        args[index + 1 :] = ["-T" if arg == "--no-TTY" else arg for arg in args[index + 1 :]]
    # Compose allows an implicit default network; podman-compose requires a declaration.
    # This runtime overlay leaves the pinned task files and network topology intact.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as overlay:
        json.dump({"networks": {"default": {}}}, overlay)
        overlay.flush()
        args[index:index] = ["-f", overlay.name]
        return subprocess.call([str(Path(sys.executable).parent / "podman-compose"), *args])


if __name__ == "__main__":
    raise SystemExit(main())
