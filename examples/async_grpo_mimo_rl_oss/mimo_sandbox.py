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
"""Shared pieces of the MiMo-V2.6-RL-oss example: one Hugging Face sandbox per rollout, the policy behind it, and
the agent that drives it. The domain modules next to this one add the task loading and the grading."""

from __future__ import annotations

import json
import os
import shlex
import tarfile
import tempfile
import time
import uuid
from functools import cache
from typing import Any

from huggingface_hub import Sandbox, hf_hub_download


os.environ.setdefault("MIMOAGENT_SILENT_STARTUP", "1")
from mimoagent.agents.base import LimitsExceeded  # noqa: E402
from mimoagent.agents.cc import CCAgent  # noqa: E402
from mimoagent.environments import TransportError  # noqa: E402
from mimoagent.models.openai_chat import OpenAIChatModel  # noqa: E402


DATASET = "XiaomiMiMo/MiMo-V2.6-RL-oss"


@cache
def _image_map() -> dict[str, str]:
    path = hf_hub_download(DATASET, "image-mapping.jsonl", repo_type="dataset")
    return {row["dataset_image"]: row["dockerhub_image"] for row in map(json.loads, open(path))}


def docker_image(dataset_image: str) -> str:
    """The published image for a row's `docker_image`, which names an image in MiMo's own registry."""
    return _image_map()[dataset_image]


# ============================================================================================================
# The sandbox as a mimoagent environment
# ============================================================================================================


class HFSandboxEnvironment:
    """mimoagent `Environment` running each command in a Hugging Face sandbox. Same contract as its
    `KubernetesEnvironment`: a fresh `bash -c` per command, stderr merged into stdout, a `reason` the Bash tool
    reads, and files shipped in by `copy_to`."""

    def __init__(self, sandbox: Sandbox, *, cwd: str, timeout: int):
        self.sandbox = sandbox
        self.config = {"cwd": cwd, "timeout": timeout}

    def start(self) -> None:
        # The sandbox is already running by the time it reaches here; the dataset environments call this before
        # their own setup.
        pass

    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> dict[str, Any]:
        # Output goes through a file read back as bytes rather than through the command's stream: the sandbox client
        # splits that stream on U+0085, U+2028 and U+2029 as well as on newlines, so a file holding one of them
        # breaks the whole command. A file also lets a command leave daemons behind, as the MCP setup does.
        output_file = f"/tmp/out-{uuid.uuid4().hex}"
        try:
            result = self.sandbox.run(
                ["bash", "-c", f"exec >{output_file} 2>&1\n" + command],
                cwd=cwd or self.config["cwd"],
                timeout=timeout or self.config["timeout"],
                check=False,
            )
            output = self.sandbox.files.read(output_file).decode("utf-8", errors="replace")
        except Exception as e:
            raise TransportError(str(e)) from e
        if result.timed_out:
            return {"output": output, "returncode": 124, "reason": "pod_timeout"}
        return {"output": output, "returncode": result.exit_code, "reason": "ok"}

    def execute_detached(self, command: str, cwd: str = "", timeout: int | None = None, **_) -> dict[str, Any]:
        return self.execute(command, cwd=cwd, timeout=timeout)

    def copy_to(self, src_path: str, dest_path: str, **_) -> None:
        if not os.path.isdir(src_path):
            self.execute(f"mkdir -p {shlex.quote(os.path.dirname(dest_path))}")
            self.sandbox.files.upload(src_path, dest_path)
            return
        archive = f"/tmp/copy_to-{uuid.uuid4().hex}.tar"
        with tempfile.NamedTemporaryFile(suffix=".tar") as f:
            with tarfile.open(f.name, "w", dereference=True) as tar:
                tar.add(src_path, arcname=".")
            self.sandbox.files.upload(f.name, archive)
        self.execute(
            f"mkdir -p {shlex.quote(dest_path)} && tar -xf {archive} --no-same-owner -C {shlex.quote(dest_path)} && rm {archive}"
        )

    def get_template_vars(self) -> dict[str, Any]:
        return dict(self.config)


class ProbeModel(OpenAIChatModel):
    """Keeps every call's prompt and completion size. A turn re-sends the whole conversation, so the sum over calls
    is what sequence packing forwards for the rollout and the last call is about what tree packing forwards."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: list[tuple[int, int]] = []

    def _record_call(self, token_count) -> None:
        super()._record_call(token_count)
        self.calls.append((token_count.input_tokens, token_count.output_tokens))


class ProbeAgent(CCAgent):
    """The upstream agent plus the wall-clock limit verl's loop imposes on it from outside."""

    def __init__(self, *args, deadline: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.deadline = deadline

    def query(self) -> dict:
        if time.monotonic() > self.deadline:
            raise LimitsExceeded("Trajectory timed out")
        return super().query()


# ============================================================================================================
# Session and factory
# ============================================================================================================


def make_model(
    *,
    base_url: str,
    api_key: str,
    model: str,
    chat_template_kwargs: dict[str, Any],
    temperature: float,
    top_p: float,
    max_turn_tokens: int,
) -> ProbeModel:
    model_kwargs = {
        "base_url": base_url,
        "api_key": api_key,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_turn_tokens,
    }
    if chat_template_kwargs:
        model_kwargs["extra_body"] = {"chat_template_kwargs": chat_template_kwargs}
    return ProbeModel(model_name=model, model_kwargs=model_kwargs)
