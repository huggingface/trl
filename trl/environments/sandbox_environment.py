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

from concurrent.futures import Future, ThreadPoolExecutor

from ..import_utils import is_huggingface_hub_available


# huggingface_hub Sandbox landed in 1.22.0.
_SANDBOX_MIN_VERSION = "1.22.0"

# Boot sandboxes off the calling thread. `reset` is driven from the trainer's rollout loop — in AsyncGRPOTrainer that
# is a single asyncio event loop — and a Sandbox boot blocks for several seconds polling the VM. Booting on this pool
# lets `reset` return immediately, so the boot overlaps generation instead of stalling every concurrent rollout.
_BOOT_POOL = ThreadPoolExecutor(thread_name_prefix="sandbox-boot")


class SandboxEnvironment:
    r"""
    Environment that gives the model a shell-execution tool backed by an isolated [Hugging Face
    Sandbox](https://huggingface.co/docs/huggingface_hub/main/guides/sandbox).

    Plugs into [`GRPOTrainer`] (and [`experimental.async_grpo.AsyncGRPOTrainer`]) via `environment_factory`. `reset`
    boots a dedicated sandbox VM (once, off the calling thread, so the boot overlaps generation) and reuses it across
    the environment instance's rollouts; the `run` method is exposed to the model as a tool during generation; the
    sandbox is torn down when the environment is garbage-collected. Reward is left entirely to the trainer's
    `reward_funcs`; this environment only provides the execution surface.

    Requires `huggingface_hub>=1.22.0`.

    Args:
        image (`str`, *optional*, defaults to `"python:3.12"`):
            Docker image the sandbox runs. Any image with `/bin/sh` works.
        flavor (`str`, *optional*, defaults to `"cpu-basic"`):
            Hardware flavor for the sandbox VM (same flavors as Jobs, e.g. `"cpu-basic"`, `"a10g-small"`).
        idle_timeout (`str`, *optional*, defaults to `"10m"`):
            Shut the sandbox down after this much inactivity, as a billing backstop for abandoned rollouts.
        command_timeout (`int`, *optional*, defaults to `60`):
            Per-command timeout, in seconds, applied to every `run` call.

    Example:

    ```python
    from trl import GRPOConfig, GRPOTrainer
    from trl.environments import SandboxEnvironment

    trainer = GRPOTrainer(
        model="Qwen/Qwen3-4B",
        args=GRPOConfig(max_tool_calling_iterations=10),
        train_dataset=dataset,
        reward_funcs=my_reward_func,
        environment_factory=SandboxEnvironment,
    )
    trainer.train()
    ```
    """

    # Cap tool output so a runaway command (e.g. `cat` on a huge file) can't blow up the context.
    _MAX_OUTPUT_CHARS = 8192

    def __init__(
        self,
        image: str = "python:3.12",
        flavor: str = "cpu-basic",
        idle_timeout: str = "10m",
        command_timeout: int = 60,
    ):
        if not is_huggingface_hub_available(_SANDBOX_MIN_VERSION):
            raise ImportError(
                f"SandboxEnvironment requires `huggingface_hub>={_SANDBOX_MIN_VERSION}` for the Sandbox API. Please "
                "upgrade with `pip install -U huggingface_hub`."
            )
        self.image = image
        self.flavor = flavor
        self.idle_timeout = idle_timeout
        self.command_timeout = command_timeout
        self._sandbox = None
        self._booting: Future | None = None

    def reset(self, **kwargs) -> None:
        # Boot the sandbox once and reuse it across this instance's rollouts (instances are pooled and reset across
        # rollouts by the trainer). The boot runs on `_BOOT_POOL` so `reset` doesn't block the rollout loop; `run`
        # waits on the result the first time it needs the sandbox.
        from huggingface_hub import Sandbox

        if self._sandbox is None and self._booting is None:
            self._booting = _BOOT_POOL.submit(
                Sandbox.create, image=self.image, flavor=self.flavor, idle_timeout=self.idle_timeout
            )

    def run(self, command: str) -> str:
        """
        Run a shell command in the sandbox and return its combined stdout and stderr. Files persist across calls, but
        each call runs in a fresh shell, so chain commands that share state with `&&` (e.g. `cd /app && python main.py`).

        Args:
            command: The shell command to run.
        """
        if self._sandbox is None:
            self._sandbox = self._booting.result()  # block until the boot started by `reset` finishes
        result = self._sandbox.run(command, timeout=self.command_timeout, check=False)
        output = (result.stdout or "") + (result.stderr or "")
        if len(output) > self._MAX_OUTPUT_CHARS:
            output = output[: self._MAX_OUTPUT_CHARS] + "\n... [truncated]"
        return output or f"(no output, exit code {result.exit_code})"

    def __del__(self):
        # Best-effort teardown. A sandbox may exist even if `run` was never called (booted by `reset`); pick it up if
        # its boot already finished, but never block the finalizer on an in-flight boot — `idle_timeout` reaps that.
        sandbox = self._sandbox
        if sandbox is None and self._booting is not None and self._booting.done():
            sandbox = self._booting.result()
        if sandbox is not None:
            try:
                sandbox.kill()
            except Exception:  # noqa: BLE001
                pass
