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

"""Shell + notes base agent — a 10-tool Harbor harness (6 shell + 4 notes).

A custom [`~trl.experimental.harbor.HarborEnv`] subclass with a richer toolset than the bash harness:
six shell tools (including background processes) and a persistent note-taking toolkit. Point a trainer
at it with ``HarborSpec(dataset, agent="examples/scripts/harbor/harnesses/terminal_notes.py:TerminalNotesEnv")``.
Submission is by writing ``/workdir/answer.txt`` (same verifier as the bash harness).

Notes live in-env (a dict) and survive across turns of the same rollout. They're recalled on demand via
``read_note`` / ``list_note`` (TRL owns the prompt under ``environment_factory``, so the env can't
inject them automatically each turn).
"""

import base64
import shlex
import uuid

from trl.experimental.harbor import HarborEnv


_PROMPT_SUFFIX = (
    "\n\nYou are an autonomous data-analysis agent in a sandboxed Linux container (Python preinstalled). "
    "Dataset files are in /home/user/input/. You have shell tools (shell_exec, "
    "shell_write_content_to_file, shell_view/wait/kill for background procs) and a persistent note system "
    "(create_note, append_note, read_note, list_note) — use notes as a scratchpad and read them back. "
    "**Submit your final answer by writing it to /workdir/answer.txt via a shell tool** (e.g. "
    "shell_write_content_to_file(path='/workdir/answer.txt', content=<answer>)). Keep it short; do not "
    "end your turn without submitting."
)


class TerminalNotesEnv(HarborEnv):
    """10-tool harness (6 shell + 4 notes) over a Harbor sandbox."""

    PROMPT_SUFFIX = _PROMPT_SUFFIX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._notes: dict[str, str] = {}
        self._bg: dict[str, dict] = {}

    def reset(self, task_dir=None, **kwargs) -> str:
        self._notes.clear()
        self._bg.clear()
        return super().reset(task_dir=task_dir, **kwargs)

    # ── shell toolkit ───────────────────────────────────────────────────────

    def shell_exec(self, command: str, blocking: bool = True) -> str:
        """
        Execute a shell command in the sandbox. If `blocking` (default), run synchronously and return
        combined stdout+stderr; otherwise detach into the background and return the new PID.

        Args:
            command: The shell command to run.
            blocking: Run synchronously (True) or in the background (False).

        Returns:
            Combined stdout+stderr (blocking) or the background PID.
        """
        if blocking:
            return self._exec(command)
        token = uuid.uuid4().hex[:8]
        log, pipe = f"/tmp/sh_{token}.log", f"/tmp/sh_{token}.in"
        out = self._exec(
            f"mkfifo {pipe} 2>/dev/null; ( nohup setsid bash -c {shlex.quote(command)} <{pipe} >{log} 2>&1 ) & echo $!",
            timeout=15,
        )
        pid = out.strip().split()[-1] if out.strip() else ""
        if not pid.isdigit():
            return f"[shell_exec bg] failed to spawn: {out}"
        self._bg[pid] = {"log": log, "pipe": pipe}
        return f"Started background process PID={pid} log={log}"

    def shell_write_content_to_file(self, path: str, content: str) -> str:
        """
        Write `content` to `path` in the sandbox (overwrites). Use to commit the final answer to
        /workdir/answer.txt.

        Args:
            path: Destination path in the sandbox.
            content: File contents.

        Returns:
            A confirmation string.
        """
        b64 = base64.b64encode(content.encode()).decode()
        out = self._exec(
            f"mkdir -p $(dirname {shlex.quote(path)}) && echo {b64} | base64 -d > {shlex.quote(path)}", timeout=30
        )
        return f"Wrote {len(content)} bytes to {path}" if "rc=" not in out else f"[write_file] {out}"

    def shell_write_to_process(self, pid: str, content: str) -> str:
        """
        Send `content` (with a trailing newline) to the stdin of a background process.

        Args:
            pid: PID returned by shell_exec(blocking=False).
            content: Text to write to stdin.

        Returns:
            A confirmation string.
        """
        proc = self._bg.get(pid)
        if proc is None:
            return f"Unknown PID={pid}. Started: {list(self._bg)}"
        b64 = base64.b64encode((content + "\n").encode()).decode()
        self._exec(f"echo {b64} | base64 -d > {shlex.quote(proc['pipe'])}", timeout=30)
        return f"Wrote {len(content)} bytes to PID={pid} stdin"

    def shell_view(self, pid: str) -> str:
        """
        Return the current captured stdout of a background process.

        Args:
            pid: PID returned by shell_exec(blocking=False).

        Returns:
            The captured stdout so far.
        """
        proc = self._bg.get(pid)
        return self._exec(f"tail -c 4000 {shlex.quote(proc['log'])} 2>/dev/null") if proc else f"Unknown PID={pid}"

    def shell_wait(self, pid: str) -> str:
        """
        Wait (up to ~5 min) for a background process to terminate, then return its captured stdout.

        Args:
            pid: PID returned by shell_exec(blocking=False).

        Returns:
            The process output after it exits.
        """
        proc = self._bg.get(pid)
        if proc is None:
            return f"Unknown PID={pid}"
        return self._exec(
            f"for i in $(seq 1 300); do [ ! -d /proc/{pid} ] && break; sleep 1; done; "
            f"echo '--- exited ---'; tail -c 4000 {shlex.quote(proc['log'])} 2>/dev/null",
            timeout=320,
        )

    def shell_kill_process(self, pid: str) -> str:
        """
        Send SIGKILL to a background process.

        Args:
            pid: PID returned by shell_exec(blocking=False).

        Returns:
            A confirmation string.
        """
        if pid not in self._bg:
            return f"Unknown PID={pid}"
        return f"Sent SIGKILL to PID={pid}: {self._exec(f'kill -9 {pid} 2>&1', timeout=10)}"

    # ── note toolkit (in-env state) ──────────────────────────────────────────

    def create_note(self, title: str, content: str) -> str:
        """
        Create a persistent note (recall it later with read_note/list_note).

        Args:
            title: Note title.
            content: Note body.

        Returns:
            A confirmation string.
        """
        self._notes[title] = content
        return f"Note '{title}' created ({len(content)} chars). Total: {len(self._notes)}."

    def append_note(self, title: str, content: str) -> str:
        """
        Append `content` (on a new line) to an existing note.

        Args:
            title: Note title.
            content: Text to append.

        Returns:
            A confirmation string.
        """
        if title not in self._notes:
            return f"Note '{title}' not found. Use create_note first."
        self._notes[title] += "\n" + content
        return f"Note '{title}' updated -> {len(self._notes[title])} chars."

    def read_note(self, title: str) -> str:
        """
        Return the full content of a note.

        Args:
            title: Note title.

        Returns:
            The note content, or a not-found message.
        """
        return self._notes.get(title, f"Note '{title}' not found.")

    def list_note(self) -> str:
        """
        List all note titles with their character counts.

        Returns:
            One line per note, or a message if there are none.
        """
        return "\n".join(f"- {t} ({len(c)} chars)" for t, c in self._notes.items()) or "(no notes yet)"
