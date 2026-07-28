# `terminal_notes` harness

A richer shell harness: **6 shell tools** (including background processes) plus a **4-tool persistent
note toolkit**. Implemented by `TerminalNotesEnv` in [`env.py`](env.py).

## Tools

### Shell (backed by the sandbox)

| Tool | Signature | What it does |
|---|---|---|
| `shell_exec` | `shell_exec(command: str, blocking: bool = True) -> str` | Run a command. Blocking → combined stdout+stderr; non-blocking → detach and return a PID. |
| `shell_write_content_to_file` | `shell_write_content_to_file(path: str, content: str) -> str` | Write `content` to `path` (used to commit `/workdir/answer.txt`). |
| `shell_write_to_process` | `shell_write_to_process(pid: str, content: str) -> str` | Write to a background process's stdin. |
| `shell_view` | `shell_view(pid: str) -> str` | Read the captured stdout of a background process. |
| `shell_wait` | `shell_wait(pid: str) -> str` | Wait (≤5 min) for a background process to exit, then return its output. |
| `shell_kill_process` | `shell_kill_process(pid: str) -> str` | SIGKILL a background process. |

### Notes (in-env state, persist across turns of a rollout)

| Tool | Signature | What it does |
|---|---|---|
| `create_note` | `create_note(title: str, content: str) -> str` | Create a note. |
| `append_note` | `append_note(title: str, content: str) -> str` | Append to an existing note. |
| `read_note` | `read_note(title: str) -> str` | Read a note's content. |
| `list_note` | `list_note() -> str` | List note titles + sizes. |

> Unlike the original SETA agent, notes are **not** auto-injected into the prompt each turn (TRL owns the
> prompt under `environment_factory`); recall them on demand with `read_note` / `list_note`.

## Submission

Write the answer to `/workdir/answer.txt`, e.g. `shell_write_content_to_file(path="/workdir/answer.txt", content="<value>")`.

## Use it

```python
from trl.experimental.harbor import HarborSpec
spec = HarborSpec(dataset, agent="examples/scripts/harbor/harnesses/terminal_notes/env.py:TerminalNotesEnv")
```
