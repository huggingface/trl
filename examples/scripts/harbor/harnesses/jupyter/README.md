# `jupyter` harness

A **stateful Python kernel** harness. Variables, imports, and side-effects persist across cells, so the
model builds up state like a notebook. Implemented by `JupyterEnv` in [`env.py`](env.py).

## How it works

On `reset`, the harness (`_setup`) uploads two helpers into the sandbox and starts a tiny kernel server:

- [`kernel_server.py`](kernel_server.py) — a local HTTP server (`127.0.0.1:8765`) holding one persistent Python kernel.
- [`run_cell.py`](run_cell.py) — sends a base64-encoded cell to the kernel server and prints its JSON result.

Each `add_and_execute_code_cell` call runs `python3 /opt/run_cell.py` against that kernel.

## Tools

| Tool | Signature | What it does |
|---|---|---|
| `add_and_execute_code_cell` | `add_and_execute_code_cell(code: str) -> str` | Execute Python in the **stateful** kernel; state persists across calls. Use for all computation. |
| `execute_shell_command` | `execute_shell_command(command: str) -> str` | Run a shell command (pip install, ls, …). **Not** stateful with the Python kernel. |

## Submission

Write the answer to `/workdir/answer.txt` — e.g. `add_and_execute_code_cell(code="open('/workdir/answer.txt','w').write(str(ans))")`.

## Use it

```python
from trl.experimental.harbor import HarborSpec
spec = HarborSpec(dataset, agent="examples/scripts/harbor/harnesses/jupyter/env.py:JupyterEnv")
```
