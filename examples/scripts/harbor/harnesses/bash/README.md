# `bash` harness

The minimal harness: a single shell tool. This is the **built-in** [`HarborBashEnv`](../../../../../trl/experimental/harbor/_env.py) (`trl.experimental.harbor.HarborBashEnv`); this folder just documents it and re-exports it as `BashEnv`.

## Tools

| Tool | Signature | What it does |
|---|---|---|
| `bash` | `bash(command: str) -> str` | Run a shell command in the sandbox; returns combined stdout+stderr (truncated to 8k). Non-stateful between calls. |

## Submission

No submit tool — write the answer to `/workdir/answer.txt`, e.g. `echo -n "<value>" > /workdir/answer.txt`. The task's verifier reads that file.

## Use it

```python
from trl.experimental.harbor import HarborSpec
spec = HarborSpec(dataset, agent="bash")   # built-in name
```
