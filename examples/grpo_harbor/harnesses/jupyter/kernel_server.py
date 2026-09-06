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

"""Tiny stateful Python execution server.

Uploaded by JupyterToolAgent into the Harbor container at /opt/kernel_server.py
and started in the background. Listens on 127.0.0.1:8765 and accepts:

  POST /  Content-Type: application/json
  Body:   {"code": "..."}
  Reply:  {"output": "<stdout+stderr+traceback>", "ok": true|false}

A single persistent globals dict survives across requests — that's the
"stateful kernel" the agent's `add_and_execute_code_cell` tool relies on.
No IPython, no jupyter_client. Just compile(...) + exec(...).
"""

from __future__ import annotations

import contextlib
import io
import json
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer


PORT = 8765
G: dict = {"__name__": "__main__"}


def _exec(code: str) -> tuple[str, bool]:
    out = io.StringIO()
    err = io.StringIO()
    ok = True
    try:
        # Try "single" mode first so a bare expression auto-prints (mimics Jupyter).
        try:
            compiled = compile(code, "<cell>", "single")
        except SyntaxError:
            compiled = compile(code, "<cell>", "exec")
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(compiled, G)
    except SystemExit:
        pass
    except BaseException:
        ok = False
        err.write(traceback.format_exc())
    return out.getvalue() + err.getvalue(), ok


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n).decode("utf-8", errors="replace")
            payload = json.loads(body)
            code = payload.get("code", "")
            output, ok = _exec(code)
        except Exception:
            output, ok = traceback.format_exc(), False
        # Cap to keep the per-cell response small.
        if len(output) > 8000:
            output = output[:8000] + f"\n... [truncated {len(output) - 8000} chars]"
        body_out = json.dumps({"output": output, "ok": ok}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body_out)))
        self.end_headers()
        self.wfile.write(body_out)

    def do_GET(self):
        # Health check.
        msg = b'{"ready": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(msg)))
        self.end_headers()
        self.wfile.write(msg)

    def log_message(self, *a, **k):
        # Quiet — Harbor agent's exec captures stdout/stderr.
        pass


if __name__ == "__main__":
    print(f"[kernel_server] starting on 127.0.0.1:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
