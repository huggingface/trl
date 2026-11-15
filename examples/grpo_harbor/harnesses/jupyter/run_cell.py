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

"""CLI client for the kernel_server, invoked by the Harbor agent via env.exec.

Usage (inside the container):
  python3 /opt/run_cell.py --code-b64 <base64-encoded-python-source>

Writes the kernel server's response payload (already JSON-encoded) to stdout.
The agent base64-decodes it on the host side. Base64 sidesteps shell escaping
for code containing quotes, newlines, etc.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request


PORT = 8765


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--code-b64", required=True)
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()

    code = base64.b64decode(args.code_b64).decode("utf-8")
    payload = json.dumps({"code": code}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            sys.stdout.write(r.read().decode("utf-8", errors="replace"))
        return 0
    except Exception as exc:
        sys.stdout.write(json.dumps({"output": f"[run_cell err] {exc}", "ok": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
