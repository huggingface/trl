#!/usr/bin/env python3
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
"""One-shot MCP streamable-http client over the standard library only.

Drop-in for the task's `mcp_bridge.py` one-shot modes: same `--url/--name/--list/--call/--args-json` flags and the
same `__MCP_ONESHOT__<json>` line. The task's own bridge imports the `mcp` package, whose import alone costs about
twelve seconds inside a sandbox, once per tool call; the protocol it needs is three JSON-RPC POSTs, so this speaks
them with `urllib` and starts in a tenth of a second.
"""

import argparse
import json
import urllib.request


MARKER = "__MCP_ONESHOT__"
PROTOCOL = "2025-06-18"


def _post(url, body, session=None, timeout=120):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": PROTOCOL,
    }
    if session:
        headers["Mcp-Session-Id"] = session
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.headers.get("Mcp-Session-Id"), response.read().decode("utf-8", "replace")


def _result(payload):
    """A streamable-http reply is either a JSON-RPC object or an SSE stream carrying one."""
    payload = payload.strip()
    if not payload:
        return None
    frames = [line[5:].strip() for line in payload.splitlines() if line.startswith("data:")]
    if not frames:  # a plain JSON body rather than an SSE stream
        return json.loads(payload)
    for frame in frames:
        message = json.loads(frame)
        if "result" in message or "error" in message:
            return message
    return None


def _call(url, method, params, timeout):
    session, body = _post(
        url,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL,
                "capabilities": {},
                "clientInfo": {"name": "trl-bridge", "version": "1"},
            },
        },
        timeout=timeout,
    )
    _post(url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session, timeout)
    message = _result(_post(url, {"jsonrpc": "2.0", "id": 2, "method": method, "params": params}, session, timeout)[1])
    if message is None:
        raise RuntimeError(f"no JSON-RPC result for {method}")
    if "error" in message:
        raise RuntimeError(str(message["error"]))
    return message["result"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--name", default="mcp")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--call", default=None)
    parser.add_argument("--args-json", default="{}")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    try:
        if args.list:
            tools = _call(args.url, "tools/list", {}, args.timeout).get("tools", [])
            payload = {
                "ok": True,
                "tools": [
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "inputSchema": t.get("inputSchema") or {"type": "object", "properties": {}},
                    }
                    for t in tools
                ],
            }
        else:
            result = _call(
                args.url,
                "tools/call",
                {"name": args.call, "arguments": json.loads(args.args_json or "{}")},
                args.timeout,
            )
            text = "\n".join(
                block.get("text", "") for block in result.get("content") or [] if block.get("type") == "text"
            )
            payload = {"ok": True, "is_error": bool(result.get("isError")), "content": text}
            if result.get("structuredContent") is not None:
                payload["structured"] = result["structuredContent"]
    except Exception as exc:
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(MARKER + json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
