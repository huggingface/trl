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

"""A call-boundary gate for full-model updates; no model call straddles a transfer."""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import secrets
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response


def token_key(prompt, completion):
    return hashlib.sha256(json.dumps([prompt, completion], separators=(",", ":")).encode()).hexdigest()


async def read_records(client, url, token):
    """Retry transient transport failures on the idempotent provenance read."""
    for attempt in range(3):
        try:
            response = await client.get(url, headers={"x-swarm-control": token})
            response.raise_for_status()
            return response.json()
        except httpx.TransportError:
            if attempt == 2:
                raise
            await asyncio.sleep(0.1)


def match_versions(turns, records):
    """Reject missing/ambiguous provenance instead of guessing a policy version."""
    versions = defaultdict(set)
    for record in records:
        versions[record["tokens"]].add(record["version"])
    result = []
    for turn in turns:
        choices = versions[token_key(turn.prompt_token_ids, turn.completion_token_ids)]
        if len(choices) != 1:
            raise ValueError(f"Missing or ambiguous policy provenance for capture node {turn.node_id}")
        result.append({"node_id": turn.node_id, "policy_version": next(iter(choices))})

    return result


class PolicyGate:
    def __init__(self):
        self.condition = asyncio.Condition()
        self.paused = False
        self.version = 0
        self.active = 0
        self.failed = False

    @asynccontextmanager
    async def call(self):
        async with self.condition:
            await self.condition.wait_for(lambda: not self.paused or self.failed)
            if self.failed:
                raise HTTPException(503, "Policy synchronization failed")
            self.active += 1
            version = self.version
        try:
            yield version
        finally:
            async with self.condition:
                self.active -= 1
                self.condition.notify_all()

    async def drain(self):
        async with self.condition:
            self.paused = True
            await self.condition.wait_for(lambda: self.active == 0)

    async def publish(self, version):
        async with self.condition:
            if self.failed or not self.paused or self.active or version != self.version + 1:
                raise HTTPException(409, "Invalid policy publication")
            self.version = version
            self.paused = False
            self.condition.notify_all()

    async def fail(self):
        async with self.condition:
            self.failed = self.paused = True
            self.condition.notify_all()


def create_app(upstream, output_dir, token, transport=None):
    gate = PolicyGate()
    records = defaultdict(list)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    client = httpx.AsyncClient(
        timeout=600,
        transport=transport,
        limits=httpx.Limits(max_connections=512, keepalive_expiry=1),
    )

    @asynccontextmanager
    async def lifespan(app):
        yield
        await client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.gate = gate

    def authorize(request):
        if not token or not secrets.compare_digest(request.headers.get("x-swarm-control", ""), token):
            raise HTTPException(403, "Control authentication required")

    @app.get("/status")
    async def status():
        return {"version": gate.version, "active_calls": gate.active, "paused": gate.paused, "failed": gate.failed}

    @app.post("/control/{operation}")
    async def control(operation: str, request: Request):
        authorize(request)
        if operation == "drain":
            try:
                await asyncio.wait_for(gate.drain(), 600)
            except TimeoutError:
                await gate.fail()
                raise HTTPException(504, "Timed out draining active calls") from None
        elif operation == "publish":
            await gate.publish((await request.json())["version"])
        elif operation == "fail":
            await gate.fail()
        else:
            raise HTTPException(404)
        return await status()

    @app.get("/records/{agent}")
    async def get_records(agent: str, request: Request):
        authorize(request)
        return records.get(agent, [])

    @app.delete("/records/{agent}")
    async def delete_records(agent: str, request: Request):
        authorize(request)
        records.pop(agent, None)
        return {"ok": True}

    async def forward(path, request, agent=None):
        headers = {}
        if request.method == "POST" and path == "v1/chat/completions":
            body = await request.json()
            if body.get("stream"):
                raise HTTPException(400, "Capture upstream must be non-streaming")
            async with gate.call() as version:
                body["cache_salt"] = f"swarm-policy-{version}"
                # A lost inference response has no agent-side effects. Keep the gate held so
                # a retry uses the same weights, and only record the response we return.
                for attempt in range(2):
                    try:
                        response = await client.post(f"{upstream}/{path}", json=body)
                        break
                    except (httpx.NetworkError, httpx.RemoteProtocolError):
                        if attempt:
                            raise
                        logging.getLogger(__name__).warning("Inference connection failed; retrying once")
                        await asyncio.sleep(0.1)
                headers["x-swarm-policy-version"] = str(version)
                if response.is_success and agent:
                    payload = response.json()
                    prompt = payload.get("prompt_token_ids")
                    for choice in payload.get("choices", []):
                        completion = choice.get("token_ids")
                        if prompt is not None and completion is not None:
                            record = {"agent": agent, "version": version, "tokens": token_key(prompt, completion)}
                            records[agent].append(record)
                            with (output / "policy-calls.jsonl").open("a") as stream:
                                stream.write(json.dumps(record) + "\n")
        elif request.method == "GET" and path in {"health", "v1/models", "metrics", "get_server_info"}:
            response = await client.get(f"{upstream}/{path}")
        else:
            raise HTTPException(404)
        return Response(
            response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
            headers=headers,
        )

    @app.api_route("/agents/{agent}/{path:path}", methods=["GET", "POST"])
    async def agent_proxy(agent: str, path: str, request: Request):
        return await forward(path, request, agent)

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        return await forward(path, request)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.upstream.rstrip("/"), args.output_dir, os.environ["SWARM_GATEWAY_TOKEN"]),
        host="0.0.0.0",
        port=args.port,
        log_level="warning",
        timeout_keep_alive=120,
    )
