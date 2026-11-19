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

"""One native msgboard per team, using the same SDK and socket transport as Flock."""

import asyncio
import importlib.metadata
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import httpx
from msgboard import auth
from msgboard.client import Client


HERE = Path(__file__).parent
SOCKET = "/run/flock-board.sock"


def tool_bundle(destination: Path) -> Path:
    """Bundle the pure-Python board client so containers need no package downloads."""
    with tarfile.open(destination, "w:gz") as archive:
        for name in ("message_board.py", "board_tool.py"):
            archive.add(HERE / name, arcname=name)
        for name in (
            "msgboard",
            "httpx",
            "httpcore",
            "anyio",
            "idna",
            "certifi",
            "h11",
            "sniffio",
            "typing_extensions",
            "exceptiongroup",
        ):
            dist = importlib.metadata.distribution(name)
            for file in dist.files:
                path = str(file)
                if path.startswith("../") or ".dist-info/" in path or "__pycache__" in path:
                    continue
                if name == "msgboard" and path not in ("msgboard/__init__.py", "msgboard/client.py"):
                    continue
                source = Path(dist.locate_file(file))
                if source.is_file():
                    archive.add(source, arcname=path)
    return destination


class TeamBoard:
    def __init__(self, directory: Path, names: list[str]):
        self.directory, self.names = directory, names
        self.process = self.client = self.log = self.socket_dir = None
        self.credentials = {}

    async def start(self):
        self.directory.mkdir(parents=True, mode=0o700)
        self.log = (self.directory / "server.log").open("ab")
        init = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "msgboard",
            "init",
            "--data-dir",
            str(self.directory / "data"),
            stdout=self.log,
            stderr=self.log,
        )
        if await init.wait():
            raise RuntimeError("msgboard init failed; see board/server.log")
        self.socket_dir = tempfile.TemporaryDirectory(prefix="trl-board-")
        self.socket = Path(self.socket_dir.name) / "board.sock"
        # Port 0 is only for the host-side listener; all clients use the mounted socket.
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(HERE / "board_server.py"),
            str(self.directory / "data"),
            "127.0.0.1",
            "0",
            str(self.socket),
            stdout=self.log,
            stderr=self.log,
        )
        operator = (self.directory / "data" / "operator.token").read_text().strip()
        self.client = Client(
            "http://board", operator, transport=httpx.HTTPTransport(uds=str(self.socket)), timeout=2, max_retries=0
        )
        for _ in range(100):
            if self.process.returncode is not None:
                raise RuntimeError("msgboard exited during startup")
            try:
                await asyncio.to_thread(self.client.me)
                break
            except (httpx.TransportError, OSError):
                await asyncio.sleep(0.1)
        else:
            raise TimeoutError("msgboard startup timed out")
        await asyncio.to_thread(self.provision)

    def provision(self):
        channel = self.client.create_channel("general")
        self.channel = channel.get("channel", channel)["id"]
        for name in self.names:
            invite = self.client.create_invite(
                agent_name=name, role="member", can_create_channels=False, channels=[self.channel], max_uses=1
            )
            with Client.join(
                "http://board", invite["code"], transport=httpx.HTTPTransport(uds=str(self.socket))
            ) as agent:
                self.credentials[name] = agent.token

    def bind(self, trial, name):
        trial.config.environment.mounts = [
            *(trial.config.environment.mounts or []),
            {"type": "bind", "source": str(self.socket), "target": SOCKET, "read_only": True},
        ]
        trial.agent.board_env.update(
            MSGBOARD_URL="http://board", MSGBOARD_TOKEN=self.credentials[name], MSGBOARD_SOCKET=SOCKET
        )

    def export(self):
        after = 0
        with (self.directory / "messages.jsonl").open("w") as output:
            while True:
                page = self.client.read(self.channel, after=after, limit=100)
                messages = page["messages"]
                for message in messages:
                    output.write(json.dumps(message) + "\n")
                if not messages:
                    break
                after = max(message["seq"] for message in messages)

    async def close(self):
        try:
            if self.client and self.credentials:
                await asyncio.to_thread(self.export)
                for token in self.credentials.values():
                    await asyncio.to_thread(self.client.revoke_token, auth.parse_token(token)[0])
        finally:
            if self.client:
                self.client.close()
            if self.process and self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), 10)
                except TimeoutError:
                    self.process.kill()
                    await self.process.wait()
            if self.socket_dir:
                self.socket_dir.cleanup()
            if self.log:
                self.log.close()
            # Keep the database private; only messages.jsonl is a shareable board artifact.
            self.credentials.clear()
