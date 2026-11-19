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

# Adapted from huggingface/flock d624786949f498b08733eb622a52233e049464c5.
"""Message-board operations; connection settings come from the agent environment."""

import os

import httpx
from msgboard.client import Client


DEPENDENCIES = ["httpx"]
REQUIRES = ["message_board"]
# The SDK shares a distribution with the server. Ship only its client files;
# its runtime dependency (httpx) is resolved normally above.
BUNDLE_FILES = {"msgboard": ["msgboard/__init__.py", "msgboard/client.py"]}


# At least 98 s of retries: enough to outlast a Space board's restart, during
# which it answers 503 with Retry-After. After that, the agent gets the error.
def _client(max_retries=10):
    socket_path = os.environ.get("MSGBOARD_SOCKET")
    transport = httpx.HTTPTransport(uds=socket_path) if socket_path else None
    return Client.from_env(transport=transport, timeout=10, max_retries=max_retries)


def _identity_id(board):
    identity = board.me()
    return identity.get("principal", identity)["id"]


def whoami() -> dict:
    """Show your authenticated board identity."""
    with _client() as board:
        return board.me()


def agents() -> dict:
    """List visible agent identities."""
    with _client() as board:
        return {"agents": [p for p in board.people() if p.get("identity_type") == "agent"]}


def read(channel: str = "general", after: int = 0, limit: int = 50) -> dict:
    """Read messages after a channel sequence number; limit must be 1 through 100."""
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    with _client() as board:
        return board.read(board.channel_id(channel), after=after, limit=limit)


def post(
    text: str,
    channel: str = "general",
    reply_to: int | None = None,
    request_id: str | None = None,
) -> dict:
    """Post a message. Reuse request_id when retrying the same logical post."""
    with _client() as board:
        return board.post(board.channel_id(channel), text, reply_to=reply_to, request_id=request_id)


def channels() -> dict:
    """List visible channels."""
    with _client() as board:
        return {"channels": board.list_channels()}


def create_channel(name: str, description: str = "", private: bool = False) -> dict:
    """Create a channel. Teammates must join or be invited separately."""
    with _client() as board:
        return board.create_channel(name, description=description, private=private)


def join_channel(channel: str) -> dict:
    """Join a channel you can access; private channels require an invitation."""
    with _client() as board:
        return {"members": board.add_member(board.channel_id(channel), _identity_id(board))}


def leave_channel(channel: str) -> dict:
    """Leave a channel."""
    with _client() as board:
        return board.remove_member(board.channel_id(channel), _identity_id(board))


def members(channel: str) -> dict:
    """List a channel's members."""
    with _client() as board:
        return {"members": board.members(board.channel_id(channel))}


def invite(channel: str, agent: str) -> dict:
    """Invite one visible agent by exact name or principal ID to a channel."""
    with _client() as board:
        matches = [
            p
            for p in board.people()
            if p.get("identity_type") == "agent" and (str(p["id"]) == agent or p.get("external_id") == agent)
        ]
        if len(matches) != 1:
            raise ValueError("agent must identify exactly one visible agent")
        return {"members": board.add_member(board.channel_id(channel), matches[0]["id"])}


def search(query: str, channel: str | None = None) -> dict:
    """Search board messages, optionally restricting the channel."""
    with _client() as board:
        return board.search(query, channel=board.channel_id(channel) if channel else None)


def wait(cursor: int | None = None, timeout: int = 25) -> dict:
    """Wait once for activity in joined channels; use activity cursors, not message sequences."""
    if not 0 <= timeout <= 25:
        raise ValueError("timeout must be between 0 and 25")
    with _client(max_retries=5) as board:  # fewer: agents repeat a long poll anyway
        return board.activity_page(after=cursor, wait=timeout, channels="joined", limit=100)


TOOLS = {
    "whoami": whoami,
    "agents": agents,
    "read": read,
    "post": post,
    "channels": channels,
    "create_channel": create_channel,
    "join_channel": join_channel,
    "leave_channel": leave_channel,
    "members": members,
    "invite": invite,
    "search": search,
    "wait": wait,
}
