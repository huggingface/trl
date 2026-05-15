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

import asyncio

from trl.experimental.async_grpo.async_rollout_worker import AsyncRolloutWorker


def sync_tool(value: int) -> str:
    return f"sync:{value}"


async def async_tool(value: int) -> str:
    await asyncio.sleep(0)
    return f"async:{value}"


async def failing_async_tool(value: int) -> str:
    await asyncio.sleep(0)
    raise RuntimeError(f"boom:{value}")


def _make_worker():
    # Bypass __init__: these tests target tool execution semantics only.
    return object.__new__(AsyncRolloutWorker)


def test_execute_tool_calls_supports_mixed_sync_and_async_tools():
    worker = _make_worker()
    tool_calls = [
        {"type": "function", "function": {"name": "sync_tool", "arguments": {"value": 1}}},
        {"type": "function", "function": {"name": "async_tool", "arguments": {"value": 2}}},
    ]

    tool_messages, n_calls, n_failures = asyncio.run(
        worker._execute_tool_calls(
            tool_calls,
            sync_tool_dict={"sync_tool": sync_tool},
            async_tool_dict={"async_tool": async_tool},
        )
    )

    assert n_calls == 2
    assert n_failures == 0
    assert tool_messages == [
        {"role": "tool", "name": "sync_tool", "content": "sync:1"},
        {"role": "tool", "name": "async_tool", "content": "async:2"},
    ]


def test_execute_tool_calls_preserves_tool_call_order_for_mixed_sync_and_async_tools():
    worker = _make_worker()
    tool_calls = [
        {"type": "function", "function": {"name": "sync_tool", "arguments": {"value": 1}}},
        {"type": "function", "function": {"name": "async_tool", "arguments": {"value": 2}}},
        {"type": "function", "function": {"name": "sync_tool", "arguments": {"value": 3}}},
    ]

    tool_messages, n_calls, n_failures = asyncio.run(
        worker._execute_tool_calls(
            tool_calls,
            sync_tool_dict={"sync_tool": sync_tool},
            async_tool_dict={"async_tool": async_tool},
        )
    )

    assert n_calls == 3
    assert n_failures == 0
    assert tool_messages == [
        {"role": "tool", "name": "sync_tool", "content": "sync:1"},
        {"role": "tool", "name": "async_tool", "content": "async:2"},
        {"role": "tool", "name": "sync_tool", "content": "sync:3"},
    ]


def test_execute_tool_calls_counts_async_failures():
    worker = _make_worker()
    tool_calls = [{"type": "function", "function": {"name": "async_tool", "arguments": {"value": 3}}}]

    tool_messages, n_calls, n_failures = asyncio.run(
        worker._execute_tool_calls(
            tool_calls,
            sync_tool_dict={},
            async_tool_dict={"async_tool": failing_async_tool},
        )
    )

    assert n_calls == 1
    assert n_failures == 1
    assert len(tool_messages) == 1
    assert tool_messages[0]["name"] == "async_tool"
    assert "boom:3" in tool_messages[0]["content"]


def test_execute_tool_calls_handles_unsupported_tool_call_type():
    worker = _make_worker()
    tool_calls = [{"type": "not-a-function", "name": "bad_tool"}]

    tool_messages, n_calls, n_failures = asyncio.run(
        worker._execute_tool_calls(
            tool_calls,
            sync_tool_dict={},
            async_tool_dict={},
        )
    )

    assert n_calls == 1
    assert n_failures == 1
    assert len(tool_messages) == 1
    assert tool_messages[0]["name"] == "bad_tool"
    assert "Unsupported tool call type" in tool_messages[0]["content"]
