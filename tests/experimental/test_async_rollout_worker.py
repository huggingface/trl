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

import pytest

from trl.experimental.async_grpo.async_rollout_worker import _make_client_session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_inflight_tasks", "expected_limit"),
    [
        (32, 100),
        (100, 100),
        (256, 256),
    ],
)
async def test_client_session_connector_follows_max_inflight_tasks(max_inflight_tasks, expected_limit):
    async with _make_client_session(max_inflight_tasks) as session:
        assert session.connector.limit == expected_limit
