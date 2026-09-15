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

import http.server
import socket
import socketserver
import threading
import time

import pytest
from accelerate import PartialState

from trl.experimental.async_distillation.vllm_client import VLLMClient as AsyncDistillationClient
from trl.experimental.async_grpo.vllm_client import VLLMClient as AsyncGRPOClient

from ..testing_utils import TrlTestCase


@pytest.mark.parametrize("client_cls", [AsyncGRPOClient, AsyncDistillationClient])
class TestAsyncClientReadiness(TrlTestCase):
    """`wait_for_server_ready` must give up after `server_timeout`, however the server fails. The two clients are
    copies of each other, so both are exercised."""

    @pytest.fixture(autouse=True)
    def accelerate_state(self):
        # The clients log through `accelerate.logging`, which refuses to log before the process state exists; the
        # trainers always create it before constructing a client.
        PartialState()

    # A poll interval far larger than the deadline, as in `TestCheckServerHealthProbe`: a wait that is bounded by the
    # poll interval, or by a fixed per-probe timeout, instead of the deadline overshoots the allowance.
    SERVER_TIMEOUT = 0.1
    POLL_INTERVAL = 5.0
    ALLOWANCE = 0.5

    def test_stalled_server_raises_within_the_deadline(self, client_cls):
        # A socket that accepts connections and never answers, so the probe itself has to be bounded.
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(8)
        try:
            client = client_cls(f"http://127.0.0.1:{server.getsockname()[1]}", server_timeout=self.SERVER_TIMEOUT)
            start = time.time()
            with pytest.raises(TimeoutError):
                client.wait_for_server_ready(poll_interval_s=self.POLL_INTERVAL)
            elapsed = time.time() - start
            assert elapsed < self.SERVER_TIMEOUT + self.ALLOWANCE, (
                f"wait_for_server_ready took {elapsed:.3f}s against a stalled server: the probe is not bounded by "
                f"the {self.SERVER_TIMEOUT}s deadline"
            )
        finally:
            server.close()

    def test_not_ready_server_raises_within_the_deadline(self, client_cls):
        # A server answering 503 while it loads: `requests` returns normally, so the sleep between polls has to be
        # clamped to the deadline rather than the poll interval.
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(503)
                self.end_headers()

            def log_message(self, *args):
                pass

        httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            client = client_cls(f"http://127.0.0.1:{httpd.server_address[1]}", server_timeout=self.SERVER_TIMEOUT)
            start = time.time()
            with pytest.raises(TimeoutError):
                client.wait_for_server_ready(poll_interval_s=self.POLL_INTERVAL)
            elapsed = time.time() - start
            assert elapsed < self.SERVER_TIMEOUT + self.ALLOWANCE, (
                f"wait_for_server_ready took {elapsed:.3f}s against a server answering 503: it slept the full poll "
                f"interval instead of stopping at the {self.SERVER_TIMEOUT}s deadline"
            )
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5.0)
