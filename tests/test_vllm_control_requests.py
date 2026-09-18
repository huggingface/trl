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

"""HTTP controls must preserve sampling-policy boundaries and accept empty acknowledgments."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from trl.experimental.async_grpo.vllm_client import VLLMClient as AsyncVLLMClient
from trl.generation.vllm_client import VLLMClient


@pytest.fixture
def server():
    responses = {"status": 200, "body": b"", "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            responses["requests"].append(urlsplit(self.path))
            self.send_response(responses["status"])
            self.send_header("Content-Length", str(len(responses["body"])))
            self.end_headers()
            self.wfile.write(responses["body"])

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", responses
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_async_weight_update_pause_drains_requests_and_clears_old_cache(server):
    url, state = server
    client = AsyncVLLMClient(url)
    client.pause()
    request = state["requests"][-1]
    assert request.path == "/pause"
    assert parse_qs(request.query) == {"mode": ["wait"], "clear_cache": ["True"]}
    state.update(status=500, body=b"pause failed")
    with pytest.raises(Exception, match="500, pause failed"):
        client.pause()


def test_reset_cache_accepts_empty_success_but_propagates_errors(server):
    url, state = server
    # The control endpoint needs no GPU communicator or readiness probing.
    client = VLLMClient.__new__(VLLMClient)
    client.base_url = url
    with requests.Session() as session:
        client.session = session
        assert client.reset_prefix_cache() is None
        assert state["requests"][-1].path == "/reset_prefix_cache"
        state.update(status=500, body=b"cache reset failed")
        with pytest.raises(Exception, match="500, cache reset failed"):
            client.reset_prefix_cache()


def test_generation_responses_still_require_json(server):
    url, state = server
    client = VLLMClient.__new__(VLLMClient)
    with requests.Session() as session:
        client.session = session
        with pytest.raises(requests.exceptions.JSONDecodeError):
            client._post(url + "/v1/completions")
        state["body"] = b'{"choices": []}'
        assert client._post(url + "/v1/completions") == {"choices": []}
