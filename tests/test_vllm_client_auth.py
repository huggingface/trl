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

import importlib
import json
import socket
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
import requests
from transformers import HfArgumentParser

from trl.generation.vllm_client import VLLMClient
from trl.import_utils import is_vllm_available
from trl.scripts.vllm_serve import ScriptArguments, build_command


@dataclass
class ServerState:
    key: str | None = "test-key"
    requests: list = field(default_factory=list)
    status: dict = field(default_factory=dict)
    delay: float = 0.0
    protected_health: bool = True
    url: str = ""
    port: int = 0


@pytest.fixture
def server(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    state = ServerState()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.respond()

        def respond(self):
            authorization = self.headers.get("Authorization")
            state.requests.append((self.command, self.path, authorization))
            needs_key = state.key and (self.path != "/health" or state.protected_health)
            status = state.status.get(self.path, 200)
            if isinstance(status, list):
                status = status.pop(0)
            if self.path == "/health":
                time.sleep(state.delay)
            if needs_key and authorization != f"Bearer {state.key}":
                status = 401
            payload = {"data": [{"id": "test-model"}], "world_size": 2, "success": True}
            if self.path == "/v1/completions":
                payload = {"choices": [{"prompt_token_ids": [1], "token_ids": [2], "logprobs": None}]}
            if status != 200:
                payload = {"error": "response-body-must-not-appear-in-auth-errors"}
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A timed-out readiness probe can close the connection before the response is sent.

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=lambda: httpd.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{httpd.server_port}"
    state.port = httpd.server_port
    yield state
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def test_explicit_key_authenticates_readiness_get_and_post(server, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "wrong-environment-key")
    client = VLLMClient(base_url=server.url, api_key=server.key)
    with client.session:
        assert client.model == "test-model"
        assert client.get_world_size() == 2
        client.reset_prefix_cache()
        assert client.generate([[1]], logprobs=None)["completion_ids"] == [[2]]
    assert [(method, path) for method, path, _ in server.requests] == [
        ("GET", "/health"),
        ("GET", "/v1/models"),
        ("GET", "/get_world_size"),
        ("POST", "/reset_prefix_cache"),
        ("POST", "/v1/completions"),
    ]
    assert all(header == "Bearer test-key" for _, _, header in server.requests)


def test_environment_is_read_when_each_client_is_created(server, monkeypatch):
    for key in ("first-key", "second-key"):
        server.key = key
        monkeypatch.setenv("VLLM_API_KEY", key)
        client = VLLMClient(base_url=server.url)
        client.session.close()
        assert server.requests[-1][2] == f"Bearer {key}"


@pytest.mark.parametrize("host", [None, "localhost"])
def test_client_loopback_default_and_explicit_host(server, host):
    kwargs = {} if host is None else {"host": host}
    client = VLLMClient(server_port=server.port, api_key=server.key, **kwargs)
    client.session.close()
    assert client.host == (host or "127.0.0.1")
    assert client.model == "test-model"


@pytest.mark.skipif(is_vllm_available(), reason="Exercises an HTTP-only client without local vLLM")
def test_weight_transfer_still_requires_local_vllm(server):
    client = VLLMClient(base_url=server.url, api_key=server.key)
    with client.session:
        assert client.generate([[1]], logprobs=None)["completion_ids"] == [[2]]
        with pytest.raises(ImportError, match="vLLM is not installed"):
            client.init_communicator()


@pytest.mark.parametrize("explicit_key", [None, ""])
def test_unauthenticated_server_compatibility(server, monkeypatch, explicit_key):
    server.key = None
    if explicit_key == "":
        monkeypatch.setenv("VLLM_API_KEY", "must-not-be-sent")
    client = VLLMClient(base_url=server.url, api_key=explicit_key)
    client.session.close()
    assert all(header is None for _, _, header in server.requests)


def test_custom_environment_does_not_fall_back_to_another_servers_key(server, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "different-server-key")
    monkeypatch.delenv("TRAINING_VLLM_API_KEY", raising=False)
    with pytest.raises(requests.HTTPError):
        VLLMClient(base_url=server.url, api_key_env="TRAINING_VLLM_API_KEY")
    assert server.requests == [("GET", "/health", None)]


@pytest.mark.timeout(5)
@pytest.mark.parametrize("path", ["/health", "/v1/models"])
@pytest.mark.parametrize("status", [401, 403])
def test_startup_auth_failure_is_immediate(server, path, status):
    server.status[path] = status
    with pytest.raises(requests.HTTPError) as error:
        VLLMClient(base_url=server.url, api_key=server.key, connection_timeout=240)
    assert error.value.response.status_code == status
    assert "response-body-must-not-appear" not in str(error.value)
    assert "test-key" not in str(error.value)
    assert sum(p == path for _, p, _ in server.requests) == 1


def test_native_public_health_does_not_hide_missing_model_credentials(server):
    server.protected_health = False
    with pytest.raises(requests.HTTPError) as error:
        VLLMClient(base_url=server.url)
    assert error.value.response.status_code == 401
    assert [path for _, path, _ in server.requests] == ["/health", "/v1/models"]


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.timeout(5)
@pytest.mark.parametrize("operation", ["get", "generate", "reset"])
def test_auth_rejection_after_startup_is_not_retried(server, operation, status):
    client = VLLMClient(base_url=server.url, api_key=server.key)
    path = {"get": "/get_world_size", "generate": "/v1/completions", "reset": "/reset_prefix_cache"}[operation]
    server.status[path] = status
    with client.session, pytest.raises(requests.HTTPError) as error:
        if operation == "get":
            client.get_world_size()
        elif operation == "generate":
            client.generate([[1]])
        else:
            client.reset_prefix_cache()
    assert error.value.response.status_code == status
    assert sum(p == path for _, p, _ in server.requests) == 1


@pytest.mark.timeout(5)
def test_unready_http_response_obeys_readiness_timeout(server):
    server.status["/health"] = 404
    with pytest.raises(requests.ConnectionError, match="HTTP 404"):
        VLLMClient(base_url=server.url, api_key=server.key, connection_timeout=0)


@pytest.mark.timeout(5)
@pytest.mark.parametrize("timeout", [0, 0.1])
def test_refused_connection_respects_readiness_budget(timeout):
    # Close a local listener to exercise the connection-refused path.
    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        endpoint.listen()
        port = endpoint.getsockname()[1]
    started = time.monotonic()
    with pytest.raises(requests.ConnectionError):
        VLLMClient(server_port=port, connection_timeout=timeout, api_key="")
    assert time.monotonic() - started < 1.0


@pytest.mark.timeout(5)
@pytest.mark.parametrize("timeout", [0, 0.1])
def test_unready_503_has_no_adapter_retries_or_polling_oversleep(server, timeout):
    server.status["/health"] = 503
    started = time.monotonic()
    with pytest.raises(requests.ConnectionError, match="HTTP 503"):
        VLLMClient(base_url=server.url, api_key=server.key, connection_timeout=timeout)
    assert time.monotonic() - started < 1.0
    assert [path for _, path, _ in server.requests] == ["/health"]


@pytest.mark.timeout(5)
def test_positive_readiness_budget_bounds_stalled_response(server):
    server.delay = 0.4
    started = time.monotonic()
    with pytest.raises(requests.ConnectionError):
        VLLMClient(base_url=server.url, api_key=server.key, connection_timeout=0.05)
    assert time.monotonic() - started < 0.3


def test_health_can_become_ready_within_budget_and_keeps_authentication(server):
    client = VLLMClient(base_url=server.url, api_key=server.key)
    server.requests.clear()
    server.status["/health"] = [503, 200]
    with client.session:
        client.check_server(total_timeout=1, retry_interval=0.01)
    assert server.requests == [("GET", "/health", "Bearer test-key")] * 2


def test_regular_api_requests_keep_their_transient_error_retry(server):
    server.status["/v1/models"] = [503, 200]
    client = VLLMClient(base_url=server.url, api_key=server.key)
    client.session.close()
    assert client.model == "test-model"
    assert [path for _, path, _ in server.requests] == ["/health", "/v1/models", "/v1/models"]


CONFIGS = [
    ("trl.trainer.grpo_config", "GRPOConfig"),
    ("trl.trainer.rloo_config", "RLOOConfig"),
    ("trl.trainer.distillation_config", "DistillationConfig"),
    ("trl.experimental.online_dpo.online_dpo_config", "OnlineDPOConfig"),
    ("trl.experimental.gold.gold_config", "GOLDConfig"),
    ("trl.experimental.ssd.ssd_config", "SSDConfig"),
    ("trl.experimental.sdpo.sdpo_config", "SDPOConfig"),
    ("trl.experimental.sdft.sdft_config", "SDFTConfig"),
    ("trl.experimental.iw_opd.iw_opd_config", "IWOPDConfig"),
]


@pytest.mark.parametrize("module,name", CONFIGS)
def test_config_parses_env_name_without_serializing_key(server, monkeypatch, tmp_path, module, name):
    server.key = "private-test-credential"
    monkeypatch.setenv("TRAINING_VLLM_API_KEY", server.key)
    config_type = getattr(importlib.import_module(module), name)
    (config,) = HfArgumentParser(config_type).parse_args_into_dataclasses(
        [
            "--output_dir",
            str(tmp_path),
            "--use_cpu",
            "true",
            "--bf16",
            "false",
            "--report_to",
            "none",
            "--vllm_server_api_key_env",
            "TRAINING_VLLM_API_KEY",
        ]
    )
    assert config.vllm_server_host == "127.0.0.1"
    assert config.to_dict()["vllm_server_api_key_env"] == "TRAINING_VLLM_API_KEY"
    assert "private-test-credential" not in config.to_json_string()
    assert "private-test-credential" not in repr(config)
    client = VLLMClient(base_url=server.url, api_key_env=config.vllm_server_api_key_env)
    client.session.close()
    assert client.model == "test-model"
    assert server.requests[-1][2] == "Bearer private-test-credential"


def test_wrapper_defaults_to_loopback_and_preserves_explicit_native_arguments():
    default = build_command(ScriptArguments(model="test-model"))
    assert default[default.index("--host") + 1] == "127.0.0.1"
    explicit = build_command(ScriptArguments(model="test-model", host="10.0.0.1"), ["--api-key", "test-key"])
    assert explicit[explicit.index("--host") + 1] == "10.0.0.1"
    assert explicit[-2:] == ["--api-key", "test-key"]
