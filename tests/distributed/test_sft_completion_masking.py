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

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from ..testing_utils import require_torch_multi_accelerator


@require_torch_multi_accelerator
@pytest.mark.parametrize("invalid_rank", [-1, 1])
def test_sft_completion_masking_ddp(tmp_path, invalid_rank):
    root = Path(__file__).resolve().parents[2]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node=2",
        "--master-addr=127.0.0.1",
        f"--master-port={port}",
        "--monitor-interval=0.1",
        str(root / "tests/distributed/scripts/sft_completion_masking.py"),
        "--output-dir",
        str(tmp_path),
        "--invalid-rank",
        str(invalid_rank),
    ]
    # Terminating the launcher first lets torchrun reap its workers if this test times out.
    with subprocess.Popen(
        command,
        cwd=root,
        env=dict(os.environ, TOKENIZERS_PARALLELISM="false"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as process:
        try:
            output, _ = process.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                output, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate()
            pytest.fail(f"torchrun timed out:\n{output}")
    if invalid_rank == -1:
        assert process.returncode == 0, output
        assert '"rank": 0' in output and '"rank": 1' in output
        assert output.count('"updated": true') == 2
    else:
        assert process.returncode != 0, output
        assert "no supervised next-token targets" in output
        assert "ChildFailedError" in output
