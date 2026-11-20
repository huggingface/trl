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

"""Build a reusable AgentEscape agent image with Pi and Flock board tools installed."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from board import tool_bundle
from harbor.agents.installed.node_install import nvm_node_install_snippet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    context = args.output_dir / "agent"
    shutil.copytree(args.task_dir / "environment/agent", context, dirs_exist_ok=True)
    tool_bundle(context / "swarm-tools.tgz")
    dockerfile = context / "Dockerfile"
    dockerfile.write_text(
        dockerfile.read_text()
        + '\nSHELL ["/bin/bash", "-c"]\nRUN '
        + nvm_node_install_snippet()
        + " && npm install -g --ignore-scripts @earendil-works/pi-coding-agent@1.4.0"
        + " && mkdir -p /opt/nvm && cp -a /root/.nvm/. /opt/nvm/"
        + ' && for f in /opt/nvm/versions/node/*/bin/*; do ln -sf "$f" /usr/local/bin/; done'
        + " && chmod -R a+rX /opt/nvm\n"
        + "COPY swarm-tools.tgz /tmp/swarm-tools.tgz\n"
        + "RUN mkdir -p /opt/swarm-tools && tar --no-same-owner -xzf /tmp/swarm-tools.tgz -C /opt/swarm-tools"
        + " && chmod -R a+rX /opt/swarm-tools && chmod a+x /opt/swarm-tools/board_tool.py"
        + " && ln -sf /opt/swarm-tools/board_tool.py /usr/local/bin/flock-tool\n"
    )
    image = "localhost/trl-swarm-pi:1.4.0"
    subprocess.run(["podman", "build", "--format", "docker", "-t", image, str(context)], check=True)
    subprocess.run(
        ["podman", "run", "--rm", image, "bash", "-c", "node --version && pi --version && flock-tool --help"],
        check=True,
    )
    (args.output_dir / "agent-overlay.json").write_text(
        json.dumps(
            {"services": {"main": {"image": image, "build": {"context": str(context), "dockerfile": "Dockerfile"}}}}
        )
    )
    subprocess.run(["podman", "save", "-o", str(args.output_dir / "agent-image.tar"), image], check=True)
