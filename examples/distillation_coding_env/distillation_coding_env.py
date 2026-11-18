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

# /// script
# dependencies = [
#     "trl[vllm]",
#     "trackio",
#     "openenv>=0.13.0",
#     "openenv-coding_env @ git+https://github.com/openenv/OpenEnv.git@v0.13.0#subdirectory=envs/coding_env",
# ]
# ///


"""
On-policy distillation of a coding agent with OpenEnv's coding environment.

The student (Qwen3-1.7B) solves MBPP problems in a Python session: it writes the function, runs it and tests it through
the `run_python` tool, over several turns, and the session keeps what it defines between calls. The teacher (Qwen3-8B)
scores every token the student wrote, with the tool results as context, and the tool results themselves are left out
of the loss. Distillation needs no reward: `get_reward` runs the MBPP tests in the student's session and is only logged
(`rewards/CodingSession/mean`), to follow how often the student ends with a correct function.

By default the script starts the coding environment server on this machine, in its own process and without the
tokens of this one, since it runs code written by the model. Pass `--env-url` to use a server you host instead; it must
accept one session per episode (32 per step).

```sh
python examples/distillation_coding_env/distillation_coding_env.py
```

On Hugging Face Jobs (1x A100, about 40 minutes for the 30 default steps):

```sh
hf jobs uv run --flavor a100-large --timeout 90m examples/distillation_coding_env/distillation_coding_env.py
```
"""

import argparse
import os
import re
import subprocess
import sys
import time
import urllib.request

from coding_env import CodeAction, CodingEnv
from datasets import load_dataset

from trl import DistillationConfig, DistillationTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="On-policy distillation of a coding agent with OpenEnv.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="Student model.")
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen3-8B", help="Teacher model.")
    parser.add_argument(
        "--env-url", type=str, default=None, help="Coding environment server. Default: start one here."
    )
    parser.add_argument("--max-steps", type=int, default=30, help="Number of training steps.")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to store outputs and checkpoints.")
    parser.add_argument("--report-to", type=str, default="trackio", help="Where to log metrics, e.g. `none`.")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the trained model to the Hub.")
    return parser.parse_args()


# The coding environment server, with one Python session per connection and up to 64 at once: OpenEnv's server allows a
# single session unless the environment is marked as safe to run several.
SERVER = """
import sys

import uvicorn
from coding_env.models import CodeAction, CodeObservation
from coding_env.server.python_codeact_env import PythonCodeActEnv
from openenv.core.env_server import create_app


class PythonSession(PythonCodeActEnv):
    SUPPORTS_CONCURRENT_SESSIONS = True  # each session has its own Python executor

    def __init__(self):
        # MBPP's reference solutions import these, which the default allowlist leaves out
        super().__init__(additional_imports=["bisect", "cmath", "heapq", "operator"])


app = create_app(PythonSession, CodeAction, CodeObservation, env_name="coding_env", max_concurrent_envs=64)
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
"""


def start_env_server(port: int = 8001) -> str:
    # The server runs code written by the model, so it gets none of the tokens and keys of this process
    env = {k: v for k, v in os.environ.items() if not re.search("TOKEN|KEY|SECRET|PASSWORD", k, re.IGNORECASE)}
    subprocess.Popen([sys.executable, "-c", SERVER, str(port)], env=env)
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(f"{url}/health", timeout=2)
            return url
        except OSError:
            time.sleep(1)
    raise RuntimeError("The coding environment server did not start.")


PROMPT = """Write a Python function for the task below. Use the `run_python` tool to define the function and to test \
it: the session keeps what you define between calls. When the function works, reply with a one-line summary.

Task: {text}
Your function must pass this test:
{test}"""


def main() -> None:
    args = parse_args()

    env_url = args.env_url or start_env_server()

    class CodingSession:
        def __init__(self):
            # No keepalive pings: the connection idles while the GPU generates, and nothing answers pings then
            self.client = CodingEnv(base_url=env_url, message_timeout_s=300, websocket_ping_interval_s=None)

        def reset(self, test_list: list[str], test_setup_code: str, **kwargs) -> None:
            self.client.reset()  # fresh Python session for this episode
            self.tests = test_list
            self.test_setup_code = test_setup_code

        def run_python(self, code: str) -> str:
            """
            Run Python code in the session and return its output.

            Args:
                code: The Python code to run.

            Returns:
                The standard output and standard error of the code.
            """
            observation = self.client.step(CodeAction(code=code)).observation
            return (observation.stdout + observation.stderr) or f"(no output, exit code {observation.exit_code})"

        def get_reward(self) -> float:
            # Logged only: the fraction of the MBPP tests that pass in the session the student left behind
            self.client.step(CodeAction(code=self.test_setup_code))
            passed = [self.client.step(CodeAction(code=test)).observation.exit_code == 0 for test in self.tests]
            return sum(passed) / len(passed)

    dataset = load_dataset("google-research-datasets/mbpp", "full", split="train")
    dataset = dataset.map(
        lambda x: {"prompt": [{"role": "user", "content": PROMPT.format(text=x["text"], test=x["test_list"][0])}]},
        remove_columns=["text", "code", "task_id", "challenge_test_list"],
    )

    trainer = DistillationTrainer(
        model=args.model,
        teacher_model=args.teacher,
        train_dataset=dataset,
        args=DistillationConfig(
            output_dir=args.output_dir or f"{args.model.split('/')[-1]}-Distillation-Coding",
            model_init_kwargs={"dtype": "bfloat16"},
            teacher_model_init_kwargs={"dtype": "bfloat16"},
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=0.25,
            chat_template_kwargs={"enable_thinking": False},
            # The budget covers the whole episode: every model turn and every tool result
            max_completion_length=4096,
            max_tool_calling_iterations=6,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=2e-6,
            max_steps=args.max_steps,
            logging_steps=1,
            log_completions=True,
            num_completions_to_print=1,
            report_to=args.report_to,
            push_to_hub=args.push_to_hub,
        ),
        environment_factory=CodingSession,
    )
    trainer.train()
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
