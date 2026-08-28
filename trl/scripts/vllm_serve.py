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

import argparse
import json
import os
import shlex
import sys
import warnings
from dataclasses import dataclass, field


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        revision (`str`, *optional*):
            Revision to use for the model. If not specified, the default branch will be used.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use.
        data_parallel_size (`int`, *optional*, defaults to `1`):
            Number of data parallel workers to use. For dense models, keep this at 1. Setting this above `1` for dense
            models is not supported/useful and will error out (see vLLM PR #30739).
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8000`):
            Port to run the server on.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        max_model_len (`int`, *optional*):
            If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        enable_prefix_caching (`bool`, *optional*):
            Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the hardware support
            this feature.
        enforce_eager (`bool`, *optional*, defaults to `False`):
            Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always execute the
            model in eager mode. If `False` (default behavior), we will use CUDA graph and eager execution in hybrid.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation to use for vLLM. Must be one of `"transformers"` or `"vllm"`. `"transformers"`: Use
            the `transformers` backend for model implementation. `"vllm"`: Use the `vllm` library for model
            implementation.
        kv_cache_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for KV cache. If set to `"auto"`, the dtype will default to the model data type.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to trust remote code when loading models. Set to `True` to allow executing code from model
            repositories. This is required for some custom models but introduces security risks.
        log_level (`str`, *optional*, defaults to `"info"`):
            Log level for uvicorn. Possible choices: `"critical"`, `"error"`, `"warning"`, `"info"`, `"debug"`,
            `"trace"`.
        distributed_executor_backend (`str` or `None`, *optional*):
            Distributed executor backend for vLLM. Set to `"ray"` to distribute tensor parallel workers across multiple
            nodes via a Ray cluster. Required when `tensor_parallel_size` exceeds the number of local GPUs. If not set,
            vLLM defaults to the multiproc backend (single-node only).
        speculative_config (`str`, *optional*):
            JSON string for vLLM speculative decoding config, forwarded to `vllm serve --speculative-config`. When
            unset, speculative decoding is disabled. Example: `'{"method": "qwen3_next_mtp", "num_speculative_tokens":
            5}'`.
    """

    model: str = field(
        metadata={"help": "Model name or path to load the model from."},
    )
    revision: str | None = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Number of data parallel workers to use. For dense models, keep this at 1. Setting this above "
            "`1` for dense models is not supported/useful and will error out (see vLLM PR #30739)."
        },
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    max_model_len: int | None = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    enable_prefix_caching: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )
    enforce_eager: bool | None = field(
        default=False,
        metadata={
            "help": "Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always "
            "execute the model in eager mode. If `False` (default behavior), we will use CUDA graph and eager "
            "execution in hybrid."
        },
    )
    kv_cache_dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for KV cache. If set to 'auto', the dtype will default to the model data type."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to trust remote code when loading models. Set to True to allow executing code from model "
            "repositories. This is required for some custom models but introduces security risks."
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "Log level for uvicorn. Possible choices: 'critical', 'error', 'warning', 'info', 'debug', "
            "'trace'."
        },
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={
            "help": "Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: "
            "Use the `transformers` backend for model implementation. `vllm`: Use the `vllm` library for "
            "model implementation."
        },
    )
    distributed_executor_backend: str | None = field(
        default=None,
        metadata={
            "help": "Distributed executor backend for vLLM. When set to 'ray', vLLM uses Ray to distribute tensor "
            "parallel workers across multiple nodes. Required when tensor_parallel_size exceeds the number of local "
            "GPUs. If not set, vLLM defaults to the multiproc backend (single-node only)."
        },
    )
    speculative_config: str | None = field(
        default=None,
        metadata={
            "help": "JSON string for vLLM speculative decoding config. "
            'Example: \'{"method": "qwen3_next_mtp", "num_speculative_tokens": 5}\''
        },
    )


def build_command(script_args: ScriptArguments, extra_args: list[str] | None = None) -> list[str]:
    """
    Build the `vllm serve` command line that serves `script_args` the way TRL trainers expect.

    Beyond the plain argument translation, three settings are imposed by TRL:

    - `--weight-transfer-config`: enables the NCCL weight-transfer engine, used by
      [`~generation.vllm_client.VLLMClient`] to push the training weights into the server.
    - `--logprobs-mode processed_logprobs`: so that temperature scaling and logit tweaking are reflected in the
      returned logprobs, which trainers use for importance sampling correction.
    - `--max-logprobs -1`: lifts the OpenAI-compatible cap of 20 logprobs per token, so that trainers can request the
      top-k teacher distribution (used for distillation).

    Args:
        script_args (`ScriptArguments`):
            Arguments of the `trl vllm-serve` command.
        extra_args (`list[str]`, *optional*):
            Additional arguments forwarded to `vllm serve` as-is.

    Returns:
        `list[str]`: Command line, starting with the Python executable.

    Examples:

    ```python
    >>> build_command(ScriptArguments(model="Qwen/Qwen3-0.6B", port=8001))[-6:]
    ['--weight-transfer-config', '{"backend": "nccl"}', '--logprobs-mode', 'processed_logprobs', '--max-logprobs', '-1']
    ```
    """
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        script_args.model,
        "--host",
        script_args.host,
        "--port",
        str(script_args.port),
        "--tensor-parallel-size",
        str(script_args.tensor_parallel_size),
        "--data-parallel-size",
        str(script_args.data_parallel_size),
        "--gpu-memory-utilization",
        str(script_args.gpu_memory_utilization),
        "--dtype",
        script_args.dtype,
        "--kv-cache-dtype",
        script_args.kv_cache_dtype,
        "--model-impl",
        script_args.vllm_model_impl,
        "--uvicorn-log-level",
        script_args.log_level,
    ]
    if script_args.revision is not None:
        command += ["--revision", script_args.revision]
    if script_args.max_model_len is not None:
        command += ["--max-model-len", str(script_args.max_model_len)]
    if script_args.enable_prefix_caching is not None:
        command += ["--enable-prefix-caching" if script_args.enable_prefix_caching else "--no-enable-prefix-caching"]
    if script_args.enforce_eager:
        command += ["--enforce-eager"]
    if script_args.trust_remote_code:
        command += ["--trust-remote-code"]
    if script_args.distributed_executor_backend is not None:
        command += ["--distributed-executor-backend", script_args.distributed_executor_backend]
    if script_args.speculative_config is not None:
        command += ["--speculative-config", script_args.speculative_config]

    command += [
        "--weight-transfer-config",
        json.dumps({"backend": "nccl"}),
        "--logprobs-mode",
        "processed_logprobs",
        "--max-logprobs",
        "-1",
    ]
    return command + list(extra_args or [])


def main(script_args: ScriptArguments, extra_args: list[str] | None = None):
    command = build_command(script_args, extra_args)
    env = os.environ.copy()
    # The weight-transfer and prefix-cache endpoints that trainers rely on live behind vLLM's dev mode.
    env["VLLM_SERVER_DEV_MODE"] = "1"
    # We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
    # error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must
    # use the 'spawn' start method
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    equivalent = shlex.join(["vllm", *command[command.index("serve") :]])
    warnings.warn(
        "`trl vllm-serve` is deprecated and will be removed in v2.0.0: it now only runs vLLM's own server. Run it "
        "directly instead:\n\n"
        f"    VLLM_SERVER_DEV_MODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn {equivalent}\n",
        FutureWarning,
        stacklevel=2,
    )
    os.execve(sys.executable, command, env)


def make_parser(subparsers: argparse._SubParsersAction | None = None, prog: str | None = None):
    from trl import TrlParser

    if subparsers is not None:
        parser = subparsers.add_parser("vllm-serve", help="Run the vLLM serve script", dataclass_types=ScriptArguments)
    else:
        parser = TrlParser(ScriptArguments, prog=prog)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, extra_args = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, extra_args)
