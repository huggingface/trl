#!/usr/bin/env python
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import HfArgumentParser

from ..trainer.grpo_config import GRPOConfig

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        revision (`str` or `None`, *optional*, defaults to `None`):
            Revision to use for the model. If not specified, the default branch will be used.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use.
        data_parallel_size (`int`, *optional*, defaults to `1`):
            Number of data parallel workers to use.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `30000`):
            Port to run the server on.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache.
            Higher values will increase the KV cache size and thus improve the model's throughput. However, if the
            value is too high, it may cause out-of-memory (OOM) errors during initialization.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration.
        max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use. This can be useful when running with reduced
            `gpu_memory_utilization`, leading to a reduced KV cache size. If not set, will use the model
            context size.
        enable_prefix_caching (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enable prefix caching. If set to `True`, ensure that the model and the hardware support
            this feature.
        enforce_eager (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always execute the
            model in eager mode.
        kv_cache_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for KV cache. If set to `"auto"`, the dtype will default to the model data type.
        log_level (`str`, *optional*, defaults to `"info"`):
            Log level for uvicorn. Possible choices: `"critical"`, `"error"`, `"warning"`, `"info"`, `"debug"`,
            `"trace"`.
        enable_torch_compile (`bool`, *optional*, defaults to `False`):
            Whether to enable torch.compile for model acceleration.
        torch_compile_mode (`str`, *optional*, defaults to `"reduce-overhead"`):
            Mode to use for torch.compile. Options: "reduce-overhead", "max-autotune".
        torch_compile_backend (`str`, *optional*, defaults to `"inductor"`):
            Backend to use for torch.compile.
        torch_compile_dynamic (`bool`, *optional*, defaults to `True`):
            Whether to use dynamic shapes for torch.compile.
        torch_compile_fullgraph (`bool`, *optional*, defaults to `False`):
            Whether to use full graph for torch.compile.
        torch_compile_disable_cudagraphs (`bool`, *optional*, defaults to `False`):
            Whether to disable CUDA graphs for torch.compile.
    """

    model: str = field(
        metadata={"help": "Model name or path to load the model from."},
    )
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of data parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=30000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache. Higher values will increase the KV cache size and thus improve the model's throughput. "
            "However, if the value is too high, it may cause out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use. This can be useful when running with reduced "
            "`gpu_memory_utilization`, leading to a reduced KV cache size. If not set, will use the model "
            "context size."
        },
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )
    enforce_eager: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always "
            "execute the model in eager mode."
        },
    )
    kv_cache_dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for KV cache. If set to 'auto', the dtype will default to the model data type."
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "Log level for uvicorn. Possible choices: 'critical', 'error', 'warning', 'info', 'debug', "
            "'trace'."
        },
    )
    enable_torch_compile: bool = field(
        default=False,
        metadata={"help": "Whether to enable torch.compile for model acceleration."},
    )
    torch_compile_mode: str = field(
        default="reduce-overhead",
        metadata={"help": "Mode to use for torch.compile. Options: 'reduce-overhead', 'max-autotune'."},
    )
    torch_compile_backend: str = field(
        default="inductor",
        metadata={"help": "Backend to use for torch.compile."},
    )
    torch_compile_dynamic: bool = field(
        default=True,
        metadata={"help": "Whether to use dynamic shapes for torch.compile."},
    )
    torch_compile_fullgraph: bool = field(
        default=False,
        metadata={"help": "Whether to use full graph for torch.compile."},
    )
    torch_compile_disable_cudagraphs: bool = field(
        default=False,
        metadata={"help": "Whether to disable CUDA graphs for torch.compile."},
    )


def main(script_args: ScriptArguments):
    try:
        import sglang
    except ImportError:
        raise ImportError(
            "SGLang is not installed. Please install it with `pip install sglang` to use the SGLang server."
        )

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, script_args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Build server arguments
    server_args = [
        "--model-path",
        script_args.model,
        "--host",
        script_args.host,
        "--port",
        str(script_args.port),
        "--tp",
        str(script_args.tensor_parallel_size),
        "--dp",
        str(script_args.data_parallel_size),
        "--mem-fraction-static",
        str(script_args.gpu_memory_utilization),
    ]

    if script_args.revision is not None:
        server_args.extend(["--revision", script_args.revision])

    if script_args.dtype != "auto":
        server_args.extend(["--dtype", script_args.dtype])

    if script_args.max_model_len is not None:
        server_args.extend(["--max-model-len", str(script_args.max_model_len)])

    if script_args.enable_prefix_caching is not None:
        if script_args.enable_prefix_caching:
            server_args.append("--enable-prefix-caching")

    if script_args.enforce_eager is not None:
        if script_args.enforce_eager:
            server_args.append("--enforce-eager")

    if script_args.kv_cache_dtype != "auto":
        server_args.extend(["--kv-cache-dtype", script_args.kv_cache_dtype])

    # Add torch.compile options if enabled
    if script_args.enable_torch_compile:
        server_args.append("--enable-torch-compile")
        if script_args.torch_compile_mode != "reduce-overhead":
            server_args.extend(["--torch-compile-mode", script_args.torch_compile_mode])
        if script_args.torch_compile_backend != "inductor":
            server_args.extend(["--torch-compile-backend", script_args.torch_compile_backend])
        if not script_args.torch_compile_dynamic:
            server_args.append("--disable-torch-compile-dynamic")
        if script_args.torch_compile_fullgraph:
            server_args.append("--enable-torch-compile-fullgraph")
        if script_args.torch_compile_disable_cudagraphs:
            server_args.append("--disable-torch-compile-cudagraphs")

    # Start the server
    logger.info(f"Starting SGLang server with arguments: {' '.join(server_args)}")
    sglang.launch_server(server_args)


def make_parser(subparsers: argparse._SubParsersAction = None):
    if subparsers is None:
        parser = HfArgumentParser((ScriptArguments,))
    else:
        parser = subparsers.add_parser("sglang-serve", help="Start a SGLang server for GRPO training.")
        # Add arguments from ScriptArguments to the existing parser
        for field in ScriptArguments.__dataclass_fields__.values():
            if field.default is not field.default_factory:
                parser.add_argument(
                    f"--{field.name.replace('_', '-')}",
                    type=type(field.default) if field.default is not None else str,
                    default=field.default,
                    help=field.metadata.get("help", ""),
                )
            else:
                parser.add_argument(
                    f"--{field.name.replace('_', '-')}",
                    type=str,
                    required=True,
                    help=field.metadata.get("help", ""),
                )
    return parser


if __name__ == "__main__":
    parser = make_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args) 
