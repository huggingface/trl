"""Launch vLLM with direct LoRA NCCL sync for async GRPO training.

Convenience wrapper around the custom LoRA vLLM server entry point.  Accepts a
small set of high-level flags (``--model``, ``--adapter``, ``--lora-name``, etc.)
and forwards them as vLLM CLI arguments to
:mod:`trl.experimental.async_grpo.lora.vllm_server`.

Usage::

    # Single GPU
    python -m trl.experimental.async_grpo.lora.run_vllm \\
        --model Qwen/Qwen3-8B --adapter /path/to/lora_adapter

    # With custom port and lora rank
    python -m trl.experimental.async_grpo.lora.run_vllm \\
        --model Qwen/Qwen3-8B --adapter /path/to/lora_adapter \\
        --port 8800 --max-lora-rank 128

Any extra arguments after ``--`` are forwarded directly to vLLM::

    python -m trl.experimental.async_grpo.lora.run_vllm \\
        --model Qwen/Qwen3-8B --adapter /path/to/lora_adapter \\
        -- --tensor-parallel-size 2 --gpu-memory-utilization 0.95
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch vLLM with direct LoRA NCCL sync for async GRPO training",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base model id or path (e.g. Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter directory (must contain adapter_config.json)",
    )
    parser.add_argument(
        "--lora-name",
        type=str,
        default="sft",
        help="OpenAI 'model' id for the adapter in vLLM (default: sft)",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=40_000)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--dtype", type=str, default="bfloat16")

    args, extra = parser.parse_known_args()

    sys.argv = [
        "trl-lora-vllm-server",
        "--model", args.model,
        "--language-model-only",
        "--enable-lora",
        "--enable-prefix-caching",
        "--generation-config", "vllm",
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--max-lora-rank", str(args.max_lora_rank),
        "--lora-modules", f"{args.lora_name}={args.adapter}",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--logprobs-mode", "processed_logprobs",
        "--dtype", args.dtype,
        *extra,
    ]

    from .vllm_server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
