#!/usr/bin/env python
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
Launch script for data parallel vLLM servers.
This script launches multiple vLLM servers, each on a different GPU, enabling data parallelism
for TRL training.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Launch data parallel vLLM servers")
    
    # Basic parameters
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model or its name on Hugging Face Hub"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision to use for the model. If not specified, the default branch will be used."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="Data type for model weights and activations"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use"
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        help="Enable prefix caching"
    )
    
    # Parallelism parameters
    parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=0,
        help="Number of data parallel instances (0 = use all available GPUs)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism per instance"
    )
    parser.add_argument(
        "--base_port",
        type=int,
        default=8000,
        help="Base port for the first instance (subsequent instances will use base_port+i)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of specific GPU indices to use (e.g., '0,1,2,3'). "
             "If not specified, all available GPUs will be used."
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model length in tokens"
    )
    
    args = parser.parse_args()
    
    # Determine GPUs to use
    if args.gpus:
        gpus = [int(gpu) for gpu in args.gpus.split(",")]
    else:
        gpus = list(range(torch.cuda.device_count()))
    
    available_gpus = len(gpus)
    
    # Calculate data parallel size if not specified
    if args.data_parallel_size <= 0:
        # If tensor parallel size is 1, use all GPUs for data parallelism
        if args.tensor_parallel_size == 1:
            data_parallel_size = available_gpus
        else:
            # Otherwise, ensure we have enough GPUs for the requested tensor parallelism
            data_parallel_size = available_gpus // args.tensor_parallel_size
            if data_parallel_size == 0:
                print(f"Error: Not enough GPUs ({available_gpus}) for tensor parallel size {args.tensor_parallel_size}")
                return 1
    else:
        data_parallel_size = min(args.data_parallel_size, available_gpus // args.tensor_parallel_size)
    
    total_gpus_needed = data_parallel_size * args.tensor_parallel_size
    
    if total_gpus_needed > available_gpus:
        print(f"Warning: Requested {total_gpus_needed} GPUs but only {available_gpus} available")
        print(f"Reducing data parallel size to {available_gpus // args.tensor_parallel_size}")
        data_parallel_size = available_gpus // args.tensor_parallel_size
    
    # Get path to vllm_serve script
    script_dir = Path(__file__).parent
    vllm_serve_path = script_dir / "vllm_serve.py"
    
    # Print configuration
    print(f"Launching {data_parallel_size} vLLM instances with tensor parallel size {args.tensor_parallel_size}")
    print(f"Using a total of {data_parallel_size * args.tensor_parallel_size} GPUs")
    print(f"Base port: {args.base_port}")
    
    # Launch each instance
    processes = []
    for i in range(data_parallel_size):
        # Calculate which GPUs to use for this instance
        start_gpu_idx = i * args.tensor_parallel_size
        gpu_range = gpus[start_gpu_idx:start_gpu_idx + args.tensor_parallel_size]
        
        # Set CUDA_VISIBLE_DEVICES
        cuda_visible_devices = ",".join(map(str, gpu_range))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        
        # Build command
        cmd = [
            sys.executable,
            str(vllm_serve_path),
            "--model", args.model,
            "--tensor_parallel_size", str(args.tensor_parallel_size),
            "--dtype", args.dtype,
            "--gpu_memory_utilization", str(args.gpu_memory_utilization),
            "--port", str(args.base_port + i),
        ]
        
        # Add optional arguments
        if args.revision:
            cmd.extend(["--revision", args.revision])
        
        if args.enable_prefix_caching:
            cmd.append("--enable_prefix_caching")
            
        if args.max_model_len:
            cmd.extend(["--max_model_len", str(args.max_model_len)])
        
        # Launch process
        print(f"Starting instance {i} on GPU(s) {cuda_visible_devices} with port {args.base_port + i}")
        process = subprocess.Popen(cmd, env=env)
        processes.append(process)
        
        # Add a small delay to prevent conflicts
        time.sleep(2)
    
    # Wait for all processes
    try:
        # Wait for all processes to complete (which they won't unless killed)
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all instances...")
        for p in processes:
            p.terminate()
        
        # Give processes time to terminate
        time.sleep(2)
        
        # Force kill any remaining processes
        for p in processes:
            if p.poll() is None:  # If process is still running
                p.kill()

if __name__ == "__main__":
    sys.exit(main()) 