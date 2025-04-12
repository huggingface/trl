# SPDX-License-Identifier: Apache-2.0
"""
Usage:
Single node:
    python examples/offline_inference/data_parallel.py \
            --model="ibm-research/PowerMoE-3b" \
            --dp-size=2 \
            --tp-size=2

Multi-node:
    Node 0 (assume the node has ip of 10.99.48.128):
            python examples/offline_inference/data_parallel.py \
                    --model="ibm-research/PowerMoE-3b" \
                    --dp-size=2 \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=0 \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
    Node 1:
            python examples/offline_inference/data_parallel.py \
                    --model="ibm-research/PowerMoE-3b" \
                    --dp-size=2 \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=1 \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
"""
import os
from time import sleep

from vllm import LLM, SamplingParams
from vllm.utils import get_open_port


def main(model, dp_size, local_dp_rank, global_dp_rank, dp_master_ip,
         dp_master_port, GPUs_per_dp_rank):
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # CUDA_VISIBLE_DEVICES for each DP rank is set automatically inside the
    # engine processes.

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Machine learning is useful for",
        "The difference between deep learning and machine learning is",
        "The most important technology trend is",
        "In the next decade, we will see",
        "Quantum computing will impact",
        "The biggest challenge in AI today is",
        "The relationship between humans and AI will",
        "Large language models can be used for",
        "The ethical implications of AI include", 
        "Climate change solutions require",
        "The future of work with AI will involve",
        "The most promising AI research areas are",
    ] * 20  

    # With DP, each rank should process different prompts.
    # Divide the prompts among the workers
    prompts_per_rank = len(prompts) // dp_size
    remainder = len(prompts) % dp_size
    
    # Calculate the start and end indices for this worker's prompts
    start_idx = global_dp_rank * prompts_per_rank + min(global_dp_rank, remainder)
    end_idx = start_idx + prompts_per_rank + (1 if global_dp_rank < remainder else 0)
    
    # Get this worker's subset of prompts
    rank_prompts = prompts[start_idx:end_idx]
    
    if len(rank_prompts) == 0:
        # If any rank has no prompts to process,
        # we need to set a placeholder prompt
        rank_prompts = ["Placeholder"]
    print(f"DP rank {global_dp_rank} will process {len(rank_prompts)} prompts from {start_idx} to {end_idx-1}")


    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=[16, 20][global_dp_rank % 2])

    llm = LLM(model=model,
              tensor_parallel_size=GPUs_per_dp_rank,
              enforce_eager=True,
              enable_expert_parallel=False)
    outputs = llm.generate(rank_prompts, sampling_params)
    # Print the generated outputs
    for i, output in enumerate(outputs):
        if i >= 5:
            # print only 5 outputs
            break
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"DP rank {global_dp_rank}, Prompt: {prompt!r}, "
              f"Generated text: {generated_text!r}")


    sleep(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data Parallel Inference")
    parser.add_argument("--model",
                        type=str,
                        default="ibm-research/PowerMoE-3b",
                        help="Model name or path")
    parser.add_argument("--dp-size",
                        type=int,
                        default=2,
                        help="Data parallel size")
    parser.add_argument("--tp-size",
                        type=int,
                        default=2,
                        help="Tensor parallel size")
    parser.add_argument("--node-size",
                        type=int,
                        default=1,
                        help="Total number of nodes")
    parser.add_argument("--node-rank",
                        type=int,
                        default=0,
                        help="Rank of the current node")
    parser.add_argument("--master-addr",
                        type=str,
                        default="",
                        help="Master node IP address")
    parser.add_argument("--master-port",
                        type=int,
                        default=0,
                        help="Master node port")
    args = parser.parse_args()

    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    from multiprocessing import Process

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
            range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)):
        proc = Process(target=main,
                       args=(args.model, dp_size, local_dp_rank,
                             global_dp_rank, dp_master_ip, dp_master_port,
                             tp_size))
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join(timeout=300)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that "
                  f"didn't stop within 5 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
