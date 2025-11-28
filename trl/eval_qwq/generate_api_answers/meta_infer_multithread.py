#!/usr/bin/env python3
# coding: utf-8
import os
import json
import torch
import argparse
import copy
import collections
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributed import init_process_group, get_rank, get_world_size, destroy_process_group, barrier
from torch import distributed as dist
from filelock import FileLock

try:
    from modeling import HFModel
except Exception:
    HFModel = None

prompt_column = 'problem'

def count_completed_samples(output_file):
    prompt_counts = collections.defaultdict(int)
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    prompt = item.get(prompt_column, '')
                    gen_count = len(item.get('gen', []))
                    prompt_counts[prompt] += gen_count
                except json.JSONDecodeError:
                    continue
    return prompt_counts


def generate_batch(messages_batch, tokenizer, model, device, args):
    """messages_batch: List[List[dict]]，每个元素是 [{'role': 'user', 'content': ...}]"""
    prompts = []

    # 如果 tokenizer 有 apply_chat_template，就用它
    if hasattr(tokenizer, "apply_chat_template"):
        for messages in messages_batch:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
    else:
        # fallback: 手动拼接
        for messages in messages_batch:
            text = ""
            for m in messages:
                role = m["role"].capitalize()
                text += f"{role}: {m['content']}\n"
            text += "Assistant:"
            prompts.append(text)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    cleaned = []
    for prompt, text in zip(prompts, decoded):
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        cleaned.append(text)

    return cleaned


def process_items_on_gpu(rank, local_data, output_file, args, world_size):
    device = torch.device(f"cuda:{rank}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name_or_path,
    #     torch_dtype=torch.bfloat16,
    #     device_map=None,
    #     low_cpu_mem_usage=False,
    #     trust_remote_code=True,
    # ).to(device)
    # model.eval()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model = model.eval().cuda()

    if HFModel:
        try:
            model = HFModel(model)
        except Exception:
            pass

    print(f"[GPU {rank}] Loaded model and start processing {len(local_data)} items")

    lock_path = output_file + ".lock"
    file_lock = FileLock(lock_path)

    batch_size = args.eval_batch_size
    for i in tqdm(range(0, len(local_data), batch_size), desc=f"Rank {rank}", position=rank, ncols=100):
        batch = local_data[i:i + batch_size]

        messages_batch = [
            [{"role": "user", "content": item[prompt_column]}] for item in batch
        ]

        try:
            generations = generate_batch(messages_batch, tokenizer, model, device, args)

            with file_lock:
                with open(output_file, 'a', encoding='utf-8') as g:
                    for item, gen_text in zip(batch, generations):
                        result = copy.deepcopy(item)
                        result["gen"] = [gen_text]
                        g.write(json.dumps(result, ensure_ascii=False) + "\n")
                    g.flush()

        except Exception as e:
            print(f"[GPU {rank}] Error on batch {i // batch_size}: {e}")

    print(f"[GPU {rank}] Finished all tasks.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=2048)
    args = parser.parse_args()

    init_process_group("nccl")
    rank = get_rank()
    world_size = get_world_size()

    if rank == 0:
        print(f"🚀 Launching distributed inference with {world_size} GPUs, batch_size={args.eval_batch_size}")

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f if l.strip()]
        # for item in data:
        #     item[prompt_column] += "\nPlease reason step by step, and put your final answer within \\boxed{}."

    if rank == 0:
        completed_counts = count_completed_samples(args.output_file)
        if completed_counts:
            print(f"Resuming, found {sum(completed_counts.values())} completed samples")
        else:
            open(args.output_file, 'w').close()
    else:
        completed_counts = {}

    barrier()

    expanded_data = []
    for item in data:
        done = completed_counts.get(item[prompt_column], 0)
        for _ in range(max(0, args.n_samples - done)):
            expanded_data.append(copy.deepcopy(item))

    if not expanded_data:
        if rank == 0:
            print("No remaining samples.")
        destroy_process_group()
        return

    per_rank = len(expanded_data) // world_size
    start = rank * per_rank
    end = (rank + 1) * per_rank if rank < world_size - 1 else len(expanded_data)
    local_data = expanded_data[start:end]

    process_items_on_gpu(rank, local_data, args.output_file, args, world_size)

    barrier()

    if rank == 0:
        print(f"✅ All GPUs completed! Results saved to {args.output_file}")

    destroy_process_group()


if __name__ == "__main__":
    main()
