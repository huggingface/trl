import json
import argparse
from tqdm import tqdm
import copy
import concurrent.futures
import threading
import os
import collections
import datasets
from datasets import load_from_disk, load_dataset
import numpy as np
import re
import time

from utils_vllm import get_content

file_lock = threading.Lock()

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def doc_to_text(doc, fewshot_prompt):
    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nLet's think step by step\n"
    )

def doc_to_text2(doc, fewshot_prompt):
    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nAnswer: "
    )

def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


def extract_answer(completion):
    try:
        last_number = re.findall(r"\d+", completion)[-1]
        return eval(last_number)
    except Exception:
        return INVALID_ANS


def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return extract_answer(completion) == gold


def count_completed_samples(output_file):
    """检查已有结果，避免重复计算"""
    prompt_counts = collections.defaultdict(int)
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    prompt = item["prompt"]
                    prompt_counts[prompt] += 1
                except json.JSONDecodeError:
                    continue
    return prompt_counts


def process_item(item, output_file, base_url, model_name, is_chat_model=False):
    """单个样本处理并写入文件"""
    result = copy.deepcopy(item)

    completion = get_content(item["prompt"], base_url, model_name, is_chat_model=is_chat_model)
    result["completion"] = completion

    answer = item["answer"]
    try:
        acc = is_correct(completion, answer)
    except AssertionError:
        acc = False

    result["acc"] = acc

    with file_lock:
        with open(output_file, "a", encoding="utf-8") as g:
            g.write(json.dumps(result, ensure_ascii=False) + "\n")
            g.flush()

    return result


def main():
    parser = argparse.ArgumentParser(description="Run GSM8K evaluation with vLLM + ThreadPoolExecutor")
    parser.add_argument("--sample_input_file", type=str, default=None, help="Optional dataset path (HF format)")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    parser.add_argument("--max_workers", type=int, default=32, help="Maximum number of worker threads")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="Base URL of vLLM server")
    parser.add_argument("--model_name", type=str, required=True, help="Model name served by vLLM")
    parser.add_argument("--few_shot_prompt_file", type=str, default="trl/evaluation/generate_api_answers/gsm8k_prompt.txt", help="Few-shot prompt file path")
    parser.add_argument("--is_chat_model", action="store_true", help="Whether the model is a chat model")
    args = parser.parse_args()

    # 加载 fewshot prompt
    fewshot_prompt = open(args.few_shot_prompt_file).read()
    # 加载数据
    if args.sample_input_file is not None:
        dataset = load_from_disk(args.sample_input_file)
    else:
        config = datasets.DownloadConfig(resume_download=True, max_retries=100)
        dataset = load_dataset("gsm8k", "main", download_config=config)

    test = dataset["test"]

    # 转换成带 prompt 的 dict 列表
    data = []
    for doc in test:
        data.append({
            "prompt": doc_to_text(doc, fewshot_prompt),
            "answer": doc["answer"],
            "question": doc["question"],
        })

    # 检查已有结果
    if os.path.exists(args.output_file):
        completed_counts = count_completed_samples(args.output_file)
        print(f"Found {len(completed_counts)} completed samples from previous run")
    else:
        with open(args.output_file, "w", encoding="utf-8") as g:
            completed_counts = dict()

    expanded_data = []
    for item in data:
        if item["prompt"] not in completed_counts:
            expanded_data.append(copy.deepcopy(item))

    total_tasks = len(expanded_data)
    print(f"Total remaining samples to process: {total_tasks}")

    acc_res = []

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_item = {
            executor.submit(process_item, item, args.output_file, args.base_url, args.model_name): i
            for i, item in enumerate(expanded_data)
        }

        with tqdm(total=len(expanded_data), desc="Processing samples") as pbar:
            for future in concurrent.futures.as_completed(future_to_item):
                idx = future_to_item[future]
                try:
                    result = future.result()
                    acc_res.append(result["acc"])
                except Exception as exc:
                    print(f"Error processing sample {idx}: {exc}")
                pbar.update(1)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")
    print(f"Acc: {np.mean(acc_res) if acc_res else 0.0}")
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
