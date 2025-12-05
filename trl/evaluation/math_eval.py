import gc
import os
import re
import json
import time
import random
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.distributed as dist

from transformers import AutoTokenizer

from trl.wandb_utils import setup_wandb
from trl.evaluation.evaluate import evaluate
from trl.evaluation.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from trl.evaluation.parser import *
from trl.evaluation.trajectory import *
from trl.evaluation.data_loader import load_data
from trl.evaluation.python_executor import PythonExecutor
from trl.evaluation.model_utils import load_hf_lm_and_tokenizer, generate_completions
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="trl/evaluation/data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--example_start", default=0, type=int)
    parser.add_argument("--example_end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    parser.add_argument(
        "--evaluate_max_workers",
        type=int,
        default=1,
        help="Maximum number of workers for evaluation.",
    )
    parser.add_argument(
        "--wandb_mode", type=str, default="disabled", help="wandb mode"
    )
    parser.add_argument(
        "--wandb_group", type=str, default="eval_math", help="wandb group"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="minillm-trl", help="wandb project"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="wandb run name"
    )
    parser.add_argument(
        "--wandb_logging_step", type=int, default=0, help="logging step on wandb"
    )
    parser.add_argument(
        "--wandb_job_type", type=str, default="eval", help="wandb job type"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="eLLM-han2024", help="wandb entity"
    )
    parser.add_argument(
        "--dummy_output_steps", type=str, default="", help="Steps to output dummy results"
    )
    parser.add_argument(
        "--log_avg", action="store_true", help="Whether to log average accuracy"
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.1, help="GPU memory utilization for vllm"
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.example_start : len(examples) if args.example_end == -1 else args.example_end]

    # get out_file name
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.example_start}_e{args.example_end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if args.resume:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def main(args):
    if os.path.exists(os.path.join(args.output_dir, f"complete_{args.data_names}.txt")) and args.resume:
        if rank == 0:
            print(f"Evaluation already completed in {args.output_dir}. Exiting...")
        return

    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(available_gpus) == 0 or available_gpus[0] == "":
        available_gpus = [str(i) for i in range(torch.cuda.device_count())]

    # setup wandb
    if rank == 0:
        wandb_cfg = {
            "project": args.wandb_project,
            "name": args.wandb_run_name.strip("/") + f"_{args.data_names}" if args.wandb_run_name else None,
            "group": args.wandb_group.strip("/") if args.wandb_group else None,
            "mode": args.wandb_mode,
            "entity": args.wandb_entity,
            "job_type": args.wandb_job_type,
        }
        if re.match(r"checkpoint-(\d+)", os.path.basename(args.output_dir)):
            wandb_dir = os.path.join(os.path.dirname(args.output_dir), f"wandb_{args.data_names}")
        else:
            wandb_dir = os.path.join(args.output_dir, f"wandb_{args.data_names}")

        wandb_logger = setup_wandb(
            wandb_cfg=wandb_cfg,
            wandb_dir=wandb_dir,
            config=vars(args),
            resume=args.resume,
        )
    else:
        wandb_logger = None

    # infer and evaluate
    data_list = args.data_names.split(",")
    llm, tokenizer = None, None
    results = []
    dummy_output_steps = list(map(int, args.dummy_output_steps.split(","))) if args.dummy_output_steps else []
    is_dummy_step = False
    if re.match(r"checkpoint-(\d+)", os.path.basename(args.output_dir)) is not None:
        is_dummy_step = int(re.match(r"checkpoint-(\d+)", os.path.basename(args.output_dir)).group(1)) in dummy_output_steps

    for data_name in data_list:
        samples, ids, processed_samples, out_file = setup_data(data_name, args)
    
        if len(samples) > 0 and llm is None and tokenizer is None and not is_dummy_step:
            if args.use_vllm:
                llm = LLM(
                    model=args.model_name_or_path,
                    tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
                    pipeline_parallel_size=args.pipeline_parallel_size,
                    trust_remote_code=True,
                    gpu_memory_utilization=args.gpu_memory_utilization
                )
                tokenizer = None
                if args.apply_chat_template:
                    tokenizer = AutoTokenizer.from_pretrained(
                        args.model_name_or_path, trust_remote_code=True
                    )
            else:
                llm, tokenizer = load_hf_lm_and_tokenizer(
                    model_name_or_path=args.model_name_or_path,
                    load_in_half=True,
                    use_fast_tokenizer=True,
                    use_safetensors=args.use_safetensors,
                )
            
            all_samples, time_use = run_generation(llm, tokenizer, data_name, args, samples)
        else:
            all_samples, time_use = [], 0
        
        if not is_dummy_step:
            result_json = postprocess_and_evaluate(all_samples, ids, processed_samples, out_file, data_name, args, time_use)
        else:
            result_json = {"acc": 0.0}
        results.append(result_json)

    # log results
    if rank == 0:
        # add "avg" result to data_list and results
        if args.log_avg:
            data_list.append("avg")
            avg_acc = sum([result["acc"] for result in results]) / len(results)
            results.append({"acc": avg_acc})

        # print all results
        pad = max([len(data_name) for data_name in data_list])
        print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
        print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))
        wandb_logging_dict = {f"math_eval/{data_name}": result["acc"] for data_name, result in zip(data_list, results)}
        if wandb_logger is not None:
            if re.match(r"checkpoint-(\d+)", os.path.basename(args.output_dir)) is not None:
                step = int(re.match(r"checkpoint-(\d+)", os.path.basename(args.output_dir)).group(1))
            else:
                step = args.wandb_logging_step
            wandb_logger.log(wandb_logging_dict, step=step)
    
    # clean up
    if wandb_logger is not None:
        wandb_logger.finish()

    # free gpu memory
    if args.use_vllm:
        destroy_model_parallel()
        destroy_distributed_environment()

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    if rank == 0 and not os.path.exists(os.path.join(args.output_dir, f"complete_{args.data_names}.txt")):
        with open(os.path.join(args.output_dir, f"complete_{args.data_names}.txt"), "w") as f:
            f.write("Evaluation completed.\n")


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def setup_data(data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)

    if rank == 0:
        print("=" * 50)
        print("data:", data_name, " ,remain samples:", len(examples))

        if len(examples) > 0:
            print(examples[0])

    examples_w_ids = [(example, i) for i, example in enumerate(examples)]
    examples_w_ids = examples_w_ids[rank::world_size]
    examples, ids = zip(*examples_w_ids) if len(examples_w_ids) > 0 else ([], [])

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.example_start and rank == 0:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    return samples, ids, processed_samples, out_file


def run_generation(llm, tokenizer, data_name, args, samples):
    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
        stop_words.append("\nYou are an AI assistant")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        if rank == 0:
            print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        if rank == 0:
            print("=== Prompts example ===")
            print(prompts[0])
            print("=======================")
        if args.use_vllm:
            outputs = llm.generate(
                prompts,
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    n=1,
                    stop=stop_words,
                    stop_token_ids=(
                        [151645, 151643]
                        if "qwen2" in args.model_name_or_path.lower()
                        else None
                    ),
                ),
            )

            outputs = sorted(
                outputs, key=lambda x: int(x.request_id)
            )  # sort outputs by request_id
            outputs = [output.outputs[0].text for output in outputs]
        else:
            outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )

        if rank == 0:
            print("=== Outputs example ===")
            print(outputs[0])
            print("=======================")

        assert len(outputs) == len(current_prompts), f"{len(outputs)=} vs {len(current_prompts)=}"

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print(f"Unsolved samples on rank {rank}: {len(remain_prompts)}")
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    torch.cuda.synchronize()
    dist.barrier()
    time_use = time.time() - start_time

    return all_samples, time_use


def postprocess_and_evaluate(
    all_samples, ids, processed_samples, out_file, data_name, args, time_use
):
    assert len(all_samples) == len(ids)
    # all gather samples
    all_samples_with_ids = list(zip(all_samples, ids))
    gathered_all_samples_w_ids = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(all_samples_with_ids, gathered_all_samples_w_ids, dst=0)
    # reorder samples because 

    result_json = None
    if rank == 0:
        all_samples = []
        for part in gathered_all_samples_w_ids:
            all_samples.extend(part)
        all_samples = sorted(all_samples, key=lambda x: x[1])
        all_samples = [item[0] for item in all_samples]

        # add processed samples
        all_samples.extend(processed_samples)
        all_samples, result_json = evaluate(
            samples=all_samples,
            data_name=data_name,
            prompt_type=args.prompt_type,
            execute=True,
            max_workers=args.evaluate_max_workers,
        )

        # save outputs
        if len(processed_samples) < len(all_samples) and args.save_outputs:
            save_jsonl(all_samples, out_file)

        result_json["time_use_in_second"] = time_use
        result_json["time_use_in_minite"] = (
            f"{int(time_use // 60)}:{int(time_use % 60):02d}"
        )

        with open(
            out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
        ) as f:
            json.dump(result_json, f, indent=4)

    dist.barrier()

    return result_json

if __name__ == "__main__":

    dist.init_process_group(backend="gloo", init_method="env://")

    args = parse_args()
    set_seed(args.seed)
    main(args)