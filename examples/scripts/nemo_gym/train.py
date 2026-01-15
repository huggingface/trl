import os
import argparse
import asyncio
import aiohttp
import json
import yaml
import requests
from omegaconf import OmegaConf
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer

def get_agent_server(
    head_server_host: str = "127.0.0.1",
    head_server_port: int = 11000,
) -> str:
    try:
        response = requests.get(
            f"http://{head_server_host}:{head_server_port}/global_config_dict_yaml",
            timeout=10
        )
        response.raise_for_status()
        global_config_yaml = response.text
        global_config_dict = OmegaConf.create(yaml.safe_load(global_config_yaml))

        for _, project_config in global_config_dict.items():
            if hasattr(project_config, 'responses_api_agents'):
                agents = project_config.responses_api_agents
                for name in agents.keys():
                    agent_config = getattr(agents, name)
                    if hasattr(agent_config, 'host') and hasattr(agent_config, 'port'):
                        agent_host = agent_config.host
                        if agent_host in ("127.0.0.1", "0.0.0.0", "localhost"):
                            agent_host = head_server_host
                        agent_server = f"http://{agent_host}:{agent_config.port}"
                        return agent_server

        raise ValueError("No agents found in global config")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to connect to head server at {head_server_host}:{head_server_port}: {e}")


@dataclass
class TrainingConfig:
    model_name: str
    dataset_path: str

    task: Optional[str] = None

    learning_rate: float = 5e-6
    max_steps: int = 100
    num_generations: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 1024
    max_prompt_length: int = None

    temperature: float = 1.0
    top_p: float = 0.999
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"

    output_dir: str = "outputs/trl_nemo_gym"
    save_steps: int = 100
    report_to: str = "none"
    run_name: str = None  # Wandb
    project_name: str = None  # Wandb
    log_completions: bool = False
    num_completions_to_print: int = None

    eval_dataset_path: Optional[str] = None
    eval_strategy: str = "no"
    eval_steps: int = 50

    vllm_importance_sampling_correction: bool = False

def reward_fn(completions: List[str], **kwargs) -> List[float]:
    env_rewards = kwargs.get("env_reward")
    assert env_rewards is not None, "env_reward not found in kwargs"
    return [float(r) for r in env_rewards]

async def call_nemo_gym_agent(
    prompts: List[str],
    dataset_items: List[Dict[str, Any]],
    agent_server: str,
    timeout: float,
    max_completion_length: int = 4096,
    temperature: float = 1.0,
    top_p: float = 0.999,
) -> List[Dict[str, Any]]:
    print(f"Calling Nemo Gym agent at {agent_server} with {len(prompts)} prompts")

    # todo: increase limits
    async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar()) as session:
        tasks = []
        for prompt, item in zip(prompts, dataset_items):
            request_body = item.copy()

            if "responses_create_params" not in request_body:
                request_body["responses_create_params"] = {
                    "input": [{"role": "user", "content": prompt}],
                }

            params = request_body["responses_create_params"]
            params.setdefault("max_output_tokens", max_completion_length)
            params["temperature"] = temperature
            params["top_p"] = top_p

            task = session.post(
                f"{agent_server}/run",
                json=request_body,
                timeout=aiohttp.ClientTimeout(total=timeout),
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for i, response in enumerate(responses):
            try:
                if isinstance(response, Exception):
                    raise response
                json_data = await response.json()
                if not isinstance(json_data, dict):
                    raise ValueError(f"Expected dict, got {type(json_data)}")
                results.append(json_data)
            except Exception as e:
                print(f"WARNING: Request {i} failed: {e}")
                results.append({"response": {"output": []}, "reward": 0.0, "error": str(e)})

        return results


def nemo_gym_rollout_func(prompts: List[str], trainer: GRPOTrainer) -> Dict[str, List]:
    current_step = trainer.state.global_step

    is_eval = not trainer.model.training
    num_generations = trainer.args.num_generations_eval if is_eval and trainer.args.num_generations_eval else trainer.args.num_generations
    dataset = trainer.eval_dataset if is_eval and trainer.eval_dataset is not None else trainer.train_dataset

    expanded_prompts = []
    expanded_dataset_items = []

    for prompt in prompts:
        matching_item = None
        for item in dataset:
            if item.get("prompt") == prompt:
                matching_item = dict(item)
                for key in ["responses_create_params", "expected_answers", "metadata", "ground_truth", "options", "template_metadata"]:
                    if key in matching_item and isinstance(matching_item[key], str):
                        matching_item[key] = json.loads(matching_item[key])
                break

        for _ in range(num_generations):
            expanded_prompts.append(prompt)
            expanded_dataset_items.append(dict(matching_item))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        responses = loop.run_until_complete(
            call_nemo_gym_agent(
                expanded_prompts,
                expanded_dataset_items,
                trainer.args.agent_server,
                trainer.args.request_timeout,
                trainer.args.max_completion_length,
                temperature=trainer.args.temperature,
                top_p=trainer.args.top_p,
            )
        )
    finally:
        loop.close()

    trajectory_file = os.path.join(trainer.args.output_dir, "trajectories.jsonl")
    os.makedirs(trainer.args.output_dir, exist_ok=True)

    with open(trajectory_file, 'a') as f:
        for i, response in enumerate(responses):
            trajectory_data = {
                "step": current_step,
                "rollout_idx": i,
                "reward": response.get("reward", 0.0) if isinstance(response, dict) else 0.0,
                "output": response.get("response", {}).get("output", []) if isinstance(response, dict) else [],
                "error": response.get("error") if isinstance(response, dict) else str(response),
            }
            f.write(json.dumps(trajectory_data) + "\n")

    tokenizer = trainer.processing_class
    
    prompt_ids: List[List[int]] = []
    completion_ids: List[List[int]] = []
    completion_mask: List[List[int]] = []  # 1 for action, 0 for observation/user
    logprobs: List[List[float]] = []
    env_rewards: List[float] = []
    num_turns_list: List[int] = []

    for i, response in enumerate(responses):
        expected_prompt = expanded_prompts[i]
        expected_prompt_ids = tokenizer.encode(expected_prompt, add_special_tokens=False)

        if not isinstance(response, dict):
            rollout_failed = True
        elif response.get("error"):
            rollout_failed = True
        else:
            output_items = response.get("response", {}).get("output", [])
            if not output_items:
                rollout_failed = True
            else:
                has_content = any(
                    item.get("type") == "function_call" or (
                        item.get("type") == "message" and
                        any(c.get("type") == "output_text" and c.get("text", "").strip()
                            for c in item.get("content", []))
                    )
                    for item in output_items
                )
                rollout_failed = not has_content

        # truncated or other failure - mask
        if rollout_failed:
            eos_token_id = tokenizer.eos_token_id or 0
            prompt_ids.append(expected_prompt_ids)
            completion_ids.append([eos_token_id])
            completion_mask.append([0])
            logprobs.append([0.0])
            env_rewards.append(0.0)
            num_turns_list.append(0)
            continue

        episode_reward = response.get("reward", 0.0)
        output_items = response.get("response", {}).get("output", [])

        # interleaved completion with mask (p,a,o,a,o...)
        # Each turn has prompt_ids, gen_ids
        # tool_result = prompt_ids - seen_token_ids (to mask it) 
        # replace_prefix_tokens done in vllm server https://docs.nvidia.com/nemo/gym/latest/contribute/rl-framework-integration/openai-compatible-http-server-on-policy-correction.html
        
        seen_token_ids: List[int] = []
        interleaved_completion: List[int] = []
        interleaved_mask: List[int] = []
        interleaved_logprobs: List[float] = []
        first_prompt = None
        num_turns = 0

        for idx, item in enumerate(output_items):
            if "prompt_token_ids" not in item or "generation_token_ids" not in item:
                continue

            num_turns += 1
            item_prompt_ids = item["prompt_token_ids"]
            item_gen_ids = item["generation_token_ids"]
            item_logprobs = item.get("generation_log_probs", [])
            tool_result_tokens = []

            if first_prompt is None:
                first_prompt = item_prompt_ids
                seen_token_ids = list(item_prompt_ids)
            else:
                if len(item_prompt_ids) > len(seen_token_ids):
                    if item_prompt_ids[:len(seen_token_ids)] != seen_token_ids:
                        raise ValueError(
                            f"[Turn {num_turns}] Non-contiguous messages (tokenization issue). "
                            f"Expected prefix len {len(seen_token_ids)}, got prompt len {len(item_prompt_ids)}"
                        )
                    tool_result_tokens = item_prompt_ids[len(seen_token_ids):]

                if tool_result_tokens:
                    interleaved_completion.extend(tool_result_tokens)
                    interleaved_mask.extend([0] * len(tool_result_tokens))
                    interleaved_logprobs.extend([0.0] * len(tool_result_tokens))

            interleaved_completion.extend(item_gen_ids)
            interleaved_mask.extend([1] * len(item_gen_ids))
            assert len(item_logprobs) == len(item_gen_ids), f"Logprobs len {len(item_logprobs)} != gen len {len(item_gen_ids)}"
            interleaved_logprobs.extend(item_logprobs)

            seen_token_ids = list(item_prompt_ids) + list(item_gen_ids)

        if not interleaved_completion or first_prompt is None:
            raise ValueError(f"Rollout {i} has no valid turns")

        prompt_ids.append(first_prompt)
        completion_ids.append(interleaved_completion)
        completion_mask.append(interleaved_mask)
        logprobs.append(interleaved_logprobs)
        env_rewards.append(episode_reward)
        num_turns_list.append(num_turns)

    if not prompt_ids:
        raise RuntimeError(
            "No valid rollouts. Check Nemo Gym and vLLM logs."
        )


    if num_turns_list:
        wandb.log({
            "train/num_turns_mean": sum(num_turns_list) / len(num_turns_list),
            "train/num_turns_min": min(num_turns_list),
            "train/num_turns_max": max(num_turns_list),
        })

    unique_prompt_ids = prompt_ids[::num_generations] # TRL re-duplicates them

    return {
        "prompt_ids": unique_prompt_ids,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "logprobs": logprobs,
        "env_reward": env_rewards,
        "num_turns": num_turns_list,
    }

def get_max_prompt_length(dataset: Dataset, tokenizer) -> int:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    return max(len(tokenizer.encode(item.get("prompt", ""))) for item in dataset if item.get("prompt"))

def load_dataset_from_jsonl(path: str) -> Dataset:
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)

                if "prompt" not in item:
                    if "responses_create_params" in item and isinstance(item["responses_create_params"], dict):
                        responses_params = item["responses_create_params"]
                        input_data = responses_params.get("input")
                        instructions = responses_params.get("instructions", "")

                        if isinstance(input_data, list) and len(input_data) > 0:
                            # list of messages format (e.g. reasoning_gym)
                            prompt_parts = []
                            if instructions:
                                prompt_parts.append(f"system: {instructions}")
                            for msg in input_data:
                                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                                    prompt_parts.append(f"{msg['role']}: {msg['content']}")
                            item["prompt"] = "\n\n".join(prompt_parts) if prompt_parts else ""
                        elif isinstance(input_data, str):
                            # prompt as string, no list of messages (e.g. google_search)
                            prompt_parts = []
                            # system prompt
                            if instructions:
                                prompt_parts.append(instructions)
                            if input_data:
                                prompt_parts.append(input_data)
                            item["prompt"] = "\n\n".join(prompt_parts) if prompt_parts else ""
                        else:
                            item["prompt"] = item.get("question", "")
                    else:
                        item["prompt"] = item.get("question", "")

                for key in ["responses_create_params", "expected_answers", "metadata", "ground_truth", "options", "template_metadata"]:
                    if key in item and isinstance(item[key], (dict, list)):
                        item[key] = json.dumps(item[key])

                data.append(item)

    return Dataset.from_list(data)

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--vllm_server_host", type=str, default="127.0.0.1",
                        help="vLLM server hostname/IP")
    parser.add_argument("--head_server_host", type=str, default="127.0.0.1",
                        help="Head server hostname/IP for ng_run")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        config = TrainingConfig(**yaml.safe_load(f))

    if isinstance(config.learning_rate, str):
        config.learning_rate = float(config.learning_rate)
    if isinstance(config.weight_decay, str):
        config.weight_decay = float(config.weight_decay)

    agent_server = get_agent_server(
        head_server_host=args.head_server_host,
        head_server_port=11000,
    )

    if config.project_name:
        os.environ["WANDB_PROJECT"] = config.project_name

    if config.run_name is None:
        task = config.task or os.path.basename(config.dataset_path).replace('.jsonl', '').replace('.json', '')
        model_short = config.model_name.split("/")[-1]
        config.run_name = (
            f"{task}_{model_short}"
            f"_rpp{config.num_generations}"
            f"_dbs{config.per_device_train_batch_size}"
            f"_ga{config.gradient_accumulation_steps}"
            f"_maxlen{config.max_seq_length}"
            f"_lr{config.learning_rate}"
            f"_temp{config.temperature}"
            f"_topp{config.top_p}"
        )

    if config.dataset_path.endswith(('.jsonl', '.json')):
        dataset = load_dataset_from_jsonl(config.dataset_path)
    else:
        dataset = load_dataset(config.dataset_path, split="train")

    print(f"Dataset has {len(dataset)} examples\n")

    eval_dataset = None
    if config.eval_dataset_path:
        eval_dataset = load_dataset_from_jsonl(config.eval_dataset_path)
        print(f"Eval dataset has {len(eval_dataset)} examples\n")

    if config.max_prompt_length is None:
        config.max_prompt_length = get_max_prompt_length(dataset, config.model_name)

    training_args = GRPOConfig(
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=8000,

        gradient_checkpointing=True,

        temperature=config.temperature,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,

        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        num_generations_eval=1,

        max_steps=config.max_steps,
        save_steps=config.save_steps,
        logging_steps=1,
        report_to=config.report_to,
        output_dir=config.output_dir,
        run_name=config.run_name,  # wandb

        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,

        vllm_importance_sampling_correction=config.vllm_importance_sampling_correction,
        epsilon=0.2,
        epsilon_high=0.28,
        loss_type="grpo",
        mask_truncated_completions=True,
        log_completions=config.log_completions,
        num_completions_to_print=config.num_completions_to_print,

        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_seq_length - config.max_prompt_length,
        shuffle_dataset=False,

        model_init_kwargs={
            "torch_dtype": "auto",
        },
    )

    training_args.agent_server = agent_server
    training_args.request_timeout = 6000

    trainer = GRPOTrainer(
        model=config.model_name,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        rollout_func=nemo_gym_rollout_func,
        args=training_args,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

if __name__ == "__main__":
    main()
