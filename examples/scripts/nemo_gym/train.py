import os
import sys 
import numpy as np

from trl import GRPOConfig, GRPOTrainer

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

from transformers import AutoTokenizer


def get_agent_server(
    head_server_host: str = "127.0.0.1",
    head_server_port: int = 11000,
    agent_name: str = None,
) -> str:
    try:
        response = requests.get(
            f"http://{head_server_host}:{head_server_port}/global_config_dict_yaml",
            timeout=10
        )
        response.raise_for_status()
        global_config_yaml = response.text
        global_config_dict = OmegaConf.create(yaml.safe_load(global_config_yaml))
        
        if agent_name:
            for project_name, project_config in global_config_dict.items():
                if hasattr(project_config, 'responses_api_agents'):
                    agents = project_config.responses_api_agents
                    if hasattr(agents, agent_name):
                        agent_config = getattr(agents, agent_name)
                        agent_server = f"http://{agent_config.host}:{agent_config.port}"
                        return agent_server
            
            raise ValueError(f"Agent '{agent_name}' not found in any project's responses_api_agents")
        
        # If no agent_name specified, find it (usually is simple_agent)
        for project_name, project_config in global_config_dict.items():
            if hasattr(project_config, 'responses_api_agents'):
                agents = project_config.responses_api_agents
                for name in agents.keys():
                    agent_config = getattr(agents, name)
                    if hasattr(agent_config, 'host') and hasattr(agent_config, 'port'):
                        agent_server = f"http://{agent_config.host}:{agent_config.port}"
                        return agent_server
        
        raise ValueError("No agents found in global config")
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to connect to head server at {head_server_host}:{head_server_port}: {e}")


@dataclass
class TrainingConfig:
    model_name: str
    dataset_path: str

    task: Optional[str] = None
    agent_name: Optional[str] = None

    learning_rate: float = 5e-6
    max_steps: int = 100
    num_generations: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16

    max_seq_length: int = 1024
    max_prompt_length: int = None

    temperature: float = 1.0
    top_p: float = 1.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"

    output_dir: str = "outputs/trl_nemo_gym"
    save_steps: int = 100
    report_to: str = "none"
    run_name: str = None  # Wandb
    project_name: str = None  # Wandb

def reward_fn(completions: List[str], **kwargs) -> List[float]:
    env_rewards = kwargs.get("env_reward", [])

    if not env_rewards:
        print(f"WARNING: No rewards from Nemo Gym, returning zeros for {len(completions)} completions")
        return [0.0] * len(completions)

    print(f"Received {len(env_rewards)} rewards from Nemo Gym")
    print(f"Mean reward: {sum(env_rewards)/len(env_rewards):.3f}")
    print(f"Reward std dev: {np.std(env_rewards):.3f}")

    return [float(r) for r in env_rewards]


def replace_prefix_tokens(
    tokenizer,
    seen_token_ids: List[int],
    new_prompt_ids: List[int],
) -> List[int]:
    """
    Extract tool result tokens when simple prefix-slicing fails due to retokenization.
    
    The last EOS in seen_token_ids marks where the
    previous model generation ended. Find that same EOS in new_prompt_ids, then return
    everything after it (the new tool results / user messages).
    
    Based on NeMo RL's _replace_prefix_tokens:
    https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/generation/vllm/vllm_worker_async.py#L40
    
    NOTE: this is old and should go in vllm_serve.py not here.
    """
    if not seen_token_ids or not new_prompt_ids:
        return []
    
    eos_token_id = tokenizer.eos_token_id
    assert eos_token_id is not None, "Your tokenizer must have an EOS token ID!"
    
    # Find last EOS in new_prompt_ids within the "prefix" region (up to len(seen_token_ids))
    # search backwards from the prefix boundary
    # EOS marks where the previous model generation ended
    new_eos_pos = -1
    search_bound = min(len(seen_token_ids), len(new_prompt_ids))
    for pos in reversed(range(search_bound)):
        if new_prompt_ids[pos] == eos_token_id:
            new_eos_pos = pos
            break
    
    if new_eos_pos < 0:
        return []
    
    new_content_start = new_eos_pos + 1
    if new_content_start >= len(new_prompt_ids):
        return []
    
    return new_prompt_ids[new_content_start:]


async def call_nemo_gym_agent(
    prompts: List[str],
    dataset_items: List[Dict[str, Any]],
    agent_server: str,
    timeout: float,
    max_completion_length: int = 4096,
    temperature: float = 1.0,
    top_p: float = 0.999,
) -> List[Dict[str, Any]]:
    print(f"Calling Nemo Gym agent: {agent_server}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Max completion length: {max_completion_length}")

    async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar()) as session:
        tasks = []
        for i, (prompt, item) in enumerate(zip(prompts, dataset_items)):
            request_body = item.copy()

            if "responses_create_params" not in request_body:
                request_body["responses_create_params"] = {
                    "input": [{"role": "user", "content": prompt}],
                }

            params = request_body["responses_create_params"]
            params.setdefault("max_output_tokens", max_completion_length)
            params["temperature"] = temperature
            params["top_p"] = top_p

            if i == 0:
                print(f"First request keys: {list(params.keys())}")

            task = session.post(
                f"{agent_server}/run",
                json=request_body,
                timeout=aiohttp.ClientTimeout(total=timeout),
            )
            tasks.append(task)

        responses = []
        with tqdm(total=len(tasks), desc="Agent requests") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                responses.append(result)
                pbar.update(1)

        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"WARNING: Request {i} failed: {response}")
                results.append({"response": {"output": []}, "reward": 0.0, "error": str(response)})
            else:
                try:
                    json_data = await response.json()
                    if isinstance(json_data, dict):
                        results.append(json_data)
                    else:
                        print(f"WARNING: Request {i} returned non-dict: {type(json_data)}")
                        results.append({"response": {"output": []}, "reward": 0.0, "error": f"Non-dict response"})
                except Exception as e:
                    print(f"WARNING: Failed to parse response {i}: {e}")
                    results.append({"response": {"output": []}, "reward": 0.0, "error": str(e)})

        return results


def nemo_gym_rollout_func(prompts: List[str], trainer: GRPOTrainer) -> Dict[str, List]:
    """
    Baseline implementation that is missing on policy tokenid correction (this would go in vllm_serve.py though)
    
    Rollout function for Nemo Gym agent within TRL GRPOTrainer
    
    Builds interleaved action/observation sequence with masking of observations.
    - prompt_ids: first turn's prompt only
    - completion_ids: interleaved [model_gen1, tool_result1, model_gen2, tool_result2, ...]
    - completion_mask: 1 for model tokens, 0 for tool results
    - logprobs: for model tokens, 0.0 for tool result tokens
    
    This ensures:
    1. Logprobs are computed on the full context, including tool results
    2. Loss is only backpropagated on model-generated tokens
    """
    
    current_step = trainer.state.global_step if hasattr(trainer, 'state') else 0

    print(f"\n{'='*80}")
    print(f"[nemo_gym_rollout_func] Starting Nemo Gym rollout (Training Step: {current_step})")
    print(f"[nemo_gym_rollout_func] Received {len(prompts)} prompts from TRL")
    print(f"[nemo_gym_rollout_func] Num generations per prompt: {trainer.args.num_generations}")

    unique_prompts_set = set(prompts)
    print(f"DEBUG: Number of unique prompts in input: {len(unique_prompts_set)}")
    print(f"DEBUG: Total number prompts: {len(prompts)}")

    print(f"\nDEBUG: All unique prompts ({len(unique_prompts_set)} total):")
    for i, prompt in enumerate(sorted(list(unique_prompts_set))[:10]):
        print(f"  [{i}] {prompt}")

    if len(unique_prompts_set) > 10:
        print(f"  ... and {len(unique_prompts_set) - 10} more unique prompts")

    print(f"{'='*80}\n")

    num_generations = trainer.args.num_generations
    print(f"[nemo_gym_rollout_func] Expanding prompts for {num_generations} generations per prompt...")

    expanded_prompts = []
    expanded_dataset_items = []

    for prompt in prompts:
        matching_item = None
        for item in trainer.train_dataset:
            if item.get("prompt") == prompt:
                matching_item = dict(item)
                for key in ["responses_create_params", "expected_answers", "metadata", "ground_truth"]:
                    if key in matching_item and isinstance(matching_item[key], str):
                        try:
                            matching_item[key] = json.loads(matching_item[key])
                        except:
                            pass
                break

        if not matching_item:
            print(f"WARNING: Could not find dataset item for prompt, using prompt only")
            matching_item = {"prompt": prompt}

        for _ in range(num_generations):
            expanded_prompts.append(prompt)
            expanded_dataset_items.append(dict(matching_item))

    print(f"[nemo_gym_rollout_func] Expanded to {len(expanded_prompts)} total requests ({len(prompts)} prompts × {num_generations} generations)")

    print("[nemo_gym_rollout_func] Calling Nemo Gym agent...")
    print(f"[nemo_gym_rollout_func] Using temperature: {trainer.args.temperature}, top_p: {trainer.args.top_p}")
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

    print(f"[nemo_gym_rollout_func] Received {len(responses)} responses from Nemo Gym")

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

    print(f"[Rollout] Saved {len(responses)} trajectories to {trajectory_file}")

    tokenizer = AutoTokenizer.from_pretrained(trainer.model.name_or_path)
    
    # interleaved completion with mask
    prompt_ids: List[List[int]] = []
    completion_ids: List[List[int]] = []
    completion_mask: List[List[int]] = []  # 1 for model tokens, 0 for tool results
    logprobs: List[List[float]] = []
    env_rewards: List[float] = []
    
    failed_count = 0
    success_count = 0

    for i, response in enumerate(responses):
        if not isinstance(response, dict):
            raise ValueError(f"Rollout {i} response is not a dict: {type(response)}")

        if "error" in response:
            raise ValueError(f"Rollout {i} had error: {response['error']}")

        episode_reward = response.get("reward", 0.0)
        output_items = response.get("response", {}).get("output", [])

        # Build interleaved completion: [model_gen1, tool_result1, model_gen2, tool_result2, ...]
        # with mask: 1 for model tokens, 0 for tool results
        # Each turn gives us (prompt_ids, gen_ids). 
        # tool_result = current_prompt - previous_seen_tokens.
        
        seen_token_ids: List[int] = []
        interleaved_completion: List[int] = []
        interleaved_mask: List[int] = []
        interleaved_logprobs: List[float] = []
        first_prompt = None
        num_turns = 0

        for item in output_items:
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
                # likely problematic due to retokenization. this is a baseline to compare against on-policy correction using _replace_prefix_tokens
                # see https://docs.nvidia.com/nemo/gym/latest/contribute/rl-framework-integration/openai-compatible-http-server-on-policy-correction.html
                # and https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/generation/vllm/vllm_worker_async.py#L40
                if len(item_prompt_ids) > len(seen_token_ids):
                    tool_result_tokens = item_prompt_ids[len(seen_token_ids):]

                # Append tool results (mask=0)
                if tool_result_tokens:
                    interleaved_completion.extend(tool_result_tokens)
                    interleaved_mask.extend([0] * len(tool_result_tokens))
                    interleaved_logprobs.extend([0.0] * len(tool_result_tokens))
            
            # Append model generation (mask=1)
            interleaved_completion.extend(item_gen_ids)
            interleaved_mask.extend([1] * len(item_gen_ids))
            interleaved_logprobs.extend(
                item_logprobs if len(item_logprobs) == len(item_gen_ids) else [0.0] * len(item_gen_ids)
            )
            
            if tool_result_tokens:
                seen_token_ids = seen_token_ids + tool_result_tokens + list(item_gen_ids)
            else:
                seen_token_ids = list(item_prompt_ids) + list(item_gen_ids)

        if not interleaved_completion or first_prompt is None:
            raise ValueError(f"Rollout {i} has no valid turns")


        success_count += 1
        
        prompt_ids.append(first_prompt)
        completion_ids.append(interleaved_completion)
        completion_mask.append(interleaved_mask)
        logprobs.append(interleaved_logprobs)
        env_rewards.append(episode_reward)
        
        model_tokens = sum(interleaved_mask)
        tool_tokens = len(interleaved_mask) - model_tokens
        
        print(f"\n{'='*60}")
        print(f"[nemo_gym_rollout_func] Turns: {num_turns}, Reward: {episode_reward:.3f}")
        print(f"[nemo_gym_rollout_func] Prompt tokens: {len(first_prompt)}")
        print(f"[nemo_gym_rollout_func] Completion tokens: {len(interleaved_completion)} (model: {model_tokens}, tool: {tool_tokens})")
        print(f"[nemo_gym_rollout_func] Completion preview: {tokenizer.decode(interleaved_completion)[:150]}...")
        print(f"{'='*60}\n")

    print(f"\n{'='*80}")
    print(f"[nemo_gym_rollout_func] Success: {success_count}, Failed: {failed_count}")
    print(f"[nemo_gym_rollout_func] Total episodes: {len(completion_ids)}")

    if not prompt_ids:
        raise RuntimeError(
            "No valid rollouts. Check Nemo Gym and vLLM logs."
        )

    mean_reward = sum(env_rewards) / len(env_rewards) if env_rewards else 0.0
    total_model_tokens = sum(sum(m) for m in completion_mask)
    total_tool_tokens = sum(len(m) - sum(m) for m in completion_mask)
    print(f"[nemo_gym_rollout_func] Mean reward: {mean_reward:.3f}")
    print(f"[nemo_gym_rollout_func] Total model generation tokens (not masked): {total_model_tokens}")
    print(f"[nemo_gym_rollout_func] Total tool tokens (masked): {total_tool_tokens}")
    
    # We need to deduplicate prompt_ids to match TRL's current code that re-duplicates prompts
    # TRL deduplicates here: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L1266 so we had to duplicate prompts for num_generations
    # TRL reduplicates here: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L1314 so we need to dedup prompts
    print(f"[nemo_gym_rollout_func] Deduplicating prompt_ids (keeping 1 per {num_generations} completions)...")
    unique_prompt_ids = []
    for idx in range(0, len(prompt_ids), num_generations):
        if idx < len(prompt_ids):
            unique_prompt_ids.append(prompt_ids[idx])

    print(f"[nemo_gym_rollout_func] Deduplicated: {len(prompt_ids)} → {len(unique_prompt_ids)} unique prompt_ids")
    print(f"[nemo_gym_rollout_func] Final counts: {len(unique_prompt_ids)} prompt_ids, {len(completion_ids)} completion_ids")
    print(f"[nemo_gym_rollout_func] Expected ratio: {len(completion_ids) / len(unique_prompt_ids) if unique_prompt_ids else 0:.1f} completions per prompt")
    print(f"{'='*80}\n")

    return {
        "prompt_ids": unique_prompt_ids,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "logprobs": logprobs,
        "env_reward": env_rewards,
    }

def get_max_prompt_length(dataset: Dataset, tokenizer) -> int:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    prompt_lengths = [len(tokenizer.encode(item.get("prompt", ""))) for item in dataset if item.get("prompt", "")]
    prompt_lengths.sort()
    max_length = prompt_lengths[-1]
    print(f"[get_max_prompt_length] Min length: {min(prompt_lengths)}")
    print(f"[get_max_prompt_length] Max length: {max(prompt_lengths)}")
    print(f"[get_max_prompt_length] Mean length: {sum(prompt_lengths) / len(prompt_lengths):.1f}")
    return max_length


def load_dataset_from_jsonl(path: str) -> Dataset:
    # TODO: standardize nemo gym dataset format or only accept 1 here (instructions field, answer field, jsonl structure...)
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)

                # Extract prompt before serializing
                if "prompt" not in item:
                    if "responses_create_params" in item and isinstance(item["responses_create_params"], dict):
                        responses_params = item["responses_create_params"]
                        input_data = responses_params.get("input")
                        instructions = responses_params.get("instructions", "")

                        # Handle both message list format and string format
                        if isinstance(input_data, list) and len(input_data) > 0:
                            # Format as messages (e.g. reasoning_gym)
                            prompt_parts = []
                            if instructions:
                                prompt_parts.append(f"system: {instructions}")
                            for msg in input_data:
                                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                                    prompt_parts.append(f"{msg['role']}: {msg['content']}")
                            item["prompt"] = "\n\n".join(prompt_parts) if prompt_parts else ""
                        elif isinstance(input_data, str):
                            # Format as string (e.g. google_search)
                            # Combine instructions field (system prompt) + input field (question)
                            prompt_parts = []
                            if instructions:
                                prompt_parts.append(instructions)
                            if input_data:
                                prompt_parts.append(input_data)
                            item["prompt"] = "\n\n".join(prompt_parts) if prompt_parts else ""
                        else:
                            item["prompt"] = item.get("question", "")
                    else:
                        item["prompt"] = item.get("question", "")

                # Serialize problematic nested structures to JSON strings
                for key in ["responses_create_params", "expected_answers", "metadata", "ground_truth"]:
                    if key in item and isinstance(item[key], (dict, list)):
                        item[key] = json.dumps(item[key])

                data.append(item)

    print(f"Loaded {len(data)} examples from {path}")

    if len(data) < 100:
        repeat_factor = 100
        print(f"Repeating dataset {repeat_factor}x: {len(data)} -> {len(data) * repeat_factor}")
        data = data * repeat_factor

    dataset = Dataset.from_list(data)
    # dataset = dataset.shuffle(seed=42)

    return dataset


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = TrainingConfig(**yaml.safe_load(f))

    agent_server = get_agent_server(
        head_server_host="127.0.0.1",
        head_server_port=11000,
        agent_name=config.agent_name,
    )

    if config.project_name:
        os.environ["WANDB_PROJECT"] = config.project_name

    if config.run_name is None:
        task = config.task or os.path.basename(config.dataset_path).replace('.jsonl', '').replace('.json', '')
        model_short = config.model_name.split("/")[-1]
        config.run_name = (
            f"{task}_{model_short}"
            f"_ng{config.num_generations}"
            f"_dbs{config.per_device_train_batch_size}"
            f"_ga{config.gradient_accumulation_steps}"
            f"_maxlen{config.max_seq_length}"
            f"_lr{config.learning_rate}"
            f"_temp{config.temperature}"
            f"_topp{config.top_p}"
            f"_wd{config.weight_decay}"
            f"_wu{config.warmup_ratio}"
        )

    print(f"\n\nModel: {config.model_name}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Nemo Gym Agent: {agent_server}")
    print(f"vLLM Server: 127.0.0.1:8000")
    print(f"Output dir: {config.output_dir}")
    print(f"Max steps: {config.max_steps}")
    print(f"Num generations: {config.num_generations}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")

    if config.dataset_path.endswith(('.jsonl', '.json')):
        dataset = load_dataset_from_jsonl(config.dataset_path)
    else:
        dataset = load_dataset(config.dataset_path, split="train")

    print(f"Dataset has {len(dataset)} examples\n")

    if config.max_prompt_length is None:
        config.max_prompt_length = get_max_prompt_length(dataset, config.model_name)

    training_args = GRPOConfig(
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="127.0.0.1",
        vllm_server_port=8000,

        temperature=config.temperature,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,

        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,

        max_steps=config.max_steps,
        save_steps=config.save_steps,
        logging_steps=1,
        report_to=config.report_to,
        output_dir=config.output_dir,
        run_name=config.run_name,  # wandb 
        
        epsilon=0.2,
        epsilon_high=0.28,
        loss_type="grpo",
        mask_truncated_completions=True,
        log_completions=False,
        
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_seq_length - config.max_prompt_length,

        shuffle_dataset=False,
    )

    training_args.agent_server = agent_server
    training_args.request_timeout = 6000

    print("\n" + "="*80)
    print("GRPO Config:\n")
    print(f"per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    print(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    print(f"num_generations: {training_args.num_generations}")
    print(f"steps_per_generation: {training_args.steps_per_generation if hasattr(training_args, 'steps_per_generation') else 'Not set (will default to gradient_accumulation_steps)'}")
    print(f"generation_batch_size: {training_args.generation_batch_size if hasattr(training_args, 'generation_batch_size') else 'Not set (will be calculated)'}")
    print(f"shuffle_dataset: {training_args.shuffle_dataset if hasattr(training_args, 'shuffle_dataset') else 'Not set (default: True)'}")
    print(f"Dataset size: {len(dataset)}")
    print("="*80 + "\n")

    print("Initializing GRPO Trainer...")

    trainer = GRPOTrainer(
        model=config.model_name,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        rollout_func=nemo_gym_rollout_func,
        args=training_args,
    )

    print("=" * 80)
    print("Starting training...")

    trainer.train()

    print("=" * 80)
    print("Training complete")

    output_dir = config.output_dir + "/final"
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    trainer.processing_class.save_pretrained(output_dir)

    print("\nFinished saving model")

if __name__ == "__main__":
    main()
