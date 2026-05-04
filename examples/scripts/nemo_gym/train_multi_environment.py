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

# /// script
# dependencies = [
#     "trl[vllm]",
#     "nemo_gym @ git+https://github.com/NVIDIA-NeMo/Gym",
# ]
# ///

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import aiohttp
import requests
import yaml
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer


@dataclass
class NeMoGymGRPOConfig(GRPOConfig):
    agent_servers: dict[str, str] | None = None
    request_timeout: float = 10800


def get_agent_servers(
    head_server_host: str = "127.0.0.1",
    head_server_port: int = 11000,
) -> dict[str, str]:
    try:
        response = requests.get(f"http://{head_server_host}:{head_server_port}/global_config_dict_yaml", timeout=10)
        response.raise_for_status()
        global_config_yaml = response.text
        global_config_dict = OmegaConf.create(yaml.safe_load(global_config_yaml))

        agent_servers = {}
        for server_name, server_config in global_config_dict.items():
            if hasattr(server_config, "responses_api_agents"):
                agents = server_config.responses_api_agents
                for agent_key in agents.keys():
                    agent_config = getattr(agents, agent_key)
                    if hasattr(agent_config, "host") and hasattr(agent_config, "port"):
                        agent_host = agent_config.host
                        if agent_host in ("127.0.0.1", "0.0.0.0", "localhost"):
                            agent_host = head_server_host
                        agent_servers[server_name] = f"http://{agent_host}:{agent_config.port}"

        if not agent_servers:
            raise ValueError("No agents found in global config")

        return agent_servers

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to connect to head server at {head_server_host}:{head_server_port}: {e}") from e


def reward_fn(completions: list[str], **kwargs) -> list[float]:
    env_rewards = kwargs.get("env_reward")
    assert env_rewards is not None, "env_reward not found in kwargs"
    return [float(r) for r in env_rewards]


async def call_nemo_gym_agents(
    prompts: list[str],
    dataset_items: list[dict[str, Any]],
    agent_servers: dict[str, str],
    timeout: float,
    max_completion_length: int = 4096,
    temperature: float = 1.0,
    top_p: float = 0.999,
) -> list[dict[str, Any]]:
    async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar()) as session:
        tasks = []
        for prompt, item in zip(prompts, dataset_items, strict=False):
            request_body = item.copy()

            if "responses_create_params" not in request_body:
                request_body["responses_create_params"] = {
                    "input": [{"role": "user", "content": prompt}],
                }

            params = request_body["responses_create_params"]
            params.setdefault("max_output_tokens", max_completion_length)
            params["temperature"] = temperature
            params["top_p"] = top_p

            agent_ref = item.get("agent_ref", {})
            agent_name = agent_ref.get("name") if isinstance(agent_ref, dict) else None
            if not agent_name or agent_name not in agent_servers:
                raise ValueError(
                    f"Missing or invalid agent_ref. Got: {agent_ref}. Available: {list(agent_servers.keys())}"
                )
            agent_url = agent_servers[agent_name]

            task = session.post(
                f"{agent_url}/run",
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


def nemo_gym_rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
    is_eval = not trainer.model.training
    num_generations = (
        trainer.args.num_generations_eval
        if is_eval and trainer.args.num_generations_eval
        else trainer.args.num_generations
    )
    dataset = trainer.eval_dataset if is_eval and trainer.eval_dataset is not None else trainer.train_dataset

    expanded_prompts = []
    expanded_dataset_items = []

    for idx_str in prompts:
        idx = int(idx_str)
        item = json.loads(dataset[idx]["metadata"])

        for _ in range(num_generations):
            expanded_prompts.append(idx_str)
            expanded_dataset_items.append(dict(item))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        responses = loop.run_until_complete(
            call_nemo_gym_agents(
                expanded_prompts,
                expanded_dataset_items,
                trainer.args.agent_servers,
                trainer.args.request_timeout,
                trainer.args.max_completion_length,
                temperature=trainer.args.temperature,
                top_p=trainer.args.top_p,
            )
        )
    finally:
        loop.close()

    tokenizer = trainer.processing_class

    prompt_ids: list[list[int]] = []
    completion_ids: list[list[int]] = []  # list of rollouts
    env_mask: list[list[int]] = []  # only train on assistant turns

    logprobs: list[list[float]] = []
    env_rewards: list[float] = []
    num_turns_list: list[int] = []

    for i, response in enumerate(responses):
        eos_token_id = tokenizer.eos_token_id or 0

        if not isinstance(response, dict) or response.get("error"):
            rollout_failed = True
        else:
            output_items = response.get("response", {}).get("output", [])
            has_content = output_items and any(
                item.get("type") == "function_call"
                or (
                    item.get("type") == "message"
                    and any(
                        c.get("type") == "output_text" and c.get("text", "").strip() for c in item.get("content", [])
                    )
                )
                for item in output_items
            )
            rollout_failed = not has_content

        if rollout_failed:
            prompt_ids.append([eos_token_id])
            completion_ids.append([eos_token_id])
            env_mask.append([0])
            logprobs.append([0.0])
            env_rewards.append(0.0)
            num_turns_list.append(0)
            continue

        episode_reward = response.get("reward", 0.0)
        output_items = response.get("response", {}).get("output", [])

        rollout_ids: list[int] = []
        rollout_mask: list[int] = []
        rollout_logprobs: list[float] = []

        seen_token_ids: list[int] = []
        first_prompt = None
        num_turns = 0

        for _idx, item in enumerate(output_items):
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
                    if item_prompt_ids[: len(seen_token_ids)] != seen_token_ids:
                        raise ValueError(
                            f"[Turn {num_turns}] Non-contiguous messages (tokenization issue). "
                            f"Expected prefix len {len(seen_token_ids)}, got prompt len {len(item_prompt_ids)}"
                        )
                    tool_result_tokens = item_prompt_ids[len(seen_token_ids) :]

                if tool_result_tokens:
                    rollout_ids.extend(tool_result_tokens)
                    rollout_mask.extend([0] * len(tool_result_tokens))
                    rollout_logprobs.extend([0.0] * len(tool_result_tokens))

            rollout_ids.extend(item_gen_ids)
            rollout_mask.extend([1] * len(item_gen_ids))
            assert len(item_logprobs) == len(item_gen_ids), (
                f"Logprobs len {len(item_logprobs)} != gen len {len(item_gen_ids)}"
            )
            rollout_logprobs.extend(item_logprobs)

            seen_token_ids = list(item_prompt_ids) + list(item_gen_ids)

        if not rollout_ids or first_prompt is None:
            raise ValueError(f"Rollout {i} has no valid turns")

        prompt_ids.append(first_prompt)  # list of prompts
        completion_ids.append(rollout_ids)  # list of rollouts
        env_mask.append(rollout_mask)
        logprobs.append(rollout_logprobs)
        env_rewards.append(episode_reward)
        num_turns_list.append(num_turns)

    if not prompt_ids:
        raise RuntimeError("No valid rollouts. Check Nemo Gym and vLLM logs.")

    if num_turns_list:
        trainer.log(
            {
                "num_turns_mean": sum(num_turns_list) / len(num_turns_list),
                "num_turns_min": min(num_turns_list),
                "num_turns_max": max(num_turns_list),
            }
        )

    unique_prompt_ids = prompt_ids[::num_generations]

    return {
        "prompt_ids": unique_prompt_ids,
        "completion_ids": completion_ids,
        "env_mask": env_mask,
        "logprobs": logprobs,
        "env_reward": env_rewards,
        "num_turns": num_turns_list,
    }


def load_dataset_from_jsonl(path: str) -> Dataset:
    data = []
    with open(path) as f:
        for idx, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                data.append(
                    {
                        "prompt": str(
                            idx
                        ),  # use index for lookup as not all nemo gym datasets have the same metadata fields. maybe not the most elegant
                        "metadata": json.dumps(item),
                    }
                )
    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--vllm_server_host", type=str, default="127.0.0.1", help="vLLM server hostname/IP")
    parser.add_argument("--head_server_host", type=str, default="127.0.0.1", help="Head server hostname/IP for ng_run")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config.pop("model_name")
    dataset_path = config.pop("dataset_path")
    eval_dataset_path = config.pop("eval_dataset_path", None)
    task = config.pop("task", None)
    project_name = config.pop("project_name", None)

    if "learning_rate" in config and isinstance(config["learning_rate"], str):
        config["learning_rate"] = float(config["learning_rate"])
    if "weight_decay" in config and isinstance(config["weight_decay"], str):
        config["weight_decay"] = float(config["weight_decay"])

    agent_servers = get_agent_servers(
        head_server_host=args.head_server_host,
        head_server_port=11000,
    )

    if project_name:
        os.environ["WANDB_PROJECT"] = project_name

    if dataset_path.endswith((".jsonl", ".json")):
        dataset = load_dataset_from_jsonl(dataset_path)
    else:
        dataset = load_dataset(dataset_path, split="train")

    eval_dataset = None
    if eval_dataset_path:
        eval_dataset = load_dataset_from_jsonl(eval_dataset_path)
        print(f"Eval dataset has {len(eval_dataset)} examples\n")

    training_args = NeMoGymGRPOConfig(
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=8000,
        gradient_checkpointing=True,
        num_generations_eval=1,
        logging_steps=1,
        epsilon=0.2,
        epsilon_high=0.28,
        loss_type="grpo",
        mask_truncated_completions=True,
        shuffle_dataset=False,
        model_init_kwargs={"torch_dtype": "auto"},
        agent_servers=agent_servers,
        request_timeout=10800,
        **config,
    )

    if training_args.run_name is None:
        task_name = task or os.path.basename(dataset_path).replace(".jsonl", "").replace(".json", "")
        model_short = model_name.split("/")[-1]
        training_args.run_name = (
            f"{task_name}_{model_short}"
            f"_rpp{training_args.num_generations}"
            f"_dbs{training_args.per_device_train_batch_size}"
            f"_ga{training_args.gradient_accumulation_steps}"
            f"_maxlen{training_args.max_completion_length}"
            f"_lr{training_args.learning_rate}"
            f"_temp{training_args.temperature}"
            f"_topp{training_args.top_p}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left", padding_side="left")

    trainer = GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        rollout_func=nemo_gym_rollout_func,
        args=training_args,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
