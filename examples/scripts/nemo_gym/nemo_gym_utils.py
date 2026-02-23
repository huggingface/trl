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

import asyncio
import json
import math
import os
from typing import List

from datasets import Dataset
from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper
from nemo_gym.rollout_collection import RolloutCollectionHelper
from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME, BaseServerConfig
from omegaconf import DictConfig

from trl import GRPOTrainer


def launch_nemo_gym(
    config_paths: List[str],
    model_name: str,
    vllm_server_host: str = "127.0.0.1",
    vllm_server_port: int = 8000,
    host: str = "0.0.0.0",
    port: int = 11000,
) -> RunHelper:
    initial_global_config_dict = {
        HEAD_SERVER_KEY_NAME: {"host": host, "port": port},
        "config_paths": config_paths,
        "policy_base_url": f"http://{vllm_server_host}:{vllm_server_port}/v1",
        "policy_api_key": "EMPTY",
        "policy_model_name": model_name,
        "global_aiohttp_connector_limit_per_host": 16_384,
        "global_aiohttp_connector_limit": 65_536,
    }

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        initial_global_config_dict["hf_token"] = hf_token

    rh = RunHelper()
    rh.start(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=DictConfig(initial_global_config_dict),
            skip_load_from_cli=True,
        )
    )
    return rh


def reward_fn(completions: list[str], **kwargs) -> list[float]:
    env_rewards = kwargs.get("env_reward")
    assert env_rewards is not None, "env_reward not found in kwargs"
    return [float(r) for r in env_rewards]


async def _collect_rollouts(
    examples: List[dict],
    rch: RolloutCollectionHelper,
    head_server_config: BaseServerConfig,
) -> List[dict]:
    nemo_gym_num_rows = len(examples)
    nemo_gym_result_iterator = rch.run_examples(
        examples=examples, head_server_config=head_server_config
    )

    rowidxs = []
    results = []
    for task in nemo_gym_result_iterator:
        nemo_gym_row, nemo_gym_result = await task
        rowidxs.append(nemo_gym_row["_rowidx"])
        results.append(nemo_gym_result)

    sorted_results = [None] * nemo_gym_num_rows
    for rowidx, result in zip(rowidxs, results):
        sorted_results[rowidx] = result

    return sorted_results


def _has_nan_logprobs(responses: List[dict]) -> bool:
    for response in responses:
        if not isinstance(response, dict):
            continue
        for output_item in response.get("response", {}).get("output", []):
            for lp in output_item.get("generation_log_probs", []):
                if math.isnan(lp):
                    return True
    return False


def nemo_gym_rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
    dataset_items = [json.loads(p) for p in prompts]

    for rowidx, item in enumerate(dataset_items):
        item["_rowidx"] = rowidx

        responses_create_params = item.get("responses_create_params", {})
        responses_create_params["temperature"] = trainer.args.temperature
        responses_create_params["top_p"] = trainer.args.top_p
        if trainer.args.max_completion_length is not None:
            responses_create_params["max_output_tokens"] = trainer.args.max_completion_length
        item["responses_create_params"] = responses_create_params

    rch = trainer.args.rch
    head_server_config = trainer.args.head_server_config

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    max_attempts = 3
    for _ in range(max_attempts):
        responses = loop.run_until_complete(
            _collect_rollouts(dataset_items, rch, head_server_config)
        )
        if not _has_nan_logprobs(responses):
            break
    else:
        print("WARNING: NaN logprobs persisted after all retries")

    tokenizer = trainer.processing_class

    prompt_ids: List[list[int]] = []
    completion_ids: List[list[int]] = []
    completion_mask: List[list[int]] = []
    logprobs: List[list[float]] = []
    env_rewards: List[float] = []
    num_turns_list: List[int] = []

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
                        c.get("type") == "output_text" and c.get("text", "").strip()
                        for c in item.get("content", [])
                    )
                )
                for item in output_items
            )
            rollout_failed = not has_content

        if rollout_failed:
            prompt_ids.append([eos_token_id])
            completion_ids.append([eos_token_id])
            completion_mask.append([0])
            logprobs.append([0.0])
            env_rewards.append(0.0)
            num_turns_list.append(0)
            continue

        episode_reward = response.get("reward", 0.0)
        output_items = response.get("response", {}).get("output", [])

        rollout_ids: list[int] = []
        rollout_mask: list[int] = []
        rollout_logprobs: list[float] = []

        seen_token_ids: List[int] = []
        first_prompt = None
        num_turns = 0

        for output_item_dict in output_items:
            if "generation_token_ids" not in output_item_dict:
                continue

            num_turns += 1
            item_prompt_ids = [int(t) for t in output_item_dict["prompt_token_ids"]]
            item_gen_ids = [int(t) for t in output_item_dict["generation_token_ids"]]
            item_logprobs = output_item_dict.get("generation_log_probs", [])

            if first_prompt is None:
                first_prompt = item_prompt_ids
                seen_token_ids = list(item_prompt_ids)
            else:
                assert (
                    seen_token_ids
                    == item_prompt_ids[: len(seen_token_ids)]
                ), (
                    f"Non-contiguous messages found! This may be a tokenization issue "
                    f"where certain tokens are combined when messages are concatenated, "
                    f"or it may be due to part of the chat history being truncated.\n"
                    f"Seen token IDs length: {len(seen_token_ids)}\n"
                    f"Output prompt token IDs length: {len(item_prompt_ids)}"
                )

                # Mask tools and env obs out of training, but retain for attention logprob recomputation
                tool_result_tokens = item_prompt_ids[len(seen_token_ids) :]
                if tool_result_tokens:
                    rollout_ids.extend(tool_result_tokens)
                    rollout_mask.extend([0] * len(tool_result_tokens))
                    rollout_logprobs.extend([0.0] * len(tool_result_tokens))

            # Train on assistant turn
            rollout_ids.extend(item_gen_ids)
            rollout_mask.extend([1] * len(item_gen_ids))
            assert len(item_logprobs) == len(item_gen_ids), (
                f"Logprobs len {len(item_logprobs)} != gen len {len(item_gen_ids)}"
            )
            rollout_logprobs.extend(item_logprobs)

            seen_token_ids = list(item_prompt_ids) + list(item_gen_ids)

        if not rollout_ids or first_prompt is None:
            raise ValueError(f"Rollout {i} has no valid turns")

        prompt_ids.append(first_prompt)
        completion_ids.append(rollout_ids)
        completion_mask.append(rollout_mask)
        logprobs.append(rollout_logprobs)
        env_rewards.append(episode_reward)
        num_turns_list.append(num_turns)

    if not prompt_ids:
        raise RuntimeError("No valid rollouts. Check NeMo Gym and vLLM logs.")

    if num_turns_list:
        trainer.log(
            {
                "num_turns_mean": sum(num_turns_list) / len(num_turns_list),
                "num_turns_min": min(num_turns_list),
                "num_turns_max": max(num_turns_list),
            }
        )

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "env_mask": completion_mask,
        "logprobs": logprobs,
        "env_reward": env_rewards,
        "num_turns": num_turns_list,
    }


def load_dataset_from_jsonl(path: str) -> Dataset:
    with open(path) as f:
        return Dataset.from_list([{"prompt": line.strip()} for line in f if line.strip()])
