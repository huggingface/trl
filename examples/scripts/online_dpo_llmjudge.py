import asyncio
import random
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from trl import ModelConfig
from trl.commands.cli_utils import TrlParser
from trl.trainer.online_dpo_trainer import OnlineDPOConfig, OnlineDPOTrainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE


"""
python examples/scripts/online_dpo_llmjudge.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_llmjudge \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-14m \
    --sft_model_path EleutherAI/pythia-14m \
    --reward_model_path EleutherAI/pythia-14m \
    --non_eos_penalty \
    --stop_token eos \
    --response_length 53 \
    --sanity_check
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/online_dpo_llmjudge.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_llmjudge \
    --per_device_train_batch_size 16 \
    --local_rollout_forward_batch_size 32 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --gradient_accumulation_steps 4 \
    --total_episodes 1000000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr  \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --save_strategy no \
    --non_eos_penalty \
    --stop_token eos \
    --beta 0.1 \
    --response_length 53 \
    --push_to_hub

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/online_dpo_llmjudge.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_llmjudge_tldr_6.9b \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --local_rollout_forward_batch_size 8 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 1000000 \
    --model_name_or_path EleutherAI/pythia-6.9b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-6.9b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-6.9b-deduped__reward__tldr \
    --save_strategy no \
    --non_eos_penalty \
    --stop_token eos \
    --beta 0.1 \
    --response_length 53 \
    --push_to_hub


python -m vllm.entrypoints.openai.api_server --model NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123
python examples/scripts/online_dpo_llmjudge.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_llmjudge \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-14m \
    --sft_model_path EleutherAI/pythia-14m \
    --reward_model_path EleutherAI/pythia-14m \
    --non_eos_penalty \
    --stop_token eos \
    --response_length 53 \
    --sanity_check \
    --base_url https://ip-26-0-166-125/v1 \
    --api_key token-abc123 \
    --model NousResearch/Meta-Llama-3-8B-Instruct
"""


@dataclass
class ScriptArguments:
    dataset_name: str = None
    dataset_text_field: str = "prompt"
    dataset_train_split: str = "train"
    dataset_test_split: Optional[str] = "validation"
    max_length: int = 512


def prepare_dataset(dataset, tokenizer, dataset_text_field):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}

    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        batched=True,
        num_proc=4,  # multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )


TEMPLATE = r"""
Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? Judge based on accuracy, coverage, and coherence.

### Post:
{{post}}

### Summary A:
{{response0}}

### Summary B:
{{response1}}

### Instructions:
FIRST provide a one-sentence comparison of the two summaries, explaining which \
you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">
"""

# 1. extract the common text as the query (this way we do not require the user to provide the query)
# 2. support different kind of judges (let us say mainly LLM judges at the moment)


@dataclass
class LLMJudgeConfig:
    n: int = 64
    model: str = "gpt-3.5-turbo-0125"
    max_parallel_requests: Optional[int] = None
    llm_judge_template: str = ""
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    def __post_init__(self):
        if "gpt-3.5" in self.model:
            # gpt-3.5 generates so fast that it will exceeds the
            # token limit per minute
            self.max_parallel_requests = 11
        elif "gpt-4" in self.model:
            self.max_parallel_requests = 13
        else:  # assume self-hosted
            self.max_parallel_requests = 11


class LLMJudge:
    def __init__(self, ljc: LLMJudgeConfig):
        self.ljc = ljc
        self.async_client = AsyncOpenAI(api_key=ljc.api_key, base_url=ljc.base_url)

    async def process_text(self, post: str, response0: str, response1: str, i: int, limiter=None):
        text = self.ljc.llm_judge_template.replace("{{post}}", post)
        text = text.replace("{{response0}}", response0)
        text = text.replace("{{response1}}", response1)

        async with limiter:
            response = None
            while response is None:
                try:
                    response = await self.async_client.chat.completions.create(
                        model=self.ljc.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": text},
                        ],
                    )
                    r = response.choices[0].message.content
                except Exception as e:
                    print(f"error in {i}: {e}")
                    time.sleep(30)
                    continue

            try:
                comparison = r.split("Comparison:")[1].split("Preferred:")[0].strip()
                preferred = r.split("Preferred:")[1].strip()
                return comparison, preferred, i, text + r
            except Exception as e:
                print(f"error in {i} {e}")
                return "", random.choice(["A", "B"]), i, text + r

    def judge(self, df: pd.DataFrame):
        async def main(ljc: LLMJudgeConfig, df: pd.DataFrame):
            limiter = asyncio.Semaphore(ljc.max_parallel_requests)
            """`df` should have columns: `prompt`, `response0`, `response1`"""
            tasks = []
            df["explanation"] = [None for _ in range(len(df))]
            df["preferred"] = [None for _ in range(len(df))]
            df["shuffled_index"] = [None for _ in range(len(df))]
            df["entire_conversation"] = [None for _ in range(len(df))]
            r = range(min(ljc.n, len(df)))
            if ljc.n == -1:
                r = range(len(df))
            for i in r:
                post = df["prompt"].iloc[i].strip()
                # shuffled the index to avoid GPT4's preference bias in the content's order
                shuffled_index = random.randint(0, 1)
                df.at[i, "shuffled_index"] = shuffled_index
                responses = [
                    df["response0"].iloc[i].strip(),
                    df["response1"].iloc[i].strip(),
                ]
                response0 = responses[shuffled_index]
                response1 = responses[1 - shuffled_index]
                task = asyncio.create_task(self.process_text(post, response0, response1, i, limiter))
                tasks.append(task)

            results = await tqdm_asyncio.gather(*tasks)

            for _, (comparison, preferred, i, entire_conversation) in enumerate(results):
                df.at[i, "explanation"] = comparison
                df.at[i, "entire_conversation"] = entire_conversation
                preferred_label = (
                    "response0"
                    if (df.at[i, "shuffled_index"] == 0 and preferred == "A")
                    or (df.at[i, "shuffled_index"] == 1 and preferred == "B")
                    else "response1"
                )
                df.at[i, "preferred"] = preferred_label
            return df

        return asyncio.run(main(self.ljc, df))


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, OnlineDPOConfig, ModelConfig, LLMJudgeConfig))
    args, config, model_config, judge_config = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    judge_config.n = -1
    judge_config.llm_judge_template = TEMPLATE
    judge = LLMJudge(judge_config)
    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name)
    if config.sanity_check:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(1024))
    train_dataset = raw_datasets[args.dataset_train_split]
    train_dataset = prepare_dataset(train_dataset, tokenizer, args.dataset_text_field)

    if args.dataset_test_split is not None:
        eval_dataset = raw_datasets[args.dataset_test_split]
        eval_dataset = prepare_dataset(eval_dataset, tokenizer, args.dataset_text_field)
    else:
        eval_dataset = None
    ################
    # Training
    ################

    trainer = OnlineDPOTrainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        judge=judge,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    if not config.sanity_check:
        trainer.save_model(config.output_dir)
        if config.push_to_hub:
            trainer.push_to_hub()
        trainer.generate_completions()
