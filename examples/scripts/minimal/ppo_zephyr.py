import multiprocessing
import shutil

import pandas as pd
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
)

from trl.trainer.ppov2_trainer import PPOConfig, PPOTrainer


"""
python -i examples/scripts/minimal/ppo_zephyr.py \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --total_episodes 10000 \
    --base_model EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped  \
    --reward_model_path EleutherAI/pythia-1b-deduped  \
    --non_eos_penalty \
    --truncate_token eos \

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/minimal/ppo_zephyr.py \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo_zephyr10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --total_episodes 200000 \
    --base_model HuggingFaceH4/mistral-7b-sft-beta \
    --sft_model_path HuggingFaceH4/mistral-7b-sft-beta \
    --reward_model_path weqweasdas/RM-Mistral-7B \
    --local_rollout_forward_batch_size 32 \
    --deepspeed3 \
    --kl_coef 0.10 \
    --non_eos_penalty \
    --truncate_token eos \
    --response_length 128 \
accelerate launch --config_file examples/accelerate_configs/fsdp.yaml \
    examples/scripts/minimal/ppo_zephyr.py \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo_zephyr10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --total_episodes 200000 \
    --base_model HuggingFaceH4/mistral-7b-sft-beta \
    --sft_model_path HuggingFaceH4/mistral-7b-sft-beta \
    --reward_model_path weqweasdas/RM-Mistral-7B \
    --local_rollout_forward_batch_size 32 \
    --kl_coef 0.10 \
    --non_eos_penalty \
    --truncate_token eos \
    --response_length 1024 \
"""


def print_rich_table(df: pd.DataFrame) -> Table:
    console = Console()
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.print(table)


if __name__ == "__main__":
    parser = HfArgumentParser(PPOConfig)
    args = parser.parse_args_into_dataclasses()[0]
    # remove output_dir if exists
    shutil.rmtree(args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        # a default chat template to simply concatenate the messages
        tokenizer.chat_template = (
            "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        )
    value_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        num_labels=1,
    )
    reward_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        num_labels=1,
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(args.sft_model_path)
    policy = AutoModelForCausalLM.from_pretrained(args.sft_model_path)
    ################
    # Dataset
    ################
    raw_datasets = load_dataset("HuggingFaceH4/ultrachat_200k")
    train_dataset = raw_datasets["train_sft"]
    eval_dataset = raw_datasets["test_sft"]
    train_dataset = train_dataset.select(range(1000))
    eval_dataset = eval_dataset.select(range(1000))

    dataset_text_field = "prompt"
    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            input_ids = tokenizer.apply_chat_template(
                element["messages"][:1],
                padding=False,
                add_generation_prompt=True,
            )
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=False,
        )
    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)
    # filtering
    train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 1024)
    eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 1024)
    ################
    # Training
    ################
    trainer = PPOTrainer(
        args=args,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.push_to_hub()
    trainer.generate_completions(True)

