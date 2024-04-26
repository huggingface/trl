from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    HfArgumentParser,
    PreTrainedModel,
)

from trl.trainer.rloo_trainer import RLOOConfig, RLOOTrainer


"""
# run ppo
python examples/scripts/minimal/ppo_bandit_rloo.py \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --base_model EleutherAI/pythia-1b-deduped \
    --non_eos_penalty \
# run REINFORCE w/ RLOO; `--epochs 1 --num_mini_batches 1` in PPO is equivalent to REINFORCE
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/minimal/ppo_bandit_rloo_large.py \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --rloo_k 4 \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --base_model mistralai/Mistral-7B-v0.1 \
    --sft_model_path mistralai/Mistral-7B-v0.1 \
    --reward_model_path mistralai/Mistral-7B-v0.1 \
    --local_rollout_forward_batch_size 1 \
    --deepspeed3 \
    --non_eos_penalty \
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
    parser = HfArgumentParser(RLOOConfig)
    print("========")
    args = parser.parse_args_into_dataclasses()[0]
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
            "{% for message in messages %}{{' ' + message['content']}}{% endfor %}{{ eos_token }}"
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
    raw_datasets = load_dataset("trl-internal-testing/descriptiveness-sentiment-trl-style", split="descriptiveness")

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False).strip()
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False).strip()
        return row

    raw_datasets = raw_datasets.map(process, load_from_cache_file=False)
    eval_samples = 20
    train_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples))
    eval_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples, len(raw_datasets)))

    dataset_text_field = "chosen"

    def prepare_dataset(dataset, tokenizer):
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

    ################
    # Training
    ################
    trainer = RLOOTrainer(
        args=args,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=prepare_dataset(train_dataset, tokenizer),
        eval_dataset=prepare_dataset(eval_dataset, tokenizer),
        train_generation_config=GenerationConfig(
            max_new_tokens=args.response_length,
            min_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        ),
    )
    trainer.train()
    trainer.save_model(args.output_dir)

    ################
    # Generate samples for visual inspection
    ################
    ref_model = trainer.ref_policy
    model = trainer.accelerator.unwrap_model(trainer.model)
    generation_config = GenerationConfig(
        max_new_tokens=100,
        temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    ref_model = ref_model.to(model.device)
    eval_batch_size = 4
    completions = defaultdict(list)
    for i in range(0, len(eval_dataset), eval_batch_size):
        batch = eval_dataset[i : i + eval_batch_size]
        input_ids, attention_mask = tokenizer(batch["prompt"], return_tensors="pt", padding=True).values()
        input_ids, attention_mask = input_ids.to(model.device), attention_mask.to(model.device)
        for m, name in zip([model, ref_model], [f"trained {args.base_model}", f"initial {args.base_model}"]):
            prompt_and_generation = m.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )
            generation = prompt_and_generation[:, input_ids.shape[1] :]
            completions[name].extend(tokenizer.batch_decode(generation, skip_special_tokens=True))

    df = pd.DataFrame({**eval_dataset.to_dict(), **completions})
    del df["rejected"]
    print_rich_table(df.iloc[0 : 0 + 5])
    if "wandb" in args.report_to:
        import wandb

        wandb.log({"completions": wandb.Table(dataframe=df)})
