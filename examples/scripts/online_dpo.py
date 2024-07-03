from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from trl import ModelConfig
from trl.commands.cli_utils import TrlParser
from trl.trainer.online_dpo_config import OnlineDPOConfig
from trl.trainer.online_dpo_trainer import OnlineDPOTrainer


@dataclass
class ScriptArguments:
    dataset_name: str = None
    dataset_text_field: str = "prompt"
    dataset_train_split: str = "train"
    dataset_test_split: Optional[str] = "test"
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


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, OnlineDPOConfig, ModelConfig))
    args, config, model_config = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_path, num_labels=1)
    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name)
    if config.sanity_check:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(1024))
        config.push_to_hub = False
        config.report_to = ""
        config.save_strategy = "no"
        config.num_sample_generations = 0
        config.total_episodes = 32
        config.per_device_train_batch_size = 8
        config.gradient_accumulation_steps = 1

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
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    if not config.sanity_check:
        trainer.save_model(config.output_dir)
        if config.push_to_hub:
            trainer.push_to_hub()
        trainer.generate_completions()
