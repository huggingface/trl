from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from trl import HfPairwiseJudge, ModelConfig
from trl.commands.cli_utils import TrlParser
from trl.trainer import OnlineDPOConfig, OnlineDPOTrainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE


"""
# Sanity check with minimal config and model
python examples/scripts/online_dpo.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --learning_rate 3e-6 \
    --output_dir online_dpo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-14m \
    --judge hf_pairwise \
    --non_eos_penalty \
    --stop_token eos \
    --response_length 53 \
    --sanity_check

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/online_dpo.py \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr  \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --local_rollout_forward_batch_size 32 \
    --num_epochs 1 \
    --total_episodes 1000000 \
    --non_eos_penalty \
    --stop_token eos
"""


@dataclass
class ScriptArguments:
    dataset_name: str = None
    dataset_text_field: str = "prompt"
    dataset_train_split: str = "train"
    dataset_test_split: Optional[str] = "validation"
    max_length: int = 512


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
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE

    ref_model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path)

    if config.reward_model_path is not None:
        reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_path, num_labels=1)
    else:
        reward_model = None

    if config.judge is not None:
        judge = HfPairwiseJudge()
    else:
        judge = None

    ################
    # Dataset
    ################
    ds = load_dataset(args.dataset_name)
    if config.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(1024))
    train_dataset = ds[args.dataset_train_split]
    if args.dataset_test_split is not None:
        eval_dataset = ds[args.dataset_test_split]
    else:
        eval_dataset = None

    ################
    # Training
    ################

    trainer = OnlineDPOTrainer(
        model=model,
        config=config,
        ref_model=ref_model,
        reward_model=reward_model,
        judge=judge,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    if not config.sanity_check:
        trainer.save_model(config.output_dir)
        if config.push_to_hub:
            trainer.push_to_hub()
        trainer.generate_completions()
