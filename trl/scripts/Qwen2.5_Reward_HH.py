import argparse
import os
import torch
from accelerate import logging
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_peft_config,
)


logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


import re
from datasets import load_dataset, DatasetDict

def convert_hh_to_conversational(example):
    """
    将 HH-RLHF 的原始文本格式转换为标准的对话 List[Dict] 格式
    """
    def parse_text(text):
        # 匹配 "Human: " 或 "\n\nHuman: " 以及 "Assistant: " 或 "\n\nAssistant: "
        # 这是一个简单的分割逻辑，适用于标准 HH 格式
        parts = re.split(r'\n\n(Human|Assistant): ', text)
        if parts[0] == '':
            parts = parts[1:]
        
        # 处理第一行可能没有 \n\n 的情况
        if not parts[0] in ["Human", "Assistant"]:
             parts = re.split(r'(Human|Assistant): ', text)
             if parts[0] == '': parts = parts[1:]

        messages = []
        for i in range(0, len(parts), 2):
            role = "user" if parts[i] == "Human" else "assistant"
            content = parts[i+1].strip()
            messages.append({"role": role, "content": content})
        return messages

    # 转换 chosen 和 rejected
    chosen_msgs = parse_text(example["chosen"])
    rejected_msgs = parse_text(example["rejected"])
    
    return {
        "chosen": chosen_msgs,
        "rejected": rejected_msgs
    }







def main(script_args, training_args, model_args, dataset_args):
    # Load the dataset
    if dataset_args.datasets and script_args.dataset_name:
        logger.warning(
            "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")


    # 应用转换逻辑，num_proc=4 利用多核加速 CPU 处理
    dataset = dataset.map(
        convert_hh_to_conversational, 
        num_proc=8, 
        remove_columns=dataset["train"].column_names, # 必须删除旧的字符串列
        desc="Formatting HH dataset"
    )

    # 只保留目标列
    dataset = dataset.select_columns(["chosen", "rejected"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
    )
    # Initialize the RewardTrainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        processing_class = tokenizer
    )

    
    # Train the model
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ Training completed.")

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: argparse._SubParsersAction | None = None):
    dataclass_types = (ScriptArguments, RewardConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "reward", help="Run the reward training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
