# 0. imports
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import DPOTrainer


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # training parameters
    model_name_or_path: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    learning_rate: Optional[float] = field(default=1e-3, metadata={"help": "optimizer learning rate"})
    per_device_train_batch_size: Optional[int] = field(default=8, metadata={"help": "batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    # instrumentation
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def split_prompt_and_responses(ex):
    prompt = extract_anthropic_prompt(ex["chosen"])
    chosen = ex["chosen"][len(prompt) :]
    rejected = ex["rejected"][len(prompt) :]

    ex["prompt"] = prompt
    responses = [chosen, rejected]
    n_responses = len(data[prompt]["responses"])
    data[prompt]["pairs"].append((n_responses, n_responses + 1))
    data[prompt]["responses"].extend(responses)
    data[prompt]["sft_target"] = chosen
    return data[prompt]


# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples, tokenizer, max_length, label_pad_token_id):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "labels_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
        "labels_rejected": [],
    }

    for prompt, responses, pairs in zip(examples["prompt"], examples["responses"], examples["pairs"]):
        input_ids_query = tokenizer(prompt)["input_ids"]
        response_j, response_k = responses[pairs[0][0]], responses[pairs[0][1]]
        input_ids_response_chosen = tokenizer(response_j)["input_ids"]
        input_ids_response_rejected = tokenizer(response_k)["input_ids"]
        len_query = len(input_ids_query)

        input_ids_chosen = input_ids_query + input_ids_response_chosen
        attention_mask_chosen = [1] * len(input_ids_chosen)
        labels_chosen = [label_pad_token_id] * len_query + input_ids_chosen[len_query:]
        input_ids_rejected = input_ids_query + input_ids_response_rejected
        attention_mask_rejected = [1] * len(input_ids_rejected)
        labels_rejected = [label_pad_token_id] * len_query + input_ids_rejected[len_query:]

        # truncate to max_length
        new_examples["input_ids_chosen"].append(input_ids_chosen[-max_length:])
        new_examples["attention_mask_chosen"].append(attention_mask_chosen[-max_length:])
        new_examples["labels_chosen"].append(labels_chosen[-max_length:])
        new_examples["input_ids_rejected"].append(input_ids_rejected[-max_length:])
        new_examples["attention_mask_rejected"].append(attention_mask_rejected[-max_length:])
        new_examples["labels_rejected"].append(labels_rejected[-max_length:])

    return new_examples


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the human stack-exchange paired dataset
    train_dataset = load_dataset("Anthropic/hh-rlhf", split="train")

    if script_args.sanity_check:
        train_dataset = train_dataset.select(range(min(len(train_dataset), 1000)))

    data = defaultdict(lambda: defaultdict(list))
    train_dataset = train_dataset.map(split_prompt_and_responses)

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=os.cpu_count(),
        fn_kwargs=dict(
            tokenizer=tokenizer,
            max_length=script_args.max_length,
            label_pad_token_id=script_args.label_pad_token_id,
        ),
    )

    # 2. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        max_steps=1000,
        remove_unused_columns=False,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        output_dir="./test",
        report_to=script_args.report_to,
    )

    # 3. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # 4. train
    dpo_trainer.train()
