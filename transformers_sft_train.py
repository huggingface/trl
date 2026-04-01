import argparse
from dataclasses import dataclass
from typing import Any

import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


@dataclass
class SFTBatch:
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    labels: list[list[int]]


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            cur_len = len(f["input_ids"])
            pad_len = max_len - cur_len

            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * cur_len + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def print_processed_examples(tokenized: DatasetDict, collator: CompletionOnlyCollator, split: str, batch_size: int) -> None:
    ds = tokenized[split]
    print(f"\n=== Processed {split} samples ===")
    for i in range(len(ds)):
        sample = ds[i]
        print(f"[{split} sample {i}] input_ids: {sample['input_ids']}")
        print(f"[{split} sample {i}] labels: {sample['labels']}")

    if len(ds) == 0:
        return

    take_n = min(batch_size, len(ds))
    batch_features = [ds[i] for i in range(take_n)]
    batch = collator(batch_features)
    print(f"\n=== Collated {split} batch (first {take_n}) ===")
    print(f"[{split} batch] input_ids: {batch['input_ids'].tolist()}")
    print(f"[{split} batch] labels: {batch['labels'].tolist()}")
    print(f"[{split} batch] attention_mask: {batch['attention_mask'].tolist()}")


def _message_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("content", ""))
    if isinstance(value, list):
        if len(value) == 0:
            return ""
        if isinstance(value[-1], dict) and "content" in value[-1]:
            return str(value[-1]["content"])
        return "\n".join(str(v) for v in value)
    return str(value)


def _to_messages(value: Any, default_role: str) -> list[dict[str, str]]:
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict) and "content" in item:
                out.append({"role": str(item.get("role", default_role)), "content": str(item["content"])})
            else:
                out.append({"role": default_role, "content": str(item)})
        return out

    if isinstance(value, dict):
        if "content" in value:
            return [{"role": str(value.get("role", default_role)), "content": str(value["content"])}]
        return [{"role": default_role, "content": str(value)}]

    return [{"role": default_role, "content": _message_to_text(value)}]


def extract_prompt_completion(example: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if "prompt" not in example or "completion" not in example:
        raise ValueError("Expected dataset fields: prompt and completion")

    prompt_messages = _to_messages(example["prompt"], default_role="user")
    completion_messages = _to_messages(example["completion"], default_role="assistant")
    return prompt_messages, completion_messages


def build_tokenized_dataset(dataset: DatasetDict, tokenizer, max_length: int, add_eos: bool) -> DatasetDict:
    eos_text = tokenizer.eos_token or ""

    def _process(example: dict[str, Any]) -> dict[str, Any]:
        prompt_messages, completion_messages = extract_prompt_completion(example)
        merged_messages = prompt_messages + completion_messages

        if add_eos and eos_text and len(merged_messages) > 0:
            last_content = merged_messages[-1].get("content", "")
            if not last_content.endswith(eos_text):
                merged_messages[-1] = {
                    "role": merged_messages[-1].get("role", "assistant"),
                    "content": last_content + eos_text,
                }

        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        input_ids = tokenizer.apply_chat_template(
            merged_messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        if all(x == -100 for x in labels):
            labels[-1] = input_ids[-1]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    train_ds = dataset["train"].map(
        _process,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing train dataset",
    )

    mapped = {"train": train_ds}
    if "test" in dataset:
        mapped["test"] = dataset["test"].map(
            _process,
            remove_columns=dataset["test"].column_names,
            desc="Tokenizing test dataset",
        )
    elif "validation" in dataset:
        mapped["validation"] = dataset["validation"].map(
            _process,
            remove_columns=dataset["validation"].column_names,
            desc="Tokenizing validation dataset",
        )

    return DatasetDict(mapped)


def main():
    parser = argparse.ArgumentParser(description="Transformers-only completion SFT")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--add_eos", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        extra_special_tokens={},
    )
    tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset(args.dataset_name, args.dataset_config)
    tokenized = build_tokenized_dataset(dataset, tokenizer, args.max_length, args.add_eos)

    eval_dataset = None
    if "test" in tokenized:
        eval_dataset = tokenized["test"]
    elif "validation" in tokenized:
        eval_dataset = tokenized["validation"]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
    )

    collator = CompletionOnlyCollator(tokenizer.pad_token_id)
    print_processed_examples(tokenized, collator, "train", args.per_device_train_batch_size)
    if eval_dataset is not None:
        if "test" in tokenized:
            print_processed_examples(tokenized, collator, "test", args.per_device_eval_batch_size)
        elif "validation" in tokenized:
            print_processed_examples(tokenized, collator, "validation", args.per_device_eval_batch_size)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
