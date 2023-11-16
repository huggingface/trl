import shutil
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import nltk
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    default_data_collator,
)

import wandb


shutil.disk_usage = lambda x: shutil._ntuple_diskusage(1, 1, 1)


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="EleutherAI/pythia-6.9b-deduped", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(
        default="CarperAI/openai_summarize_tldr", metadata={"help": "the dataset name"}
    )
    split: Optional[str] = field(default="valid[:20]", metadata={"help": "the dataset name"})
    dataset_text_field: Optional[str] = field(default="prompt")
    dataset_label_field: Optional[str] = field(default="label")
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    seed: Optional[int] = field(default=0)
    batch_size: Optional[int] = field(default=1)
    bf16: Optional[bool] = field(default=False)
    fp16: Optional[bool] = field(default=False)
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    max_new_tokens: Optional[int] = field(default=48, metadata={"help": "Max new tokens to generate"})
    num_logged_samples: int = field(default=100, metadata={"help": "Max samples to log to wandb"})
    temperature: Optional[float] = field(default=0.0)
    sample: Optional[bool] = field(default=False)

    strip: Optional[bool] = field(default=False)
    log_with: Optional[str] = field(default="wandb")


parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

print("Loading the model")
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif args.load_in_8bit or args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
    device_map = {"": Accelerator().local_process_index}
else:
    device_map = None
    quantization_config = None

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16 if args.bf16 else None,
)

print("Loading dataset")
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name if args.tokenizer_name is None else args.tokenizer_name, padding_side="left"
)
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token


def create_dataset(tokenizer, args):
    eval_data = load_dataset(
        args.dataset_name,
        split=args.split,
    )

    padding = "max_length"
    max_source_length = args.seq_length
    max_target_length = 52

    def preprocess_function(example):
        inputs = example[args.dataset_text_field]
        targets = example[args.dataset_label_field]

        if args.strip:
            inputs = inputs.strip()
            targets = targets.strip()

        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    eval_dataset = eval_data.map(
        preprocess_function,
        remove_columns=eval_data.column_names,
    )

    return eval_dataset


eval_dataset = create_dataset(tokenizer, args)

rouge = evaluate.load("rouge")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


accelerator = Accelerator()
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)

model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

model.eval()

gen_kwargs = {
    "max_new_tokens": args.max_new_tokens,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "do_sample": args.sample,
}

wandb.init(project="trl")
table = wandb.Table(columns=["prompt", "prediction", "label"])

for batch in tqdm(eval_dataloader):
    with torch.no_grad():
        output_tokens = accelerator.unwrap_model(model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **gen_kwargs,
        )

    # get just the generated tokens
    generated_tokens = output_tokens[:, batch["input_ids"].shape[1] :]

    generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
    labels = batch["labels"]
    # if not args.pad_to_max_length:
    #     # If we did not pad to max length, we need to pad the labels too
    #     labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

    generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
    generated_tokens = generated_tokens.cpu().numpy()
    labels = labels.cpu().numpy()

    # if args.ignore_pad_token_for_loss:
    #     # Replace -100 in the labels as we can't decode them.
    #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    if isinstance(generated_tokens, tuple):
        generated_tokens = generated_tokens[0]
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # print(f"Label {decoded_labels}")
    # print(f"Pred {decoded_preds}")

    rouge.add_batch(
        predictions=decoded_preds,
        references=decoded_labels,
    )

    # log samples
    if accelerator.is_main_process:
        if len(table.data) < args.num_logged_samples:
            num_samples_to_add = min(args.num_logged_samples - len(table.data), len(batch["input_ids"]))
            for i in range(num_samples_to_add):
                table.add_data(
                    tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True),
                    decoded_preds[i],
                    decoded_labels[i],
                )

            if len(table.data) == args.num_logged_samples:
                if args.log_with == "wandb":
                    wandb.log({"examples": table})
                else:
                    for row in table.iterrows():
                        print("PROMPT")
                        print(row[1][0])
                        print("\n")
                        print("PRED")
                        print(row[1][1])
                        print("\n")
                        print("LABEL")
                        print(row[1][2])
                        print("\n")
                        print("\n")

result = rouge.compute()
print(result)

for key, value in result.items():
    wandb.run.summary[key] = value
