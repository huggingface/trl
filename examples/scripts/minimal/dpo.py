from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from trl import DPOTrainer


def print_rich_table(df: pd.DataFrame) -> Table:
    console = Console()
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.print(table)


training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-05,
    logging_steps=10,
    evaluation_strategy="epoch",
    num_train_epochs=1,
    output_dir="dpo_descriptiveness",
    report_to="wandb",
)

################
# Model & Tokenizer
################
model_name = "gpt2"
dataset_tokenizer = AutoTokenizer.from_pretrained("gpt2")  # of the dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)
left_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")  # for generation
model = AutoModelForCausalLM.from_pretrained(model_name)
model_ref = AutoModelForCausalLM.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    left_tokenizer.pad_token = left_tokenizer.eos_token

################
# Dataset
################
file = hf_hub_download(
    repo_id="vwxyzjn/lm-human-preferences",
    repo_type="dataset",
    filename="descriptiveness/offline_5k.json",  # or "sentiment/offline_5k.json"
)
ds = Dataset.from_json(file)


# columns are `['sample2', 'sample3', 'sample0', 'query', 'sample1', 'best']`
def modify(row):
    for j in range(4):
        row[f"sample{j}"] = dataset_tokenizer.batch_decode(row[f"sample{j}"])
    row["prompt"] = dataset_tokenizer.batch_decode(row["query"])
    chosen = []
    rejected = []
    for i in range(len(row["best"])):
        best_idx = row["best"][i]
        chosen.append(row[f"sample{best_idx}"][i])
        rejected_ids = [k for k in [0, 1, 2, 3] if k != best_idx]
        rejected_idx = np.argmin(rejected_ids)  # select the first rejected sample for reproducibility
        rejected.append(row[f"sample{rejected_idx}"][i])
    row["chosen"] = chosen
    row["rejected"] = rejected
    return row


ds = ds.map(modify, batched=True, load_from_cache_file=False)
ds = ds.remove_columns(["sample0", "sample1", "sample2", "sample3", "best", "query"])
ds = ds.shuffle(seed=2)
df = ds.to_pandas()
eval_dataset = ds.select(range(0, 20))
train_dataset = ds.select(range(20, len(ds)))
print_rich_table(eval_dataset.to_pandas().iloc[0 : 0 + 2])
################
# Training
################
trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model(training_args.output_dir)
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
print(metrics)

################
# Generate samples for visual inspection
################
eval_batch_size = 4
completions = defaultdict(list)
for i in range(0, len(eval_dataset), eval_batch_size):
    batch = eval_dataset[i : i + eval_batch_size]
    input_ids, attention_mask = left_tokenizer(batch["prompt"], return_tensors="pt", padding=True).values()
    input_ids, attention_mask = input_ids.to(model.device), attention_mask.to(model.device)
    for m, name in zip([model, model_ref], [f"trained {model_name}", f"initial {model_name}"]):
        prompt_and_generation = m.generate(
            input_ids, attention_mask=attention_mask, max_new_tokens=None, max_length=100
        )
        generation = prompt_and_generation[:, input_ids.shape[1] :]
        completions[name].extend(left_tokenizer.batch_decode(generation, skip_special_tokens=True))

df = pd.DataFrame({**eval_dataset.to_dict(), **completions})
del df["rejected"]
print_rich_table(df.iloc[0 : 0 + 5])
if "wandb" in training_args.report_to:
    import wandb

    wandb.log({"completions": wandb.Table(dataframe=df)})
