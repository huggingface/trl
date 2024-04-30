# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict
from datasets import load_dataset
import pandas as pd
from transformers import (
    AutoTokenizer, TrainingArguments, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, GenerationConfig
from trl import SFTTrainer


def print_rich_table(df: pd.DataFrame) -> Table:
    console = Console()
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.print(table)


if __name__ == "__main__":
    ################
    # Model & Tokenizer
    ################
    base_model = "gpt2-large"
    model = AutoModelForCausalLM.from_pretrained(base_model)
    ref_model = AutoModelForCausalLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token_id = tokenizer.eos_token_id
    left_tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left") # for generation
    left_tokenizer.pad_token = left_tokenizer.eos_token
    if tokenizer.chat_template is None:
        # a default chat template to simply concatenate the messages
        tokenizer.chat_template = "{% for message in messages %}{{' ' + message['content']}}{% endfor %}{{ eos_token }}"

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

    ################
    # Training
    ################
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-05,
        logging_steps=10,
        evaluation_strategy="epoch",
        num_train_epochs=5,
        output_dir="minimal/sft",
        report_to=None,
    )
    # treats the EOS token and the padding token distinctively
    default_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    def data_collator(x):
        batch = default_collator(x)
        batch["input_ids"].masked_fill_(~batch["attention_mask"].bool(), 0)
        return batch
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="chosen",
        max_seq_length=1000,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)

    ################
    # Generate samples for visual inspection
    ################
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
        batch = eval_dataset[i:i+eval_batch_size]
        input_ids, attention_mask = left_tokenizer(batch["prompt"], return_tensors="pt", padding=True).values()
        input_ids, attention_mask = input_ids.to(model.device), attention_mask.to(model.device)
        for m, name in zip([model, ref_model], [f"trained {base_model}", f"initial {base_model}"]):
            prompt_and_generation = m.generate(input_ids, attention_mask=attention_mask, generation_config=generation_config, return_dict_in_generate=True)
            generation = prompt_and_generation[:, input_ids.shape[1]:]
            completions[name].extend(left_tokenizer.batch_decode(generation, skip_special_tokens=True))

    df = pd.DataFrame({**eval_dataset.to_dict(), **completions})
    del df["rejected"]
    print_rich_table(df.iloc[0:0+5])
    if "wandb" in training_args.report_to:
        import wandb
        wandb.log({"completions": wandb.Table(dataframe=df)})