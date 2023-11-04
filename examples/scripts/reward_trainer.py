# coding=utf-8
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
from dataclasses import dataclass, field
from typing import Optional

from accelerate import Accelerator
from datasets import load_dataset
import mlflow
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
import evaluate

from trl import RewardConfig, RewardTrainer
from trl.trainer.moreh_utils import TBTrainerCallback, get_num_parameters


tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    Hyperparameters to fine-tune a reward model on a given dataset with the `RewardTrainer`.
    """

    model_name_or_path: Optional[str] = field(default="camembert-base", metadata={"help": "the model name"})
    dataset_name_or_path: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the dataset name"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    eval_split: Optional[str] = field(
        default="test", metadata={"help": "the dataset split to evaluate on; default to 'none' (no evaluation)"}
    )
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})


parser = HfArgumentParser((ScriptArguments, RewardConfig))
script_args, training_args = parser.parse_args_into_dataclasses()

# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    )
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}
else:
    device_map = None
    quantization_config = None

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    num_labels=1,
)

num_params = get_num_parameters(model)
mlflow.log_param('num_params', num_params)


# Step 2: Load the dataset and pre-process it
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
train_dataset = load_dataset(script_args.dataset_name_or_path, split="train")


# Tokenize chosen/rejected pairs of inputs
# Adapt this section to your needs for custom datasets
def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen, truncation=True)
        tokenized_rejected = tokenizer(rejected, truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


# Preprocess the dataset and filter out examples that are longer than script_args.max_length
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= script_args.seq_length
    and len(x["input_ids_rejected"]) <= script_args.seq_length
)

if script_args.eval_split == "none":
    eval_dataset = None
else:
    eval_dataset = load_dataset(script_args.dataset_name_or_path, split=script_args.eval_split)

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= script_args.seq_length
        and len(x["input_ids_rejected"]) <= script_args.seq_length
    )

# Step 4: Define the LoraConfig
if script_args.use_peft:
    peft_config = LoraConfig(r=16, lora_alpha=16, bias="none", task_type="SEQ_CLS", modules_to_save=["scores"])
else:
    peft_config = None

# Step 5: Define the Trainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)

trainer.add_callback(TBTrainerCallback())

train_result = trainer.train()

trainer.save_model()
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)

mlflow.end_run()
