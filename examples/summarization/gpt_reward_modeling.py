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
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import bitsandbytes as bnb
import torch
from accelerate import Accelerator
from datasets import DatasetDict, builder, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TrainingArguments,
)

from trl import RewardTrainer


tqdm.pandas()
builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True
# torch.autograd.set_detect_anomaly(True)

### fix from https://github.com/huggingface/trl/issues/274


class GPTRewardTrainer(RewardTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
    ):
        rewards = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_chosen = rewards[jidx]
        rewards_rejected = rewards[kidx]
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss


@dataclass
class GPTRewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        # features_chosen = []
        # features_rejected = []
        merged_features = []
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            merged_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }
        return batch


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with RewardTrainer
    """

    model_name: Optional[str] = field(
        default="/home/toolkit/huggingface/tldr_sft_pythia7b", metadata={"help": "the model name"}
    )
    dataset_name: Optional[str] = field(
        default="mnoukhov/openai_summarize_comparisons_tldrprompt", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="prompt", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the number of update steps between two logs"})
    train_split: Optional[str] = field(
        default="train", metadata={"help": "the dataset split to evaluate on; default to 'none' (no evaluation)"}
    )
    eval_split: Optional[str] = field(
        default="test[:5000]", metadata={"help": "the dataset split to evaluate on; default to 'none' (no evaluation)"}
    )
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    weight_decay: Optional[float] = field(default=0.001)
    num_warmup_steps: Optional[int] = field(default=100)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})
    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    seq_length: Optional[int] = field(default=560, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    fp16_model: Optional[bool] = field(
        default=False,
        metadata={},
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_lora: Optional[bool] = field(
        default=True,
    )
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_all_linear: Optional[bool] = field(default=False, metadata={"help": "lora adapter on all linear layers"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="results", metadata={"help": "the output directory"})
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    mode: Optional[str] = field(default="train")
    eval_steps: Optional[float] = field(default=None)
    pretrained_adapter: Optional[str] = field(default=None)
    padding: Optional[str] = field(
        default="max_length", metadata={"help": "padding to use for preprocessing the dataset"}
    )
    save_strategy: Optional[str] = field(default="steps")


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.load_in_4bit else (bnb.nn.Linear8bitLt if args.load_in_8bit else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    if "score" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("score")

    return list(lora_module_names)


def create_and_prepare_model(args):
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
        device_map = {"": Accelerator().local_process_index}
    else:
        device_map = None
        quantization_config = None

    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16_model:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        num_labels=1,
        torch_dtype=torch_dtype,
    )

    model.config.torch_dtype = torch_dtype
    model.config.use_cache = not args.gradient_checkpointing

    # if script_args.ignore_bias_buffers:
    # torch distributed hack
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        args.gradient_checkpointing = False

    if args.use_lora:
        # we add `score` to the list of modules to save to
        # correctly save the score head.
        if args.pretrained_adapter is not None:
            model = PeftModel.from_pretrained(model, args.pretrained_adapter)
        else:
            if args.lora_all_linear:
                target_modules = find_all_linear_names(args, model)
            else:
                target_modules = None

            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="SEQ_CLS",
                target_modules=target_modules,
                modules_to_save=["score"],
            )

            model = get_peft_model(model, peft_config)

        modules_to_save = ["score"]
        for key, _ in model.named_modules():
            target_module_found = any(key.endswith(target_key) for target_key in modules_to_save)
            if target_module_found:
                model.get_submodule(key + ".original_module").requires_grad_(False)

        if torch_dtype == torch.bfloat16:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch_dtype)
                if "norm" in name:
                    module = module.to(torch.float32)
                if "score" in name or "embed_tokens" in name:
                    if hasattr(module, "weight") and module.weight.dtype == torch.float32:
                        module = module.to(torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def prepare_dataset(args, dataset, tokenizer, num_proc=2):
    # def summary_filter(example):
    #     return (example["chosen"] != example["rejected"]) and (
    #         len(example["chosen"].split()) >= 5 or len(example["rejected"].split()) >= 5
    #     )
    #
    # pre_filter = len(dataset)
    # dataset = dataset.filter(summary_filter)
    # print(f"filtered {pre_filter - len(dataset)} samples from {split}")
    original_columns = dataset.column_names

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(
                prompt + " " + chosen, padding=args.padding, truncation=True, max_length=script_args.seq_length
            )
            tokenized_rejected = tokenizer(
                prompt + " " + rejected, padding=args.padding, truncation=True, max_length=script_args.seq_length
            )
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    dataset = dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns)

    return dataset


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    model, tokenizer = create_and_prepare_model(script_args)
    if script_args.mode != "eval":
        train_data = load_dataset(script_args.dataset_name, split=script_args.train_split)
        train_dataset = prepare_dataset(script_args, train_data, tokenizer)
    else:
        train_dataset = None

    if script_args.eval_split is not None and script_args.eval_split != "None":
        eval_data = load_dataset(script_args.dataset_name, split=script_args.eval_split)
        eval_dataset = prepare_dataset(script_args, eval_data, tokenizer)
    else:
        eval_dataset = None

    # don't include gradient_checkpointing here, see trl#728
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        bf16=script_args.bf16,
        fp16=script_args.fp16,
        num_train_epochs=script_args.num_train_epochs,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        report_to=script_args.log_with,
        remove_unused_columns=False,
        lr_scheduler_type=script_args.lr_scheduler_type,
        weight_decay=script_args.weight_decay,
        optim=script_args.optimizer_type,
        warmup_steps=script_args.num_warmup_steps,
        logging_steps=script_args.logging_steps,
        evaluation_strategy=("steps" if script_args.eval_steps is not None else "epoch"),
        eval_steps=script_args.eval_steps,
        save_strategy="epoch",
        gradient_checkpointing=script_args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
    )

    data_collator = GPTRewardDataCollatorWithPadding(
        tokenizer, max_length=script_args.seq_length, pad_to_multiple_of=8
    )

    trainer = GPTRewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_length=script_args.seq_length,
        data_collator=data_collator,
    )

    if script_args.mode == "train":
        print("Training")
        trainer.train()
        trainer.evaluate()

        print("Saving last checkpoint of the model")
        trainer.save_model(script_args.output_dir)

        output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
        trainer.model.save_pretrained(output_dir)
    elif script_args.mode == "eval":
        print("Evaluating")
        # results = trainer.evaluate()
        results = trainer.evaluate()
        print(results)
    elif script_args.mode == "relabel":

        def relabel_with_preds(batch: Dict[str, List]):
            relabel_batch = {
                "prompt": [],
                "chosen": [],
                "rejected": [],
                "pred_chosen": [],
                "pred_rejected": [],
            }
            for prompt, chosen, rejected, pred_chosen, pred_rejected in zip(
                batch["prompt"],
                batch["chosen"],
                batch["rejected"],
                batch["pred_chosen"],
                batch["pred_rejected"],
            ):
                relabel_batch["prompt"].append(prompt)
                if pred_chosen >= pred_rejected:
                    relabel_batch["chosen"].append(chosen)
                    relabel_batch["rejected"].append(rejected)
                    relabel_batch["pred_chosen"].append(pred_chosen)
                    relabel_batch["pred_rejected"].append(pred_rejected)
                else:
                    relabel_batch["chosen"].append(rejected)
                    relabel_batch["rejected"].append(chosen)
                    relabel_batch["pred_chosen"].append(pred_rejected)
                    relabel_batch["pred_rejected"].append(pred_chosen)

            return relabel_batch

        relabel_dataset = DatasetDict()
        for split, pred_dataset in [("train", train_dataset), ("test", eval_dataset)]:
            if pred_dataset is None:
                continue
            trainer.accelerator.print(f"Prediction {split}")
            preds, _, metrics = trainer.predict(pred_dataset)
            trainer.accelerator.print(f"metrics {metrics}")

            if trainer.accelerator.is_local_main_process:
                print("Relabelling Dataset and Saving")
                ds_split = script_args.train_split if split == "train" else script_args.eval_split
                dataset = load_dataset(script_args.dataset_name, split=ds_split)
                dataset = dataset.add_column("pred_chosen", preds[:, 0])
                dataset = dataset.add_column("pred_rejected", preds[:, 1])

                dataset = dataset.map(relabel_with_preds, batched=True)

                dataset._info.description = f"{script_args.dataset_name} relabelled with {script_args.model_name}"
                relabel_dataset[split] = dataset

        if trainer.accelerator.is_local_main_process:
            print("Saving")
            relabel_dataset.save_to_disk(script_args.output_dir)
            print("Pushing")
            relabel_dataset.push_to_hub(os.path.basename(script_args.output_dir))
    elif script_args.mode == "predict":
        relabel_dataset = DatasetDict()
        for split, pred_dataset in [("train", train_dataset), ("test", eval_dataset)]:
            if pred_dataset is None:
                continue
            trainer.accelerator.print(f"Prediction {split}")
            preds, _, metrics = trainer.predict(pred_dataset)
            trainer.accelerator.print(f"metrics {metrics}")

            if trainer.accelerator.is_local_main_process:
                print("Relabelling Dataset and Saving")
                ds_split = script_args.train_split if split == "train" else script_args.eval_split
                dataset = load_dataset(script_args.dataset_name, split=ds_split)
                model_basename = script_args.model_name.rsplit("/", 1)[-1]
                dataset = dataset.add_column(f"pred_chosen_{model_basename}", preds[:, 0])
                dataset = dataset.add_column(f"pred_rejected_{model_basename}", preds[:, 1])

                dataset._info.description = f"{script_args.dataset_name} relabelled with {script_args.model_name}"
                relabel_dataset[split] = dataset

        if trainer.accelerator.is_local_main_process:
            print("Saving")
            relabel_dataset.save_to_disk(script_args.output_dir)
            print("Pushing")
            relabel_dataset.push_to_hub(os.path.basename(script_args.output_dir))
    else:
        raise Exception(f"incorrect mode {script_args.mode}")
        # TODO this freezes for some reason
        # for split, dataset in relabel_dataset.items():
        #     if trainer.accelerator.is_local_main_process:
        #         eval_dataset = prepare_dataset(script_args, dataset, tokenizer)
        #     trainer.accelerator.print(f"Re-evaluating relabel {split} dataset of size {len(dataset)}")
        #     trainer.accelerator.wait_for_everyone()
        #     results = trainer.evaluate(eval_dataset)
        #     trainer.accelerator.print(results)
