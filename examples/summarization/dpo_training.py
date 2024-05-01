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
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional

import torch
from accelerate import Accelerator
from callbacks import GoldModelRewardCallback, PerplexityCallback, PerplexityGenCallback
from datasets import builder, concatenate_datasets, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftConfig, get_peft_model, prepare_model_for_kbit_training
from scalar_rm_model import ScalarModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    HfArgumentParser,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

import wandb
from trl import DPOTrainer


builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    dataset_name: Optional[str] = field(
        default="mnoukhov/openai_summarize_comparisons_tldrprompt_relabel1b", metadata={"help": "the dataset name"}
    )
    train_split: Optional[str] = field(default="train", metadata={"help": "the dataset split to train on"})
    eval_split: Optional[str] = field(
        default="test", metadata={"help": "the dataset split to evaluate on; default to 'none' (no evaluation)"}
    )
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    pseudo_dataset_name: Optional[str] = field(default=None, metadata={"help": "the dataset name"})
    pseudo_dataset_split: Optional[str] = field(default="train", metadata={"help": "the dataset name"})
    prompt_field: Optional[str] = field(default="prompt")

    # model parameters
    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    model_revision: Optional[str] = field(default=None, metadata={"help": "the model name"})
    ref_model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    ref_model_revision: Optional[str] = field(default=None, metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    bf16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    fp16_model: Optional[bool] = field(
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
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    # training parameters
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})
    warmup_steps: Optional[int] = field(default=150)
    learning_rate: Optional[float] = field(default=1e-3, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="linear")
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=8, metadata={"help": "batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_length: Optional[int] = field(default=560, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=48, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1)
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing"}
    )
    gradient_checkpointing_use_reentrant: bool = field(
        default=True, metadata={"help": "Whether to apply `use_reentrant` for gradient_checkpointing"}
    )

    # instrumentation
    seed: Optional[int] = field(default=0)
    output_dir: Optional[str] = field(default="results", metadata={"help": "the output directory"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the number of update steps between two logs"})
    log_n_samples_during_eval: Optional[int] = field(default=100)
    eval_steps: Optional[float] = field(default=None, metadata={"help": "the number of steps to eval at"})
    save_steps: Optional[float] = field(default=1000, metadata={"help": "the number of steps to save at"})
    save_strategy: Optional[str] = field(default="steps")
    report_to: Optional[str] = field(
        default="wandb",
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
    push_to_hub: Optional[bool] = field(default=False)
    push_to_hub_organization: Optional[str] = field(default=None)

    # gold model
    gold_eval: Literal["full", "gen", "ppl", "none"] = field(default="full")
    gold_model_name: str = field(default=None, metadata={"help": "the gold reward model name"})
    gold_model_revision: Optional[str] = field(default=None, metadata={"help": "the model name"})
    gold_in_8bit: Optional[bool] = field(default=False, metadata={"help": "gold the model in 8 bits precision"})
    gold_in_4bit: Optional[bool] = field(default=False, metadata={"help": "gold the model in 4 bits precision"})
    gold_bf16: Optional[bool] = field(
        default=False,
    )
    gold_fp16: Optional[bool] = field(
        default=False,
    )
    generate_greedy: Optional[bool] = field(default=True)
    gold_dataset_name: Optional[str] = field(
        default="CarperAI/openai_summarize_tldr", metadata={"help": "the dataset name"}
    )
    gold_eval_split: Optional[str] = field(default="valid")
    gold_prompt_field: Optional[str] = field(default="prompt")
    gold_target_field: Optional[str] = field(default="label")
    gold_load_and_unload: Optional[str] = field(default=False)
    mode: Literal["train", "eval", "predict", "relabel"] = field(default="train")
    eval_first_step: Optional[bool] = field(default=True)
    strip_prompt: Optional[bool] = field(default=True)


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
        dtype = torch.bfloat16
    elif args.fp16_model:
        dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer_name = args.tokenizer_name

    if "adapter" in args.model_name:
        model_cls = AutoPeftModelForCausalLM
        config = PeftConfig.from_pretrained(args.model_name)
        if tokenizer_name is None:
            tokenizer_name = config.base_model_name_or_path
    else:
        model_cls = AutoModelForCausalLM
        if tokenizer_name is None:
            tokenizer_name = args.model_name

    model = model_cls.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=dtype,
    )

    model.config.torch_dtype = dtype
    model.config.use_cache = not script_args.gradient_checkpointing
    # if script_args.ignore_bias_buffers:
    # torch distributed hack
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)

    # we add `score` to the list of modules to save to
    # correctly save the score head.
    # set target modules to be query_key_value for Pythia
    if args.use_peft and args.mode == "train":
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

        model = get_peft_model(model, peft_config)

        ref_model = None
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model_name,
            revision=args.ref_model_revision,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer_name.startswith("EleutherAI"):
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    elif getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, ref_model


def create_and_prepare_gold_model(args):
    if script_args.gold_in_8bit or script_args.gold_in_4bit:
        gold_quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.gold_in_8bit, load_in_4bit=script_args.gold_in_4bit
        )
        gold_device_map = {"": Accelerator().local_process_index}
    else:
        gold_device_map = None
        gold_quantization_config = None

    if script_args.gold_bf16:
        torch_dtype = torch.bfloat16
    elif script_args.gold_fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if script_args.gold_model_name.startswith("vwxyzjn"):
        gold_model_cls = ScalarModel
    else:
        gold_model_cls = AutoModelForSequenceClassification

    gold_model = gold_model_cls.from_pretrained(
        script_args.gold_model_name,
        revision=script_args.gold_model_revision,
        quantization_config=gold_quantization_config,
        torch_dtype=torch_dtype,
        device_map=gold_device_map,
    )

    # if getattr(gold_model.config, "pad_token_id", None) is None:
    #     gold_model.config.pad_token_id = gold_model.config.eos_token_id

    return gold_model


def strip_prompt(examples):
    examples["prompt"] = [prompt.strip() for prompt in examples["prompt"]]

    return examples


def create_and_prepare_dataset(args, tokenizer):
    train_dataset = load_dataset(args.dataset_name, split=args.train_split)
    eval_dataset = load_dataset(args.dataset_name, split=args.eval_split)

    if args.prompt_field != "prompt":
        train_dataset = train_dataset.rename_column(args.prompt_field, "prompt")
        eval_dataset = eval_dataset.rename_column(args.prompt_field, "prompt")

    if args.pseudo_dataset_name is not None:
        all_train_datasets = [train_dataset]
        pseudo_dataset_names = args.pseudo_dataset_name.split(",")
        for ds_name in pseudo_dataset_names:
            dataset = load_dataset(ds_name, split=args.pseudo_dataset_split)
            if args.strip_prompt:
                dataset = dataset.map(strip_prompt, batched=True)
            all_train_datasets.append(dataset)

        train_dataset = concatenate_datasets(all_train_datasets)

    if args.dataset_name.startswith("vwxyzjn"):
        # remove eos token from end of chosen
        def remove_eos(example):
            example["chosen"] = example["chosen"].removesuffix(tokenizer.eos_token)
            example["rejected"] = example["rejected"].removesuffix(tokenizer.eos_token)

            return example

        train_dataset = train_dataset.map(remove_eos)
        eval_dataset = eval_dataset.map(remove_eos)

    return train_dataset, eval_dataset


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. load a pretrained model
    model, tokenizer, ref_model = create_and_prepare_model(script_args)

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    train_dataset, eval_dataset = create_and_prepare_dataset(script_args, tokenizer)

    if script_args.push_to_hub:
        # configname_wandbid
        model_id = os.getenv("WANDB_NAME", "config_name") + "_" + os.getenv("WANDB_RUN_ID", "xxxxx")
        hub_model_id = f"{script_args.push_to_hub_organization}/{model_id}"
        print(f"pushing model to {hub_model_id}")
    else:
        hub_model_id = None

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        remove_unused_columns=False,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        lr_scheduler_type=script_args.lr_scheduler_type,
        evaluation_strategy="epoch" if script_args.eval_steps is None else "steps",
        save_strategy=script_args.save_strategy,
        logging_first_step=True,
        logging_steps=script_args.logging_steps,
        eval_steps=script_args.eval_steps,
        save_steps=script_args.save_steps,
        optim=script_args.optimizer_type,
        warmup_steps=script_args.warmup_steps,
        report_to=script_args.report_to,
        bf16=script_args.bf16,
        fp16=script_args.fp16,
        ddp_find_unused_parameters=False,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
    )

    if dpo_trainer.accelerator.is_local_main_process:
        wandb.init(reinit=True)
        wandb.config.update(asdict(script_args), allow_val_change=True)

    # Gold Eval
    if script_args.gold_eval != "none" and script_args.mode in ["train", "eval"]:
        gold_eval_dataset = load_dataset(
            script_args.gold_dataset_name,
            split=script_args.gold_eval_split,
        )

        if script_args.strip_prompt:
            gold_eval_dataset = gold_eval_dataset.map(strip_prompt, batched=True)

        if script_args.generate_greedy:
            generation_config = GenerationConfig(
                max_new_tokens=script_args.max_target_length,
                do_sample=False,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            generation_config = GenerationConfig(
                max_new_tokens=script_args.max_target_length,
                min_length=-1,
                top_k=0.0,
                top_p=1.0,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        if script_args.gold_eval == "full":
            gold_model = create_and_prepare_gold_model(script_args)

            callback = GoldModelRewardCallback(
                training_args,
                gold_model,
                gold_eval_dataset,
                tokenizer,
                dpo_trainer.accelerator,
                script_args.max_length,
                script_args.max_prompt_length,
                script_args.gold_prompt_field,
                script_args.gold_target_field,
                script_args.gold_load_and_unload,
                script_args.log_n_samples_during_eval,
                generation_config,
            )
        else:
            if script_args.gold_eval == "gen":
                callback_cls = PerplexityGenCallback
            elif script_args.gold_eval == "ppl":
                callback_cls = PerplexityCallback
            else:
                raise NotImplementedError

            callback = callback_cls(
                args=training_args,
                dataset=gold_eval_dataset,
                tokenizer=tokenizer,
                accelerator=dpo_trainer.accelerator,
                max_length=script_args.max_length,
                max_prompt_length=script_args.max_prompt_length,
                prompt_field=script_args.gold_prompt_field,
                target_field=script_args.gold_target_field,
                log_n_samples_during_eval=script_args.log_n_samples_during_eval,
                generation_config=generation_config,
                hub_model_id=hub_model_id,
            )

        dpo_trainer.add_callback(callback)

    if script_args.eval_first_step:

        class EvaluateFirstStepCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    control.should_evaluate = True

        dpo_trainer.add_callback(EvaluateFirstStepCallback())

    # 6. train
    if script_args.mode == "train":
        last_checkpoint = get_last_checkpoint(script_args.output_dir)
        dpo_trainer.train(resume_from_checkpoint=last_checkpoint)
    elif script_args.mode == "eval":
        print("evaluating")
        results = dpo_trainer.evaluate()
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

        dpo_trainer.accelerator.print(f"Prediction {script_args.eval_split}")
        preds, _, metrics = dpo_trainer.predict(eval_dataset)
        (
            chosen_rewards,
            rejected_rewards,
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        ) = preds
        dpo_trainer.accelerator.print(f"metrics {metrics}")

        if dpo_trainer.accelerator.is_local_main_process:
            print("Relabelling Dataset and Saving")
            dataset = load_dataset(script_args.dataset_name, split=script_args.eval_split)
            dataset = dataset.add_column("pred_chosen", chosen_rewards)
            dataset = dataset.add_column("pred_rejected", rejected_rewards)

            relabel_dataset = dataset.map(
                relabel_with_preds,
                batched=True,
            )

            description = f"{script_args.dataset_name} relabelled with {script_args.model_name}"
            relabel_dataset._info.description = description

        if dpo_trainer.accelerator.is_local_main_process:
            # print("Saving")
            # relabel_dataset.save_to_disk(script_args.output_dir)
            print("Pushing")
            # repo_id = f"MilaRLHF/{os.path.basename(script_args.output_dir)}"
            relabel_dataset.push_to_hub(os.path.basename(script_args.output_dir), split=script_args.eval_split)
            # relabel_dataset_card = DatasetCard.load(repo_id)
            # relabel_dataset_card.text = description
            # relabel_dataset_card.push_to_hub(repo_id)
    elif script_args.mode == "predict":
        dpo_trainer.accelerator.print(f"Prediction {script_args.eval_split}")
        preds, _, metrics = dpo_trainer.predict(eval_dataset)
        (
            chosen_rewards,
            rejected_rewards,
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        ) = preds
        dpo_trainer.accelerator.print(f"metrics {metrics}")

        if dpo_trainer.accelerator.is_local_main_process:
            print("Relabelling Dataset and Saving")
            dataset = load_dataset(script_args.dataset_name, split=script_args.eval_split)
            model_basename = script_args.model_name.rsplit("/", 1)[-1]
            dataset = dataset.add_column(f"pred_chosen_{model_basename}", chosen_rewards)
            dataset = dataset.add_column(f"pred_rejected_{model_basename}", rejected_rewards)

        if dpo_trainer.accelerator.is_local_main_process:
            # print("Saving")
            # relabel_dataset.save_to_disk(script_args.output_dir)
            print("Pushing")
            dataset.push_to_hub(os.path.basename(script_args.output_dir), split=script_args.eval_split)
