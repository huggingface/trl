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
# import random
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import bitsandbytes as bnb
import torch
from accelerate import Accelerator
from datasets import concatenate_datasets, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

import wandb
from trl import DPOTrainer
from trl.trainer.utils import pad_to_length


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

    # model parameters
    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
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
    lora_all_linear: Optional[bool] = field(default=False, metadata={"help": "lora adapter on all linear layers"})

    # training parameters
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})
    warmup_steps: Optional[int] = field(default=150)
    learning_rate: Optional[float] = field(default=1e-3, metadata={"help": "optimizer learning rate"})
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

    # instrumentation
    output_dir: Optional[str] = field(default="results", metadata={"help": "the output directory"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the number of update steps between two logs"})
    log_n_samples_during_eval: Optional[int] = field(default=100)
    eval_steps: Optional[int] = field(default=None, metadata={"help": "the number of steps to eval at"})
    save_steps: Optional[int] = field(default=1000, metadata={"help": "the number of steps to save at"})
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

    # gold model
    gold_model_name: str = field(default=None, metadata={"help": "the gold reward model name"})
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
    mode: Optional[str] = field(default="train")
    eval_first_step: Optional[bool] = field(default=True)


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
        dtype = torch.bfloat16
    elif args.fp16_model:
        dtype = torch.float16
    else:
        dtype = torch.float32

    if "adapter" in args.model_name:
        model_cls = AutoPeftModelForCausalLM
        config = PeftConfig.from_pretrained(args.model_name)
        tokenizer_name = config.base_model_name_or_path
    else:
        model_cls = AutoModelForCausalLM
        tokenizer_name = args.model_name

    model = model_cls.from_pretrained(
        args.model_name,
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
    if args.lora_all_linear:
        modules = find_all_linear_names(args, model)
    else:
        modules = None

    if args.use_peft:
        modules_to_save = ["lm_head"]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules,
            modules_to_save=modules_to_save,
        )

        model = get_peft_model(model, peft_config)

        for key, _ in model.named_modules():
            target_module_found = any(key.endswith(target_key) for target_key in modules_to_save)
            if target_module_found:
                model.get_submodule(key + ".original_module").requires_grad_(False)

    if args.bf16:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "score" in name or "embed_tokens" in name:
                if hasattr(module, "weight") and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    # tokenizer_name = script_args.model_name if script_args.tokenizer_name is None else script_args.tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # tokenizer.truncation_side = "left"
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


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

    gold_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.gold_model_name,
        quantization_config=gold_quantization_config,
        torch_dtype=torch_dtype,
        device_map=gold_device_map,
    )

    if getattr(gold_model.config, "pad_token_id", None) is None:
        gold_model.config.pad_token_id = gold_model.config.eos_token_id

    return gold_model


def strip_prompt(examples):
    examples["prompt"] = [prompt.strip() for prompt in examples["prompt"]]

    return examples


def create_and_prepare_dataset(args):
    train_dataset = load_dataset(args.dataset_name, split=args.train_split)
    eval_dataset = load_dataset(args.dataset_name, split=args.eval_split)

    if args.pseudo_dataset_name is not None:
        all_train_datasets = [train_dataset]
        pseudo_dataset_names = args.pseudo_dataset_name.split(",")
        for ds_name in pseudo_dataset_names:
            dataset = load_dataset(ds_name, split=args.pseudo_dataset_split)
            dataset = dataset.map(strip_prompt, batched=True)
            all_train_datasets.append(dataset)

        train_dataset = concatenate_datasets(all_train_datasets)

    return train_dataset, eval_dataset


@dataclass
class PromptCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_prompt_length: Optional[int] = None
    prompt_field: str = "prompt"
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [feat[self.prompt_field] for feat in features]

        original_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        tokenized_batch = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=self.max_prompt_length,
            return_tensors=self.return_tensors,
        )
        tokenized_batch["prompt"] = prompts

        self.tokenizer.padding_side = original_side

        return tokenized_batch


class GoldModelRewardCallback(TrainerCallback):
    def __init__(
        self,
        args,
        gold_model,
        gold_eval_dataset,
        tokenizer,
        accelerator,
        max_length,
        max_prompt_length,
        log_n_samples_during_eval=0,
        generation_config=None,
    ):
        self.max_length = max_length
        self.log_n_samples_during_eval = log_n_samples_during_eval
        self.generation_config = generation_config

        # data_collator = DataCollatorWithPadding(tokenizer)
        data_collator = PromptCollator(
            tokenizer,
            max_prompt_length=max_prompt_length,
            prompt_field="prompt",
        )
        dataloader_params = {
            "batch_size": args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": args.dataloader_num_workers,
            "pin_memory": args.dataloader_pin_memory,
        }
        dataloader = DataLoader(gold_eval_dataset, **dataloader_params)
        self.gold_model, self.dataloader = accelerator.prepare(gold_model, dataloader)
        self.accelerator = accelerator

    def on_evaluate(self, args, state, control, model, tokenizer, metrics, **kwargs):
        samples_to_log = []
        gold_reward_sum = 0.0
        total_samples = 0

        for inputs in tqdm(
            self.dataloader, desc="Gold Eval", dynamic_ncols=True, disable=not state.is_local_process_zero
        ):
            policy_output_decoded, ref_output_decoded, policy_output_ids = self.get_batch_samples(
                model,
                tokenizer,
                inputs["input_ids"],
                inputs["attention_mask"],
                return_ids=True,
            )

            # gold reward
            policy_output_attention_mask = (policy_output_ids != tokenizer.pad_token_id).to(torch.int64)
            with torch.no_grad():
                gold_rewards = self.gold_model(
                    input_ids=policy_output_ids, attention_mask=policy_output_attention_mask
                )[0]

            gold_rewards = self.accelerator.gather_for_metrics(gold_rewards)

            if state.is_local_process_zero:
                gold_reward_sum += gold_rewards.sum().item()
                total_samples += gold_rewards.size(0)

                # Sample and save to game log if requested (for one batch to save time)
                for i, (prompt, pol, ref) in enumerate(
                    zip(inputs["prompt"], policy_output_decoded, ref_output_decoded)
                ):
                    if len(samples_to_log) < self.log_n_samples_during_eval:
                        samples_to_log.append([prompt, pol[len(prompt) :], ref[len(prompt) :]])
                    else:
                        break

        if state.is_world_process_zero:
            print(f"gold reward mean {gold_reward_sum / total_samples}")
            gold_log = {
                "eval/gold_rewards_mean": gold_reward_sum / total_samples,
            }
            if state.epoch:
                gold_log["epoch"] = round(state.epoch, 2)
                gold_log["step"] = state.global_step
            if samples_to_log:
                gold_log["game_log"] = (
                    wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=samples_to_log,
                    ),
                )
            wandb.log(gold_log)

    def get_batch_samples(self, model, tokenizer, input_ids, attention_mask, return_ids=False) -> Tuple[str, str]:
        """Reduce inputs to unseen prompts, and maximum batch size if necessary
        Generate samples from the model and reference model for the given batch of inputs."""
        policy_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
        )

        # if self.ref_model is None:
        with self.accelerator.unwrap_model(model).disable_adapter():
            reference_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
            )
        # else:
        #     reference_output = self.ref_model.generate(
        #         **inputs,
        #         generation_config=self.generation_config,
        #     )

        policy_output = pad_to_length(policy_output, self.max_length, tokenizer.pad_token_id)
        policy_output_decoded = tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, tokenizer.pad_token_id)
        reference_output_decoded = tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        if return_ids:
            return policy_output_decoded, reference_output_decoded, policy_output
        else:
            return policy_output_decoded, reference_output_decoded


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. load a pretrained model
    model, tokenizer = create_and_prepare_model(script_args)

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    train_dataset, eval_dataset = create_and_prepare_dataset(script_args)

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
        ddp_find_unused_parameters=(script_args.gradient_checkpointing),
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
    )

    # Gold Eval
    if script_args.gold_model_name is not None:
        gold_model = create_and_prepare_gold_model(script_args)
        gold_eval_dataset = load_dataset(
            script_args.gold_dataset_name,
            split=script_args.gold_eval_split,
        )

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

        gold_eval_callback = GoldModelRewardCallback(
            training_args,
            gold_model,
            gold_eval_dataset,
            tokenizer,
            dpo_trainer.accelerator,
            script_args.max_length,
            script_args.max_prompt_length,
            script_args.log_n_samples_during_eval,
            generation_config,
        )

        dpo_trainer.add_callback(gold_eval_callback)

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
        results = dpo_trainer.evaluate()
        print(results)
    elif script_args.mode == "relabel":

        def relabel_with_preds(batch: Dict[str, List]):
            relabel_batch = {
                "prompt": [],
                "chosen": [],
                "rejected": [],
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
                else:
                    relabel_batch["chosen"].append(rejected)
                    relabel_batch["rejected"].append(chosen)

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
                relabel_with_preds, batched=True, remove_columns=["pred_chosen", "pred_rejected"]
            )

            relabel_dataset._info.description = f"{script_args.dataset_name} relabelled with {script_args.model_name}"

        if dpo_trainer.accelerator.is_local_main_process:
            print("Saving")
            relabel_dataset.save_to_disk(script_args.output_dir)
            print("Pushing")
            relabel_dataset.push_to_hub(os.path.basename(script_args.output_dir), split=script_args.eval_split)
