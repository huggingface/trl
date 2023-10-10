import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)


@dataclass
class ScoreArguments:
    reward_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "flan t5 or a finetuned teacher."}
    )
    scorer_bs: Optional[int] = field(default=8)
    num_grow_steps: Optional[int] = field(default=2)
    num_improve_steps: Optional[int] = field(default=4)
    save_dataset_path: Optional[str] = field(default=None)
    concat_init_dataset: Optional[bool] = field(default=True)


@dataclass
class GenArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="Dahoas/full-hh-rlhf", metadata={"help": "the HF data path"})

    gen_bs: Optional[int] = field(default=8, metadata={"help": "the generation batch size"})
    max_prompt_length: Optional[int] = field(default=256, metadata={"help": "The maximum prompt length"})

    max_new_tokens: Optional[int] = field(
        default=256, metadata={"help": "the maximum number of tokens generated per sample"}
    )
    temperature: Optional[float] = field(default=1.0)
    top_p: Optional[float] = field(default=1.0)
    top_k: Optional[float] = field(default=0)
    num_return_sequences: Optional[int] = field(default=8)

    sanity_check: Optional[bool] = field(default=False)


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    **generation_kwargs,
) -> Dataset:
    all_predictions = []
    all_prompts = []
    pbar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)

    for batch in dataloader:
        sequence_length = batch["input_ids"].shape[1]

        all_tokens = accelerator.unwrap_model(model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **generation_kwargs,
        )

        generated_tokens = all_tokens[:, sequence_length:]

        generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
        input_ids = torch.repeat_interleave(batch["input_ids"], generation_kwargs["num_return_sequences"], dim=0)

        prompt_tokens = accelerator.pad_across_processes(input_ids, dim=1, pad_index=tokenizer.pad_token_id)

        generated_tokens = accelerator.gather(generated_tokens)
        generated_tokens = generated_tokens.cpu()
        prompt_tokens = accelerator.gather(prompt_tokens)
        prompt_tokens = prompt_tokens.cpu()

        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
            prompt_tokens = prompt_tokens[0]

        all_predictions.extend(generated_tokens)
        all_prompts.extend(prompt_tokens)
        pbar.update(1)

    accelerator.wait_for_everyone()

    all_predictions = tokenizer.batch_decode(all_predictions, skip_special_tokens=True)

    def filter_text(text):
        return text.split("Human:")[0].strip()

    # postprocessing
    all_predictions = [filter_text(generated_text) for generated_text in all_predictions]
    all_prompts = tokenizer.batch_decode(all_prompts, skip_special_tokens=True)

    text = [prompt + " " + preds for prompt, preds in zip(all_prompts, all_predictions)]

    generated_dataset = Dataset.from_dict({"text": text, "prompt": all_prompts, "gen": all_predictions})

    return generated_dataset


@torch.no_grad()
def score(
    args: ScoreArguments,
    dataset: Dataset,
    max_length: int,
    accelerator: Accelerator,
) -> Tuple[Dataset, List[float]]:
    model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name_or_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Avoid truncating the model response
    tokenizer.padding_side = "left"

    reward_data_collator = DataCollatorWithPadding(tokenizer)

    def tokenize_fn(samples):
        return {**tokenizer(samples["text"], truncation=True, padding=False, max_length=max_length)}

    reward_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=list(dataset.features))
    dataloader = DataLoader(reward_dataset, shuffle=False, collate_fn=reward_data_collator)

    model, dataloader = accelerator.prepare(model, dataloader)
    rewards = []
    pbar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)

    for batch in dataloader:
        scores = model(**batch).logits.squeeze(1)
        scores = accelerator.gather(scores)
        rewards.extend(scores)
        pbar.update(1)

    rewards = rewards[: len(dataset)]
    rewards = [reward.item() for reward in rewards]

    dataset = dataset.add_column("rewards", rewards)

    return dataset, rewards


def main():
    parser = HfArgumentParser((ScoreArguments, GenArguments, TrainingArguments))
    score_args, gen_args, train_args = parser.parse_args_into_dataclasses()

    if score_args.num_improve_steps > 4:
        warnings.warn(
            "Overiding the number of improve steps from {score_args.improve_steps} to 4 as the filtering strategy used in this script do not work for more than 4 improve_steps"
        )

    # load policy and reward model
    tokenizer = AutoTokenizer.from_pretrained(gen_args.model_name_or_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # for generation
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(gen_args.model_name_or_path)

    # define gen_kwargs
    generation_kwargs = {
        "top_k": gen_args.top_k,
        "top_p": gen_args.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "temperature": gen_args.temperature,
        "max_new_tokens": gen_args.max_new_tokens,
        "num_return_sequences": gen_args.num_return_sequences,
    }

    # load and preprocess the dataset
    dataset = load_dataset(gen_args.dataset_name)

    init_train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    if gen_args.sanity_check:
        init_train_dataset = init_train_dataset.select(range(min(len(init_train_dataset), 100)))
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), 100)))

    def tokenize_fn(samples):
        model_inputs = tokenizer(samples["prompt"])

        return {
            **model_inputs,
        }

    train_dataset = init_train_dataset.map(tokenize_fn, batched=True, remove_columns=list(init_train_dataset.features))
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) <= gen_args.max_prompt_length)

    eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=list(eval_dataset.features))
    eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) <= gen_args.max_prompt_length)

    max_length = gen_args.max_prompt_length + gen_args.max_new_tokens
    data_collator = DataCollatorForSeq2Seq(tokenizer, max_length=max_length, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )

    accelerator = trainer.accelerator

    gen_dataloader = DataLoader(train_dataset, batch_size=gen_args.gen_bs, shuffle=True, collate_fn=data_collator)

    model, gen_dataloader = accelerator.prepare(model, gen_dataloader)
    max_length = gen_args.max_prompt_length + gen_args.max_new_tokens

    if score_args.concat_init_dataset:
        init_train_dataset = init_train_dataset.map(lambda x: {"text": x["prompt"] + " " + x["chosen"]})
        init_train_dataset = init_train_dataset.rename_column("chosen", "gen")

    for grow_step in range(score_args.num_grow_steps):
        accelerator.print(f"*** Grow step number: {grow_step} ***")

        accelerator.print("Starting Grow phase")
        generated_dataset = generate(model, gen_dataloader, tokenizer, accelerator, **generation_kwargs)

        if score_args.concat_init_dataset:
            generated_dataset = concatenate_datasets([init_train_dataset, generated_dataset])

        accelerator.print("Starting Improve phase")
        reward_dataset, rewards = score(score_args, generated_dataset, max_length, accelerator)
        print(reward_dataset.features)
        # save the dataset
        save_dataset_path = os.path.join(score_args.save_dataset_path, f"grow_{grow_step}")
        accelerator.print(f"Saving dataset to {save_dataset_path}")
        reward_dataset.save_to_disk(save_dataset_path)

        reward_stats = {
            "train/mean_reward": np.mean(rewards).item(),
            "train/max_reward": np.max(rewards).item(),
            "train/min_reward": np.min(rewards).item(),
        }

        accelerator.print(f"Reward Statistics: {reward_stats}")
        trainer.state.log_history = []
        trainer.state.global_step = grow_step
        trainer.log(reward_stats)

        def preprocess_fn(samples):
            prompt_ids = tokenizer(samples["prompt"], truncation=False, padding=False)["input_ids"]
            responses = tokenizer(samples["gen"], truncation=False, padding=False)["input_ids"]

            input_ids = [prompt + response for prompt, response in zip(prompt_ids, responses)]
            attention_mask = [[1 for _ in range(len(inp))] for inp in input_ids]
            labels = [[-100 for _ in range(len(prompt))] + response for prompt, response in zip(prompt_ids, responses)]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "rewards": samples["rewards"],
            }

        accelerator.print("Preprocessing")
        reward_dataset = reward_dataset.map(preprocess_fn, batched=True, remove_columns=list(reward_dataset.features))

        def filter_dataset(dataset: Dataset, step: int) -> Dataset:
            rewards = dataset["rewards"]

            # Filtering step. You should implement your own based on your dataset and needs.
            thresholds = [
                np.percentile(rewards, 75),
                np.percentile(rewards, 90),
                np.percentile(rewards, 95),
                np.percentile(rewards, 99),
            ]
            if step >= len(thresholds):
                step = len(thresholds) - 1

            dataset = dataset.filter(lambda example: example["rewards"] > thresholds[step])

            dataset = dataset.remove_columns(["rewards"])

            return dataset

        accelerator.print("Starting Improve steps")
        for improve_step in range(score_args.num_improve_steps):
            accelerator.print(f"Improve steps number {improve_step}")

            temp_dataset = filter_dataset(reward_dataset, improve_step)

            trainer.train_dataset = temp_dataset

            _ = trainer.train(resume_from_checkpoint=False)

            if train_args.output_dir is not None:
                save_model_path = os.path.join(train_args.output_dir, f"rest_grow_{grow_step}_improve_{improve_step}")
                accelerator.print(f"*** Saving the model to {save_model_path}***")
                trainer.save_model(save_model_path)


if __name__ == "__main__":
    main()
