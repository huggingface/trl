import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from numpy import array
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollator,
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
    num_raft_iterations: Optional[int] = field(default=10)
    log_reward_stats: Optional[int] = field(default=20)


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
    batch: Dict,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    **generation_kwargs,
) -> Dataset:
    input_len = batch["input_ids"].shape[1]

    generated_tokens = accelerator.unwrap_model(model).generate(**batch, **generation_kwargs)
    generated_texts = tokenizer.batch_decode(generated_tokens[:, input_len:], skip_special_tokens=True)

    def filter_text(text):
        return text.split("Human:")[0].strip()

    generated_texts = [filter_text(generated_text) for generated_text in generated_texts]
    input_ids = torch.repeat_interleave(batch["input_ids"], generation_kwargs["num_return_sequences"], dim=0)
    input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    texts = [q + " " + r for q, r in zip(input_texts, generated_texts)]

    reward_dataset = Dataset.from_dict({"text": texts})

    return reward_dataset


@torch.no_grad()
def score_and_filter(
    dataset: Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    data_collator: DataCollator,
    max_length: int,
    accelerator: Accelerator,
    num_return_sequences: int,
    prompt_ids: torch.Tensor,
) -> Tuple[Dataset, array]:
    def tokenize_fn(samples):
        return {**tokenizer(samples["text"], truncation=True, padding=False, max_length=max_length)}

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=list(dataset.features))
    dataloader = DataLoader(dataset, shuffle=False, collate_fn=data_collator)

    rewards = []
    all_input_ids = []
    for batch in dataloader:
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        scores = model(**batch).logits.squeeze(1)

        input_ids = accelerator.pad_across_processes(batch["input_ids"], dim=1, pad_index=tokenizer.pad_token_id)
        input_ids = accelerator.gather(input_ids)
        scores = accelerator.gather(scores)
        rewards.extend(scores)
        all_input_ids.extend(input_ids)

    prompt_ids = accelerator.pad_across_processes(prompt_ids, dim=1, pad_index=tokenizer.pad_token_id)
    prompt_ids = accelerator.gather(prompt_ids)
    rewards = torch.stack(rewards).view(-1)
    rewards = torch.Tensor(rewards).reshape(-1, num_return_sequences)

    chosen_reward_inds = torch.argmax(rewards, dim=1) + torch.arange(
        0, len(all_input_ids), num_return_sequences, dtype=torch.int32, device=accelerator.device
    )

    top_k = [all_input_ids[i] for i in chosen_reward_inds]
    top_k = tokenizer.batch_decode(top_k, skip_special_tokens=True)
    prompts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)

    top_k_generated_texts = [gen.replace(prompt, " ") for prompt, gen in zip(prompts, top_k)]

    return (
        Dataset.from_dict({"prompt": prompts, "gen": top_k_generated_texts}),
        rewards.view(-1).cpu().detach().tolist(),
    )


def main():
    parser = HfArgumentParser((ScoreArguments, GenArguments, TrainingArguments))
    score_args, gen_args, train_args = parser.parse_args_into_dataclasses()

    # load policy and reward model
    tokenizer = AutoTokenizer.from_pretrained(gen_args.model_name_or_path)
    reward_tokenizer = AutoTokenizer.from_pretrained(score_args.reward_model_name_or_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if reward_tokenizer.pad_token_id is None:
        reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id
    # Avoid truncating the model response
    reward_tokenizer.padding_side = "left"
    # for generation
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(gen_args.model_name_or_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(score_args.reward_model_name_or_path)

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

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    if gen_args.sanity_check:
        train_dataset = train_dataset.select(range(min(len(train_dataset), 500)))
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), 500)))

    def tokenize_fn(samples):
        model_inputs = tokenizer(samples["prompt"])

        return {
            **model_inputs,
        }

    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=list(train_dataset.features))
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) <= gen_args.max_prompt_length)

    eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=list(eval_dataset.features))
    eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) <= gen_args.max_prompt_length)

    max_length = gen_args.max_prompt_length + gen_args.max_new_tokens
    data_collator = DataCollatorForSeq2Seq(tokenizer, max_length=max_length, pad_to_multiple_of=8)

    reward_data_collator = DataCollatorWithPadding(tokenizer)

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

    model, reward_model, gen_dataloader = accelerator.prepare(model, reward_model, gen_dataloader)

    steps = 0
    all_rewards = []

    progress_bar = tqdm(
        total=len(gen_dataloader) * score_args.num_raft_iterations, disable=not accelerator.is_local_main_process
    )
    max_length = gen_args.max_prompt_length + gen_args.max_new_tokens

    for iteration in range(score_args.num_raft_iterations):
        for batch in gen_dataloader:
            reward_dataset = generate(model, batch, tokenizer, accelerator, **generation_kwargs)

            accelerator.wait_for_everyone()

            prompt_ids = batch["input_ids"]

            reward_dataset, rewards = score_and_filter(
                reward_dataset,
                reward_model,
                reward_tokenizer,
                reward_data_collator,
                max_length,
                accelerator,
                gen_args.num_return_sequences,
                prompt_ids,
            )

            # compute reward statistics & log them

            all_rewards.extend(rewards)

            if (steps + 1) % score_args.log_reward_stats == 0 or (steps - 1) == len(gen_dataloader):
                reward_stats = {
                    "train/mean_reward": np.mean(all_rewards).item(),
                    "train/max_reward": np.max(all_rewards).item(),
                    "train/min_reward": np.min(all_rewards).item(),
                }
                accelerator.print(f"Reward Statistics: {reward_stats}")

                trainer.state.log_history = []
                trainer.state.global_step = steps
                trainer.log(reward_stats)
                all_rewards = []

            # save the dataset
            def preprocess_fn(samples):
                prompt_ids = tokenizer(samples["prompt"], truncation=False, padding=False)["input_ids"]
                responses = tokenizer(samples["gen"], truncation=False, padding=False)["input_ids"]

                input_ids = [prompt + response for prompt, response in zip(prompt_ids, responses)]
                attention_mask = [[1 for _ in range(len(inp))] for inp in input_ids]
                labels = [
                    [-100 for _ in range(len(prompt))] + response for prompt, response in zip(prompt_ids, responses)
                ]

                return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

            reward_dataset = reward_dataset.map(
                preprocess_fn, batched=True, remove_columns=list(reward_dataset.features)
            )

            trainer.train_dataset = reward_dataset

            trainer.train(resume_from_checkpoint=False)

            progress_bar.update(1)
            steps += 1

        if train_args.output_dir is not None:
            accelerator.print("*** Saving the model ***")
            save_model_path = os.path.join(train_args.output_dir, f"raft_iter_{iteration}")
            trainer.save_model(save_model_path)


if __name__ == "__main__":
    main()
