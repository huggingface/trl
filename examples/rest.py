# [WIP]
import os
from dataclasses import dataclass, field
from typing import Optional

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
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
)


@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "the model name"})
    reward_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the HF data path"})
    save_grow_dataset: Optional[str] = field(default=None, metadata={"help": "the HF data path"})
    train_bs: Optional[int] = field(default=8)
    gen_bs: Optional[int] = field(default=4)
    eval_bs: Optional[int] = field(default=16)
    reward_bs: Optional[int] = field(default=16)
    grow_steps: Optional[int] = field(default=1)
    improve_steps: Optional[int] = field(default=1)
    max_length: Optional[int] = field(default=512)
    max_new_tokens: Optional[int] = field(default=256)
    temperature: Optional[float] = field(default=0.7)
    top_p: Optional[float] = field(default=0.9)
    num_return_sequences: Optional[int] = field(default=8)
    is_test: Optional[bool] = field(default=False)


def main():

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Create the accelerator
    accelerator = Accelerator()

    gen_kwargs = {
        "max_new_tokens": script_args.max_new_tokens,
        "temperature": script_args.temperature,
        "num_return_sequences": script_args.num_return_sequences,
        "do_sample": True,
        "top_p": script_args.top_p,
    }

    # Load model, reward model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(script_args.reward_model_name_or_path)
    reward_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load and extract the prompt in the dataset. We do not need this step if we already have a dataset with prompts separated from the answers.
    train_dataset = load_dataset(script_args.dataset_name, split="train")
    eval_dataset = load_dataset(script_args.dataset_name, split="test")

    if script_args.is_test:
        train_dataset = train_dataset.select([i for i in range(int(len(train_dataset) * 0.1))])
        eval_dataset = eval_dataset.select([i for i in range(int(len(eval_dataset) * 0.1))])

    def preprocess_function(sample):
        def extract_prompt(chosen, rejected):
            for i, (c, r) in enumerate(zip(chosen, rejected)):
                if c != r:
                    return chosen[:i].strip()
            return chosen

        prompts = []
        for chosen, rejected in zip(sample["chosen"], sample["rejected"]):
            prompts.append(extract_prompt(chosen, rejected))
        model_inputs = tokenizer(prompts, max_length=512, truncation=True)

        return model_inputs

    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=list(train_dataset.features))
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=list(eval_dataset.features))

    # set padding side to left for generation
    tokenizer.padding_side = "left"
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    reward_collator = DataCollatorWithPadding(tokenizer, padding=True, max_length=script_args.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=script_args.gen_bs, shuffle=False, collate_fn=collator)
    eval_dataloader = DataLoader(eval_dataset, batch_size=script_args.eval_bs, shuffle=False, collate_fn=collator)

    (
        model,
        train_dataloader,
        eval_dataloader,
    ) = accelerator.prepare(model, train_dataloader, eval_dataloader)

    accelerator.wait_for_everyone()

    num_gen_steps = len(train_dataloader)
    pbar = tqdm(total=num_gen_steps)

    # Start Grow step. Looks like only one grow step is enough.
    for _ in range(script_args.grow_steps):
        # generate
        predictions = []
        for _, batch in tqdm(enumerate(train_dataloader)):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )

                generated_tokens = accelerator.gather_for_metrics(generated_tokens)
                generated_tokens = generated_tokens.cpu().numpy()

                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                pbar.update(1)

        encoded_predictions = tokenizer(predictions, truncation=True, max_length=script_args.max_length)["input_ids"]

        # Create the new dataset Dg
        generated_dataset = Dataset.from_dict(
            {
                "input_ids": encoded_predictions,
                "attention_mask": [[1 for _ in range(len(preds))] for preds in encoded_predictions],
            }
        )

        d_g = concatenate_datasets([train_dataset, generated_dataset])

        # Reward scoring
        # set padding side to right for training the model and make inference with the reward model
        tokenizer.padding_side = "right"
        reward_dataloader = DataLoader(
            d_g, shuffle=False, batch_size=script_args.reward_bs, collate_fn=reward_collator
        )
        reward_model, reward_dataloader = accelerator.prepare(reward_model, reward_dataloader)

        num_gen_steps = len(reward_dataloader)
        pbar = tqdm(total=num_gen_steps)
        all_rewards = []
        for batch in reward_dataloader:
            with torch.no_grad():
                rewards = reward_model(**batch).logits
                rewards = accelerator.gather(rewards)
                all_rewards.extend(rewards)
                pbar.update(1)

        all_rewards = [reward.item() for reward in all_rewards][: len(d_g)]
        d_g = d_g.add_column("scores", all_rewards)

        if script_args.save_grow_dataset is not None:
            accelerator.print("*** Saving the dataset ***")
            d_g.save_to_disk(os.path.join(script_args.save_grow_dataset, "train"))

        # Create thresholds for the filtering step
        thresholds = [
            np.percentile(all_rewards, 50),
            np.percentile(all_rewards, 75),
            np.percentile(all_rewards, 90),
            np.percentile(all_rewards, 95),
        ]

        accelerator.wait_for_everyone()

        # Improve step
        for improve_step in range(script_args.improve_steps):
            d_g = d_g.filter(lambda example: example["scores"] > thresholds[improve_step])

            # Train the model here using an Iterative trainer.
            improve_dataloader = DataLoader(d_g, batch_size=script_args.train_bs, shuffle=True, collate_fn=collator)

            # Evaluation step


if __name__ == "__main__":
    main()
