# this is a work in progress

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
)

from trl import RewardTrainer


@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default="sft", metadata={"help": "the model name"})
    reward_model_name_or_path: Optional[str] = field(default="sft", metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the HF data path"})
    train_bs: Optional[int] = field(default=8)
    eval_bs: Optional[int] = field(default=16)
    grow_steps: Optional[int] = field(default=1)
    improve_steps: Optional[int] = field(default=1)
    max_length: Optional[int] = field(default=512)
    max_new_tokens: Optional[int] = field(default=256)
    temperature: Optional[float] = field(default=0.9)
    num_return_sequences: Optional[int] = field(default=8)


def main():

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    generation_kwargs = {
        "max_new_tokens": script_args.max_new_tokens,
        "temperature": script_args.temperature,
        "num_return_sequences": script_args.temperature,
    }

    model = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_name_or_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(script_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

    train_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    eval_dataset = load_dataset("Anthropic/hh-rlhf", split="test")

    def preprocess_function(sample):

        # tokenize inputs
        # TO DO: get only the prompt
        model_inputs = tokenizer(sample["chosen"], max_length=script_args.max_length, truncation=True)
        model_inputs["labels"] = model_inputs["input_ids"]

        return model_inputs

    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=list(train_dataset.features))
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=list(eval_dataset.features))

    train_dataloader = DataLoader(train_dataset, batch_size=script_args.gen_bs, shuffle=False)

    eval_dataloader = DataLoader(eval_dataset, batch_size=script_args.gen_bs, shuffle=False)

    reward_trainer = RewardTrainer(reward_model)
    trainer = Trainer(model)

    for _ in script_args.grow_steps:
        # generate
        predictions = []
        for _, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            generated = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)
            gen_decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            gen_no_pad = tokenizer(gen_decoded, truncation=True, max_length=generation_kwargs["max_new_tokens"])[
                "input_ids"
            ]

            predictions.append(gen_no_pad)

        # create the new dataset Dg
        d = Dataset.from_dict(
            {"input_ids": predictions, "attention_mask": torch.where(predictions == tokenizer.pad_token_id, 0, 1)}
        )
        d_g = concatenate_datasets([train_dataset, d])
        # score sentences
        scores = reward_trainer.predict(d_g).predictions

        d_g = d_g.add_column("scores", scores)

        # filtering step
        thresholds = [
            np.percentile(scores, 50),
            np.percentile(scores, 75),
            np.percentile(scores, 90),
            np.percentile(scores, 95),
        ]
        # improve step
        for improve_step in script_args.improve_steps:
            d_g = d_g.filter(lambda example: example["scores"] > thresholds[improve_step])

        # train
        trainer.train()


if __name__ == "__main__":
    main()
