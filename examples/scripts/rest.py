# [WIP]
# TO DO:
# * Save datasets after each grow step and load the right one
# * Load and save the right model at each grow step
# * reformat
from dataclasses import dataclass, field
from typing import Optional

import generate
import numpy as np
import score
import torch
from datasets import load_from_disk
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import IterativeConfig, IterativeTrainer


@dataclass
class ScriptArguments:

    learning_rate: Optional[float] = field(default=2e-5)
    num_grow_steps: Optional[int] = field(default=1)
    num_improve_steps: Optional[int] = field(default=1)
    evaluation_step: Optional[str] = field(
        default="no", metadata={"help": "Choose between 'grow_step', 'improve_step', 'no'."}
    )
    save_model_step: Optional[str] = field(
        default="no", metadata={"help": "Choose between 'grow_step', 'improve_step', 'no'."}
    )
    train_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the grow dataset, if generation and scoring were already run"}
    )

    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "the model name"})
    reward_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "the reward model name"})
    save_model_path: Optional[str] = field(
        default=None, metadata={"help": "where to save your model after each improve step"}
    )
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the HF data path"})

    save_dataset_path: Optional[str] = field(default=None, metadata={"help": "the save dataset path"})
    generation_column_name: Optional[str] = field(default="generated")

    eval_bs: Optional[int] = field(default=16, metadata={"help": "the generation batch size"})
    gen_bs: Optional[int] = field(default=16, metadata={"help": "the generation batch size"})
    step_bs: Optional[int] = field(default=8, metadata={"help": "the generation batch size"})
    bf16: Optional[bool] = field(
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        metadata={"help": "whether to use bf16."},
    )
    fp16: Optional[bool] = field(
        default=True if not torch.cuda.get_device_capability()[0] == 8 else False,
        metadata={"help": "whether to use fp16."},
    )
    log_with: Optional[str] = field(default="tensorboard")
    logging_dir: Optional[str] = field(default=None)
    max_prompt_length: Optional[int] = field(default=256, metadata={"help": "The maximum prompt length"})
    max_length: Optional[int] = field(
        default=512, metadata={"help": "The maximum number of tokens for training and reward scoring"}
    )

    truncation_side: Optional[str] = field(
        default="right",
        metadata={"help": "the side to truncate the prompt if the prompt is longer than max_prompt_length"},
    )
    max_new_tokens: Optional[int] = field(
        default=256, metadata={"help": "the maximum number of tokens generated per sample"}
    )
    temperature: Optional[float] = field(default=1.0)
    top_p: Optional[float] = field(default=1.0)
    top_k: Optional[float] = field(default=50)
    num_return_sequences: Optional[int] = field(default=1)

    sanity_check: Optional[bool] = field(
        default=False, metadata={"help": "Percentage of the dataset you want to make generation on."}
    )


def main():

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    for grow_step in range(script_args.num_grow_steps):

        if script_args.train_dataset_path is not None or grow_step != 0:
            generated_dataset = generate.generate(script_args)
            dataset = score.score(script_args, reward_dataset=generated_dataset)
        else:
            dataset = load_from_disk(script_args.save_dataset_path)
            if script_args.sanity_check:
                dataset = dataset.select(range(min(len(dataset), 500)))

        if grow_step != 0:
            model = AutoModelForCausalLM.from_pretrained(script_args.save_model_path)
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=script_args.learning_rate,
            )
            model, optimizer, data_collator = trainer.accelerator.prepare(model, optimizer, trainer.data_collator)
            trainer.model, trainer.optimizer, trainer.data_collator = model, optimizer, data_collator

        else:
            model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)

        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

        if grow_step == 0:
            config = IterativeConfig(
                model_name=script_args.model_name_or_path,
                step_batch_size=script_args.step_bs,
                log_with=script_args.log_with,
                project_kwargs={"logging_dir": script_args.logging_dir},
                learning_rate=script_args.learning_rate,
            )

            trainer = IterativeTrainer(config, model, tokenizer)

        def preprocess_function(samples):
            model_inputs = tokenizer(samples, max_length=script_args.max_length, truncation=True)
            return model_inputs

        dataset = dataset.map(preprocess_function, batched=True, remove_columns=list(dataset.features))

        rewards = dataset["rewards"]

        # Filtering step. You should implement your own based on your dataset and needs.
        thresholds = [
            np.percentile(rewards, 75),
            np.percentile(rewards, 90),
            np.percentile(rewards, 95),
            np.percentile(rewards, 99),
        ]

        for improve_step in range(script_args.num_improve_steps):
            dataset = dataset.filter(lambda example: example["rewards"] > thresholds[improve_step])
            dataset.set_format("torch")

            stats = trainer.step(input_ids=dataset["input_ids"], attention_mask=dataset["attention_mask"])

            if script_args.evaluation_step == "improve_step":
                # Do the evaluation step. Just need to call the generate & score function then use the iterative trainer to log the results.
                pass

            if script_args.save_model_step == "improve_step":
                model.save_pretrained(script_args.save_model_path)

        if script_args.evaluation_step == "grow_step":
            # Do the evaluation step. Just need to call the generate & score function then use the iterative trainer to log the results.
            pass

        if script_args.save_model_step == "grow_step":
            model.save_pretrained(script_args.save_model_path)

        # Add the mean reward of the dataset to the stats.
        trainer.log_stats(stats)

        # free memory to keep the number of models up during training to 1.
        trainer.accelerator.free_memory()
        del model, trainer.optimizer, dataset, generated_dataset


if __name__ == "__main__":
    main()
