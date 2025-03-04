import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, HfArgumentParser, TrainingArguments, AutoProcessor, Trainer
from dataclasses import dataclass
from trl import ModelConfig, ScriptArguments
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


@dataclass
class LeRobotConfig(TrainingArguments):
    r"""
    Configuration class for the [`LeRobotTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        my_custom_arg (`int`, *optional*, defaults to `1`):
            A custom argument that you can use in your script.
    """

    my_custom_arg: int = 1

class LeRobotTrainer(Trainer):
    def __init__(self, model, args, train_dataset, processing_class):
        super().__init__(model=model, args=args, train_dataset=train_dataset, processing_class=processing_class)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, LeRobotConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path)
    processor = AutoProcessor.from_pretrained(model_config.model_name_or_path)

    dataset = load_dataset(script_args.dataset_name)

    # Training
    trainer = LeRobotTrainer(
        model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        processing_class=processor,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
