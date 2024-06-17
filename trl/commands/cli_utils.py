# This file is a copy of trl/examples/scripts/sft.py so that we could
# use it together with rich and the TRL CLI in a more customizable manner.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import os
import sys
from argparse import Namespace
from dataclasses import dataclass, field

import yaml
from transformers import HfArgumentParser


logger = logging.getLogger(__name__)


class YamlConfigParser:
    def parse_and_set_env(self, config_path):
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)

        if "env" in config:
            env_vars = config.pop("env")
            if isinstance(env_vars, dict):
                for key, value in env_vars.items():
                    os.environ[key] = str(value)
            else:
                raise ValueError("`env` field should be a dict in the YAML file.")

        return config

    def to_string(self, config):
        final_string = """"""
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                if len(value) != 0:
                    value = str(value)
                    value = value.replace("'", '"')
                    value = f"'{value}'"
                else:
                    continue

            final_string += f"--{key} {value} "
        return final_string


def init_zero_verbose():
    """
    Perform zero verbose init - use this method on top of the CLI modules to make
    """
    import logging
    import warnings

    from rich.logging import RichHandler

    FORMAT = "%(message)s"
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.ERROR)

    # Custom warning handler to redirect warnings to the logging system
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logging.warning(f"{filename}:{lineno}: {category.__name__}: {message}")

    # Add the custom warning handler - we need to do that before importing anything to make sure the loggers work well
    warnings.showwarning = warning_handler


@dataclass
class SFTScriptArguments:
    dataset_name: str = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "the dataset name"},
    )
    dataset_train_split: str = field(default="train", metadata={"help": "The dataset split to train on"})
    dataset_test_split: str = field(default="test", metadata={"help": "The dataset split to evaluate on"})
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={"help": "Whether to apply `use_reentrant` for gradient_checkpointing"},
    )


@dataclass
class DPOScriptArguments:
    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    dataset_train_split: str = field(default="train", metadata={"help": "The dataset split to use for training"})
    dataset_test_split: str = field(default="test", metadata={"help": "The dataset split to use for evaluation"})
    sanity_check: bool = field(default=False, metadata={"help": "only train on 1000 samples"})
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "debug argument for distributed training;"
            "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={"help": "Whether to apply `use_reentrant` for gradient_checkpointing"},
    )


@dataclass
class ChatArguments:
    # general settings
    model_name_or_path: str = field(metadata={"help": "Name of the pre-trained model"})
    user: str = field(default=None, metadata={"help": "Username to display in chat interface"})
    system_prompt: str = field(default=None, metadata={"help": "System prompt"})
    save_folder: str = field(default="./chat_history/", metadata={"help": "Folder to save chat history"})
    device: str = field(
        default="cpu",
        metadata={"help": "device to use for inference."},
    )
    config: str = field(
        default="default",
        metadata={
            "help": "Config file used for setting the configs. If `default` uses examples/scripts/config/default_chat_config.yaml"
        },
    )
    examples: str = field(default=None, metadata={"help": "Empty placeholder needs to be set via config."})
    # generation settings
    max_new_tokens: int = field(default=256, metadata={"help": "Maximum number of tokens to generate"})
    do_sample: bool = field(default=True, metadata={"help": "Whether to sample outputs during generation"})
    num_beams: int = field(default=1, metadata={"help": "Number of beams for beam search"})
    temperature: float = field(default=1.0, metadata={"help": "Temperature parameter for generation"})
    top_k: int = field(default=50, metadata={"help": "Value of k for top-k sampling"})
    top_p: float = field(default=1.0, metadata={"help": "Value of p for nucleus sampling"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "Repetition penalty"})
    eos_tokens: str = field(
        default=None,
        metadata={"help": "EOS tokens to stop the generation. If multiple they should be comma separated"},
    )
    eos_token_ids: str = field(
        default=None,
        metadata={"help": "EOS token IDs to stop the generation. If multiple they should be comma separated"},
    )
    # model loading
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    torch_dtype: str = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code when loading a model."})
    attn_implementation: str = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use; you can run --attn_implementation=flash_attention_2, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "use 8 bit precision for the base model - works only with LoRA"},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "use 4 bit precision for the base model - works only with LoRA"},
    )

    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"})
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})


class TrlParser(HfArgumentParser):
    def __init__(self, parsers, ignore_extra_args=False):
        """
        The TRL parser parses a list of parsers (TrainingArguments, trl.ModelConfig, etc.), creates a config
        parsers for users that pass a valid `config` field and merge the values that are set in the config
        with the processed parsers.

        Args:
            parsers (`List[argparse.ArgumentParser`]):
                List of parsers.
            ignore_extra_args (`bool`):
                Whether to ignore extra arguments passed by the config
                and not raise errors.
        """
        super().__init__(parsers)
        self.yaml_parser = YamlConfigParser()
        self.ignore_extra_args = ignore_extra_args

    def post_process_dataclasses(self, dataclasses):
        # Apply additional post-processing in case some arguments needs a special
        # care
        training_args = trl_args = None
        training_args_index = None

        for i, dataclass_obj in enumerate(dataclasses):
            if dataclass_obj.__class__.__name__ == "TrainingArguments":
                training_args = dataclass_obj
                training_args_index = i
            elif dataclass_obj.__class__.__name__ in ("SFTScriptArguments", "DPOScriptArguments"):
                trl_args = dataclass_obj
            else:
                ...

        if trl_args is not None and training_args is not None:
            training_args.gradient_checkpointing_kwargs = dict(
                use_reentrant=trl_args.gradient_checkpointing_use_reentrant
            )
            dataclasses[training_args_index] = training_args

        return dataclasses

    def parse_args_and_config(self, return_remaining_strings=False):
        yaml_config = None
        if "--config" in sys.argv:
            config_index = sys.argv.index("--config")

            _ = sys.argv.pop(config_index)  # --config
            config_path = sys.argv.pop(config_index)  # path to config
            yaml_config = self.yaml_parser.parse_and_set_env(config_path)

            self.set_defaults_with_config(**yaml_config)

        outputs = self.parse_args_into_dataclasses(return_remaining_strings=return_remaining_strings)

        if yaml_config is None:
            return outputs

        if return_remaining_strings:
            # if we have extra yaml config and command line strings
            # outputs[-1] is remaining command line strings
            # outputs[-2] is remaining yaml config as Namespace
            # combine them into remaining strings object
            remaining_strings = outputs[-1] + [f"{key}: {value}" for key, value in vars(outputs[-2]).items()]
            return outputs[:-2], remaining_strings
        else:
            # outputs[-1] is either remaining yaml config as Namespace or parsed config as Dataclass
            if isinstance(outputs[-1], Namespace) and not self.ignore_extra_args:
                remaining_args = vars(outputs[-1])
                raise ValueError(f"Some specified config arguments are not used by the TrlParser: {remaining_args}")

            return outputs

    def set_defaults_with_config(self, **kwargs):
        """Defaults we're setting with config allow us to change to required = False"""
        self._defaults.update(kwargs)

        # if these defaults match any existing arguments, replace
        # the previous default on the object with the new one
        for action in self._actions:
            if action.dest in kwargs:
                action.default = kwargs[action.dest]
                action.required = False
