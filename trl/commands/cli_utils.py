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
import inspect
import logging
import os
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from typing import Any, List

import yaml
from transformers import HfArgumentParser


logger = logging.getLogger(__name__)


class YamlConfigParser:
    def __init__(self, config_path: str = None, dataclasses: List[Any] = None):
        self.config = None

        if config_path is not None:
            with open(config_path) as yaml_file:
                self.config = yaml.safe_load(yaml_file)
        else:
            self.config = {}

        if dataclasses is None:
            dataclasses = []

        # We create a dummy training args to compare the values before / after
        # __post_init__
        # Here we import `TrainingArguments` from the local level to not
        # break TRL lazy imports.
        from transformers import TrainingArguments

        self._dummy_training_args = TrainingArguments(output_dir="dummy-training-args")

        self.parse_and_set_env()
        self.merge_dataclasses(dataclasses)

    def parse_and_set_env(self):
        if "env" in self.config:
            env_vars = self.config["env"]
            if isinstance(env_vars, dict):
                for key, value in env_vars.items():
                    os.environ[key] = str(value)
            else:
                raise ValueError("`env` field should be a dict in the YAML file.")

    def merge_dataclasses(self, dataclasses):
        from transformers import TrainingArguments

        dataclasses_copy = [deepcopy(dataclass) for dataclass in dataclasses]

        if len(self.config) > 0:
            for i, dataclass in enumerate(dataclasses):
                is_hf_training_args = False

                for data_class_field in fields(dataclass):
                    # Get the field here
                    field_name = data_class_field.name
                    field_value = getattr(dataclass, field_name)

                    if not isinstance(dataclass, TrainingArguments):
                        default_value = data_class_field.default
                    else:
                        default_value = (
                            getattr(self._dummy_training_args, field_name)
                            if field_name != "output_dir"
                            else field_name
                        )
                        is_hf_training_args = True

                    default_value_changed = field_value != default_value

                    if field_value is not None or field_name in self.config:
                        if field_name in self.config:
                            # In case the field value is not different from default, overwrite it
                            if not default_value_changed:
                                value_to_replace = self.config[field_name]
                                setattr(dataclasses_copy[i], field_name, value_to_replace)
                        # Otherwise do nothing

                # Re-init `TrainingArguments` to handle all post-processing correctly
                if is_hf_training_args:
                    init_signature = list(inspect.signature(TrainingArguments.__init__).parameters)
                    dict_dataclass = asdict(dataclasses_copy[i])
                    new_dict_dataclass = {k: v for k, v in dict_dataclass.items() if k in init_signature}
                    dataclasses_copy[i] = TrainingArguments(**new_dict_dataclass)

        return dataclasses_copy

    def to_string(self):
        final_string = """"""
        for key, value in self.config.items():
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
    dataset_name: str = field(default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"})
    dataset_train_split: str = field(default="train", metadata={"help": "The dataset split to train on"})
    dataset_test_split: str = field(default="test", metadata={"help": "The dataset split to evaluate on"})
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    gradient_checkpointing_use_reentrant: bool = field(
        default=False, metadata={"help": "Whether to apply `use_reentrant` for gradient_checkpointing"}
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
        default=False, metadata={"help": "Whether to apply `use_reentrant` for gradient_checkpointing"}
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
        default=False, metadata={"help": "use 8 bit precision for the base model - works only with LoRA"}
    )
    load_in_4bit: bool = field(
        default=False, metadata={"help": "use 4 bit precision for the base model - works only with LoRA"}
    )

    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"})
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})


class TrlParser(HfArgumentParser):
    def __init__(self, parsers):
        """
        The TRL parser parses a list of parsers (TrainingArguments, trl.ModelConfig, etc.), creates a config
        parsers for users that pass a valid `config` field and merge the values that are set in the config
        with the processed parsers.

        Args:
            parsers (`List[argparse.ArgumentParser`]):
                List of parsers.
        """
        super().__init__(parsers)

        self.config_parser = None

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

    def parse_args_and_config(self):
        # Hack to force-replace the `output_dir` from the YAML file if one did not passed
        # output_dir in the command line
        if "--config" in sys.argv:
            config_index = sys.argv.index("--config") + 1
            config_path = sys.argv[config_index]

            self.config_parser = YamlConfigParser(config_path)
            output_dir = self.config_parser.config.get("output_dir")

            if output_dir is not None:
                if "--output_dir" in sys.argv:
                    output_dir_index = sys.argv.index("--output_dir")
                    passed_output_dir = sys.argv[output_dir_index + 1]
                    self.config_parser.config["output_dir"] = passed_output_dir
                else:
                    sys.argv.extend(["--output_dir", output_dir])

        dataclasses = self.parse_args_into_dataclasses(return_remaining_strings=True)

        if len(dataclasses[-1]) > 0:
            # It is expected that `config` is in that list but not ignored
            # let's simply remove them
            list_ignored = dataclasses[-1]
            if "--config" in list_ignored:
                config_index = list_ignored.index("--config") + 1
                config_path = list_ignored[config_index]

                list_ignored.remove(config_path)
                list_ignored.remove("--config")

            if len(list_ignored) > 0:
                logger.warning(
                    f"Detected extra arguments that are going to be ignored: {list_ignored} - make sure to double check what you are doing"
                )

        # Pop the last element which should be the remaining strings
        dataclasses = self.update_dataclasses_with_config(dataclasses[:-1])
        return dataclasses

    def update_dataclasses_with_config(self, dataclasses):
        for parser_dataclass in dataclasses:
            if hasattr(parser_dataclass, "config") and self.config_parser is None:
                self.config_parser = YamlConfigParser(parser_dataclass.config)

        if self.config_parser is not None:
            dataclasses = self.config_parser.merge_dataclasses(dataclasses)
        dataclasses = self.post_process_dataclasses(dataclasses)
        return dataclasses
