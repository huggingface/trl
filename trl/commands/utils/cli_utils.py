# flake8: noqa
# This file is a copy of trl/examples/scripts/sft.py so that we could
# use it together with rich and the TRL CLI in a more customizable manner.
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
from dataclasses import dataclass, field


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
class SftScriptArguments:
    dataset_name: str = field(default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"})
    dataset_text_field: str = field(default="text", metadata={"help": "the text field of the dataset"})
    max_seq_length: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})
    packing: bool = field(default=False, metadata={"help": "Whether to apply data packing or not during training"})
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    gradient_checkpointing_use_reentrant: bool = field(
        default=False, metadata={"help": "Whether to apply `use_reentrant` for gradient_checkpointing"}
    )
