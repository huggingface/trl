# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

from transformers import TrainingArguments


@dataclass
class BaseConfig(TrainingArguments):
    # Override fields from TrainingArguments whose help strings contain unescaped "%" characters.
    # argparse interprets "%" as a format specifier, raising TypeError when rendering --help output.
    # Fixed upstream in transformers v5.3.0, but overridden here to support older versions.
    # - Introduced in v5.2.0; fixed in v5.3.0
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training."
        },
    )
    # - Introduced in v5.2.0; fixed in v5.3.0
    use_liger_kernel: bool = field(
        default=False,
        metadata={
            "help": "Enable Liger Kernel optimizations. Increases throughput by ~20%% and reduces memory by ~60%%."
        },
    )
    # - Introduced in v4.54.1; fixed in v5.3.0
    torch_empty_cache_steps: int | None = field(
        default=None,
        metadata={
            "help": "Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied."
        },
    )
