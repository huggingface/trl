# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import functools
import time

from transformers import is_wandb_available


if is_wandb_available():
    import wandb


def profiling_decorator(func):
    """
    Decorator to profile a function and log the time taken to execute it.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time

        if "wandb" in self.args.report_to and wandb.run is not None and self.accelerator.is_main_process:
            wandb.log({f"profiling/Time taken: {self.__class__.__name__}.{func.__name__}": duration})
        return result

    return wrapper
