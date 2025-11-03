# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import os
from typing import Optional

from transformers import TrainerCallback

from ...import_utils import is_mergekit_available
from ...trainer.utils import get_config_model_id
from .mergekit_utils import MergeConfig, merge_models, upload_model_to_hf


class MergeModelCallback(TrainerCallback):
    r"""
    A [`~transformers.TrainerCallback`] that merges the policy model (the model being trained) with another model based
    on a merge configuration.

    Args:
        merge_config ([`MergeConfig`], *optional*):
            Configuration used for the merging process. If not provided, the default [`MergeConfig`] is used.
        merge_at_every_checkpoint (`bool`, *optional*, defaults to `False`):
            Whether to merge the model at every checkpoint.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the merged model to the Hub after merging.

    Example:

    ```python
    from trl.experimental.mergekit import MergeConfig, MergeModelCallback

    config = MergeConfig()
    merge_callback = MergeModelCallback(config)
    trainer = DPOTrainer(..., callbacks=[merge_callback])
    ```
    """

    def __init__(
        self,
        merge_config: Optional["MergeConfig"] = None,
        merge_at_every_checkpoint: bool = False,
        push_to_hub: bool = False,
    ):
        if not is_mergekit_available():
            raise ImportError(
                "MergeModelCallback requires the `mergekit` extra. To install, run `pip install mergekit`."
            )
        self.merge_config = merge_config or MergeConfig()
        self.merge_at_every_checkpoint = merge_at_every_checkpoint
        self.push_to_hub = push_to_hub

    def _merge_and_maybe_push(self, output_dir, global_step, model):
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        self.merge_config.policy_model_path = checkpoint_path
        if self.merge_config.target_model_path is None:
            self.merge_config.target_model_path = get_config_model_id(model.config)
        merge_path = os.path.join(checkpoint_path, "merged")

        merge_models(self.merge_config.create(), merge_path)

        if self.push_to_hub:
            repo_name = f"{output_dir}_checkpoint-{global_step}_merged"
            upload_model_to_hf(merge_path, repo_name)

    def on_save(self, args, state, control, model=None, **kwargs):
        if self.merge_at_every_checkpoint:
            self._merge_and_maybe_push(args.output_dir, state.global_step, model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not self.merge_at_every_checkpoint:
            self._merge_and_maybe_push(args.output_dir, state.global_step, model)
