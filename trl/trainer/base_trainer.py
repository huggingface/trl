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

import os

from transformers import Trainer, is_wandb_available

from .utils import generate_model_card, get_comet_experiment_url, get_config_model_id, get_trackio_space_url


if is_wandb_available():
    import wandb


class _BaseTrainer(Trainer):
    _tag_names = []
    _name = "Base"
    _paper = {}
    _template_file = None

    def create_model_card(
        self,
        model_name: str | None = None,
        dataset_name: str | None = None,
        tags: str | list[str] | None = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str`, *optional*):
                Name of the model.
            dataset_name (`str`, *optional*):
                Name of the dataset used for training.
            tags (`str`, `list[str]`, *optional*):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        model_name_or_path = get_config_model_id(self.model.config)
        if model_name_or_path and not os.path.isdir(model_name_or_path):
            base_model = model_name_or_path
        else:
            base_model = None

        # Normalize tags
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)
        if hasattr(self.model.config, "unsloth_version"):
            tags.add("unsloth")
        if "JOB_ID" in os.environ:
            tags.add("hf_jobs")
        tags.update(self._tag_names)

        trackio_url = get_trackio_space_url()
        # Pop existing Trackio tag and re-add the one with the proper url parameters
        if trackio_url is not None:
            for tag in list(tags):
                if tag.startswith("trackio:"):
                    tags.remove(tag)
            tags.add(f"trackio:{trackio_url}")

        tags = list(tags)

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.url if is_wandb_available() and wandb.run is not None else None,
            trackio_url=trackio_url,
            comet_url=get_comet_experiment_url(),
            trainer_name=self._name,
            trainer_citation=self._paper.get("citation"),
            template_file=self._template_file,
            paper_title=self._paper.get("title"),
            paper_id=self._paper.get("id"),
        )
        model_card.save(os.path.join(self.args.output_dir, "README.md"))
