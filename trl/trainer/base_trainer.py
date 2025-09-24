import os
from typing import Optional, Union, List

from transformers import Trainer, is_wandb_available

from .utils import generate_model_card, get_comet_experiment_url


if is_wandb_available():
    import wandb


class BaseTrainer(Trainer):
    _tag_names = []

    def _create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, List[str], None] = None,
        trainer_name: Optional[str] = None,
        trainer_citation: Optional[str] = None,
        paper_title: Optional[str] = None,
        paper_id: Optional[str] = None,
    ):
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
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
        tags = list(tags)

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.url if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name=trainer_name,
            trainer_citation=trainer_citation,
            paper_title=paper_title,
            paper_id=paper_id,
        )
        model_card.save(os.path.join(self.args.output_dir, "README.md"))

