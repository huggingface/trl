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

import torch
from accelerate.utils import is_peft_model
from huggingface_hub.utils import send_telemetry
from transformers import CONFIG_MAPPING, Trainer, is_wandb_available

from .. import __version__
from .utils import generate_model_card, get_comet_experiment_url, get_config_model_id, get_trackio_space_url


if is_wandb_available():
    import wandb


# Trainer class names that may appear in telemetry topics. Any class outside this set — internal helpers,
# in-flight subclasses not yet shipped, user-defined subclasses — is reported as "other" so unreleased or
# private names never leak. Adding a new trainer requires an explicit entry here.
_TELEMETRY_TRAINERS = {
    # Stable
    "DPOTrainer",
    "GRPOTrainer",
    "KTOTrainer",
    "RewardTrainer",
    "RLOOTrainer",
    "SFTTrainer",
    # Experimental
    "AsyncGRPOTrainer",
    "BCOTrainer",
    "CPOTrainer",
    "DistillationTrainer",
    "DPPOTrainer",
    "GFPOTrainer",
    "GKDTrainer",
    "GOLDTrainer",
    "GRPOWithReplayBufferTrainer",
    "MiniLLMTrainer",
    "NashMDTrainer",
    "OnlineDPOTrainer",
    "ORPOTrainer",
    "PAPOTrainer",
    "PPOTrainer",
    "PRMTrainer",
    "SDFTTrainer",
    "SDPOTrainer",
    "SSDTrainer",
    "TPOTrainer",
    "XPOTrainer",
}


class _BaseTrainer(Trainer):
    _tag_names = []
    _name = "Base"
    _paper = {}
    _template_file = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._send_telemetry()

    def _send_telemetry(self):
        # Only send from rank 0 to avoid multiplying pings by world size, and skip CI runs so automated tests don't
        # bias the data. Honors `HF_HUB_DISABLE_TELEMETRY=1` and `HF_HUB_OFFLINE=1` (handled by `send_telemetry`).
        if not self.accelerator.is_main_process or os.environ.get("CI"):
            return
        if self.is_deepspeed_enabled:
            distributed = "deepspeed"
        elif self.is_fsdp_enabled:
            distributed = "fsdp"
        elif self.accelerator.num_processes > 1:
            distributed = "ddp"
        else:
            distributed = "none"
        device = self.accelerator.device.type
        if device == "cuda":
            gpu = torch.cuda.get_device_name(0)
        elif device == "xpu":
            gpu = torch.xpu.get_device_name(0)
        elif device == "npu":
            gpu = torch.npu.get_device_name(0)
        elif device == "mlu":
            gpu = torch.mlu.get_device_name(0)
        else:
            gpu = "other"
        # Bucketed to avoid fingerprinting individual deployments by their exact cluster size.
        n = self.accelerator.num_processes
        world_size = "1" if n == 1 else "2-8" if n <= 8 else "9-64" if n <= 64 else "65+"
        # Trainer class and model arch are reported only if they come from a known upstream allowlist (TRL trainers,
        # transformers `CONFIG_MAPPING`); anything else is reported as "other" so we never leak the names of internal,
        # custom trainer subclasses or private model architectures.
        cls = type(self)
        trainer = (
            cls.__name__ if cls.__name__ in _TELEMETRY_TRAINERS and cls.__module__.startswith("trl.") else "other"
        )
        model_type = self.model.config.model_type
        model_arch = model_type if model_type in CONFIG_MAPPING else "other"
        send_telemetry(
            topic=f"trl/{trainer}",
            library_name="trl",
            library_version=__version__,
            user_agent={
                "model_arch": model_arch,
                "peft": str(is_peft_model(self.model)).lower(),
                "distributed": distributed,
                "world_size": world_size,
                "device": device,
                "gpu": gpu,
            },
        )

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
