# Copyright 2023 metric-space, The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import tyro
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPModel, CLIPProcessor

from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
from trl.import_utils import is_xpu_available

from io import BytesIO
from PIL import Image


@dataclass
class ScriptArguments:
    hf_user_access_token: str
    pretrained_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    """the pretrained model to use"""
    pretrained_revision: str = "main"
    """the pretrained model revision to use"""
    sdxl_vae: str = "madebyollin/sdxl-vae-fp16-fix"
    """the name of the pretrained SDXL VAE model to use"""
    hf_hub_model_id: str = "ddpo-finetuned-stable-diffusion-xl"
    """HuggingFace repo to save model weights to"""
    compression_quality: int = 80
    """JPEG compression quality"""
    mode: str = "compressibility" # or 'incompressibility'
    """Whether we are looking to make images more compressible or incompressible"""

    ddpo_config: DDPOConfig = field(
        default_factory=lambda: DDPOConfig(
            width=1024,
            height=1024,
            sdxl=True,
            num_epochs=200,
            train_gradient_accumulation_steps=1,
            sample_num_steps=50,
            sample_batch_size=6,
            train_batch_size=3,
            sample_num_batches_per_epoch=4,
            per_prompt_stat_tracking=True,
            per_prompt_stat_tracking_buffer_size=32,
            tracker_project_name="stable_diffusion_training",
            log_with="wandb",
            project_kwargs={
                "logging_dir": "./logs",
                "automatic_checkpoint_naming": True,
                "total_limit": 5,
                "project_dir": "./save",
            },
        )
    )

def scorer(compression_quality = 80, mode = 'compressibility'):

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores = []
        for i in range(images.shape[0]):
            img = Image.fromarray(images[i].cpu().detach().numpy())

            # Convert the image to a JPEG format with the specified compression quality
            output_buffer = BytesIO()
            img.save(output_buffer, 'JPEG', quality=compression_quality)
            output_buffer.seek(0) # Reset the output buffer position
            output_buffer = output_buffer.getbuffer()

            # Get the byte representation of the original image
            original_bytes = BytesIO()
            img.save(original_bytes, 'PNG')
            original_bytes.seek(0)
            original_bytes = original_bytes.getbuffer()

            # Calculate the ratio between the sizes of the two images
            ratio = original_bytes.nbytes / output_buffer.nbytes
            if mode == 'incompressibility':
                ratio = 1 / ratio

            scores.append(ratio)

        return scores, {}

    return _fn


# list of example prompts to feed stable diffusion
animals = [
    "cat",
    "dog",
    "horse",
    "monkey",
    "rabbit",
    "zebra",
    "spider",
    "bird",
    "sheep",
    "deer",
    "cow",
    "goat",
    "lion",
    "frog",
    "chicken",
    "duck",
    "goose",
    "bee",
    "pig",
    "turkey",
    "fly",
    "llama",
    "camel",
    "bat",
    "gorilla",
    "hedgehog",
    "kangaroo",
]


def prompt_fn():
    return np.random.choice(animals), {}


def image_outputs_logger(image_data, global_step, accelerate_logger):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    images, prompts, _, rewards, _ = image_data[-1]

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()
        result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0)

    accelerate_logger.log_images(
        result,
        step=global_step,
    )


if __name__ == "__main__":
    args = tyro.cli(ScriptArguments)

    pipeline = DefaultDDPOStableDiffusionPipeline(
        args.pretrained_model, pretrained_model_revision=args.pretrained_revision, use_lora=True, sdxl_vae=args.sdxl_vae,
    )

    trainer = DDPOTrainer(
        args.ddpo_config,
        scorer(args.compression_quality, args.mode),
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()

    trainer.push_to_hub(args.hf_hub_model_id, token=args.hf_user_access_token)
