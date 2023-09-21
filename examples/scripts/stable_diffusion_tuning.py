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


import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPModel, CLIPProcessor

from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    """
    This model attempts to predict the aesthetic score of an image. The aesthetic score
    is a numerical approximation of how much a specific image is liked by humans on average.
    This is from https://github.com/christophschuhmann/improved-aesthetic-predictor
    """

    def __init__(self, *, dtype, model_id, model_filename):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        try:
            cached_path = hf_hub_download(model_id, model_filename)
        except EntryNotFoundError:
            cached_path = os.path.join(model_id, model_filename)
        state_dict = torch.load(cached_path)
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


def aesthetic_scorer(hub_model_id, model_filename):
    scorer = AestheticScorer(
        model_id=hub_model_id,
        model_filename=model_filename,
        dtype=torch.float32,
    ).cuda()

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
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


def parse_arguments():
    parser = argparse.ArgumentParser(description="DDPOConfig settings and pretrained model details.")

    # DDPOConfig arguments
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--train_gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--sample_num_steps", type=int, default=50)
    parser.add_argument("--sample_batch_size", type=int, default=6)
    parser.add_argument("--train_batch_size", type=int, default=3)
    parser.add_argument("--sample_num_batches_per_epoch", type=int, default=4)
    parser.add_argument("--per_prompt_stat_tracking", action="store_true", default=True)
    parser.add_argument("--per_prompt_stat_tracking_buffer_size", type=int, default=32)
    parser.add_argument("--tracker_project_name", default="stable_diffusion_training")
    parser.add_argument("--log_with", default="none")

    parser.add_argument("--logging_dir", default="./logs")
    parser.add_argument("--automatic_checkpoint_naming", action="store_true", default=True)
    parser.add_argument("--total_limit", type=int, default=5)
    parser.add_argument("--project_dir", default="./save")

    parser.add_argument("--pretrained_model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--pretrained_revision", default="main")
    parser.add_argument("--hf_user_access_token", required=False)
    parser.add_argument(
        "--hf_hub_model_id",
        help="HuggingFace repo to save model weights to",
        default="ddpo-finetuned-stable-diffusion",
    )

    parser.add_argument(
        "--hf_hub_aesthetic_model_id",
        help="HuggingFace model ID for aesthetic scorer model weights",
        default="trl-lib/ddpo-aesthetic-predictor",
    )

    parser.add_argument(
        "--hf_hub_aesthetic_model_filename",
        default="aesthetic-model.pth",
        help="HuggingFace model filename for aesthetic scorer model weights",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    project_kwargs = {
        "logging_dir": args.logging_dir,
        "automatic_checkpoint_naming": args.automatic_checkpoint_naming,
        "total_limit": args.total_limit,
        "project_dir": args.project_dir,
    }

    config = DDPOConfig(
        num_epochs=args.num_epochs,
        train_gradient_accumulation_steps=args.train_gradient_accumulation_steps,
        sample_num_steps=args.sample_num_steps,
        sample_batch_size=args.sample_batch_size,
        train_batch_size=args.train_batch_size,
        sample_num_batches_per_epoch=args.sample_num_batches_per_epoch,
        per_prompt_stat_tracking=args.per_prompt_stat_tracking,
        per_prompt_stat_tracking_buffer_size=args.per_prompt_stat_tracking_buffer_size,
        tracker_project_name=args.tracker_project_name,
        log_with=args.log_with,
        project_kwargs=project_kwargs,
    )

    pipeline = DefaultDDPOStableDiffusionPipeline(
        args.pretrained_model, pretrained_model_revision=args.pretrained_revision, use_lora=True
    )

    trainer = DDPOTrainer(
        config,
        aesthetic_scorer(args.hf_hub_aesthetic_model_id, args.hf_hub_aesthetic_model_filename),
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()

    trainer.push_to_hub(args.hf_hub_model_id, token=args.hf_user_access_token)
