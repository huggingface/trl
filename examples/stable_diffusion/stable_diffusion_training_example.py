import os

import numpy as np
import requests
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

from trl import DDPOConfig, DDPOTrainer, DefaultDDPOPipeline, DefaultDDPOScheduler


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
    def __init__(self, dtype, cache=".", weights_fname="sac+logos+ava1-l14-linearMSE.pth"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        self.loadpath = os.path.join(cache, weights_fname)
        if not os.path.exists(self.loadpath):
            url = (
                "https://github.com/christophschuhmann/"
                f"improved-aesthetic-predictor/blob/main/{weights_fname}?raw=true"
            )
            r = requests.get(url)

            with open(self.loadpath, "wb") as f:
                f.write(r.content)

        state_dict = torch.load(self.loadpath)
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


def aesthetic_score():
    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


# prompt function
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
    # extract the last one
    result = {}
    images, prompts, _, rewards = image_data[-1]

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()
        result[f"{prompt:.25} | {reward:.2f}"] = image.cpu().unsqueeze(0)

    accelerate_logger.log_images(
        result,
        step=global_step,
    )


if __name__ == "__main__":
    config = DDPOConfig(
        num_epochs=200,
        train_gradient_accumulation_steps=1,
        per_prompt_stat_tracking_buffer_size=32,
        sample_num_batches_per_epoch=2,
        sample_batch_size=1,
        tracker_project_name="stable_diffusion_training",
        log_with="tensorboard",
        project_kwargs={
            "logging_dir": "./logs",
            "automatic_checkpoint_naming": True,
            "total_limit": 5,
            "project_dir": "./save",
        },
    )
    pretrained_model = "runwayml/stable-diffusion-v1-5"
    # revision of the model to load.
    pretrained_revision = "main"

    pipeline = DefaultDDPOPipeline.from_pretrained(pretrained_model, revision=pretrained_revision)

    pipeline.scheduler = DefaultDDPOScheduler.from_config(pipeline.scheduler.config)

    trainer = DDPOTrainer(
        config,
        aesthetic_score(),
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.run()
