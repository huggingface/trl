import os
import tempfile

import numpy as np
import requests
import torch
import torch.nn as nn
import wandb
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from trl import DDPOConfig, DDPOTrainer, DefaultDDPOPipeline, DefaultDDPOScheduler


# reward function
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
    "tiger",
    "bear",
    "raccoon",
    "fox",
    "wolf",
    "lizard",
    "beetle",
    "ant",
    "butterfly",
    "fish",
    "shark",
    "whale",
    "dolphin",
    "squirrel",
    "mouse",
    "rat",
    "snake",
    "turtle",
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


def image_outputs_hook(images_and_prompts, rewards, global_step, accelerate_logger):
    with tempfile.TemporaryDirectory() as tmpdir:
        # extract the last one
        images, prompts, _ = images_and_prompts[-1]
        for i, image in enumerate(images):
            pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            pil = pil.resize((256, 256))
            pil.save(os.path.join(tmpdir, f"{i}.jpg"))
        accelerate_logger(
            {
                "images": [
                    wandb.Image(
                        os.path.join(tmpdir, f"{i}.jpg"),
                        caption=f"{prompt:.25} | {reward:.2f}",
                    )
                    for i, (prompt, reward) in enumerate(zip(prompts, rewards))
                ],
            },
            step=global_step,
        )


if __name__ == "__main__":
    config = DDPOConfig(
        num_epochs=200,
        train_gradient_accumulation_steps=1,
        per_prompt_stat_tracking_buffer_size=32,
        sample_batch_size=1,
        tracker_project_name="gintkoi",
        log_with="tensorboard",
        project_kwargs={"logging_dir": "./logs", "automatic_checkpoint_naming": True, "total_limit": 5},
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
        image_outputs_hook=image_outputs_hook,
    )

    trainer.run()
