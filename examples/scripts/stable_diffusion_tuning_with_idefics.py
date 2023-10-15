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
from huggingface_hub import hf_hub_download, HfApi
from datasets import load_dataset

from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline

import tempfile
from text_generation import Client

from PIL import Image

import time

from httpx import HTTPError


endpoint = "https://api-inference.huggingface.co/models/HuggingFaceM4/idefics-80b-instruct"


def query_Template(prompt, image_url):
    return f"User: The reference aesthetic is depicted here ![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/Portrait_assis_de_l%27empereur_Tianqi.jpg/372px-Portrait_assis_de_l%27empereur_Tianqi.jpg) . The following picture should depict a {prompt} in the same aesthetic. Rate the the closeness and aesthetic of this image from 1 to 10 ![]({image_url}) Start and end conversation with the numerical rating and nothing else<end_of_utterance>\nAssistant:"


class IDEFICS:
    def __init__(self, hub_token, repo_name, endpoint):
        self.endpoint = endpoint
        self.headers = {"Authorization": f"Bearer {hub_token}", "Content-Type": "application/json"}
        self.repo_name = repo_name
        self.hf_api = HfApi(token=hub_token)
        self.client = Client(
            base_url=endpoint,
            headers={"x-use-cache": "0", "Authorization": f"Bearer {hub_token}"},
        )
        self.generation_kwargs = {
            "temperature": 1.0,
            "do_sample": True,
            "top_p": 0.95,
        }

    # Note: This is a hacky way to extract the answer from the generated text
    def extract_ans(self, generated_text):
        generated_text = generated_text.strip()

        if generated_text.isdigit():
            return int(generated_text)
        elif generated_text[-1] == "." and generated_text[-2].isdigit():
            return int(generated_text[-2])
        elif generated_text[-1].isdigit():
            return int(generated_text[-1])
        else:
            return None

    def http_io_to_scores(self, prompts, image_urls, number_of_retries_limit=3, backoff_time=3):
        input_queue = [(prompt, url, 0) for prompt, url in zip(prompts, image_urls)]
        outputs = {}

        while input_queue:
            prompt, image_url, number_of_retries = input_queue.pop(0)
            prompt = query_Template(prompt, image_url)
            print(prompt)
            # write a catch for HttPError
            try:
                response = self.client.generate(prompt, **self.generation_kwargs)
                score = self.extract_ans(response.generated_text)
                if score is None:
                    input_queue.append((prompt, image_url, number_of_retries + 1))
                else:
                    outputs[image_url] = score
            except HTTPError as e:
                if number_of_retries > number_of_retries_limit:
                    # warn user that the image could not be scored
                    print(f"Image {image_url} could not be scored. Defaulting to 0")
                    outputs[image_url] = 0
                else:
                    # append to the end of the tuple the timestamp
                    time.sleep(backoff_time)
                    input_queue.append((prompt, image_url, number_of_retries + 1))

        # rearrange outputs based on order of image_urls
        return [outputs[image_url] for image_url in image_urls]

    def rate_images(self, images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        # create a temp directory to store the images
        temp_dir = tempfile.mkdtemp()

        # Save images to the temporary directory and collect filenames
        stored_filenames = []
        for idx, img in enumerate(images):
            # Create a filename for the image (e.g., image0.jpg, image1.jpg, ...)
            filename = f"image{idx}.jpg"
            filepath = os.path.join(temp_dir, filename)

            img = Image.fromarray(img)
            img.save(filepath)
            stored_filenames.append(filename)

        # push to hub
        url = self.hf_api.upload_folder(folder_path=temp_dir, repo_id=self.repo_name, repo_type="dataset")
        url = url.split("/tree")[0]

        image_urls = [f"{url}/resolve/main/{filename}" for filename in stored_filenames]

        scores = self.http_io_to_scores(prompts, image_urls)

        print(scores)

        return torch.tensor(scores).cuda(), {}


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
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--train_gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--sample_num_steps", type=int, default=50)
    parser.add_argument("--sample_batch_size", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--sample_num_batches_per_epoch", type=int, default=1)
    parser.add_argument("--per_prompt_stat_tracking", action="store_true", default=True)
    parser.add_argument("--per_prompt_stat_tracking_buffer_size", type=int, default=32)
    parser.add_argument("--tracker_project_name", default="stable_diffusion_training")
    parser.add_argument("--log_with", default="wandb")

    parser.add_argument("--logging_dir", default="./logs")
    parser.add_argument("--automatic_checkpoint_naming", action="store_true", default=True)
    parser.add_argument("--total_limit", type=int, default=5)
    parser.add_argument("--project_dir", default="./save")

    parser.add_argument("--pretrained_model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--pretrained_revision", default="main")
    parser.add_argument("--hf_hub_save_to_model_id", required=True)
    parser.add_argument("--hf_user_access_token", required=True)
    parser.add_argument(
        "--hf_hub_dataset_id",
        help="HuggingFace repo to save sampled images to",
        default="repo to stash images",
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

    idefics = IDEFICS(args.hu_user_access_token, args.hf_hub_dataset_id, endpoint)

    trainer = DDPOTrainer(
        config,
        idefics.rate_images,
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()

    trainer.push_to_hub(args.hf_save_to_model_id, token=args.hf_user_access_token)
