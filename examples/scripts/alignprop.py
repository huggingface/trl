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

"""
Total Batch size = 128 = 4 (num_gpus) * 8 (per_device_batch) * 4 (accumulation steps)
Feel free to reduce batch size or increasing truncated_rand_backprop_min to a higher value to reduce memory usage.

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/scripts/alignprop.py \
    --num_epochs=20 \
    --train_gradient_accumulation_steps=4 \
    --sample_num_steps=50 \
    --train_batch_size=8 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb"

"""

from dataclasses import dataclass, field

import numpy as np
from transformers import HfArgumentParser

from trl import AlignPropConfig, AlignPropTrainer, DefaultDDPOStableDiffusionPipeline
from trl.models.auxiliary_modules import aesthetic_scorer


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        pretrained_model (`str`, *optional*, defaults to `"runwayml/stable-diffusion-v1-5"`):
            Pretrained model to use.
        pretrained_revision (`str`, *optional*, defaults to `"main"`):
            Pretrained model revision to use.
        hf_hub_model_id (`str`, *optional*, defaults to `"alignprop-finetuned-stable-diffusion"`):
            HuggingFace repo to save model weights to.
        hf_hub_aesthetic_model_id (`str`, *optional*, defaults to `"trl-lib/ddpo-aesthetic-predictor"`):
            Hugging Face model ID for aesthetic scorer model weights.
        hf_hub_aesthetic_model_filename (`str`, *optional*, defaults to `"aesthetic-model.pth"`):
            Hugging Face model filename for aesthetic scorer model weights.
        use_lora (`bool`, *optional*, defaults to `True`):
            Whether to use LoRA.
    """

    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5", metadata={"help": "Pretrained model to use."}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "Pretrained model revision to use."})
    hf_hub_model_id: str = field(
        default="alignprop-finetuned-stable-diffusion", metadata={"help": "HuggingFace repo to save model weights to."}
    )
    hf_hub_aesthetic_model_id: str = field(
        default="trl-lib/ddpo-aesthetic-predictor",
        metadata={"help": "Hugging Face model ID for aesthetic scorer model weights."},
    )
    hf_hub_aesthetic_model_filename: str = field(
        default="aesthetic-model.pth",
        metadata={"help": "Hugging Face model filename for aesthetic scorer model weights."},
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})


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


def image_outputs_logger(image_pair_data, global_step, accelerate_logger):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    images, prompts, _ = [image_pair_data["images"], image_pair_data["prompts"], image_pair_data["rewards"]]
    for i, image in enumerate(images[:4]):
        prompt = prompts[i]
        result[f"{prompt}"] = image.unsqueeze(0).float()
    accelerate_logger.log_images(
        result,
        step=global_step,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, AlignPropConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": "./save",
    }

    pipeline = DefaultDDPOStableDiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )
    trainer = AlignPropTrainer(
        training_args,
        aesthetic_scorer(script_args.hf_hub_aesthetic_model_id, script_args.hf_hub_aesthetic_model_filename),
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
