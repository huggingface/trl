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
import gc
import unittest
from dataclasses import dataclass

import torch

from trl import DDPOConfig, DDPOTrainer, DefaultDDPOPipeline, DefaultDDPOScheduler


def scorer_function(images, prompts, metadata):
    return torch.randn(1) * 3.0, {}


def prompt_function():
    return ("cabbages", {})


@dataclass
class DummyConfig(object):
    sample_size: int = 64
    in_channels: int = 4


@dataclass
class DummyOutput(object):
    sample: torch.Tensor


class DummyUnet(torch.nn.Module):
    def __init__(self):
        super(DummyUnet, self).__init__()
        # Create a dummy parameter so that this module has parameters and can have gradients.
        self.dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        self.config = DummyConfig()

    def forward(self, x):
        # Create a tensor filled with ones (or any other value or randomness you wish) of the desired shape
        output = torch.ones((x.shape[0], 4, 64, 64), requires_grad=True)
        output = output.to(x.device)

        # Multiply the output by the dummy parameter to ensure backward passes can flow through this module
        return output * self.dummy_param

    def __call__(self, x, *args, return_dict=True, **kwargs):
        output = self.forward(x)
        if return_dict:
            return DummyOutput(output)
        return (output,)

    @property
    def device(self):
        return next(self.parameters()).device


class DDPOTrainerTester(unittest.TestCase):
    """
    Test the DDPOTrainer class.
    """

    def setUp(self):
        self.ddpo_config = DDPOConfig(
            num_epochs=2,
            train_gradient_accumulation_steps=1,
            per_prompt_stat_tracking_buffer_size=32,
            sample_num_batches_per_epoch=2,
            sample_batch_size=2,
            mixed_precision=None,
            save_freq=1000000,
            use_lora=False,
        )
        pretrained_model = "hf-internal-testing/tiny-stable-diffusion-torch"
        pretrained_revision = "main"

        pipeline = DefaultDDPOPipeline.from_pretrained(pretrained_model, revision=pretrained_revision)
        # pipeline.unet = DummyUnet()
        pipeline.scheduler = DefaultDDPOScheduler.from_config(pipeline.scheduler.config)

        self.trainer = DDPOTrainer(self.ddpo_config, scorer_function, prompt_function, pipeline)

        return super().setUp()

    def tearDown(self) -> None:
        gc.collect()

    def test_loss(self):
        advantage = torch.tensor([-1.0])
        clip_range = 0.0001
        ratio = torch.tensor([1.0])
        loss = self.trainer.loss(advantage, clip_range, ratio)
        self.assertEqual(loss.item(), 1.0)

    def test_generate_samples(self):
        samples, output_pairs = self.trainer._generate_samples(1, 2)
        self.assertEqual(len(samples), 1)
        self.assertEqual(len(output_pairs), 1)
        self.assertEqual(len(output_pairs[0][0]), 2)

    def test_calculate_loss(self):
        samples, _ = self.trainer._generate_samples(1, 2)
        sample = samples[0]

        latents = sample["latents"][0, 0].unsqueeze(0)
        next_latents = sample["next_latents"][0, 0].unsqueeze(0)
        log_probs = sample["log_probs"][0, 0].unsqueeze(0)
        timesteps = sample["timesteps"][0, 0].unsqueeze(0)
        prompt_embeds = sample["prompt_embeds"]
        advantage = torch.tensor([1.0], device=prompt_embeds.device)

        self.assertEqual(latents.shape, (1, 4, 64, 64))
        self.assertEqual(next_latents.shape, (1, 4, 64, 64))
        self.assertEqual(log_probs.shape, (1,))
        self.assertEqual(timesteps.shape, (1,))
        self.assertEqual(prompt_embeds.shape, (2, 77, 32))
        loss, approx_kl, clipfrac = self.trainer.calculate_loss(
            latents, timesteps, next_latents, log_probs, advantage, prompt_embeds
        )

        self.assertTrue(torch.isclose(loss.cpu(), torch.tensor([-0.9994]), 1e-04))
