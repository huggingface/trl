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

import torch

from trl import is_diffusers_available, is_peft_available

from .testing_utils import require_diffusers


if is_diffusers_available() and is_peft_available():
    from trl import AlignPropConfig, AlignPropTrainer, DefaultDDPOStableDiffusionPipeline


def scorer_function(images, prompts, metadata):
    return torch.randn(1) * 3.0, {}


def prompt_function():
    return ("cabbages", {})


@require_diffusers
class AlignPropTrainerTester(unittest.TestCase):
    """
    Test the AlignPropTrainer class.
    """

    def setUp(self):
        self.alignprop_config = AlignPropConfig(
            num_epochs=2,
            train_gradient_accumulation_steps=1,
            train_batch_size=2,
            truncated_backprop_rand=False,
            mixed_precision=None,
            save_freq=1000000,
        )
        pretrained_model = "hf-internal-testing/tiny-stable-diffusion-torch"
        pretrained_revision = "main"

        pipeline = DefaultDDPOStableDiffusionPipeline(
            pretrained_model, pretrained_model_revision=pretrained_revision, use_lora=False
        )

        self.trainer = AlignPropTrainer(self.alignprop_config, scorer_function, prompt_function, pipeline)

        return super().setUp()

    def tearDown(self) -> None:
        gc.collect()

    def test_generate_samples(self):
        output_pairs = self.trainer._generate_samples(2, with_grad=True)
        assert len(output_pairs.keys()) == 3
        assert len(output_pairs["images"]) == 2

    def test_calculate_loss(self):
        sample = self.trainer._generate_samples(2)

        images = sample["images"]
        prompts = sample["prompts"]

        assert images.shape == (2, 3, 128, 128)
        assert len(prompts) == 2

        rewards = self.trainer.compute_rewards(sample)
        loss = self.trainer.calculate_loss(rewards)

        assert torch.isfinite(loss.cpu())


@require_diffusers
class AlignPropTrainerWithLoRATester(AlignPropTrainerTester):
    """
    Test the AlignPropTrainer class.
    """

    def setUp(self):
        self.alignprop_config = AlignPropConfig(
            num_epochs=2,
            train_gradient_accumulation_steps=1,
            mixed_precision=None,
            truncated_backprop_rand=False,
            save_freq=1000000,
        )

        pretrained_model = "hf-internal-testing/tiny-stable-diffusion-torch"
        pretrained_revision = "main"

        pipeline = DefaultDDPOStableDiffusionPipeline(
            pretrained_model, pretrained_model_revision=pretrained_revision, use_lora=True
        )

        self.trainer = AlignPropTrainer(self.alignprop_config, scorer_function, prompt_function, pipeline)

        return super().setUp()
