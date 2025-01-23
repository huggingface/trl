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

import gc
import unittest

import torch
from parameterized import parameterized
from transformers.utils import is_peft_available

from trl import is_diffusers_available

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
        training_args = AlignPropConfig(
            num_epochs=2,
            train_gradient_accumulation_steps=1,
            train_batch_size=2,
            truncated_backprop_rand=False,
            mixed_precision=None,
            save_freq=1000000,
        )
        pretrained_model = "hf-internal-testing/tiny-stable-diffusion-torch"
        pretrained_revision = "main"
        pipeline_with_lora = DefaultDDPOStableDiffusionPipeline(
            pretrained_model, pretrained_model_revision=pretrained_revision, use_lora=True
        )
        pipeline_without_lora = DefaultDDPOStableDiffusionPipeline(
            pretrained_model, pretrained_model_revision=pretrained_revision, use_lora=False
        )
        self.trainer_with_lora = AlignPropTrainer(training_args, scorer_function, prompt_function, pipeline_with_lora)
        self.trainer_without_lora = AlignPropTrainer(
            training_args, scorer_function, prompt_function, pipeline_without_lora
        )

    def tearDown(self) -> None:
        gc.collect()

    @parameterized.expand([True, False])
    def test_generate_samples(self, use_lora):
        trainer = self.trainer_with_lora if use_lora else self.trainer_without_lora
        output_pairs = trainer._generate_samples(2, with_grad=True)
        self.assertEqual(len(output_pairs.keys()), 3)
        self.assertEqual(len(output_pairs["images"]), 2)

    @parameterized.expand([True, False])
    def test_calculate_loss(self, use_lora):
        trainer = self.trainer_with_lora if use_lora else self.trainer_without_lora
        sample = trainer._generate_samples(2)

        images = sample["images"]
        prompts = sample["prompts"]

        self.assertTupleEqual(images.shape, (2, 3, 128, 128))
        self.assertEqual(len(prompts), 2)

        rewards = trainer.compute_rewards(sample)
        loss = trainer.calculate_loss(rewards)

        self.assertTrue(torch.isfinite(loss.cpu()))
