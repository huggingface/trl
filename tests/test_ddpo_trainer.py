import unittest
import numpy as np
import gc

from trl import DDPOTrainer, DDPOConfig, DefaultDDPOPipeline, DefaultDDPOScheduler

def scorer_function(images, prompts, metadata):
    return np.random.randint(6), {}

def prompt_function():
    return ("cabbages", {})

import torch


class DDPOTrainerTester(unittest.TestCase):
    """
    Test the DDPOTrainer class.
    """

    def setUp(self):
        self.ddpo_config = DDPOConfig(
            num_epochs=200,
            train_gradient_accumulation_steps=1,
            per_prompt_stat_tracking_buffer_size=32,
            sample_num_batches_per_epoch=2,
            sample_batch_size=1
        )
        pretrained_model = "runwayml/stable-diffusion-v1-5"
        # revision of the model to load.
        pretrained_revision = "main"

        pipeline = DefaultDDPOPipeline.from_pretrained(pretrained_model, revision=pretrained_revision)
        pipeline.scheduler = DefaultDDPOScheduler.from_config(pipeline.scheduler.config)

        self.trainer = DDPOTrainer(
            self.ddpo_config,
            scorer_function,
            prompt_function,
            pipeline)

        return super().setUp()

    def tearDown(self) -> None:
        gc.collect()

    def test_loss(self):
        self.trainer.loss()

        pass





