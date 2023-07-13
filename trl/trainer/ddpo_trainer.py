import contextlib
import datetime
import os
import tempfile
import time
from concurrent import futures
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional

import torch
import tqdm
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from PIL import Image

from . import BaseTrainer, DDPOConfig
from .utils import PerPromptStatTracker


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
logger = get_logger(__name__)


@dataclass
class DDPOPipelineOutput(object):
    images: torch.Tensor
    latents: torch.Tensor
    log_probs: torch.Tensor


@dataclass
class DDPOSchedulerOutput(object):
    # timesteps: torch.Tensor huh?
    latents: torch.Tensor
    log_probs: torch.Tensor


# NOTE: possible problematic area
class DDPOScheduler(DDIMScheduler):
    def step(self, *args) -> DDPOSchedulerOutput:
        raise NotImplementedError


class DDPOStableDiffusionPipeline(StableDiffusionPipeline):
    def __call__(self, **kwargs) -> DDPOPipelineOutput:
        raise NotImplementedError

class DDPOTrainer(BaseTrainer):
    def __init__(
        self,
        config: DDPOConfig,
        reward_function,
        prompt_function,
        sd_pipeline: DDPOStableDiffusionPipeline,
    ):
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        if not config.run_name:
            config.run_name = unique_id
        else:
            config.run_name += "_" + unique_id

        self.prompt_fn = prompt_function  # TODO: move this elsewhere
        self.reward_fn = reward_function  # TODO: move this elsewhere
        self.config = config

        if config.resume_from:
            config.resume_from = os.path.normpath(
                os.path.expanduser(config.resume_from)
            )
            if "checkpoint_" not in os.path.basename(config.resume_from):
                # get the most recent checkpoint in this directory
                checkpoints = list(
                    filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
                )
                if len(checkpoints) == 0:
                    raise ValueError(f"No checkpoints found in {config.resume_from}")
                config.resume_from = os.path.join(
                    config.resume_from,
                    sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
                )

        # number of timesteps within each trajectory to train on
        self.num_train_timesteps = int(
            config.sample_num_steps * config.train_timestep_fraction
        )

        accelerator_config = ProjectConfiguration(
            project_dir=os.path.join(self.config.logdir, self.config.run_name),
            automatic_checkpoint_naming=True,
            total_limit=config.num_checkpoint_limit,
        )

        self.accelerator = Accelerator(
            log_with="wandb",
            mixed_precision=config.mixed_precision,
            project_config=accelerator_config,
            # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
            # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
            # the total number of optimizer steps to accumulate across.
            gradient_accumulation_steps=config.train_gradient_accumulation_steps
            * self.num_train_timesteps,
        )

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name="ddpo-pytorch",
                config=asdict(config),
                init_kwargs={"wandb": {"name": config.run_name}},
            )

        logger.info(f"\n{config}")

        # set seed (device_specific is very important to get different prompts on different devices)
        set_seed(self.config.seed, device_specific=True)

        # load scheduler, tokenizer and models. (NOTE: should the user do this?)
        self.sd_pipeline = sd_pipeline
        # freeze parameters of models to save more memory (NOTE: should the user do this?)
        self.sd_pipeline.vae.requires_grad_(False)
        # NOTE: should the user do this?
        self.sd_pipeline.text_encoder.requires_grad_(False)
        # NOTE: should the user do this?
        self.sd_pipeline.unet.requires_grad_(not config.use_lora)
        # disable safety checker (NOTE: should the user do this?)
        self.sd_pipeline.safety_checker = None
        # make the progress bar nicer
        self.sd_pipeline.set_progress_bar_config(
            position=1,
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )
        # switch to DDIM scheduler (NOTE: THE USER SHOULD DO THIS)
        # self.sd_pipeline_wrapper.pipeline.scheduler = DDIMScheduler.from_config(
        #    self.pipeline.scheduler.config
        # )

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        inference_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to inference_dtype
        self.sd_pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.text_encoder.to(self.accelerator.device, dtype=inference_dtype)
        if config.use_lora:
            self.sd_pipeline.unet.to(self.accelerator.device, dtype=inference_dtype)

        if config.use_lora:
            # Set correct lora layers
            lora_attn_procs = {}
            for name in self.sd_pipeline.unet.attn_processors.keys():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else self.sd_pipeline.unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = self.sd_pipeline.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(
                        reversed(self.sd_pipeline.unet.config.block_out_channels)
                    )[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.sd_pipeline.unet.config.block_out_channels[
                        block_id
                    ]

                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                )
            self.sd_pipeline.unet.set_attn_processor(lora_attn_procs)
            trainable_layers = AttnProcsLayers(self.sd_pipeline.unet.attn_processors)
        else:
            trainable_layers = self.sd_pipeline.unet

        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Initialize the optimizer
        if self.config.train_use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            trainable_layers.parameters(),
            lr=config.train_learning_rate,
            betas=(config.train_adam_beta1, config.train_adam_beta2),
            weight_decay=config.train_adam_weight_decay,
            eps=config.train_adam_epsilon,
        )

        neg_prompt_embed = self.sd_pipeline.text_encoder(
            self.sd_pipeline.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
        )[0]
        self.sample_neg_prompt_embeds = neg_prompt_embed.repeat(
            config.sample_batch_size, 1, 1
        )
        self.train_neg_prompt_embeds = neg_prompt_embed.repeat(
            config.train_batch_size, 1, 1
        )

        # initialize stat tracker
        if config.per_prompt_stat_tracking:
            self.stat_tracker = PerPromptStatTracker(
                config.per_prompt_stat_tracking_buffer_size,
                config.per_prompt_stat_tracking_min_count,
            )

        # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
        # more memory
        self.autocast = (
            contextlib.nullcontext if config.use_lora else self.accelerator.autocast
        )

        # Prepare everything with our `accelerator`.
        self.trainable_layers, self.optimizer = self.accelerator.prepare(
            trainable_layers, self.optimizer
        )

        # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
        # remote server running llava inference.
        self.executor = futures.ThreadPoolExecutor(max_workers=2)

        # Train!
        self.samples_per_epoch = (
            config.sample_batch_size
            * self.accelerator.num_processes
            * config.sample_num_batches_per_epoch
        )
        self.total_train_batch_size = (
            config.train_batch_size
            * self.accelerator.num_processes
            * config.train_gradient_accumulation_steps
        )

    def _save_model_hook(self, models, weights, output_dir):
        assert len(models) == 1
        if self.config.use_lora and isinstance(models[0], AttnProcsLayers):
            self.sd_pipeline.unet.save_attn_procs(output_dir)
        elif not self.config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    # throw this on to the user?
    def _load_model_hook(self, models, input_dir):
        assert len(models) == 1
        if self.config.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            # THOUGHT: this should be taken care of by the user
            tmp_unet = UNet2DConditionModel.from_pretrained(
                self.config.pretrained_model,
                revision=self.config.pretrained_revision,
                subfolder="unet",
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(
                AttnProcsLayers(tmp_unet.attn_processors).state_dict()
            )
            del tmp_unet
        elif not self.config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    def step(self, epoch: int, global_step: int):
        self.sd_pipeline.unet.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(self.config.sample_num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not self.accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts, prompt_metadata = zip(
                *[
                    self.prompt_fn() for _ in range(self.config.sample_batch_size)
                ]
            )

            # encode prompts
            prompt_ids = self.sd_pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
            prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]

            # sample
            with self.autocast():
                sd_output = self.sd_pipeline(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=self.sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    output_type="pt",
                )

                images = sd_output.images
                latents = sd_output.latents
                log_probs = sd_output.log_probs

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 4, 64, 64)
            # (batch_size, num_steps, 1)
            log_probs = torch.stack(log_probs, dim=1)
            timesteps = self.sd_pipeline.scheduler.timesteps.repeat(
                self.config.sample_batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = self.executor.submit(
                self.reward_fn, images, prompts, prompt_metadata
            )
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not self.accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = torch.as_tensor(rewards, device=self.accelerator.device)

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        # gather rewards across processes
        rewards = self.accelerator.gather(samples["rewards"]).cpu().numpy()

        # log rewards and images
        self.accelerator.log(
            {
                "reward": rewards,
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=global_step,
        )
        # Do we need this hack?
        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray(
                    (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            self.accelerator.log(
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

        # per-prompt mean/std tracking
        if self.config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = self.accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = self.sd_pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = self.stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]
            .to(self.accelerator.device)
        )

        del samples["rewards"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (
            total_batch_size
            == self.config.sample_batch_size * self.config.sample_num_batches_per_epoch
        )
        assert num_timesteps == self.config.sample_num_steps

        #################### TRAINING ####################
        for inner_epoch in range(self.config.train_num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=self.accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=self.accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=self.accelerator.device)[
                        :, None
                    ],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, self.config.train_batch_size, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            self.sd_pipeline.unet.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not self.accelerator.is_local_main_process,
            ):
                if self.config.train_cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [self.train_neg_prompt_embeds, sample["prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]

                for j in tqdm(
                    range(self.num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not self.accelerator.is_local_main_process,
                ):
                    with self.accelerator.accumulate(self.sd_pipeline.unet):
                        with self.autocast():
                            if self.config.train_cfg:
                                noise_pred = self.sd_pipeline.unet(
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = (
                                    noise_pred_uncond
                                    + self.config.sample_guidance_scale
                                    * (noise_pred_text - noise_pred_uncond)
                                )
                            else:
                                noise_pred = self.sd_pipeline.unet(
                                    sample["latents"][:, j],
                                    sample["timesteps"][:, j],
                                    embeds,
                                ).sample
                            # compute the log prob of next_latents given latents under the current model

                            # this should be setup somewhere
                            scheduler_step_output = self.sd_pipeline.scheduler.step(
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=self.config.sample_eta,
                                prev_sample=sample["next_latents"][:, j],
                            )

                            log_prob = scheduler_step_output.log_probs

                        advantages = torch.clamp(
                            sample["advantages"],
                            -self.config.train_adv_clip_max,
                            self.config.train_adv_clip_max,
                        )

                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])

                        loss = self.loss(sample, self.config.train_clip_range, ratio)

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0)
                                    > self.config.train_clip_range
                                ).float()
                            )
                        )
                        info["loss"].append(loss)

                        # backward pass
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.trainable_layers.parameters(),
                                self.config.train_max_grad_norm,
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if self.accelerator.sync_gradients:
                        assert (j == self.num_train_timesteps - 1) and (
                            i + 1
                        ) % self.config.train_gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = self.accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        self.accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert self.accelerator.sync_gradients

        if (
            epoch != 0
            and epoch % self.config.save_freq == 0
            and self.accelerator.is_main_process
        ):
            self.accelerator.save_state()

    def loss(
        self,
        advantages: torch.Tensor,
        clip_range: float,
        ratio: torch.Tensor,
    ):
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )
        return torch.mean(torch.maximum(unclipped_loss, clipped_loss))
