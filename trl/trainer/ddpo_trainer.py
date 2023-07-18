# NOTE: code heavily inspired by https://github.com/kvablack/ddpo-pytorch

import contextlib
import datetime
import os
from collections import defaultdict
from concurrent import futures
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

from . import BaseTrainer, DDPOConfig
from .utils import PerPromptStatTracker


logger = get_logger(__name__)


@dataclass
class DDPOPipelineOutput(object):
    images: torch.Tensor
    latents: torch.Tensor
    log_probs: torch.Tensor


@dataclass
class DDPOSchedulerOutput(object):
    latents: torch.Tensor
    log_probs: torch.Tensor


class DDPOStableDiffusionPipeline(StableDiffusionPipeline):
    def __call__(self, *args, **kwargs) -> DDPOPipelineOutput:
        raise NotImplementedError

    def scheduler_step(self, *args, **kwargs) -> DDPOSchedulerOutput:
        raise NotImplementedError


class DDPOTrainer(BaseTrainer):
    def __init__(
        self,
        config: DDPOConfig,
        reward_function: Callable[[torch.Tensor, Tuple[str], Tuple[Any]], torch.Tensor],
        prompt_function: Callable[[], Tuple[str, Any]],
        sd_pipeline: DDPOStableDiffusionPipeline,
        image_outputs_hook: Callable[[Any, Any, Any, int], Any],  # should make a default for this
    ):
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        if not config.run_name:
            config.run_name = unique_id
        else:
            config.run_name += "_" + unique_id

        self.prompt_fn = prompt_function
        self.reward_fn = reward_function
        self.config = config
        self.image_outputs_callback = image_outputs_hook

        if self.config.resume_from:
            self.config.resume_from = os.path.normpath(os.path.expanduser(self.config.resume_from))
            if "checkpoint_" not in os.path.basename(self.config.resume_from):
                # get the most recent checkpoint in this directory
                checkpoints = list(
                    filter(
                        lambda x: "checkpoint_" in x,
                        os.listdir(self.config.resume_from),
                    )
                )
                if len(checkpoints) == 0:
                    raise ValueError(f"No checkpoints found in {self.config.resume_from}")
                self.config.resume_from = os.path.join(
                    self.config.resume_from,
                    sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
                )

        # number of timesteps within each trajectory to train on
        self.num_train_timesteps = int(self.config.sample_num_steps * self.config.train_timestep_fraction)

        accelerator_config = ProjectConfiguration(
            project_dir=os.path.join(self.config.logdir, self.config.run_name),
            automatic_checkpoint_naming=True,
            total_limit=config.num_checkpoint_limit,
        )

        self.accelerator = Accelerator(
            log_with=self.config.log_with,
            mixed_precision=self.config.mixed_precision,
            project_config=accelerator_config,
            # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
            # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
            # the total number of optimizer steps to accumulate across.
            gradient_accumulation_steps=self.config.train_gradient_accumulation_steps * self.num_train_timesteps,
        )

        self._config_check()

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
                    None if name.endswith("attn1.processor") else self.sd_pipeline.unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = self.sd_pipeline.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.sd_pipeline.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.sd_pipeline.unet.config.block_out_channels[block_id]

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

        self.optimizer = self._initialize_optimizer(trainable_layers.parameters())

        neg_prompt_embed = self.sd_pipeline.text_encoder(
            self.sd_pipeline.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
        )[0]
        self.sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample_batch_size, 1, 1)
        self.train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train_batch_size, 1, 1)

        # initialize stat tracker
        if config.per_prompt_stat_tracking:
            self.stat_tracker = PerPromptStatTracker(
                config.per_prompt_stat_tracking_buffer_size,
                config.per_prompt_stat_tracking_min_count,
            )

        # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
        # more memory
        self.autocast = contextlib.nullcontext if config.use_lora else self.accelerator.autocast

        # Prepare everything with our `accelerator`.
        self.trainable_layers, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)

        # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
        # remote server running llava inference.
        if self.config.async_reward_computation:
            self.executor = futures.ThreadPoolExecutor(max_workers=2)

        if config.resume_from:
            logger.info(f"Resuming from {config.resume_from}")
            self.accelerator.load_state(config.resume_from)
            self.first_epoch = int(config.resume_from.split("_")[-1]) + 1
        else:
            self.first_epoch = 0

    def compute_rewards(self, prompt_image_pairs, is_async=False):
        # if not async no need to loop twice
        if not is_async:
            rewards = []
            for images, prompts, prompt_metadata in prompt_image_pairs:
                reward, reward_metadata = self.reward_fn(images, prompts, prompt_metadata)
                print(reward)
                rewards.append(torch.as_tensor(reward, device=self.accelerator.device))
        else:
            # submit all jobs
            rewards = self.executor.map(lambda x: self.reward_fn(*x), prompt_image_pairs)
            # wait for all jobs to finish
            rewards = [
                torch.as_tensor(reward.result(), device=self.accelerator.device) for reward, _ in rewards
            ]  # ignoring metadata
        rewards = torch.cat(rewards)  # problematic area
        return self.accelerator.gather(rewards)

    def step(self, epoch: int, global_step: int):
        samples, prompt_image_pairs = self._generate_samples(
            iterations=self.config.sample_num_batches_per_epoch,
            batch_size=self.config.sample_batch_size,
            is_async=self.config.async_reward_computation,
        )

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        rewards = self.compute_rewards(prompt_image_pairs, is_async=self.config.async_reward_computation)

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
        self.image_outputs_callback(prompt_image_pairs, rewards, global_step, self.accelerator.log)

        # per-prompt mean/std tracking
        if self.config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = self.accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = self.sd_pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = self.stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]
            .to(self.accelerator.device)
        )

        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert total_batch_size == self.config.sample_batch_size * self.config.sample_num_batches_per_epoch
        assert num_timesteps == self.config.sample_num_steps

        for inner_epoch in range(self.config.train_num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=self.accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [torch.randperm(num_timesteps, device=self.accelerator.device) for _ in range(total_batch_size)]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=self.accelerator.device)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, self.config.train_batch_size, *v.shape[1:]) for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            # train
            self.sd_pipeline.unet.train()
            global_step = self._train_batched_samples((global_step, epoch, inner_epoch), samples_batched)
            # make sure we did an optimization step at the end of the inner epoch
            assert self.accelerator.sync_gradients

        if epoch != 0 and epoch % self.config.save_freq == 0 and self.accelerator.is_main_process:
            self.accelerator.save_state()

        return global_step

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

    def _initialize_optimizer(self, trainable_layers_parameters):
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

        return optimizer_cls(
            trainable_layers_parameters,
            lr=self.config.train_learning_rate,
            betas=(self.config.train_adam_beta1, self.config.train_adam_beta2),
            weight_decay=self.config.train_adam_weight_decay,
            eps=self.config.train_adam_epsilon,
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
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not self.config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    def _generate_samples(self, iterations, batch_size, is_async=False):
        samples = []
        prompt_image_pairs = []
        self.sd_pipeline.unet.eval()
        for _ in range(iterations):
            # generate prompts
            prompts, prompt_metadata = zip(*[self.prompt_fn() for _ in range(batch_size)])

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

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
            # (batch_size, num_steps, 1)
            log_probs = torch.stack(log_probs, dim=1)
            timesteps = self.sd_pipeline.scheduler.timesteps.repeat(
                self.config.sample_batch_size, 1
            )  # (batch_size, num_steps)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # each entry is the latent before timestep t
                    "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                }
            )
            prompt_image_pairs.append((images, prompts, prompt_metadata))

        return samples, prompt_image_pairs

    def _train_batched_samples(self, step_coordinates, batched_samples):
        global_step, epoch, inner_epoch = step_coordinates
        info = defaultdict(list)
        for i, sample in enumerate(batched_samples):
            if self.config.train_cfg:
                # concat negative prompts to sample prompts to avoid two forward passes
                embeds = torch.cat([self.train_neg_prompt_embeds, sample["prompt_embeds"]])
            else:
                embeds = sample["prompt_embeds"]

            for j in range(self.num_train_timesteps):
                with self.accelerator.accumulate(self.sd_pipeline.unet):
                    with self.autocast():
                        # latents, timesteps, next_latents, advantages
                        if self.config.train_cfg:
                            noise_pred = self.sd_pipeline.unet(
                                torch.cat([sample["latents"][:, j]] * 2),
                                torch.cat([sample["timesteps"][:, j]] * 2),
                                embeds,
                            ).sample
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.config.sample_guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )
                        else:
                            noise_pred = self.sd_pipeline.unet(
                                sample["latents"][:, j],
                                sample["timesteps"][:, j],
                                embeds,
                            ).sample
                        # compute the log prob of next_latents given latents under the current model

                        # this should be setup somewhere
                        scheduler_step_output = self.sd_pipeline.scheduler_step(
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

                    loss = self.loss(advantages, self.config.train_clip_range, ratio)

                    # debugging values
                    # John Schulman says that (ratio - 1) - log(ratio) is a better
                    # estimator, but most existing code uses this so...
                    # http://joschu.net/blog/kl-approx.html
                    info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                    info["clipfrac"].append(
                        torch.mean((torch.abs(ratio - 1.0) > self.config.train_clip_range).float())
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
        return global_step

    def _config_check(self):
        # TODO: make this IO free
        samples_per_epoch = (
            self.config.sample_batch_size * self.accelerator.num_processes * self.config.sample_num_batches_per_epoch
        )
        total_train_batch_size = (
            self.config.train_batch_size
            * self.accelerator.num_processes
            * self.config.train_gradient_accumulation_steps
        )

        assert self.config.sample_batch_size >= self.config.train_batch_size
        assert self.config.sample_batch_size % self.config.train_batch_size == 0
        assert samples_per_epoch % total_train_batch_size == 0
        return

    def run(self, epochs: Optional[int] = None):
        global_step = 0
        if epochs is None:
            epochs = self.config.num_epochs
        for epoch in range(self.first_epoch, epochs):
            global_step = self.step(epoch, global_step)
