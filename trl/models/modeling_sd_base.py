# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import contextlib
import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from transformers.utils import is_peft_available

from ..core import randn_tensor
from .sd_utils import convert_state_dict_to_diffusers


if is_peft_available():
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict


@dataclass
class DDPOPipelineOutput:
    """
    Output class for the diffusers pipeline to be finetuned with the DDPO trainer

    Args:
        images (`torch.Tensor`):
            The generated images.
        latents (`list[torch.Tensor]`):
            The latents used to generate the images.
        log_probs (`list[torch.Tensor]`):
            The log probabilities of the latents.

    """

    images: torch.Tensor
    latents: torch.Tensor
    log_probs: torch.Tensor


@dataclass
class DDPOSchedulerOutput:
    """
    Output class for the diffusers scheduler to be finetuned with the DDPO trainer

    Args:
        latents (`torch.Tensor`):
            Predicted sample at the previous timestep. Shape: `(batch_size, num_channels, height, width)`
        log_probs (`torch.Tensor`):
            Log probability of the above mentioned sample. Shape: `(batch_size)`
    """

    latents: torch.Tensor
    log_probs: torch.Tensor


class DDPOStableDiffusionPipeline:
    """
    Main class for the diffusers pipeline to be finetuned with the DDPO trainer
    """

    def __call__(self, *args, **kwargs) -> DDPOPipelineOutput:
        raise NotImplementedError

    def scheduler_step(self, *args, **kwargs) -> DDPOSchedulerOutput:
        raise NotImplementedError

    @property
    def unet(self):
        """
        Returns the 2d U-Net model used for diffusion.
        """
        raise NotImplementedError

    @property
    def vae(self):
        """
        Returns the Variational Autoencoder model used from mapping images to and from the latent space
        """
        raise NotImplementedError

    @property
    def tokenizer(self):
        """
        Returns the tokenizer used for tokenizing text inputs
        """
        raise NotImplementedError

    @property
    def scheduler(self):
        """
        Returns the scheduler associated with the pipeline used for the diffusion process
        """
        raise NotImplementedError

    @property
    def text_encoder(self):
        """
        Returns the text encoder used for encoding text inputs
        """
        raise NotImplementedError

    @property
    def autocast(self):
        """
        Returns the autocast context manager
        """
        raise NotImplementedError

    def set_progress_bar_config(self, *args, **kwargs):
        """
        Sets the progress bar config for the pipeline
        """
        raise NotImplementedError

    def save_pretrained(self, *args, **kwargs):
        """
        Saves all of the model weights
        """
        raise NotImplementedError

    def get_trainable_layers(self, *args, **kwargs):
        """
        Returns the trainable parameters of the pipeline
        """
        raise NotImplementedError

    def save_checkpoint(self, *args, **kwargs):
        """
        Light wrapper around accelerate's register_save_state_pre_hook which is run before saving state
        """
        raise NotImplementedError

    def load_checkpoint(self, *args, **kwargs):
        """
        Light wrapper around accelerate's register_lad_state_pre_hook which is run before loading state
        """
        raise NotImplementedError


def _left_broadcast(input_tensor, shape):
    """
    As opposed to the default direction of broadcasting (right to left), this function broadcasts
    from left to right
        Args:
            input_tensor (`torch.FloatTensor`): is the tensor to broadcast
            shape (`tuple[int]`): is the shape to broadcast to
    """
    input_ndim = input_tensor.ndim
    if input_ndim > len(shape):
        raise ValueError(
            "The number of dimensions of the tensor to broadcast cannot be greater than the length of the shape to broadcast to"
        )
    return input_tensor.reshape(input_tensor.shape + (1,) * (len(shape) - input_ndim)).broadcast_to(shape)


def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu()).to(timestep.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance


def scheduler_step(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
) -> DDPOSchedulerOutput:
    """

    Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
            predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
            `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
            coincide with the one provided as input and `use_clipped_model_output` will have not effect.
        generator: random number generator.
        variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
            can directly provide the noise for the variance itself. This is useful for methods such as
            CycleDiffusion. (https://huggingface.co/papers/2210.05559)

    Returns:
        `DDPOSchedulerOutput`: the predicted sample at the previous timestep and the log probability of the sample
    """

    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://huggingface.co/papers/2010.02502
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    # to prevent OOB on gather
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device)

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://huggingface.co/papers/2010.02502
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = _get_variance(self, timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://huggingface.co/papers/2010.02502
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://huggingface.co/papers/2010.02502
    prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    # log prob of prev_sample given prev_sample_mean and std_dev_t
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(np.pi)))
    )
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return DDPOSchedulerOutput(prev_sample.type(sample.dtype), log_prob)


# 1. The output type for call is different as the logprobs are now returned
# 2. An extra method called `scheduler_step` is added which is used to constraint the scheduler output
@torch.no_grad()
def pipeline_step(
    self,
    prompt: Optional[Union[str, list[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, list[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
):
    r"""
    Function invoked when calling the pipeline for generation.  Args: prompt (`str` or `list[str]`, *optional*): The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.  instead.  height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor): The height in pixels of the generated image.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        negative_prompt (`str` or `list[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        guidance_rescale (`float`, *optional*, defaults to 0.7):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.

    Examples:

    Returns:
        `DDPOPipelineOutput`: The generated image, the predicted latents used to generate the image and the associated log probabilities
    """
    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    all_latents = [latents]
    all_log_probs = []
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://huggingface.co/papers/2305.08891
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = scheduler_step(self.scheduler, noise_pred, t, latents, eta)
            latents = scheduler_output.latents
            log_prob = scheduler_output.log_probs

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if not output_type == "latent":
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    return DDPOPipelineOutput(image, all_latents, all_log_probs)


def pipeline_step_with_grad(
    pipeline,
    prompt: Optional[Union[str, list[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    truncated_backprop: bool = True,
    truncated_backprop_rand: bool = True,
    gradient_checkpoint: bool = True,
    truncated_backprop_timestep: int = 49,
    truncated_rand_backprop_minmax: tuple = (0, 50),
    negative_prompt: Optional[Union[str, list[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
):
    r"""
    Function to get RGB image with gradients attached to the model weights.

    Args:
        prompt (`str` or `list[str]`, *optional*, defaults to `None`):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds` instead.
        height (`int`, *optional*, defaults to `pipeline.unet.config.sample_size * pipeline.vae_scale_factor`):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to `pipeline.unet.config.sample_size * pipeline.vae_scale_factor`):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to `50`):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to `7.5`):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://huggingface.co/papers/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        truncated_backprop (`bool`, *optional*, defaults to True):
            Truncated Backpropation to fixed timesteps, helps prevent collapse during diffusion reward training as shown in AlignProp (https://huggingface.co/papers/2310.03739).
        truncated_backprop_rand (`bool`, *optional*, defaults to True):
            Truncated Randomized Backpropation randomizes truncation to different diffusion timesteps, this helps prevent collapse during diffusion reward training as shown in AlignProp (https://huggingface.co/papers/2310.03739).
            Enabling truncated_backprop_rand allows adapting earlier timesteps in diffusion while not resulting in a collapse.
        gradient_checkpoint (`bool`, *optional*, defaults to True):
            Adds gradient checkpointing to Unet forward pass. Reduces GPU memory consumption while slightly increasing the training time.
        truncated_backprop_timestep (`int`, *optional*, defaults to 49):
            Absolute timestep to which the gradients are being backpropagated. Higher number reduces the memory usage and reduces the chances of collapse.
            While a lower value, allows more semantic changes in the diffusion generations, as the earlier diffusion timesteps are getting updated.
            However it also increases the chances of collapse.
        truncated_rand_backprop_minmax (`Tuple`, *optional*, defaults to (0,50)):
            Range for randomized backprop. Here the value at 0 index indicates the earlier diffusion timestep to update (closer to noise), while the value
            at index 1 indicates the later diffusion timestep to update.
        negative_prompt (`str` or `list[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `pipeline.processor` in
            [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        guidance_rescale (`float`, *optional*, defaults to 0.7):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.

    Examples:

    Returns:
        `DDPOPipelineOutput`: The generated image, the predicted latents used to generate the image and the associated log probabilities
    """
    # 0. Default height and width to unet
    height = height or pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = width or pipeline.unet.config.sample_size * pipeline.vae_scale_factor

    with torch.no_grad():
        # 1. Check inputs. Raise error if not correct
        pipeline.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = pipeline._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = pipeline._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipeline.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
    # 6. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
    all_latents = [latents]
    all_log_probs = []
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            if gradient_checkpoint:
                noise_pred = checkpoint.checkpoint(
                    pipeline.unet,
                    latent_model_input,
                    t,
                    prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    use_reentrant=False,
                )[0]
            else:
                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

            #  truncating backpropagation is critical for preventing overoptimization (https://huggingface.co/papers/2304.05977).
            if truncated_backprop:
                # Randomized truncation randomizes the truncation process (https://huggingface.co/papers/2310.03739)
                # the range of truncation is defined by truncated_rand_backprop_minmax
                # Setting truncated_rand_backprop_minmax[0] to be low will allow the model to update earlier timesteps in the diffusion chain, while setitng it high will reduce the memory usage.
                if truncated_backprop_rand:
                    rand_timestep = random.randint(
                        truncated_rand_backprop_minmax[0], truncated_rand_backprop_minmax[1]
                    )
                    if i < rand_timestep:
                        noise_pred = noise_pred.detach()
                else:
                    # fixed truncation process
                    if i < truncated_backprop_timestep:
                        noise_pred = noise_pred.detach()

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://huggingface.co/papers/2305.08891
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = scheduler_step(pipeline.scheduler, noise_pred, t, latents, eta)
            latents = scheduler_output.latents
            log_prob = scheduler_output.log_probs

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if not output_type == "latent":
        image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        image, has_nsfw_concept = pipeline.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = pipeline.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload last model to CPU
    if hasattr(pipeline, "final_offload_hook") and pipeline.final_offload_hook is not None:
        pipeline.final_offload_hook.offload()

    return DDPOPipelineOutput(image, all_latents, all_log_probs)


class DefaultDDPOStableDiffusionPipeline(DDPOStableDiffusionPipeline):
    def __init__(self, pretrained_model_name: str, *, pretrained_model_revision: str = "main", use_lora: bool = True):
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name, revision=pretrained_model_revision
        )

        self.use_lora = use_lora
        self.pretrained_model = pretrained_model_name
        self.pretrained_revision = pretrained_model_revision

        try:
            self.sd_pipeline.load_lora_weights(
                pretrained_model_name,
                weight_name="pytorch_lora_weights.safetensors",
                revision=pretrained_model_revision,
            )
            self.use_lora = True
        except OSError:
            if use_lora:
                warnings.warn(
                    "Trying to load LoRA weights but no LoRA weights found. Set `use_lora=False` or check that "
                    "`pytorch_lora_weights.safetensors` exists in the model folder.",
                    UserWarning,
                )

        self.sd_pipeline.scheduler = DDIMScheduler.from_config(self.sd_pipeline.scheduler.config)
        self.sd_pipeline.safety_checker = None

        # memory optimization
        self.sd_pipeline.vae.requires_grad_(False)
        self.sd_pipeline.text_encoder.requires_grad_(False)
        self.sd_pipeline.unet.requires_grad_(not self.use_lora)

    def __call__(self, *args, **kwargs) -> DDPOPipelineOutput:
        return pipeline_step(self.sd_pipeline, *args, **kwargs)

    def rgb_with_grad(self, *args, **kwargs) -> DDPOPipelineOutput:
        return pipeline_step_with_grad(self.sd_pipeline, *args, **kwargs)

    def scheduler_step(self, *args, **kwargs) -> DDPOSchedulerOutput:
        return scheduler_step(self.sd_pipeline.scheduler, *args, **kwargs)

    @property
    def unet(self):
        return self.sd_pipeline.unet

    @property
    def vae(self):
        return self.sd_pipeline.vae

    @property
    def tokenizer(self):
        return self.sd_pipeline.tokenizer

    @property
    def scheduler(self):
        return self.sd_pipeline.scheduler

    @property
    def text_encoder(self):
        return self.sd_pipeline.text_encoder

    @property
    def autocast(self):
        return contextlib.nullcontext if self.use_lora else None

    def save_pretrained(self, output_dir):
        if self.use_lora:
            state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.sd_pipeline.unet))
            self.sd_pipeline.save_lora_weights(save_directory=output_dir, unet_lora_layers=state_dict)
        self.sd_pipeline.save_pretrained(output_dir)

    def set_progress_bar_config(self, *args, **kwargs):
        self.sd_pipeline.set_progress_bar_config(*args, **kwargs)

    def get_trainable_layers(self):
        if self.use_lora:
            lora_config = LoraConfig(
                r=4,
                lora_alpha=4,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.sd_pipeline.unet.add_adapter(lora_config)

            # To avoid accelerate unscaling problems in FP16.
            for param in self.sd_pipeline.unet.parameters():
                # only upcast trainable parameters (LoRA) into fp32
                if param.requires_grad:
                    param.data = param.to(torch.float32)
            return self.sd_pipeline.unet
        else:
            return self.sd_pipeline.unet

    def save_checkpoint(self, models, weights, output_dir):
        if len(models) != 1:
            raise ValueError("Given how the trainable params were set, this should be of length 1")
        if self.use_lora and hasattr(models[0], "peft_config") and getattr(models[0], "peft_config", None) is not None:
            state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(models[0]))
            self.sd_pipeline.save_lora_weights(save_directory=output_dir, unet_lora_layers=state_dict)
        elif not self.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")

    def load_checkpoint(self, models, input_dir):
        if len(models) != 1:
            raise ValueError("Given how the trainable params were set, this should be of length 1")
        if self.use_lora:
            lora_state_dict, network_alphas = self.sd_pipeline.lora_state_dict(
                input_dir, weight_name="pytorch_lora_weights.safetensors"
            )
            self.sd_pipeline.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=models[0])

        elif not self.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
