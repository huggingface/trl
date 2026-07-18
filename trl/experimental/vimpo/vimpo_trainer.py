# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import textwrap

import torch
from accelerate.utils import is_peft_model

from ...extras.profiling import profiling_decorator
from ...models.utils import disable_gradient_checkpointing
from ...trainer.grpo_trainer import GRPOTrainer
from ...trainer.utils import (
    entropy_from_logits,
    get_config_model_id,
    nanmax,
    nanmin,
    selective_log_softmax,
    use_adapter,
)
from .vimpo_config import VIMPOConfig


def _compute_vimpo_gae(token_advantages: torch.Tensor, mask: torch.Tensor, lam: float) -> torch.Tensor:
    """Compute VIMPO's GAE-style reverse accumulation over valid response tokens."""
    advantages = torch.zeros_like(token_advantages)
    lastgaelam = torch.zeros(token_advantages.size(0), dtype=token_advantages.dtype, device=token_advantages.device)
    for t in reversed(range(token_advantages.size(1))):
        lastgaelam = token_advantages[:, t] + lam * lastgaelam
        lastgaelam = lastgaelam * mask[:, t]
        advantages[:, t] = lastgaelam
    return advantages


class VIMPOTrainer(GRPOTrainer):
    """
    Trainer for Value-Implicit Policy Optimization (VIMPO).

    VIMPO (https://huggingface.co/papers/2606.20008) is a critic-free RLVR method that uses a policy-implied value
    recurrence derived from a KL-regularized optimality condition. It reuses GRPO's rollout and reward computation, then
    trains with a terminal value loss and an optional PPO-style actor loss using token-level policy-implied advantages.
    """

    _tag_names = ["trl", "vimpo"]
    _name = "VIMPO"
    _paper = {
        "title": "VIMPO: Value-Implicit Policy Optimization for LLMs",
        "id": "2606.20008",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{kang2026vimpo,
                title        = {{VIMPO: Value-Implicit Policy Optimization for LLMs}},
                author       = {Zhewei Kang and Aosong Feng and Sergey Levine and Dawn Song and Xuandong Zhao},
                year         = 2026,
                eprint       = {arXiv:2606.20008},
            }"""),
    }

    def __init__(self, model, reward_funcs, args=None, **kwargs):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            args = VIMPOConfig(f"{model_name.split('/')[-1]}-VIMPO")
        elif args.beta != 0.0:
            raise ValueError("VIMPO uses `vimpo_beta`; leave the inherited GRPO `beta` set to 0.0.")
        if args.sync_ref_model:
            raise ValueError("VIMPO uses the frozen-reference setting from the paper; set `sync_ref_model=False`.")
        if args.use_liger_kernel:
            raise ValueError("VIMPO does not support `use_liger_kernel=True`.")
        if args.multi_objective_aggregation != "sum_then_normalize":
            raise ValueError("VIMPO currently supports `multi_objective_aggregation='sum_then_normalize'` only.")
        if args.scale_rewards != "none":
            raise ValueError("VIMPO uses the paper target `R - mean_group(R)`; set `scale_rewards='none'`.")

        self.vimpo_beta = args.vimpo_beta
        self.vimpo_actor_coeff = args.vimpo_actor_coeff
        self.vimpo_gae_lambda = args.vimpo_gae_lambda

        # GRPO only creates/copies a reference model when `beta != 0.0`. Temporarily enable that path, then disable
        # GRPO's own KL-penalty semantics after initialization. VIMPO uses `vimpo_beta` inside its objective instead.
        args.beta = args.vimpo_beta
        super().__init__(model, reward_funcs, args=args, **kwargs)
        self.beta = 0.0
        self.args.beta = 0.0

    def _masked_whiten(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        local = values[mask.bool()].float()
        if local.numel() == 0:
            return torch.zeros_like(values)

        pad_value = torch.finfo(local.dtype).max
        padded = self.accelerator.pad_across_processes(local, dim=0, pad_index=pad_value)
        gathered = self.accelerator.gather(padded)
        gathered = gathered[gathered != pad_value]
        if gathered.numel() <= 1:
            return torch.zeros_like(values)

        mean = gathered.mean()
        std = gathered.std(unbiased=False)
        return ((values.float() - mean) / (std + 1e-8)).to(values.dtype) * mask

    def _build_model_inputs(
        self,
        logits_to_keep,
        input_ids_batch,
        attention_mask_batch,
        start,
        batch_size,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        spatial_shapes=None,
        num_tiles=None,
        image_sizes=None,
        token_type_ids=None,
        mm_token_type_ids=None,
        image_position_ids=None,
        compute_aux_loss=False,
    ):
        model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
        if image_grid_thw is not None and pixel_values is not None:
            rows_per_image = image_grid_thw.prod(dim=-1)
            rows_per_sample = torch.split(rows_per_image, num_images)
            rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
            cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
            row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
            model_inputs["pixel_values"] = pixel_values[row_start:row_end]
            cum_imgs = torch.tensor([0] + num_images).cumsum(0)
            img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
            model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
        elif image_position_ids is not None and pixel_values is not None:
            cum_imgs = torch.tensor([0] + num_images).cumsum(0)
            img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
            model_inputs["pixel_values"] = pixel_values[img_start:img_end]
            model_inputs["image_position_ids"] = image_position_ids[img_start:img_end]
        elif spatial_shapes is not None and pixel_values is not None:
            # LFM2-VL tensors are tile-indexed.
            cum_tiles = torch.tensor([0] + num_tiles).cumsum(0)
            tile_start, tile_end = cum_tiles[start], cum_tiles[start + batch_size]
            model_inputs["pixel_values"] = pixel_values[tile_start:tile_end]
            model_inputs["pixel_attention_mask"] = pixel_attention_mask[tile_start:tile_end]
            model_inputs["spatial_shapes"] = spatial_shapes[tile_start:tile_end]
        elif pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
        if pixel_attention_mask is not None and spatial_shapes is None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]
        if mm_token_type_ids is not None:
            model_inputs["mm_token_type_ids"] = mm_token_type_ids[start : start + batch_size]

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        # MoE models: request router logits so the model returns `outputs.aux_loss`. VLM wrappers honor this only
        # as a forward kwarg (not from the model config), so it must be passed here.
        if compute_aux_loss:
            model_inputs["output_router_logits"] = True

        return model_inputs

    def _get_ref_outputs(self, model_inputs):
        if self.ref_model is not None:
            return self.ref_model(**model_inputs)
        if not is_peft_model(self.model):
            raise RuntimeError("VIMPO requires a reference model or a PEFT model whose adapter can be disabled.")

        model = self.accelerator.unwrap_model(self.model)
        with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
            return self.model(**model_inputs)

    @profiling_decorator
    def _get_per_token_logps_entropies_and_kls(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_aux_loss=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        spatial_shapes=None,
        num_tiles=None,
        image_sizes=None,
        token_type_ids=None,
        mm_token_type_ids=None,
        image_position_ids=None,
    ):
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_ref_logps = []
        all_token_kls = []
        all_entropies = []
        all_aux_losses = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]
            model_inputs = self._build_model_inputs(
                logits_to_keep,
                input_ids_batch,
                attention_mask_batch,
                start,
                batch_size,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                num_images=num_images,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
                num_tiles=num_tiles,
                image_sizes=image_sizes,
                token_type_ids=token_type_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_position_ids=image_position_ids,
                compute_aux_loss=compute_aux_loss,
            )

            outputs = model(**model_inputs)
            logits = outputs.logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits.div_(self.temperature)
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)

            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                ref_outputs = self._get_ref_outputs(model_inputs)
                ref_logits = ref_outputs.logits
                ref_logits = ref_logits[:, :-1, :]
                ref_logits = ref_logits[:, -logits_to_keep:, :]
                ref_logits.div_(self.temperature)
                ref_logps = selective_log_softmax(ref_logits, completion_ids)
                ref_log_probs = torch.log_softmax(ref_logits.float(), dim=-1)
                token_kls = (log_probs.exp() * (log_probs - ref_log_probs)).sum(dim=-1).to(logps.dtype)

            all_logps.append(logps)
            all_ref_logps.append(ref_logps)
            all_token_kls.append(token_kls)

            with torch.no_grad():
                all_entropies.append(entropy_from_logits(logits))

            if compute_aux_loss:
                all_aux_losses.append(outputs.aux_loss)

        logps = torch.cat(all_logps, dim=0)
        ref_logps = torch.cat(all_ref_logps, dim=0)
        token_kls = torch.cat(all_token_kls, dim=0)
        entropies = torch.cat(all_entropies, dim=0)
        aux_loss = torch.stack(all_aux_losses).mean() if compute_aux_loss else None
        return logps, ref_logps, token_kls, entropies, aux_loss

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        per_token_logps, ref_per_token_logps, token_kls, entropies, aux_loss = (
            self._get_per_token_logps_entropies_and_kls(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                compute_aux_loss=self.aux_loss_enabled,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                num_images=inputs.get("num_images"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                spatial_shapes=inputs.get("spatial_shapes"),
                num_tiles=inputs.get("num_tiles"),
                image_sizes=inputs.get("image_sizes"),
                token_type_ids=inputs.get("token_type_ids"),
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
                image_position_ids=inputs.get("image_position_ids"),
            )
        )

        rho = self.vimpo_beta * (per_token_logps - ref_per_token_logps)
        kappa = self.vimpo_beta * token_kls
        value_terms = (rho - kappa.detach()) * mask
        terminal_value = value_terms.sum(dim=-1)
        # VIMPO rejects reward scaling, so inherited GRPO advantages are the unscaled centered rewards.
        centered_rewards = inputs["advantages"].to(dtype=terminal_value.dtype, device=terminal_value.device)
        value_residual = terminal_value - centered_rewards
        value_loss = 0.5 * value_residual.square().mean()

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        actor_loss = torch.zeros_like(value_loss)
        if self.vimpo_actor_coeff != 0.0:
            actor_advantages = rho.detach() - kappa.detach()
            actor_advantages = _compute_vimpo_gae(actor_advantages, mask, self.vimpo_gae_lambda)
            raw_actor_advantages = actor_advantages
            actor_advantages = self._masked_whiten(actor_advantages, mask)
            actor_advantages = actor_advantages.detach()

            log_ratio = per_token_logps - old_per_token_logps
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_actor_loss1 = ratio * actor_advantages
            per_token_actor_loss2 = clipped_ratio * actor_advantages
            per_token_actor_loss = -torch.min(per_token_actor_loss1, per_token_actor_loss2)
            if self.use_vllm and self.vllm_importance_sampling_correction:
                per_token_actor_loss = per_token_actor_loss * inputs["importance_sampling_ratio"]

            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            actor_loss = (per_token_actor_loss * mask).sum() / normalizer

        mode = "train" if self.model.training else "eval"
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
        loss = value_loss / normalizer + self.vimpo_actor_coeff * actor_loss

        # The policy loss above is scaled for gradient accumulation (HF auto-scaling is off here), so scale aux too
        if self.aux_loss_enabled:
            loss = loss + self.router_aux_loss_coef * aux_loss / normalizer
            self._metrics[mode]["aux_loss"].append(self.accelerator.gather_for_metrics(aux_loss).mean().item())

        completion_token_count = mask.sum().clamp(min=1.0)
        mean_exact_kl = (token_kls * mask).sum() / completion_token_count
        mean_entropy = (entropies * mask).sum() / completion_token_count
        mean_rho = (rho * mask).sum() / completion_token_count
        mean_kappa = (kappa * mask).sum() / completion_token_count

        self._metrics[mode]["vimpo/value_loss"].append(self.accelerator.gather(value_loss.detach()).nanmean().item())
        self._metrics[mode]["vimpo/actor_loss"].append(self.accelerator.gather(actor_loss.detach()).nanmean().item())
        self._metrics[mode]["vimpo/terminal_value"].append(
            self.accelerator.gather(terminal_value.detach()).nanmean().item()
        )
        self._metrics[mode]["vimpo/value_residual"].append(
            self.accelerator.gather(value_residual.detach()).nanmean().item()
        )
        self._metrics[mode]["vimpo/rho"].append(self.accelerator.gather(mean_rho.detach()).nanmean().item())
        self._metrics[mode]["vimpo/kappa"].append(self.accelerator.gather(mean_kappa.detach()).nanmean().item())
        self._metrics[mode]["kl"].append(self.accelerator.gather(mean_exact_kl.detach()).nanmean().item())
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy.detach()).nanmean().item())

        if self.vimpo_actor_coeff != 0.0:
            mean_raw_actor_advantage = (raw_actor_advantages * mask).sum() / completion_token_count
            self._metrics[mode]["vimpo/actor_advantage"].append(
                self.accelerator.gather(mean_raw_actor_advantage.detach()).nanmean().item()
            )

            is_low_clipped = (ratio < 1 - self.epsilon_low) & (actor_advantages < 0)
            is_high_clipped = (ratio > 1 + self.epsilon_high) & (actor_advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped
            low_clip = (is_low_clipped.float() * mask).sum() / completion_token_count
            high_clip = (is_high_clipped.float() * mask).sum() / completion_token_count
            clip_ratio = (is_region_clipped.float() * mask).sum() / completion_token_count

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss
