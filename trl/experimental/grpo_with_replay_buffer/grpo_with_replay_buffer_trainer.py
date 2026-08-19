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

import heapq
import inspect
import itertools
import math
from typing import Any

import torch
from accelerate.logging import get_logger
from accelerate.utils import gather_object

from ...data_utils import apply_chat_template, is_conversational, prepare_multimodal_messages
from ...models.utils import disable_gradient_checkpointing
from ...trainer.grpo_trainer import GRPOTrainer
from ...trainer.utils import get_config_model_id, nanmax, nanmin, nanstd, pad, use_adapter
from .grpo_with_replay_buffer_config import GRPOWithReplayBufferConfig


logger = get_logger(__name__)

# Fields sliced/padded on the completion axis when a group is buffered or replayed: (padding value, description).
# `completion_ids` is handled separately because its padding value is the tokenizer's pad token.
_COMPLETION_FIELD_PADDING = {
    "completion_mask": 0,
    "tool_mask": 1,  # matches the generation-time padding; the loss also multiplies by completion_mask
    "old_per_token_logps": 0.0,
    "ref_per_token_logps": 0.0,
    "sampling_per_token_logps": 0.0,
    "importance_sampling_ratio": 1.0,  # neutral correction on padding positions (masked in the loss anyway)
}

# Multimodal keys that the replay machinery does not support (see `update_with_replay_buffer`).
_VISION_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_attention_mask",
    "spatial_shapes",
    "image_sizes",
    "token_type_ids",
    "mm_token_type_ids",
    "image_position_ids",
    "num_images",
    "num_tiles",
)


class ReplayBuffer:
    """
    A simple replay buffer to store and sample previously seen rollouts.

    Args:
        max_size (`int`):
            Maximum number of groups the buffer holds. When full, a new group evicts the lowest-scored one only if it
            scores higher.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.heap = []  # Min-heap of (score, insertion_index, data) tuples
        # Unique tiebreaker so equal scores never fall through to comparing the data dicts (which raises TypeError)
        self._counter = itertools.count()

    def add(self, scores: list[float], data: list[dict]):
        for score, datum in zip(scores, data, strict=True):
            entry = (score, next(self._counter), datum)
            if len(self.heap) < self.max_size:
                heapq.heappush(self.heap, entry)
            elif score > self.heap[0][0]:
                # Only add if score is better than worst (minimum) item
                heapq.heapreplace(self.heap, entry)

    def sample(self, num_samples: int) -> list[dict[str, torch.Tensor]] | None:
        if not self.heap:
            return None

        # Sample by normalized scores
        scores = torch.tensor([entry[0] for entry in self.heap], dtype=torch.float32)
        probabilities = scores / scores.sum()
        replacement = num_samples > len(self.heap)
        chosen_indices = torch.multinomial(probabilities, num_samples, replacement=replacement).tolist()
        return [self.heap[i][2] for i in chosen_indices]


def _widen(tensor: torch.Tensor, target_len: int, value: float | int, side: str) -> torch.Tensor:
    """Pad a (B, T) tensor along dim 1 up to `target_len` with `value`, on the given side."""
    diff = target_len - tensor.size(1)
    if diff <= 0:
        return tensor
    fill = tensor.new_full((tensor.size(0), diff), value)
    return torch.cat([fill, tensor], dim=1) if side == "left" else torch.cat([tensor, fill], dim=1)


class GRPOWithReplayBufferTrainer(GRPOTrainer):
    """
    Trainer for GRPO with a replay buffer.

    Groups whose rewards have nonzero variance are stored in a replay buffer, scored by `(|advantage| *
    reward_std).sum()`. Groups with zero reward variance (which contribute no gradient) are replaced with groups
    sampled from the buffer, so every optimizer step trains on informative rollouts.

    The only changes w.r.t. [`GRPOTrainer`] are the replay hook at the end of `_generate_and_score_completions` and
    the unconditional tracking of `old_per_token_logps` (replayed groups come from earlier policy versions, so the
    surrogate objective needs their behavior logps). Multimodal (VLM) batches are not supported by the replay
    machinery and pass through unchanged.
    """

    def __init__(self, model, reward_funcs=None, args: GRPOWithReplayBufferConfig | None = None, **kwargs):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            args = GRPOWithReplayBufferConfig(f"{model_name.split('/')[-1]}-GRPO")
        super().__init__(model, reward_funcs, args=args, **kwargs)
        self.replay_buffer = ReplayBuffer(args.replay_buffer_size) if args.replay_buffer_size > 0 else None
        self._warned_multimodal_replay = False

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # `prompt` is optional only when an environment owns the data (e.g. a multi-environment routing dataset that
        # carries only an `environment` column); each rollout's `reset()` then supplies it. Default it here rather than
        # writing it back onto the row, so the placeholder stays out of the `reset()` kwargs built below, while every
        # prompt-derived check downstream (conversational detection, multimodal handling) stays consistent. Without an
        # environment, a missing `prompt` is a malformed dataset and must still fail fast.
        if self.environment_factories is not None:
            prompts = [x.get("prompt", [{"role": "user", "content": ""}]) for x in inputs]
        else:
            prompts = [x["prompt"] for x in inputs]

        # Resolve each example's environment and draw one reusable instance per rollout from the pool, creating more
        # only when this batch needs more concurrent instances of an environment than exist. `_batch_environments`
        # records each example's environment so `_tokenize_prompts` can render the matching tool schema.
        if self.environment_factories is not None:
            self._batch_environments = [x.get("environment") if self._multi_environment else None for x in inputs]
            if self._multi_environment:
                for name in set(self._batch_environments):
                    if name not in self.environment_factories:
                        raise ValueError(
                            f"Example has `environment={name!r}`, which is not among the environments passed to "
                            f"`environment_factory`. Expected one of: {list(self.environment_factories)}."
                        )
            self.environments = []
            pool_cursor = {}  # how many instances of each environment have been handed out so far this batch
            for name in self._batch_environments:
                pool = self._environment_pool[name]
                index = pool_cursor.get(name, 0)
                if index == len(pool):
                    pool.append(self.environment_factories[name]())
                pool_cursor[name] = index + 1
                self.environments.append(pool[index])

        # Build the per-rollout tool dicts for this batch: the standalone tools plus, for each rollout, the methods of
        # its environment. Done here (not at init) because each example's environment, hence its tools, is data-dependent.
        if self.tools:
            self._sync_tool_dicts = []
            self._async_tool_dicts = []
            for i in range(len(inputs)):
                methods = []
                if self.environments:
                    methods = [
                        member
                        for member_name, member in inspect.getmembers(self.environments[i], predicate=inspect.ismethod)
                        if member_name not in ("reset", "get_reward") and not member_name.startswith("_")
                    ]
                sync_tool_dict, async_tool_dict = {}, {}
                for tool in self._standalone_tools + methods:
                    if inspect.iscoroutinefunction(tool):
                        async_tool_dict[tool.__name__] = tool
                    else:
                        sync_tool_dict[tool.__name__] = tool
                self._sync_tool_dicts.append(sync_tool_dict)
                self._async_tool_dicts.append(async_tool_dict)

        if self.environments:
            for i, (prompt, environment, x) in enumerate(zip(prompts, self.environments, inputs, strict=True)):
                # `environment` is a control field in multi-environment mode, so it is not forwarded to `reset`.
                reset_kwargs = {k: v for k, v in x.items() if k != "environment"} if self._multi_environment else x
                observation = environment.reset(**reset_kwargs)
                if observation is None:
                    continue
                content = prompt[-1]["content"]
                if isinstance(observation, list) and isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                if isinstance(observation, str) and isinstance(content, list):
                    observation = [{"type": "text", "text": observation}]
                # Rebuild the last message rather than mutating it in place, so the input example is left untouched.
                prompts[i] = prompt[:-1] + [{**prompt[-1], "content": content + observation}]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            if not is_conversational(inputs[0]):
                raise ValueError(
                    "Multimodal training requires conversational prompts. It looks like the dataset contains "
                    "non-conversational inputs, likely because a chat template was applied before passing the dataset "
                    "to the trainer. Please provide the raw conversational prompts and let the trainer apply the chat "
                    "template internally."
                )
            prompts = [
                prepare_multimodal_messages(prompt, images=image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        dataset_images = images  # preserve dataset images before _generate may overwrite
        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            sampling_per_token_logps_list,
            extra_fields,
            images,
            tool_images,
        ) = self._generate(prompts)
        if images is None:
            images = dataset_images  # restore dataset images (rollout_func path returns None)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(
            prompt_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        prompt_mask = pad(
            prompt_mask, padding_value=0, padding_side="left", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device=device)
        completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(
            completion_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        completion_mask = pad(
            completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device=device)
        if sampling_per_token_logps_list is not None:
            # vLLM replaces a NaN token logprob with `None` (see `extract_logprobs`); map it back to NaN so the
            # tensor builds (`torch.tensor([..., None, ...])` raises "Could not infer dtype of NoneType"). The NaN
            # positions are neutralized in the importance-sampling ratio below so they contribute no correction.
            sampling_per_token_logps = [
                torch.tensor([float("nan") if x is None else x for x in logps])
                for logps in sampling_per_token_logps_list
            ]
            sampling_per_token_logps = pad(
                sampling_per_token_logps,
                padding_value=0.0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            ).to(device=device)
        else:
            sampling_per_token_logps = None
        if tool_mask_list is not None:
            tool_mask = [torch.tensor(mask) for mask in tool_mask_list]
            tool_mask = pad(
                tool_mask, padding_value=1, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            ).to(device=device)
        else:
            tool_mask = None

        # If mask_truncated_completions is enabled, zero out truncated completions for attention and loss masking
        if self.mask_truncated_completions:
            eos_and_pad = [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            # Mask completion_mask for attention masking
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
            # Also mask tool_mask for consistency in multi-turn training
            if tool_mask is not None:
                tool_mask = tool_mask * (~is_truncated).unsqueeze(1).int()

        loss_mask = completion_mask if tool_mask is None else completion_mask * tool_mask
        num_items_in_batch = self.accelerator.gather(loss_mask.sum()).sum()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) if img_list else 0 for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs.
        # When tool images are present (from _tool_call_loop), use image_processor directly and build
        # mm_token_type_ids from prompt_completion_ids. Otherwise, use the full processor pipeline
        # which returns model-specific keys (image_sizes, pixel_attention_mask, etc.).
        if self.tools and any(imgs for imgs in tool_images) and self._is_vlm:
            flat_images = [img for img_list in images if img_list for img in img_list]
            image_inputs = self.processing_class.image_processor(images=flat_images, return_tensors="pt")
            image_inputs = super()._prepare_inputs(image_inputs)
            forward_kwargs = dict(image_inputs)
        elif images is not None:
            if self.environment_factories is not None:
                per_prompt_tools = [self._env_tools[name] for name in self._batch_environments]
            else:
                per_prompt_tools = [self.tools] * len(prompts)
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt}, self.processing_class, tools=tools, **self.chat_template_kwargs
                )["prompt"]
                for prompt, tools in zip(prompts, per_prompt_tools, strict=True)
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # Recover LFM2-VL tile counts; the full processor drops row/column metadata.
        num_tiles = None
        if images is not None and "spatial_shapes" in forward_kwargs:
            image_info = self.processing_class.image_processor(
                images=images, return_tensors="pt", return_row_col_info=True
            )
            tiles_per_image = image_info["image_rows"] * image_info["image_cols"]
            if self.processing_class.image_processor.use_thumbnail:
                tiles_per_image = tiles_per_image + (tiles_per_image > 1).to(tiles_per_image.dtype)
            num_tiles = [group.sum().item() for group in torch.split(tiles_per_image, num_images)]

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            if self.pad_to_multiple_of is not None:
                # Needed only with pad_to_multiple_of: otherwise prompt_ids and token_type_ids must have equal len
                padding_size = prompt_ids.size(1) - token_type_ids.size(1)
                if padding_size > 0:
                    token_type_ids = torch.cat(
                        [token_type_ids.new_zeros((token_type_ids.size(0), padding_size)), token_type_ids], dim=1
                    )
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )
        # If mm_token_type_ids are used, extend them with zeros for the completion part
        if "mm_token_type_ids" in forward_kwargs:
            mm_token_type_ids = forward_kwargs["mm_token_type_ids"]
            if self.pad_to_multiple_of is not None:
                # Needed only with pad_to_multiple_of: otherwise prompt_ids and mm_token_type_ids must have equal len
                padding_size = prompt_ids.size(1) - mm_token_type_ids.size(1)
                if padding_size > 0:
                    mm_token_type_ids = torch.cat(
                        [mm_token_type_ids.new_zeros((mm_token_type_ids.size(0), padding_size)), mm_token_type_ids],
                        dim=1,
                    )
            forward_kwargs["mm_token_type_ids"] = torch.cat(
                [mm_token_type_ids, mm_token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # For VLM tool images: build token type IDs from the full prompt_completion_ids.
        # This must happen AFTER the token_type_ids/mm_token_type_ids extension blocks above,
        # because our version already covers the full sequence (images are in the completion,
        # not just the prompt).
        if self.tools and any(imgs for imgs in tool_images) and self._is_vlm:
            mm_ids = torch.zeros_like(prompt_completion_ids)
            if self._image_pad_token_id is not None:
                mm_ids[prompt_completion_ids == self._image_pad_token_id] = 1
            if self._video_pad_token_id is not None:
                mm_ids[prompt_completion_ids == self._video_pad_token_id] = 2

            # Use the same key the model expects: token_type_ids for models like Gemma,
            # mm_token_type_ids for models like Qwen.
            image_grid_thw = forward_kwargs.get("image_grid_thw")
            if image_grid_thw is not None:
                forward_kwargs["mm_token_type_ids"] = mm_ids
            else:
                forward_kwargs["token_type_ids"] = mm_ids

            # Truncation safety (Qwen-style models with image_grid_thw only): if
            # max_completion_length truncated some image tokens, the number of image pad tokens
            # in input_ids won't match pixel_values features. Check per-sample and drop ALL
            # images for any sample with a mismatch (safe fallback).
            if image_grid_thw is not None and num_images is not None:
                merge_length = getattr(self.processing_class.image_processor, "merge_size", 2) ** 2
                img_offset = 0
                has_mismatch = False
                for b in range(mm_ids.shape[0]):
                    sample_tokens = (mm_ids[b] == 1).sum().item()
                    sample_features = 0
                    for i in range(num_images[b]):
                        grid_idx = img_offset + i
                        if grid_idx < image_grid_thw.shape[0]:
                            sample_features += image_grid_thw[grid_idx].prod().item() // merge_length
                    if sample_tokens != sample_features:
                        has_mismatch = True
                        break
                    img_offset += num_images[b]

                if has_mismatch:
                    # Drop all images: safer than partial trim which is error-prone
                    forward_kwargs.pop("pixel_values", None)
                    forward_kwargs.pop("image_grid_thw", None)
                    mm_ids.zero_()
                    forward_kwargs["mm_token_type_ids"] = mm_ids
                    num_images = None

        # When gradient checkpointing is enabled with use_reentrant=True (non default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            # Unlike GRPOTrainer, also compute old_per_token_logps whenever the replay buffer is active: replayed
            # groups come from earlier policy versions, so the surrogate objective needs their behavior logps.
            if (
                (self.replay_buffer is not None and mode == "train")
                or self.args.gradient_accumulation_steps % generate_every != 0
                or (self.use_vllm and self.vllm_importance_sampling_correction)
            ):
                old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    num_tiles=num_tiles,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask, spatial_shapes, image_sizes, image_position_ids
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                mask = completion_mask if tool_mask is None else completion_mask * tool_mask
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask
                # Tokens whose sampling logprob was NaN (unavailable from vLLM) get a zero difference, so their
                # importance ratio is exactly 1 (no correction) rather than propagating NaN through the loss.
                per_token_logps_diff = torch.nan_to_num(per_token_logps_diff, nan=0.0)

                sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                if sequence_level_is:
                    per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                    logps_diff = per_sequence_logps_diff
                else:
                    logps_diff = per_token_logps_diff

                vllm_importance_sampling_ratio = torch.exp(logps_diff)

                # vllm_importance_sampling_ratio.shape:
                #   token_* modes:     (B, T)  (per-token ratio)
                #   sequence_* modes:  (B, 1)  (per-sequence ratio)

                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio,
                        min=self.vllm_importance_sampling_clip_min,
                        max=self.vllm_importance_sampling_clip_max,
                    )
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    min_val = (
                        self.vllm_importance_sampling_clip_min
                        if self.vllm_importance_sampling_clip_min is not None
                        else -math.inf
                    )
                    max_val = (
                        self.vllm_importance_sampling_clip_max
                        if self.vllm_importance_sampling_clip_max is not None
                        else math.inf
                    )

                    invalid_mis_mask = (vllm_importance_sampling_ratio < min_val) | (
                        vllm_importance_sampling_ratio > max_val
                    )
                    vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                        invalid_mis_mask, value=0.0
                    )
                else:
                    raise ValueError(
                        f"Unknown vLLM importance sampling level: {self.vllm_importance_sampling_mode}. Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', and 'sequence_mask'."
                    )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        num_tiles=num_tiles,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask, spatial_shapes, image_sizes, image_position_ids
                    )
                else:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    model = self.accelerator.unwrap_model(self.model)
                    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                        ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            num_tiles=num_tiles,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask, spatial_shapes, image_sizes, image_position_ids
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        # A completion for which every reward function returned None is unscorable. nansum would collapse it to 0,
        # which both biases the per-group baseline and hands the completion a spurious advantage. Mark these rows NaN
        # so they're excluded from the (nan-aware) baseline below; their advantage is forced to 0 afterwards.
        unscorable_mask = torch.isnan(rewards_per_func).all(dim=1)

        if self.multi_objective_aggregation == "sum_then_normalize":
            # Apply weights to each reward function's output and sum
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            rewards[unscorable_mask] = torch.nan
            mean_grouped_rewards = torch.nanmean(rewards.view(-1, num_generations), dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            if self.scale_rewards in ["group", "none"]:
                # If self.scale_rewards = "none", we'll only use std_rewards to check for zero std for logging
                if num_generations > 1:
                    std_rewards = nanstd(rewards.view(-1, num_generations), dim=1)
                    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
                else:  # doesn't occur during training, but could occur in eval when num_generations_eval=1
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "batch":
                # Compute global std
                if rewards.numel() > 1:
                    std_rewards = nanstd(rewards).expand_as(rewards)
                else:  # doesn't occur during training, but could occur in eval when num_generations_eval=batch_size=1
                    std_rewards = torch.zeros_like(rewards)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                )

            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging

        elif self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            std_k = nanstd(grouped, dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(mean_k)
            reward_k = (grouped - mean_k) / (std_k + 1e-4)
            reward_k = reward_k.view(-1, len(self.reward_funcs))
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            rewards[unscorable_mask] = torch.nan
            std_rewards = nanstd(rewards).expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            advantages = (rewards - torch.nanmean(rewards)) / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging

        else:
            raise ValueError(
                f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}. Must be "
                "'sum_then_normalize' or 'normalize_then_sum'."
            )

        # Unscorable completions (every reward func returned None) carry no learning signal: their reward is NaN here,
        # so zero their advantage to keep them from moving the policy.
        advantages = torch.nan_to_num(advantages, nan=0.0)

        # Per-group reward std (nan-aware), used by the replay buffer to identify zero-variance groups. Computed here,
        # before `rewards` is reassigned for logging below. NaN stds (single scorable completion) compare as
        # not-greater-than-zero, so those groups fall in the replacement pool alongside zero-variance ones.
        group_std_rewards = nanstd(rewards.view(-1, num_generations), dim=1).repeat_interleave(num_generations, dim=0)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]
        group_std_rewards = group_std_rewards[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        rewards = (rewards_per_func * self.reward_weights.to(rewards_per_func.device).unsqueeze(0)).nansum(dim=1)
        rewards[unscorable_mask] = torch.nan  # exclude unscorable rows from the logged reward stats
        self._metrics[mode]["reward"].append(torch.nanmean(rewards).item())
        self._metrics[mode]["reward_std"].append(nanstd(rewards).item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        # Flush user-logged extra columns (from log_extra), gathering across processes.
        # Keys must be sorted so that all ranks call gather_object in the same order, otherwise values
        # get mis-attributed across columns (dict insertion order may differ between processes).
        for column in sorted(self._pending_extra_logs):
            self._logs["extra"][column].extend(gather_object(self._pending_extra_logs[column]))
        self._pending_extra_logs.clear()

        # Flush user-logged metrics (from log_metric), averaging across processes.
        # Keys must be sorted so that all ranks call accelerator.gather in the same order, otherwise values
        # get mis-attributed across metrics (dict insertion order may differ between processes).
        for name in sorted(self._pending_metrics):
            values = self._pending_metrics[name]
            local_mean = sum(values) / len(values)
            global_mean = self.accelerator.gather(torch.tensor(local_mean, device=device)).mean().item()
            self._metrics[mode][name].append(global_mean)
        self._pending_metrics.clear()

        if images is not None and self.log_multimodal:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            mask = completion_mask.bool() if tool_mask is None else (completion_mask * tool_mask).bool()
            # Tokens vLLM could not score carry NaN, so exclude them rather than let them turn the reported
            # divergence into NaN. Counting them as zero instead would understate the divergence.
            delta = delta[mask & ~torch.isnan(delta)]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )
            if sequence_level_is:
                flat_is_ratio = vllm_importance_sampling_ratio.flatten()
            else:
                flat_is_ratio = vllm_importance_sampling_ratio[mask]

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if sampling_per_token_logps is not None:
            output["sampling_per_token_logps"] = sampling_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "spatial_shapes" in forward_kwargs:
            output["spatial_shapes"] = forward_kwargs["spatial_shapes"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if "mm_token_type_ids" in forward_kwargs:
            output["mm_token_type_ids"] = forward_kwargs["mm_token_type_ids"]
        if "image_position_ids" in forward_kwargs:
            output["image_position_ids"] = forward_kwargs["image_position_ids"]
        if images is not None:
            output["num_images"] = num_images
            if num_tiles is not None:
                output["num_tiles"] = num_tiles
        if tool_mask is not None:
            output["tool_mask"] = tool_mask

        # Replay buffer: buffer scorable groups with reward variance and replace zero-variance groups with
        # previously seen high-signal groups. Train-only, so eval batches neither pollute the buffer nor get
        # replaced. The advantages/completions logged above intentionally reflect what was generated this batch.
        if self.replay_buffer is not None and mode == "train":
            output = self.update_with_replay_buffer(output, group_std_rewards)
            # Recompute the loss normalizer from the final masks: replacement may change the active token count
            loss_mask = (
                output["completion_mask"]
                if "tool_mask" not in output
                else output["completion_mask"] * output["tool_mask"]
            )
            output["num_items_in_batch"] = self.accelerator.gather(loss_mask.sum()).sum()
        return output

    def _slice_group(self, output: dict[str, torch.Tensor], group_idx: int) -> dict[str, torch.Tensor]:
        """
        Extract one group's rows from the batch `output`, trimmed to the group's own maximum prompt length (prompts
        are left-padded, so trimming keeps the last columns) and maximum completion length (completions are
        right-padded, so trimming keeps the first columns).
        """
        rows = slice(group_idx * self.num_generations, (group_idx + 1) * self.num_generations)
        prompt_mask = output["prompt_mask"][rows]
        prompt_len = max(int(prompt_mask.sum(dim=1).max()), 1)
        completion_mask = output["completion_mask"][rows]
        completion_len = max(int(completion_mask.sum(dim=1).max()), 1)

        item = {
            "prompt_ids": output["prompt_ids"][rows][:, -prompt_len:].clone(),
            "prompt_mask": prompt_mask[:, -prompt_len:].clone(),
            "completion_ids": output["completion_ids"][rows][:, :completion_len].clone(),
            "advantages": output["advantages"][rows].clone(),
        }
        for key in _COMPLETION_FIELD_PADDING:
            if key in output:
                tensor = output[key][rows]
                # Sequence-level vLLM IS modes produce a (B, 1) ratio; keep it as-is
                item[key] = tensor.clone() if tensor.size(1) == 1 else tensor[:, :completion_len].clone()
        return item

    def update_with_replay_buffer(
        self, output: dict[str, torch.Tensor], group_std_rewards: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Buffer this batch's groups that have reward variance, then replace its zero-variance groups with groups
        sampled from the replay buffer.

        Args:
            output (`dict[str, torch.Tensor]`):
                Batch output of `_generate_and_score_completions`, holding this process's rows. Modified in place and
                returned.
            group_std_rewards (`torch.Tensor`):
                Per-sample group reward std of shape `(batch_size,)` (each group's std repeated `num_generations`
                times). NaN stds count as zero variance.

        Returns:
            `dict[str, torch.Tensor]`: `output`, with zero-variance groups replaced when the buffer could serve them.
        """
        # The replay machinery slices and re-pads text tensors only. Vision tensors index differently per model
        # family (patch rows, tiles, grids), so buffering them correctly requires per-sample bookkeeping that this
        # trainer does not implement. Pass multimodal batches through unchanged.
        if any(key in output for key in _VISION_KEYS):
            if not self._warned_multimodal_replay:
                logger.warning(
                    "GRPOWithReplayBufferTrainer does not support multimodal batches; the replay buffer is disabled "
                    "for them."
                )
                self._warned_multimodal_replay = True
            return output

        num_groups = output["advantages"].size(0) // self.num_generations
        group_std = group_std_rewards.view(num_groups, self.num_generations)[:, 0]  # one std per group
        has_variance = group_std > 0  # NaN compares False, so unscorable groups land in the replacement pool

        # Buffer the groups with variance, scored by summed |advantage| weighted by the group's reward std
        group_indices = has_variance.nonzero(as_tuple=True)[0].tolist()
        if group_indices:
            group_advantages = output["advantages"].view(num_groups, self.num_generations)
            scores = (group_advantages.abs() * group_std.unsqueeze(1)).sum(dim=1)
            self.replay_buffer.add(
                scores[has_variance].tolist(), [self._slice_group(output, g) for g in group_indices]
            )

        # Replace the zero-variance groups with sampled ones
        indices_to_replace = (~has_variance).nonzero(as_tuple=True)[0].tolist()
        if not indices_to_replace:
            return output
        sampled = self.replay_buffer.sample(num_samples=len(indices_to_replace))
        if sampled is None:  # empty buffer: keep the groups; their advantages are zero, so they add no gradient
            return output

        # Widen the batch (or the sampled groups) so both sides share the same prompt/completion widths
        pad_token_id = self._tokenizer.pad_token_id
        target_prompt_len = max([output["prompt_ids"].size(1)] + [item["prompt_ids"].size(1) for item in sampled])
        target_completion_len = max(
            [output["completion_ids"].size(1)] + [item["completion_ids"].size(1) for item in sampled]
        )
        widen_specs = {
            "prompt_ids": (pad_token_id, "left", target_prompt_len),
            "prompt_mask": (0, "left", target_prompt_len),
            "completion_ids": (pad_token_id, "right", target_completion_len),
            **{
                key: (value, "right", target_completion_len)
                for key, value in _COMPLETION_FIELD_PADDING.items()
                if key in output
            },
        }
        for key, (value, side, target_len) in widen_specs.items():
            if output[key].size(1) == 1 and key == "importance_sampling_ratio":
                continue  # sequence-level IS ratio stays (B, 1)
            output[key] = _widen(output[key], target_len, value, side)
            for item in sampled:
                item[key] = _widen(item[key], target_len, value, side)

        for item, group_idx in zip(sampled, indices_to_replace, strict=True):
            rows = slice(group_idx * self.num_generations, (group_idx + 1) * self.num_generations)
            output["advantages"][rows] = item["advantages"].to(output["advantages"].device)
            for key in widen_specs:
                output[key][rows] = item[key].to(output[key].device)

        return output
