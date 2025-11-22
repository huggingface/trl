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

import heapq
from typing import Any

import torch
from accelerate.utils import gather_object

from ...data_utils import apply_chat_template, is_conversational, prepare_multimodal_messages
from ...trainer.grpo_trainer import GRPOTrainer
from ...trainer.utils import nanmax, nanmin, nanstd, pad
from .grpo_with_replay_buffer_config import GRPOWithReplayBufferConfig


class ReplayBuffer:
    """
    A simple replay buffer to store and sample previously seen rollouts.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.heap = []  # Min-heap of (score, data) tuples

    def add(self, scores: list[float], data: list[dict]):
        for score, datum in zip(scores, data, strict=True):
            if len(self.heap) < self.max_size:
                heapq.heappush(self.heap, (score, datum))
            else:
                # Only add if score is better than worst (minimum) item
                if score > self.heap[0][0]:
                    heapq.heapreplace(self.heap, (score, datum))

    def sample(self, num_samples: int) -> list[dict[str, torch.Tensor]]:
        if not self.heap:
            return None

        # Sample by normalized scores
        scores = torch.tensor([item[0] for item in self.heap], dtype=torch.float32)
        probabilities = scores / scores.sum()
        replacement = False
        if num_samples > len(self.heap):
            replacement = True
        chosen_indices = torch.multinomial(probabilities, num_samples, replacement=replacement).tolist()
        return [self.heap[i][1] for i in chosen_indices]


class GRPOWithReplayBufferTrainer(GRPOTrainer):
    def __init__(self, args: GRPOWithReplayBufferConfig | None = None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.replay_buffer = ReplayBuffer(args.replay_buffer_size) if args.replay_buffer_size > 0 else None

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

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
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        prompt_ids_list, completion_ids_list, num_items_in_batch, sampling_per_token_logps_list, extra_fields = (
            self._generate(prompts)
        )

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs
        if images is not None:
            prompts_text = [
                apply_chat_template({"prompt": prompt}, self.processing_class, **self.chat_template_kwargs)["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=True):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                if isinstance(bootstrap, list):  # for VLM, the format might be [{"type": "text", "text": "..."}]
                    assert len(bootstrap) == 1 and bootstrap[0]["type"] == "text"
                    bootstrap = bootstrap[0]["text"]
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

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

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        grouped_std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        grouped_std_rewards = grouped_std_rewards.repeat_interleave(self.num_generations, dim=0)

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            std_rewards = grouped_std_rewards.clone()
        elif self.scale_rewards == "batch":
            # Compute global std
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]
        grouped_std_rewards = grouped_std_rewards[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
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
        outputs_after_sampling_buffer = self.update_with_replay_buffer(
            advantages,
            grouped_std_rewards,
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            forward_kwargs,
            num_items_in_batch,
            old_per_token_logps,
            ref_per_token_logps,
            importance_sampling_ratio if self.use_vllm and self.vllm_importance_sampling_correction else None,
        )
        if outputs_after_sampling_buffer is not None:
            return outputs_after_sampling_buffer
        else:
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
                output["importance_sampling_ratio"] = importance_sampling_ratio
            if ref_per_token_logps is not None:
                output["ref_per_token_logps"] = ref_per_token_logps
            if "pixel_values" in forward_kwargs:
                output["pixel_values"] = forward_kwargs["pixel_values"]
            if "image_grid_thw" in forward_kwargs:
                output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
            if "pixel_attention_mask" in forward_kwargs:
                output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
            if "image_sizes" in forward_kwargs:
                output["image_sizes"] = forward_kwargs["image_sizes"]
            if "token_type_ids" in forward_kwargs:
                output["token_type_ids"] = forward_kwargs["token_type_ids"]
            if images is not None:
                output["num_images"] = num_images
            return output

    def slice_group_data(
        self, data: torch.Tensor, mask: torch.Tensor, group_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Slices the input data and mask tensors for a specific group index. Also trims the sequence length to the
        maximum length in the group based on the mask.

        Args:
            data: Tensor of shape (num_groups * num_generations, seq_length)
            mask: Tensor of shape (num_groups * num_generations, seq_length)
            group_idx: Index of the group to slice
        Returns:
            Tuple of (sliced_data, sliced_mask) for the specified group, with sequence length trimmed to the maximum
            length in the group.
        """
        start_idx = group_idx * self.num_generations
        end_idx = (group_idx + 1) * self.num_generations
        group_data = data[start_idx:end_idx]
        group_mask = mask[start_idx:end_idx]
        group_max_len = group_mask.sum(dim=1).max().item()
        return group_data[:, :group_max_len], group_mask[:, :group_max_len]

    def update_replay_buffer(
        self,
        groups_with_variance: torch.Tensor,
        group_advantages: torch.Tensor,
        group_std_rewards: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        forward_kwargs: dict,
        optional_vision_fields: list[str] = None,
        old_per_token_logps: torch.Tensor | None = None,
        ref_per_token_logps: torch.Tensor | None = None,
        importance_sampling_ratio: float | None = None,
    ) -> None:
        """
        Update the replay buffer with groups that have reward variance (std > 0).

        Args:
            groups_with_variance: Boolean tensor indicating which groups have reward variance
            group_advantages: Tensor of shape (num_groups, num_generations) containing advantage values
            std_rewards: Tensor of shape (num_groups, num_generations) containing std of rewards per group
            prompt_ids: Tensor containing prompt token IDs
            prompt_mask: Tensor containing prompt attention masks
            completion_ids: Tensor containing completion token IDs
            completion_mask: Tensor containing completion attention masks
            forward_kwargs: Dictionary containing additional prompt inputs (vision data, etc.)
            optional_vision_fields: List of optional vision-related fields to include if present in forward_kwargs
            old_per_token_logps: Optional tensor of old per-token log probabilities
            ref_per_token_logps: Optional tensor of reference per-token log probabilities
            importance_sampling_ratio: Optional importance sampling correction ratio
        """
        # Prepare buffered outputs for groups with variance
        buffered_outputs = []
        for _, group_idx in enumerate(groups_with_variance.nonzero(as_tuple=True)[0].unique().tolist()):
            group_prompt_ids, group_prompt_mask = self.slice_group_data(prompt_ids, prompt_mask, group_idx)
            group_completion_ids, group_completion_mask = self.slice_group_data(
                completion_ids, completion_mask, group_idx
            )

            # Store unpadded data in the buffer
            buffered_output = {
                "prompt_ids": group_prompt_ids,
                "completion_ids": group_completion_ids,
                "advantages": group_advantages[group_idx].tolist(),
                "prompt_mask": group_prompt_mask,
                "completion_mask": group_completion_mask,
            }

            # Add optional fields if they exist
            optional_fields = {
                "old_per_token_logps": old_per_token_logps if old_per_token_logps is not None else None,
                "ref_per_token_logps": ref_per_token_logps if ref_per_token_logps is not None else None,
            }

            for field_name, field_data in optional_fields.items():
                if field_data is not None:
                    buffered_output[field_name] = self.slice_group_data(field_data, completion_mask, group_idx)[0]

            # Add importance sampling if needed
            if self.use_vllm and self.vllm_importance_sampling_correction:
                buffered_output["importance_sampling_ratio"] = importance_sampling_ratio

            if optional_vision_fields:
                # Add vision-related fields if they exist
                for field_name in optional_vision_fields:
                    if field_name in forward_kwargs:
                        buffered_output[field_name] = self.slice_group_data(
                            forward_kwargs[field_name], prompt_mask, group_idx
                        )[0]

            buffered_outputs.append(buffered_output)

        if groups_with_variance.any():
            # Calculate replay buffer scores for groups with variance
            replay_buffer_scores = (group_advantages.abs() * group_std_rewards).sum(dim=-1)[groups_with_variance]
            # Add all groups to replay buffer at once (batch operation)
            self.replay_buffer.add(replay_buffer_scores.tolist(), buffered_outputs)

    def sample_from_replay_buffer(
        self, num_samples: int, optional_vision_fields: list[str] = None, optional_tensor_fields: list[str] = None
    ) -> list[dict]:
        """
        Sample groups from the replay buffer.

        Args:
            num_samples: Number of samples to draw from the replay buffer
            optional_vision_fields: List of optional vision-related fields to include if present in sampled data
            optional_tensor_fields: List of optional tensor fields to include if present in sampled data
        Returns:
            List of sampled data dictionaries from the replay buffer
        """
        sampled = self.replay_buffer.sample(num_samples=num_samples)

        # Extract and concatenate sampled data
        sampled_data = {
            "prompt_ids": [],
            "prompt_mask": [],
            "completion_ids": [],
            "completion_mask": [],
            "advantages": [],
        }

        all_optional_fields = (optional_tensor_fields or []) + (optional_vision_fields or [])
        # Initialize containers for optional fields if they exist in sampled data
        for field in all_optional_fields:
            if sampled and field in sampled[0]:
                sampled_data[field] = []

        # Extract data from each sampled item
        for item in sampled:
            # Handle core fields
            for key in ["prompt_ids", "prompt_mask", "completion_ids", "completion_mask"]:
                sampled_data[key].append(item[key])

            # Handle advantages (list, not tensor)
            sampled_data["advantages"].append(item["advantages"])

            # Handle optional fields
            for field in all_optional_fields:
                if field in item:
                    sampled_data[field].append(item[field])

        return sampled_data

    def update_with_replay_buffer(
        self,
        group_advantages: torch.Tensor,
        group_std_rewards: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        forward_kwargs: dict,
        num_items_in_batch: int,
        old_per_token_logps: torch.Tensor | None = None,
        ref_per_token_logps: torch.Tensor | None = None,
        importance_sampling_ratio: float | None = None,
    ) -> None:
        """
        Update current batch data with samples from replay buffer.

        Groups with reward variance (std > 0) are added to the replay buffer and then replaced with samples from the
        buffer to improve training stability.

        Args:
            group_advantages: Tensor of shape (num_groups, num_generations) containing advantage values
            std_rewards: Tensor of shape (num_groups, num_generations) containing std of rewards per group
            prompt_ids: Tensor containing prompt token IDs
            prompt_mask: Tensor containing prompt attention masks
            completion_ids: Tensor containing completion token IDs
            completion_mask: Tensor containing completion attention masks
            forward_kwargs: Dictionary containing additional prompt inputs (vision data, etc.)
            num_items_in_batch: Number of items in the current batch
            old_per_token_logps: Optional tensor of old per-token log probabilities
            ref_per_token_logps: Optional tensor of reference per-token log probabilities
            importance_sampling_ratio: Optional importance sampling correction ratio
        """
        if self.replay_buffer.max_size <= 0:
            return

        # Groups to consider for adding to the replay buffer
        groups_with_variance = group_std_rewards.max(dim=0).values > 0
        # Groups to replace from the replay buffer
        groups_without_variance = ~groups_with_variance

        # Track which optional fields are present in sampled data
        optional_tensor_fields = ["old_per_token_logps", "ref_per_token_logps"]
        vision_fields = ["pixel_values", "image_grid_thw", "pixel_attention_mask", "image_sizes"]

        self.update_replay_buffer(
            groups_with_variance,
            group_advantages,
            group_std_rewards,
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            forward_kwargs,
            vision_fields,
            old_per_token_logps,
            ref_per_token_logps,
            importance_sampling_ratio,
        )

        # Sample from replay buffer to replace groups with variance
        num_groups_to_replace = groups_without_variance.sum().item()
        if not num_groups_to_replace:
            return

        sampled_data = self.sample_from_replay_buffer(
            num_samples=num_groups_to_replace,
            optional_vision_fields=vision_fields,
            optional_tensor_fields=optional_tensor_fields,
        )

        # Pad sampled data if they are shorter than the current batch sequences
        # Or pad the current batch if sampled are longer
        current_batch_prompt_seq_len = prompt_ids.size(1)
        current_batch_completion_seq_len = completion_ids.size(1)

        groups_to_replace_idxs = groups_with_variance.logical_not().nonzero(as_tuple=True)[0].unique().tolist()

        # Determine target (max) sequence lengths once
        sampled_prompt_lengths = [t.size(1) for t in sampled_data["prompt_ids"]]
        sampled_completion_lengths = [t.size(1) for t in sampled_data["completion_ids"]]
        target_prompt_len = max([current_batch_prompt_seq_len] + sampled_prompt_lengths)
        target_completion_len = max([current_batch_completion_seq_len] + sampled_completion_lengths)

        # If any sampled prompt is longer, pad the whole batch prompt tensors once (left padding)
        if target_prompt_len > current_batch_prompt_seq_len:
            prompt_ids = pad(
                list(prompt_ids.unbind(0)),
                padding_value=self.pad_token_id,
                pad_to_multiple_of=target_prompt_len,
                padding_side="left",
            )
            prompt_mask = pad(
                list(prompt_mask.unbind(0)), padding_value=0, pad_to_multiple_of=target_prompt_len, padding_side="left"
            )
        # If any sampled completion is longer, pad the whole batch completion tensors once (right padding)
        if target_completion_len > current_batch_completion_seq_len:
            completion_ids = pad(
                list(completion_ids.unbind(0)),
                padding_value=self.pad_token_id,
                pad_to_multiple_of=target_completion_len,
                padding_side="right",
            )
            completion_mask = pad(
                list(completion_mask.unbind(0)),
                padding_value=0,
                pad_to_multiple_of=target_completion_len,
                padding_side="right",
            )
            if old_per_token_logps is not None:
                old_per_token_logps = pad(
                    list(old_per_token_logps.unbind(0)),
                    padding_value=0.0,
                    pad_to_multiple_of=target_completion_len,
                    padding_side="right",
                )
            if ref_per_token_logps is not None:
                ref_per_token_logps = pad(
                    list(ref_per_token_logps.unbind(0)),
                    padding_value=0.0,
                    pad_to_multiple_of=target_completion_len,
                    padding_side="right",
                )

        # Replace per-group data, padding only sampled groups that are shorter than the target
        for i, group_idx in enumerate(groups_to_replace_idxs):
            start_idx = group_idx * self.num_generations
            end_idx = (group_idx + 1) * self.num_generations
            idx_range = slice(start_idx, end_idx)

            # Pad sampled prompt to target length if needed
            if sampled_data["prompt_ids"][i].size(1) < target_prompt_len:
                sampled_data["prompt_ids"][i] = pad(
                    sampled_data["prompt_ids"][i],
                    padding_value=self.pad_token_id,
                    pad_to_multiple_of=target_prompt_len,
                    padding_side="left",
                )
                sampled_data["prompt_mask"][i] = pad(
                    sampled_data["prompt_mask"][i],
                    padding_value=0,
                    pad_to_multiple_of=target_prompt_len,
                    padding_side="left",
                )

            # Pad sampled completion to target length if needed
            if sampled_data["completion_ids"][i].size(1) < target_completion_len:
                sampled_data["completion_ids"][i] = pad(
                    sampled_data["completion_ids"][i],
                    padding_value=self.pad_token_id,
                    pad_to_multiple_of=target_completion_len,
                    padding_side="right",
                )
                sampled_data["completion_mask"][i] = pad(
                    sampled_data["completion_mask"][i],
                    padding_value=0,
                    pad_to_multiple_of=target_completion_len,
                    padding_side="right",
                )
                if "old_per_token_logps" in sampled_data:
                    sampled_data["old_per_token_logps"][i] = pad(
                        sampled_data["old_per_token_logps"][i],
                        padding_value=0.0,
                        pad_to_multiple_of=target_completion_len,
                        padding_side="right",
                    )
                if "ref_per_token_logps" in sampled_data:
                    sampled_data["ref_per_token_logps"][i] = pad(
                        sampled_data["ref_per_token_logps"][i],
                        padding_value=0.0,
                        pad_to_multiple_of=target_completion_len,
                        padding_side="right",
                    )

            # Assign (replace) group slice
            prompt_ids[idx_range] = sampled_data["prompt_ids"][i]
            prompt_mask[idx_range] = sampled_data["prompt_mask"][i]
            completion_ids[idx_range] = sampled_data["completion_ids"][i]
            completion_mask[idx_range] = sampled_data["completion_mask"][i]
            group_advantages[group_idx] = sampled_data["advantages"][i]

            if "old_per_token_logps" in sampled_data:
                old_per_token_logps[idx_range] = sampled_data["old_per_token_logps"][i]
            if "ref_per_token_logps" in sampled_data:
                ref_per_token_logps[idx_range] = sampled_data["ref_per_token_logps"][i]

            for field in vision_fields:
                if field in sampled_data and field in forward_kwargs:
                    forward_kwargs[field][idx_range] = sampled_data[field][i]

        # Prepare final outputs after sampling and replacement
        outputs_after_sampling_buffer = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": group_advantages,
        }

        # Replace optional tensor fields if they exist
        for field in optional_tensor_fields:
            if field in sampled_data:
                outputs_after_sampling_buffer[field] = (
                    old_per_token_logps if field == "old_per_token_logps" else ref_per_token_logps
                )

        # Replace vision fields if they exist
        for field in vision_fields:
            if field in sampled_data and field in forward_kwargs:
                outputs_after_sampling_buffer[field] = forward_kwargs[field]

        outputs_after_sampling_buffer["num_items_in_batch"] = num_items_in_batch
        if self.use_vllm and self.vllm_importance_sampling_correction:
            outputs_after_sampling_buffer["importance_sampling_ratio"] = importance_sampling_ratio

        return outputs_after_sampling_buffer
