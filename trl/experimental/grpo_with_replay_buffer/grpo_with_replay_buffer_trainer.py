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

import copy
import heapq
import re
from contextlib import nullcontext
from typing import Any, Optional, Union

import torch
from accelerate.utils import broadcast_object_list, gather_object
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.utils import is_flash_attn_2_available

from trl.data_utils import is_conversational, maybe_apply_chat_template, prepare_multimodal_messages
from trl.extras.profiling import profiling_context
from trl.import_utils import is_vllm_available
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.utils import (
    nanmax,
    nanmin,
    nanstd,
    pad,
    truncate_with_protected_tokens,
)

from .grpo_with_replay_buffer_config import GRPOWithReplayBufferConfig


if is_vllm_available():
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams


class ReplayBuffer:
    """
    A simple replay buffer to store and sample previously seen rollouts.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.heap = []  # Min-heap of (score, data) tuples

    def add(self, scores: list[float], data: list[dict]):
        for score, datum in zip(scores, data):
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
    def __init__(self, args: Optional[GRPOWithReplayBufferConfig] = None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.replay_buffer = ReplayBuffer(args.replay_buffer_size) if args.replay_buffer_size > 0 else None

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        # We don't yet support visual reward models/function, so we keep a copy of the original text-only prompts for
        # later use in the reward computation. If images are present, we insert {"type": "image"} as required by the
        # VLM chat template.
        original_prompts = copy.deepcopy(prompts)

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]}]
        kwargs = {}
        has_images = "image" in inputs[0]
        image_split_sizes = None
        if has_images:
            images = [example.get("image") for example in inputs]
            kwargs = {"images": [[img] for img in images]}
            for prompt in prompts:
                if isinstance(prompt, list):  # i.e., when using conversational data
                    prepare_multimodal_messages(prompt, num_images=1)

            if hasattr(self.processing_class, "_get_num_multimodal_tokens"):
                image_sizes = [(image.height, image.width) for image in images]
                multimodal_extra_data = self.processing_class._get_num_multimodal_tokens(image_sizes)
                image_split_sizes = multimodal_extra_data.num_image_patches

        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if "image_grid_thw" in prompt_inputs and image_split_sizes is None:
            # Fallback for VLMs that require image_grid_thw but don't provide _get_num_multimodal_tokens
            image_split_sizes = prompt_inputs["image_grid_thw"].prod(dim=1).tolist()

        if self.max_prompt_length is not None:
            # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
            # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
            # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
            protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
            protected = [token for token in protected if token is not None]
            prompt_ids, prompt_mask = truncate_with_protected_tokens(
                prompt_ids, prompt_mask, self.max_prompt_length, protected
            )

            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            prompts_text = [re.sub(rf"^({re.escape(self.pad_token)})+", "", text) for text in prompts_text]

            # The chat template sometimes inserts a single image token into the prompt text. However, when this text is
            # later tokenized, the single image token string is expanded into multiple image token IDs, depending on the
            # image size. Since we're detokenizing here, we may see repeated image tokens in the decoded text. We
            # collapse them back into a single token string to match the original chat template in case it originally
            # applies it. Otherwise, it assumes that the chat template uses only vision_start_token_id to indicate images
            # (e.g. Gemma 3) and removes all image_token instances and vision_end_token_id as well, leaving only
            # the vision_start_token_id (e.g. <start_of_image>).
            if self.image_token is not None:
                escaped_img_token = re.escape(self.image_token)
                # Search for the image token in the chat template
                if re.search(escaped_img_token, self.processing_class.chat_template):
                    prompts_text = [
                        re.sub(rf"({escaped_img_token})+", self.image_token, text) for text in prompts_text
                    ]
                else:
                    # If the chat template doesn't use the image token, we remove all instances of it + vision_end_token_id
                    if self.vision_end_token_id is not None:
                        escaped_eoi_token = re.escape(
                            self.processing_class.tokenizer.decode([self.vision_end_token_id])
                        )
                        prompts_text = [
                            re.sub(rf"({escaped_img_token})+{escaped_eoi_token}", "", text) for text in prompts_text
                        ]
                    else:
                        # If vision_end_token_id is None, just remove the image tokens
                        prompts_text = [re.sub(rf"({escaped_img_token})+", "", text) for text in prompts_text]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                # wake up colocated vLLM instances if needed
                torch.cuda.empty_cache()  # required to avoid OOM in some cases
                self.llm.wake_up()

            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if has_images:
                    all_images = gather_object(images)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

                    if has_images:
                        ordered_set_of_images = all_images[:: self.num_generations]
                    else:
                        ordered_set_of_images = None

                    with profiling_context(self, "vLLM.generate"):
                        output = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                        payload = (output["completion_ids"], output["logprobs"])
                else:
                    payload = None

                # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                completion_ids, all_logprobs = obj_list[0]

                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]
                all_logprobs = all_logprobs[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "guided_decoding": guided_decoding,
                    "logprobs": 0,  # only return the logprob of the generated token
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]

                    if has_images:
                        gathered_images = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_images, images, group=self.tp_group)
                        all_images = [img for sublist in gathered_images for img in sublist]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images if has_images else None

                if has_images and all_images:
                    vllm_inputs = []
                    for prompt, image in zip(all_prompts_text, all_images):
                        if image is not None:
                            vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
                        else:
                            vllm_inputs.append(prompt)
                else:
                    vllm_inputs = all_prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                all_logprobs = [
                    [next(iter(lp.values())).logprob for lp in output.logprobs]
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]
                    all_logprobs = all_logprobs[tp_slice]

                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            sampling_per_token_logps = [
                torch.tensor(logprobs, device=device, dtype=torch.float32) for logprobs in all_logprobs
            ]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0)

        elif self.use_transformers_paged:
            # Re-process inputs for paged generation if needed
            # Note: images are already validated and preprocessed above
            paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            previous_attn = self.model_wrapped.config._attn_implementation

            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        paged_prompt_inputs.input_ids, generation_config=self.generation_config, progress_bar=False
                    )
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
            prompt_ids = [torch.tensor(ids, device=device) for ids in paged_prompt_inputs.input_ids]
            prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            # Restore the original attention implementation, training mode
            self.model_wrapped.config._attn_implementation = previous_attn
        else:
            # Regular generation path
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config, disable_compile=True
                )
            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [row[mask_row].tolist() for row, mask_row in zip(completion_ids, completion_mask.bool())]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        num_items_in_batch = agg_completion_lengths.sum()  # this is required for the DAPO loss

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

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
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs.get("image_sizes"),
                    image_split_sizes=image_split_sizes,
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
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                        image_sizes=prompt_inputs.get("image_sizes"),
                        image_split_sizes=image_split_sizes,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            pixel_values=prompt_inputs.get("pixel_values"),
                            image_grid_thw=prompt_inputs.get("image_grid_thw"),
                            pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                            image_sizes=prompt_inputs.get("image_sizes"),
                            image_split_sizes=image_split_sizes,
                        )
            else:
                ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, original_prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        std_rewards = None
        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
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
        if std_rewards is None:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        std_rewards = std_rewards[process_slice] if std_rewards is not None else None

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

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

        if has_images:
            self._logs["image"].extend(gather_object(images))

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
            std_rewards,
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            prompt_inputs,
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
            optional_vision_fields = [
                "pixel_values",
                "image_grid_thw",
                "pixel_attention_mask",
                "image_sizes",
            ]
            for field in optional_vision_fields:
                if field in prompt_inputs:
                    output[field] = prompt_inputs[field]
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
        prompt_inputs: dict,
        optional_vision_fields: list[str] = None,
        old_per_token_logps: Optional[torch.Tensor] = None,
        ref_per_token_logps: Optional[torch.Tensor] = None,
        importance_sampling_ratio: Optional[float] = None,
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
            prompt_inputs: Dictionary containing additional prompt inputs (vision data, etc.)
            optional_vision_fields: List of optional vision-related fields to include if present in prompt_inputs
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
                    if field_name in prompt_inputs:
                        buffered_output[field_name] = self.slice_group_data(
                            prompt_inputs[field_name], prompt_mask, group_idx
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
        prompt_inputs: dict,
        num_items_in_batch: int,
        old_per_token_logps: Optional[torch.Tensor] = None,
        ref_per_token_logps: Optional[torch.Tensor] = None,
        importance_sampling_ratio: Optional[float] = None,
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
            prompt_inputs: Dictionary containing additional prompt inputs (vision data, etc.)
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
            prompt_inputs,
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
                if field in sampled_data and field in prompt_inputs:
                    prompt_inputs[field][idx_range] = sampled_data[field][i]

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
            if field in sampled_data and field in prompt_inputs:
                outputs_after_sampling_buffer[field] = prompt_inputs[field]

        outputs_after_sampling_buffer["num_items_in_batch"] = num_items_in_batch
        if self.use_vllm and self.vllm_importance_sampling_correction:
            outputs_after_sampling_buffer["importance_sampling_ratio"] = importance_sampling_ratio

        return outputs_after_sampling_buffer
