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

import textwrap
from collections import defaultdict, deque
from collections.abc import Callable
from functools import partial
from typing import Any

import datasets
import jinja2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from accelerate import logging
from accelerate.utils import (
    gather_object,
)
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader, Sampler
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_trackio_available,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import (
    is_datasets_available,
    is_peft_available,
    is_rich_available,
)

from ...data_utils import (
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages,
)
from ...models import prepare_deepspeed
from ...trainer.grpo_trainer import GRPOTrainer
from ...trainer.utils import (
    SIMPLE_CHAT_TEMPLATE,
    RepeatSampler,
    get_config_model_id,
    nanstd,
    pad,
)
from ..judges import BasePairwiseJudge
from .online_dpo_config import OnlineDPOConfig
from .utils import print_pairwise_prompt_completions_sample


if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

if is_trackio_available():
    import trackio


logger = logging.get_logger(__name__)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = str | PreTrainedModel | Callable[[list, list], list[float]]

# What we call a rollout function is a callable that takes prompts (list) and the trainer instance as parameters and
# returns a dict of generation results. Those results must include "prompt_ids", "completion_ids", and "logprobs"
# fields. Any extra fields (per-completion) are forwarded to the reward functions.
RolloutFunc = Callable[[list[str], "OnlineDPOTrainer"], dict[str, Any]]


class OnlineDPOTrainer(GRPOTrainer):
    """
    Trainer for the Online Direct Preference Optimization (DPO) algorithm.

    It is implemented as a subclass of [`GRPOTrainer`].

    Args:
        model (`str | nn.Module | PreTrainedModel`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        judge ([`experimental.judges.BasePairwiseJudge`]):
            The judge to use for pairwise comparison of model completions.
        reward_funcs (`RewardFunc | list[RewardFunc]`, *optional*):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function: Can be a string (path to model), a [`~transformers.PreTrainedModel`], or a
              custom callable function.
            - A list of reward functions: Must all be of compatible types.
            - None: If `judge` is provided, `reward_funcs` is set to `None`.

            Note: Only one of `judge`, or `reward_funcs` should be provided.
        args ([`experimental.online_dpo.OnlineDPOConfig`], *optional*):
            The online DPO config arguments to use for training. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        reward_processing_classes ([`~transformers.PreTrainedTokenizerBase`] or `list[PreTrainedTokenizerBase]`, *optional*):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using
            [`~transformers.AutoTokenizer.from_pretrained`]. For elements in `reward_funcs` that are custom reward
            functions (not [`~transformers.PreTrainedModel`]), the corresponding entries in `reward_processing_classes`
            are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        rollout_func (`RolloutFunc`, *optional*):
            Function to use for generating completions. It receives the list of prompts allocated to the current
            process and the trainer instance. It must return a dict with `"prompt_ids"`, `"completion_ids"`, and
            `"logprobs"` fields. Any other fields are forwarded to the reward functions. This feature is experimental
            and may change or be removed at any time without prior notice.
    """

    _tag_names = ["trl", "online-dpo"]
    _name = "Online DPO"
    _paper = {
        "title": "Direct Language Model Alignment from Online AI Feedback",
        "id": "2402.04792",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{guo2024direct,
                title        = {{Direct Language Model Alignment from Online AI Feedback}},
                author       = {Shangmin Guo and Biao Zhang and Tianlin Liu and Tianqi Liu and Misha Khalman and Felipe Llinares and Alexandre Ram{\'{e}} and Thomas Mesnard and Yao Zhao and Bilal Piot and Johan Ferret and Mathieu Blondel},
                year         = 2024,
                eprint       = {arXiv:2402.04792}
            }"""),
    }

    def __init__(
        self,
        model: str | PreTrainedModel,
        judge: BasePairwiseJudge | None = None,
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
        args: OnlineDPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        rollout_func: RolloutFunc | None = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = OnlineDPOConfig(f"{model_name}-OnlineDPO")

        if args.use_liger_kernel:
            raise ValueError("Liger kernel is not supported in Online DPO Trainer.")

        # Validate reward configuration - must have exactly one of: judge, or reward_funcs
        reward_configs = sum(x is not None for x in [judge, reward_funcs])
        if reward_configs == 0:
            raise ValueError("One of `judge` or `reward_funcs` must be provided.")
        elif reward_configs > 1:
            if judge is not None:
                logger.warning(
                    "Both `judge` and `reward_funcs` are provided. Using `judge` and ignoring `reward_funcs`.",
                    UserWarning,
                )
                reward_funcs = None

        if reward_funcs is None:
            # Add a dummy reward function that returns zero rewards for correctness of GRPO initialization
            def dummy_reward_func(prompts: list[str], completions: list[str]) -> list[float]:
                return [0.0] * len(prompts)

            reward_funcs = dummy_reward_func
            reward_processing_classes = None

        # Call GRPO initializer
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            rollout_func=rollout_func,
        )
        # Reference model was already initialized in GRPOTrainer.__init__

        # Updated structure of `_logs` to accommodate chosen and rejected completions separately
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "chosen_completion": deque(maxlen=args.generation_batch_size),
            "rejected_completion": deque(maxlen=args.generation_batch_size),
            "chosen_rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "rejected_rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
        }

        self.judge = judge
        # Initialize judge
        if self.judge is not None:
            # If our judge is implemented by a pairwise reward model, prepare it
            if hasattr(self.judge, "model") and isinstance(self.judge.model, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.judge.model = prepare_deepspeed(self.judge.model, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.judge.model = self.accelerator.prepare_model(
                        self.judge.model, evaluation_mode=True, device_placement=True
                    )

    # This method overrides `GRPOTrainer.get_train_dataloader` to support even more custom batching strategy.
    # GRPO trainer returns batches of size `self._train_batch_size * self.args.steps_per_generation`.
    # In Online DPO, since we are going to merge two generations per prompt, we need to return batches of size
    # `self._train_batch_size * self.args.steps_per_generation * self.num_generations`.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        generation_batch_size = (
            self.num_generations * self._train_batch_size * self.args.steps_per_generation
        )  # < this is the change
        dataloader_params = {
            "batch_size": generation_batch_size,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        # One-line modification from GRPOTrainer: generated completions for each pair of duplicates are merged in post-processing.
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size,  # <- this is the change
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

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

        # Generation completions using the same method as in GRPOTrainer
        prompt_ids_list, completion_ids_list, _, sampling_per_token_logps_list, extra_fields = self._generate(prompts)
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
            sampling_logps = (sampling_per_token_logps * completion_mask).sum(dim=1)  # (B,)
        else:
            sampling_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        # Twice the batch size since each prompt has 2 completions
        batch_size = (
            2 * self.args.per_device_train_batch_size if mode == "train" else 2 * self.args.per_device_eval_batch_size
        )

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs
        if images is not None:
            prompts_text = [
                apply_chat_template({"prompt": prompt}, self.processing_class, **self.chat_template_kwargs)["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            # call BaseTrainer (grandparent) implementation of _prepare_inputs
            prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
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
            # Compute the per-token log probabilities for the reference model
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
            # Compute the sequence-level log probabilities for the reference model
            ref_logps = (ref_per_token_logps * completion_mask).sum(dim=1)  # (B,)

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

        # We must be guaranteed to have 2 completions per prompt in Online DPO for each device
        assert all(prompts[i] == prompts[i + 1] for i in range(0, len(prompts), 2)), (
            "Each prompt must have 2 completions."
        )
        unique_prompts = prompts[::2]

        if self.judge is not None:
            # Once formatted, conversational data may contain special tokens (such as <|im_start|>) that are not
            # directly understandable by the judge and could alter its judgment. To avoid this and make the judge
            # independent of the model's chat template, we use the raw conversation data, and apply our own chat
            # template to it.
            if is_conversational({"prompt": prompts[0]}):
                environment = jinja2.Environment()
                template = environment.from_string(SIMPLE_CHAT_TEMPLATE)
                unique_prompts = [template.render(messages=prompt) for prompt in unique_prompts]
                completions = [template.render(messages=completion) for completion in completions]

            ranks_of_first_completion = self.judge.judge(
                unique_prompts, list(zip(completions[::2], completions[1::2], strict=True))
            )
            # True if first completion is preferred
            mask = torch.tensor([rank == 0 for rank in ranks_of_first_completion], device=device)
        else:
            # Compute rewards for each completion using the reward functions
            all_processes_rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
            # Actually, we don't need to aggregate rewards since each prompt has exactly 2 completions
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            rewards_per_func = all_processes_rewards_per_func[process_slice]
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            # Reshape rewards to (num_samples, num_generations)
            rewards = rewards.view(-1, self.num_generations)
            # Determine which completion is preferred based on rewards
            mask = torch.argmax(rewards, dim=1) == 0  # True if first completion is preferred

        # Expand the per-prompt preference mask to all completions in the batch.
        # Example: mask = [True, False, False, True] -> chosen_mask = [True, False, False, True, False, True, True, False]
        chosen_mask = torch.stack([mask, ~mask], dim=1).reshape(-1)
        rejected_mask = ~chosen_mask

        # Log prompt and completion texts
        prompt_logs = gather_object(prompts_text[:: self.num_generations])
        chosen_logs = gather_object([completions_text[i] for i, chosen in enumerate(chosen_mask) if chosen])
        rejected_logs = gather_object([completions_text[i] for i, rejected in enumerate(rejected_mask) if rejected])

        self._logs["prompt"].extend(prompt_logs)
        self._logs["chosen_completion"].extend(chosen_logs)
        self._logs["rejected_completion"].extend(rejected_logs)

        if self.judge is None:
            # Also log chosen and rejected rewards per reward function
            for i, reward_func_name in enumerate(self.reward_func_names):
                func_chosen_rewards = all_processes_rewards_per_func[:, i][chosen_mask]
                func_rejected_rewards = all_processes_rewards_per_func[:, i][rejected_mask]
                self._logs["chosen_rewards"][reward_func_name].extend(func_chosen_rewards.cpu().numpy().tolist())
                self._logs["rejected_rewards"][reward_func_name].extend(func_rejected_rewards.cpu().numpy().tolist())

                chosen_mean_rewards = torch.nanmean(func_chosen_rewards).item()
                chosen_std_func_rewards = nanstd(func_chosen_rewards).item()
                self._metrics[mode][f"rewards/{reward_func_name}/chosen_mean"].append(chosen_mean_rewards)
                self._metrics[mode][f"rewards/{reward_func_name}/chosen_std"].append(chosen_std_func_rewards)

                rejected_mean_rewards = torch.nanmean(func_rejected_rewards).item()
                rejected_std_func_rewards = nanstd(func_rejected_rewards).item()
                self._metrics[mode][f"rewards/{reward_func_name}/rejected_mean"].append(rejected_mean_rewards)
                self._metrics[mode][f"rewards/{reward_func_name}/rejected_std"].append(rejected_std_func_rewards)

                margin = func_chosen_rewards - func_rejected_rewards
                margin_mean = torch.nanmean(margin).item()
                margin_std = nanstd(margin).item()
                self._metrics[mode][f"rewards/{reward_func_name}/margin_mean"].append(margin_mean)
                self._metrics[mode][f"rewards/{reward_func_name}/margin_std"].append(margin_std)

        if images is not None:
            self._logs["images"].extend(gather_object(images[:: self.num_generations]))

        output = {
            "prompt_ids": prompt_ids[::2],
            "prompt_mask": prompt_mask[::2],
            "chosen_completion_ids": completion_ids[chosen_mask],
            "chosen_completion_mask": completion_mask[chosen_mask],
            "chosen_ref_logps": ref_logps[chosen_mask],
            "rejected_completion_ids": completion_ids[rejected_mask],
            "rejected_completion_mask": completion_mask[rejected_mask],
            "rejected_ref_logps": ref_logps[rejected_mask],
        }

        if sampling_logps is not None:
            # Include to have an unbiased estimate of KL-divergence for logging
            output["chosen_sampling_logps"] = sampling_logps[chosen_mask]
            output["rejected_sampling_logps"] = sampling_logps[rejected_mask]
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

    def _compute_loss(self, model: PreTrainedModel | nn.Module, inputs: dict[str, torch.Tensor | Any]) -> torch.Tensor:
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        chosen_completion_ids, chosen_completion_mask = (
            inputs["chosen_completion_ids"],
            inputs["chosen_completion_mask"],
        )
        rejected_completion_ids, rejected_completion_mask = (
            inputs["rejected_completion_ids"],
            inputs["rejected_completion_mask"],
        )

        chosen_input_ids = torch.cat([prompt_ids, chosen_completion_ids], dim=1)
        chosen_attention_mask = torch.cat([prompt_mask, chosen_completion_mask], dim=1)
        rejected_input_ids = torch.cat([prompt_ids, rejected_completion_ids], dim=1)
        rejected_attention_mask = torch.cat([prompt_mask, rejected_completion_mask], dim=1)

        # Concatenate chosen and rejected inputs for efficient processing
        input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
        logits_to_keep = chosen_completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        completion_mask = torch.cat([chosen_completion_mask, rejected_completion_mask], dim=0)
        logps = (per_token_logps * completion_mask).sum(dim=1)
        sum_entropy = (entropies * completion_mask).sum(dim=1)
        mean_entropy = sum_entropy / completion_mask.sum(dim=1)

        chosen_logps, rejected_logps = torch.split(
            logps, [chosen_completion_ids.size(0), rejected_completion_ids.size(0)], dim=0
        )
        chosen_sum_entropies, rejected_sum_entropies = torch.split(
            sum_entropy, [chosen_completion_ids.size(0), rejected_completion_ids.size(0)], dim=0
        )
        chosen_mean_entropy, rejected_mean_entropy = torch.split(
            mean_entropy, [chosen_completion_ids.size(0), rejected_completion_ids.size(0)], dim=0
        )

        # Gather the reference log probabilities for chosen and rejected completions
        chosen_ref_logps, rejected_ref_logps = inputs["chosen_ref_logps"], inputs["rejected_ref_logps"]

        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = chosen_ref_logps - rejected_ref_logps

        logits = pi_logratios - ref_logratios

        # Support beta provided as a list/tuple (schedule per epoch) or a scalar
        beta = self.beta
        if isinstance(beta, (list, tuple)):
            epoch_idx = self.state.epoch
            beta = beta[min(epoch_idx, len(beta) - 1)]

        if self.args.loss_type == "sigmoid":
            losses = -F.logsigmoid(beta * logits)
        elif self.args.loss_type == "ipo":
            losses = (logits - 1 / (2 * beta)) ** 2
        else:
            raise NotImplementedError(f"invalid loss type {self.loss_type}")

        loss = losses.mean()

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        with torch.no_grad():
            chosen_kl = torch.exp(chosen_ref_logps - chosen_logps) - (chosen_ref_logps - chosen_logps) - 1
            chosen_sampling_logps = inputs.get("chosen_sampling_logps")
            if chosen_sampling_logps is not None:
                # Perform an IS-correction if sampling was performed from a different distribution
                importance_weights = torch.exp(chosen_logps - chosen_sampling_logps)
                chosen_kl = chosen_kl * importance_weights

            rejected_kl = torch.exp(rejected_ref_logps - rejected_logps) - (rejected_ref_logps - rejected_logps) - 1
            rejected_sampling_logps = inputs.get("rejected_sampling_logps")
            if rejected_sampling_logps is not None:
                # Perform an IS-correction if sampling was performed from a different distribution
                importance_weights = torch.exp(rejected_logps - rejected_sampling_logps)
                rejected_kl = rejected_kl * importance_weights

            chosen_rewards = self.beta * (chosen_logps - chosen_ref_logps)
            gathered_chosen_rewards = self.accelerator.gather_for_metrics(chosen_rewards)
            self._metrics[mode]["rewards/chosen"].append(gathered_chosen_rewards.mean().item())

            rejected_rewards = self.beta * (rejected_logps - rejected_ref_logps)
            gathered_rejected_rewards = self.accelerator.gather_for_metrics(rejected_rewards)
            self._metrics[mode]["rewards/rejected"].append(gathered_rejected_rewards.mean().item())

            margin = gathered_chosen_rewards - gathered_rejected_rewards
            self._metrics[mode]["rewards/margin"].append(margin.mean().item())
            accuracy = margin > 0
            self._metrics[mode]["rewards/accuracies"].append(accuracy.float().mean().item())

        self._metrics[mode]["kl/chosen"].append(self.accelerator.gather(chosen_kl).nanmean().item())
        self._metrics[mode]["kl/rejected"].append(self.accelerator.gather(rejected_kl).nanmean().item())

        self._metrics[mode]["mean_entropy/chosen"].append(
            self.accelerator.gather(chosen_mean_entropy).nanmean().item()
        )
        self._metrics[mode]["mean_entropy/rejected"].append(
            self.accelerator.gather(rejected_mean_entropy).nanmean().item()
        )

        self._metrics[mode]["sum_entropy/chosen"].append(
            self.accelerator.gather(chosen_sum_entropies).nanmean().item()
        )
        self._metrics[mode]["sum_entropy/rejected"].append(
            self.accelerator.gather(rejected_sum_entropies).nanmean().item()
        )
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super(GRPOTrainer, self).log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            prompts = list(self._logs["prompt"])
            chosen = list(self._logs["chosen_completion"])
            rejected = list(self._logs["rejected_completion"])
            row_count = min(len(prompts), len(chosen), len(rejected))

            if row_count == 0:
                return

            prompts = prompts[:row_count]
            chosen = chosen[:row_count]
            rejected = rejected[:row_count]

            def _pad(values: list[Any]) -> list[Any]:
                values = list(values)
                if len(values) >= row_count:
                    return values[:row_count]
                return values + [None] * (row_count - len(values))

            chosen_rewards = {name: _pad(vals) for name, vals in self._logs["chosen_rewards"].items()}
            rejected_rewards = {name: _pad(vals) for name, vals in self._logs["rejected_rewards"].items()}

            reward_keys = list(chosen_rewards.keys())
            for name in rejected_rewards.keys():
                if name not in reward_keys:
                    reward_keys.append(name)

            for name in reward_keys:
                chosen_rewards.setdefault(name, [None] * row_count)
                rejected_rewards.setdefault(name, [None] * row_count)

            if is_rich_available():
                print_pairwise_prompt_completions_sample(
                    prompts,
                    chosen,
                    chosen_rewards,
                    None,
                    self.state.global_step,
                    self.num_completions_to_print,
                    rejected_completions=rejected,
                    rejected_rewards=rejected_rewards,
                )

            logging_backends = []
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                logging_backends.append(wandb)
            if self.args.report_to and "trackio" in self.args.report_to:
                logging_backends.append(trackio)

            table = {
                "step": [str(self.state.global_step)] * row_count,
                "prompt": prompts,
                "chosen_completion": chosen,
                "rejected_completion": rejected,
            }
            for name in reward_keys:
                table[f"chosen_{name}"] = chosen_rewards[name]
                table[f"rejected_{name}"] = rejected_rewards[name]

            df_base = pd.DataFrame(table)
            images_raw = list(self._logs["images"])[:row_count] if self._logs["images"] else []

            for logging_backend in logging_backends:
                if images_raw:
                    images = []
                    for image_list in images_raw:
                        images.append([logging_backend.Image(image) for image in image_list] if image_list else None)
                    df = pd.concat(
                        [df_base, pd.Series(images, name="image")],
                        axis=1,
                        copy=False,
                    )
                else:
                    df = df_base

                if self.log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])

                logging_backend.log({"completions": logging_backend.Table(dataframe=df)})
