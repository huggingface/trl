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

import os
import textwrap
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollator,
    EvalPrediction,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_peft_available

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..import_utils import is_vllm_available
from ..models import create_reference_model, unwrap_model_for_generation
from .grpo_config import GRPOConfig
from .utils import generate_model_card, get_comet_experiment_url, truncate_right


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb


def prepare_fsdp(model, accelerator):
    if not isinstance(model, FSDP):
        accelerator.state.fsdp_plugin.set_auto_wrap_policy(model)
        fsdp_plugin = accelerator.state.fsdp_plugin
        kwargs = {
            "sharding_strategy": fsdp_plugin.sharding_strategy,
            "cpu_offload": fsdp_plugin.cpu_offload,
            "auto_wrap_policy": fsdp_plugin.auto_wrap_policy,
            "mixed_precision": fsdp_plugin.mixed_precision_policy,
            "sync_module_states": fsdp_plugin.sync_module_states,
            "backward_prefetch": fsdp_plugin.backward_prefetch,
            "forward_prefetch": fsdp_plugin.forward_prefetch,
            "use_orig_params": fsdp_plugin.use_orig_params,
            "param_init_fn": fsdp_plugin.param_init_fn,
            "ignored_modules": fsdp_plugin.ignored_modules,
            "limit_all_gathers": fsdp_plugin.limit_all_gathers,
            "device_id": accelerator.device,
        }
        model = FSDP(model, **kwargs)
    return model


class GRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel, nn.Module] = None,
        reward_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: GRPOConfig = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_class: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward model
        if isinstance(reward_model, str):
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                reward_model, num_labels=1, **model_init_kwargs
            )
        self.reward_model = reward_model

        # Reward processing class
        if reward_processing_class is None:
            reward_processing_class = AutoTokenizer.from_pretrained(reward_model.config._name_or_path)
        if reward_processing_class.pad_token_id is None:
            reward_processing_class.pad_token = reward_processing_class.eos_token
        self.reward_processing_class = reward_processing_class
        # The reward model computes the reward for the latest non-padded token in the input sequence.
        # So it's important to set the pad token ID to the padding token ID of the processing class.
        self.reward_model.config.pad_token_id = reward_processing_class.pad_token_id

        # Data loading and preprocessing
        if data_collator is None:

            def data_collator(features):  # No data collation is needed in GRPO
                return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.generation_config = SamplingParams(
                n=self.num_generations,  # 2 generations per prompt
                max_tokens=args.max_completion_length,
                temperature=args.temperature,
                top_k=50,
                top_p=1.0,
                detokenize=False,  # to avoid vllm to decode (we don't need it)
            )
            # vLLM dynamically adjusts the size of the key-value cache based on available GPU memory at instantiation.
            # A larger cache size improves speed, so we would expect gpu_memory_utilization=1.
            # However, at this stage, the optimizer's weights are not yet loaded onto the GPU; they will be loaded
            # after the first optimizer step and remain in GPU memory throughout training. So we must reserve enough
            # space for them. Setting gpu_memory_utilization to 0.55 seems to work well in practice.
            self.llm = LLM(
                model=model.name_or_path,
                gpu_memory_utilization=0.55,
                dtype=torch.float32,
                # When release by vLLM, we would be able to distribute the model on multiple GPUs
                # tensor_parallel_size=torch.cuda.device_count(),
                # distributed_executor_backend="external_launcher",
            )
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=self.num_generations,
                pad_token_id=processing_class.pad_token_id,
            )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"kl": [], "reward": [], "reward_std": []}

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Prepare the ref model
        if self.ref_model is not None:
            if self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        # Prepare the reward model
        if self.is_fsdp_enabled:
            self.reward_model = prepare_fsdp(self.reward_model, self.accelerator)
        else:
            self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def _generate_vllm(self, model, prompts):
        eos_token_id = self.processing_class.eos_token_id
        pad_token_id = self.processing_class.pad_token_id

        # Load the latest weights
        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(model.state_dict().items())

        if is_conversational({"prompt": prompts[0]}):
            outputs = self.llm.chat(prompts, self.generation_config, use_tqdm=False)
        else:
            outputs = self.llm.generate(prompts, self.generation_config, use_tqdm=False)

        completion_ids = [list(output.outputs[i].token_ids) for i in range(2) for output in outputs]
        prompt_ids = [list(output.prompt_token_ids) for _ in range(2) for output in outputs]

        # Create mask and pad the prompt and completion
        max_prompt_length = max(len(ids) for ids in prompt_ids)
        prompt_mask = [[0] * (max_prompt_length - len(ids)) + [1] * len(ids) for ids in prompt_ids]
        prompt_ids = [[pad_token_id] * (max_prompt_length - len(ids)) + ids for ids in prompt_ids]
        max_tokens = self.generation_config.max_tokens
        completion_mask = [[1] * len(ids) + [0] * (max_tokens - len(ids)) for ids in completion_ids]
        completion_ids = [
            ids + [eos_token_id] if ids[-1] != eos_token_id and len(ids) < max_tokens else ids
            for ids in completion_ids
        ]
        completion_ids = [ids + [pad_token_id] * (max_tokens - len(ids)) for ids in completion_ids]

        # Convert to tensors
        prompt_ids = torch.tensor(prompt_ids, device=self.accelerator.device)
        prompt_mask = torch.tensor(prompt_mask, device=self.accelerator.device)
        completion_ids = torch.tensor(completion_ids, device=self.accelerator.device)
        completion_mask = torch.tensor(completion_mask, device=self.accelerator.device)

        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def _generate(self, model, prompts):
        eos_token_id = self.processing_class.eos_token_id
        pad_token_id = self.processing_class.pad_token_id

        inputs = [{"prompt": prompt} for prompt in prompts]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions
        if self.is_fsdp_enabled:
            # From https://github.com/databricks/Compose-RL/blob/36c7a859128240efd6e1c7d2f2ca7f69f323c5f4/compose_rl/ppo/model.py#L158
            with FSDP.summon_full_params(model, writeback=False, recurse=False):
                prompt_completion_ids = model.generate(
                    input_ids=prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    input_ids=prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

        prompt_ids = prompt_ids.repeat_interleave(self.num_generations, dim=0)
        prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completion_ids, completion_mask = truncate_right(completion_ids, eos_token_id, pad_token_id)
        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # Generate completions
        prompts = [x["prompt"] for x in inputs]
        if self.use_vllm:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate_vllm(model, prompts)
        else:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate(model, prompts)

        prompt_length = prompt_ids.size(1)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        prompt_completion_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Get the per-token log probabilities for the completions for the model and the reference model
        def get_per_token_logps(model, input_ids, attention_mask):
            logits = model(input_ids, attention_mask).logits  # (B, L, V)
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        per_token_logps = get_per_token_logps(model, prompt_completion_ids, prompt_completion_mask)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(
                    self.ref_model, prompt_completion_ids, prompt_completion_mask
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids, prompt_completion_mask)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
            messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
            texts = [apply_chat_template(x, self.reward_processing_class)["text"] for x in messages]
            reward_inputs = self.reward_processing_class(
                texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
            )
        else:
            texts = [p + c for p, c in zip(prompts, completions)]
            reward_inputs = self.reward_processing_class(
                texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
            )
        reward_inputs = super()._prepare_inputs(reward_inputs)
        with torch.inference_mode():
            rewards = self.reward_model(**reward_inputs).logits[:, 0]  # Shape (B*G,)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        advantages = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(advantages - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items() if val}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics = {key: [] for key in self._metrics}

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
