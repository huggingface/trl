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
import torch.nn.functional as F
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

from ..data_utils import is_conversational, maybe_apply_chat_template
from ..import_utils import is_vllm_available
from ..models import create_reference_model, unwrap_model_for_generation
from .prime_config import PrimeConfig
from .utils import generate_model_card, get_comet_experiment_url, truncate_right


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

if is_vllm_available():
    from vllm import LLM, SamplingParams


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


class PrimeTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel, nn.Module] = None,
        reward_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        verifier_function: Optional[Callable] = None,
        args: PrimeConfig = None,
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
            args = PrimeConfig(f"{model_name}-PRIME")

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
                    "Invalid `torch_dtype` passed to `PrimeConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `PrimeConfig`, but your model is already instantiated. "
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

            def data_collator(features):  # No data collation is needed in PRIME
                return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |y_i| in the Prime doc
        self.num_generations = args.num_generations  # = K in the Prime doc

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
                # See https://github.com/vllm-project/vllm/pull/12071
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
        # input tensor associated with the key "input_ids". However, in PRIME, the sampled data does not include the
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
        if self.reward_model is not None:
            if self.is_fsdp_enabled:
                self.reward_model = prepare_fsdp(self.reward_model, self.accelerator)
            else:
                self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

        self.verifier_function = verifier_function

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In PrimeTrainer, we preprocess data, so using the model's signature columns doesn't work.
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

    def _generate_candidates(self, model, prompts):
        """Generate candidate completions using either vLLM or standard generation."""
        if self.use_vllm:
            return self._generate_vllm(model, prompts)
        else:
            return self._generate(model, prompts)

    def _compute_process_rewards(
        self,
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
    ):
        """
        Compute implicit process rewards using KL divergence between PRM and reference model.

        As per PRIME algorithm:
        r_φ(y) = β log(π_φ(y)/π_ref(y)) = β [log π_φ(y) - log π_ref(y)]
        where:
        - π_φ is the PRM
        - π_ref is the reference model
        - β is the KL penalty weight
        """
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        prompt_completion_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        def get_sequence_logprobs(model, input_ids, attention_mask):
            logits = model(input_ids, attention_mask).logits
            logits = logits[:, :-1, :]  # exclude last prediction
            input_ids = input_ids[:, 1:]  # exclude first input ID

            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, dim=2, index=input_ids.unsqueeze(-1)).squeeze(-1)

            # Sum log probs over sequence length to get sequence log prob
            sequence_log_probs = (token_log_probs * attention_mask[:, 1:]).sum(dim=1)
            return sequence_log_probs

        # Get PRM log probs
        with torch.inference_mode():
            prm_log_probs = get_sequence_logprobs(self.reward_model, prompt_completion_ids, prompt_completion_mask)

            # Get reference model log probs
            if self.ref_model is not None:
                ref_log_probs = get_sequence_logprobs(self.ref_model, prompt_completion_ids, prompt_completion_mask)
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter() as unwrapped_model:
                    ref_log_probs = get_sequence_logprobs(
                        unwrapped_model, prompt_completion_ids, prompt_completion_mask
                    )

        # Compute process rewards as KL divergence
        process_rewards = self.args.beta * (prm_log_probs - ref_log_probs)

        return process_rewards

    def _compute_prm_loss(
        self,
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
        verifier_rewards,
    ):
        """
        Compute Cross-Entropy loss for updating the Process Reward Model (PRM).

        As per the PRIME algorithm:
        L_CE(φ) = E_(x,y,r)~T [r log σ(r_φ(y)) + (1-r) log(1-σ(r_φ(y)))]
        where:
        - r_φ(y) is the process reward from PRM
        - r is the verifier (outcome) reward
        - σ is the sigmoid function
        """
        # Get PRM logits
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        prompt_completion_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Forward pass through reward model to get process rewards
        prm_outputs = self.reward_model(
            input_ids=prompt_completion_ids,
            attention_mask=prompt_completion_mask,
        )
        prm_logits = prm_outputs.logits[:, 0]  # Shape: [batch_size]

        # Convert verifier rewards to binary labels (0 or 1)
        binary_labels = (verifier_rewards > self.args.reward_threshold).float()

        # Compute binary cross-entropy loss
        return F.binary_cross_entropy_with_logits(prm_logits, binary_labels)

    def _compute_rewards(self, prompts, completion_ids, completion_mask, prompt_ids, prompt_mask):
        """
        Compute both verifier rewards and implicit process rewards.

        Returns:
            tuple: (verifier_rewards, process_rewards, mean_verifier_rewards, std_verifier_rewards)
        """
        # 1. Get verifier rewards (outcome rewards)
        verifier_rewards = []
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        for prompt, completion in zip(prompts, completions):
            if self.verifier_function is not None:
                reward = self.verifier_function(prompt, completion)
                verifier_rewards.append(reward)

        verifier_rewards = torch.tensor(verifier_rewards, device=self.accelerator.device)

        # 2. Get process rewards using KL divergence
        process_rewards = self._compute_process_rewards(
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
        )

        # Compute grouped rewards statistics
        mean_verifier_rewards = verifier_rewards.view(-1, self.num_generations).mean(dim=1)
        std_verifier_rewards = verifier_rewards.view(-1, self.num_generations).std(dim=1)

        # Repeat for each generation
        mean_verifier_rewards = mean_verifier_rewards.repeat_interleave(self.num_generations, dim=0)
        std_verifier_rewards = std_verifier_rewards.repeat_interleave(self.num_generations, dim=0)

        return (
            verifier_rewards,
            process_rewards,
            mean_verifier_rewards,
            std_verifier_rewards,
        )

    def _compute_prime_loss(
        self,
        model,
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
        verifier_rewards,
        process_rewards,
        old_logprobs=None,
    ):
        """
        Compute the PRIME loss using PPO with the combined advantages from verifier and process rewards.

        The advantage is computed as:
        A_i = r_i - (1/(K-1)) * sum_{j≠i} r_j + sum_{t} gamma^(t) * (r_i^t - (1/(K-1)) * sum_{j≠i} r_j^t)
        where:
        - r_i is the verifier (outcome) reward for sequence i
        - r_i^t is the process reward at token t for sequence i
        """
        # Get current policy logprobs
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        prompt_completion_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits = model(prompt_completion_ids, attention_mask=prompt_completion_mask).logits
        logits = logits[:, :-1]  # Remove last logit
        current_logprobs = F.log_softmax(logits, dim=-1)

        # Gather only the logprobs for the completion tokens
        completion_ids_shifted = completion_ids[:, 1:]  # Remove first token
        current_logprobs = torch.gather(current_logprobs, dim=2, index=completion_ids_shifted.unsqueeze(-1)).squeeze(
            -1
        )

        # If old_logprobs not provided (first iteration), use current logprobs
        if old_logprobs is None:
            old_logprobs = current_logprobs.detach()

        # Compute advantages using RLOO for both reward types
        K = self.args.num_generations
        batch_size = verifier_rewards.size(0) // K

        # Reshape rewards for RLOO computation
        grouped_verifier_rewards = verifier_rewards.view(batch_size, K)
        grouped_process_rewards = process_rewards.view(batch_size, K)

        # Compute leave-one-out means
        loo_verifier_means = (grouped_verifier_rewards.sum(dim=1, keepdim=True) - grouped_verifier_rewards) / (K - 1)
        loo_process_means = (grouped_process_rewards.sum(dim=1, keepdim=True) - grouped_process_rewards) / (K - 1)

        # Compute advantages for each sequence
        verifier_advantages = (grouped_verifier_rewards - loo_verifier_means).view(-1)  # Shape: [batch_size * K]
        process_advantages = (grouped_process_rewards - loo_process_means).view(-1)  # Shape: [batch_size * K]

        # Compute discounted process rewards
        discount_factors = self.args.gamma ** torch.arange(
            completion_mask.size(1), device=completion_mask.device
        ).unsqueeze(0)

        # Combine advantages as per PRIME paper
        advantages = (
            verifier_advantages.unsqueeze(1)  # Shape: [batch_size * K, 1]
            + process_advantages.unsqueeze(1) * discount_factors  # Shape: [batch_size * K, seq_len]
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute PPO policy loss
        ratio = torch.exp(current_logprobs - old_logprobs)
        clip_range = self.args.epsilon

        # Policy loss with clipping
        pg_loss1 = advantages * ratio
        pg_loss2 = advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        pg_loss = -torch.min(pg_loss1, pg_loss2)

        # Compute KL divergence for logging
        approx_kl = 0.5 * ((current_logprobs - old_logprobs) ** 2).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_range).float().mean()

        # Mask out padding tokens
        pg_loss = (pg_loss * completion_mask[:, 1:]).sum(dim=1) / completion_mask[:, 1:].sum(dim=1)
        pg_loss = pg_loss.mean()

        # Add KL penalty term from reference model
        if self.ref_model is not None:
            with torch.no_grad():
                ref_logits = self.ref_model(prompt_completion_ids, attention_mask=prompt_completion_mask).logits[
                    :, :-1
                ]
                ref_logprobs = F.log_softmax(ref_logits, dim=-1)
                ref_logprobs = torch.gather(ref_logprobs, dim=2, index=completion_ids_shifted.unsqueeze(-1)).squeeze(
                    -1
                )
        else:
            with self.accelerator.unwrap_model(model).disable_adapter():
                ref_logits = model(prompt_completion_ids, attention_mask=prompt_completion_mask).logits[:, :-1]
                ref_logprobs = F.log_softmax(ref_logits, dim=-1)
                ref_logprobs = torch.gather(ref_logprobs, dim=2, index=completion_ids_shifted.unsqueeze(-1)).squeeze(
                    -1
                )

        kl_div = (current_logprobs - ref_logprobs) * completion_mask[:, 1:]
        kl_div = kl_div.sum(dim=1) / completion_mask[:, 1:].sum(dim=1)
        kl_div = kl_div.mean()

        # Combine losses
        loss = pg_loss + self.args.beta * kl_div

        return loss, {
            "policy_loss": pg_loss.item(),
            "kl_div": kl_div.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": clipfrac.item(),
        }

    def train(self):
        """
        Main training loop for PRIME algorithm.

        Algorithm steps:
        1. Initialize policy, PRM, and reference models from SFT model
        2. For N iterations:
            a. Sample batch of prompts
            b. Generate K candidates per prompt
            c. Filter based on correct response ratio
            d. For M PPO epochs:
                - Update PRM with cross-entropy loss
                - Compute advantages and update policy with PPO
        """
        args = self.args
        accelerator = self.accelerator
        model = self.model
        optimizer = self.optimizer

        # Initialize models and metrics
        model.train()
        self.ref_model.eval()
        self.ref_model.to(accelerator.device)
        self.reward_model.train()
        self.reward_model.to(accelerator.device)

        # Create separate optimizer for PRM
        prm_optimizer = torch.optim.AdamW(
            self.reward_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # Initialize metrics to include both reward types
        self._metrics = {
            "policy_loss": [],
            "prm_loss": [],
            "kl_div": [],
            "approx_kl": [],
            "clipfrac": [],
            "verifier_reward": [],
            "process_reward": [],
        }

        # Training state initialization
        self.state.global_step = 0
        self.state.epoch = 0
        self.state.max_steps = args.num_total_batches
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Main training loop
        for epoch in range(args.num_train_epochs):
            for step, batch in enumerate(self.get_train_dataloader()):
                # Step 1: Generate candidates and compute initial metrics
                prompts = [x["prompt"] for x in batch]
                generation_outputs = self._generate_candidates(model, prompts)
                prompt_ids, prompt_mask, completion_ids, completion_mask = generation_outputs

                # Step 2: Filter based on correct response ratio
                with torch.no_grad():
                    # Compute rewards to determine correct responses
                    reward_outputs = self._compute_rewards(
                        prompts, completion_ids, completion_mask, prompt_ids, prompt_mask
                    )
                    verifier_rewards, process_rewards, mean_verifier_rewards, std_verifier_rewards = reward_outputs

                    # Filter based on verifier rewards
                    grouped_verifier_rewards = verifier_rewards.view(-1, args.num_generations)
                    correct_responses = (grouped_verifier_rewards > args.reward_threshold).sum(dim=1)
                    correct_ratios = correct_responses.float() / args.num_generations

                    # Apply filtering
                    valid_mask = (correct_ratios >= args.correct_ratio_min) & (
                        correct_ratios <= args.correct_ratio_max
                    )
                    if not valid_mask.any():
                        continue

                    # Update tensors with valid mask
                    prompt_ids = prompt_ids[valid_mask.repeat_interleave(args.num_generations)]
                    prompt_mask = prompt_mask[valid_mask.repeat_interleave(args.num_generations)]
                    completion_ids = completion_ids[valid_mask.repeat_interleave(args.num_generations)]
                    completion_mask = completion_mask[valid_mask.repeat_interleave(args.num_generations)]
                    verifier_rewards = verifier_rewards[valid_mask.repeat_interleave(args.num_generations)]
                    process_rewards = process_rewards[valid_mask.repeat_interleave(args.num_generations)]

                # Get initial logprobs for PPO
                with torch.no_grad():
                    old_logprobs = None  # Will be computed in first iteration

                # Step 3: PPO Training Loop
                for _ in range(args.num_ppo_epochs):
                    # Shuffle data for each PPO epoch
                    perm = torch.randperm(len(prompt_ids), device=prompt_ids.device)
                    prompt_ids = prompt_ids[perm]
                    prompt_mask = prompt_mask[perm]
                    completion_ids = completion_ids[perm]
                    completion_mask = completion_mask[perm]
                    verifier_rewards = verifier_rewards[perm]
                    process_rewards = process_rewards[perm]
                    if old_logprobs is not None:
                        old_logprobs = old_logprobs[perm]

                    # Process in mini-batches
                    for mb_start in range(0, len(prompt_ids), args.per_device_train_batch_size):
                        mb_end = min(mb_start + args.per_device_train_batch_size, len(prompt_ids))
                        mb_slice = slice(mb_start, mb_end)

                        # 1. Update PRM with Cross-Entropy loss
                        with self.accelerator.accumulate(self.reward_model):
                            prm_loss = self._compute_prm_loss(
                                prompt_ids[mb_slice],
                                prompt_mask[mb_slice],
                                completion_ids[mb_slice],
                                completion_mask[mb_slice],
                                verifier_rewards[mb_slice],
                            )

                            self.accelerator.backward(prm_loss)
                            if args.max_grad_norm is not None:
                                self.accelerator.clip_grad_norm_(self.reward_model.parameters(), args.max_grad_norm)
                            prm_optimizer.step()
                            prm_optimizer.zero_grad()

                            self._metrics["prm_loss"].append(prm_loss.item())

                        # 2. Update policy with PPO loss
                        with accelerator.accumulate(model):
                            # Get updated process rewards after PRM update
                            with torch.no_grad():
                                _, process_rewards_updated = self._compute_rewards(
                                    prompts[mb_slice], completion_ids[mb_slice], completion_mask[mb_slice]
                                )[:2]

                            # Compute PPO loss with PRIME advantages
                            policy_loss, metrics = self._compute_prime_loss(
                                model,
                                prompt_ids[mb_slice],
                                prompt_mask[mb_slice],
                                completion_ids[mb_slice],
                                completion_mask[mb_slice],
                                verifier_rewards[mb_slice],
                                process_rewards_updated,
                                old_logprobs[mb_slice] if old_logprobs is not None else None,
                            )

                            # Backward pass and optimization
                            accelerator.backward(policy_loss)
                            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()

                            # Store metrics
                            for k, v in metrics.items():
                                self._metrics[k].append(v)

                # Update training state
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / len(self.get_train_dataloader())

                # Handle callbacks and saving
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                if self.control.should_save:
                    self._save_checkpoint(model, trial=None)

                if self.control.should_training_stop:
                    break

            if self.control.should_training_stop:
                break

        # End of training
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
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
                TODO
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
            trainer_name="PRIME",
            trainer_citation=citation,
            paper_title="Process Reinforcement through Implicit Rewards",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
