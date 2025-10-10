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

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "math-verify",
#     "latex2sympy2_extended",
#     "trackio",
#     "kernels",
# ]
# ///

"""
LoRA Without Regret - Online DPO Training Script
Based on: https://thinkingmachines.ai/blog/lora/

Trains a language model using Online DPO with LoRA adapters for parameter-efficient
fine-tuning on mathematical reasoning tasks.

Features:
- Online DPO training with custom reward functions
- LoRA for parameter-efficient fine-tuning
- BEMA (Bias-Corrected Exponential Moving Average) for dynamic reference model updating
  - **PEFT/LoRA Compatible**: Updates base model with smoothed merged weights
  - Creates a dynamic reference that follows policy in a smoothed, lagged manner
  - For LoRA: Policy = BEMA(base) + LoRA adapters, Reference = BEMA(base)
  - Improves training stability compared to fixed reference model
  - The final saved BEMA model often generalizes better than the last checkpoint
  - Default BEMA settings (can be modified in the script):
    * update_freq=50: Update BEMA weights every 50 steps
    * ema_power=0.5: EMA decay factor power (κ in BEMA paper)
    * bias_power=0.2: BEMA scaling factor power (η in BEMA paper)
    * lag=10: Initial offset for smoothness (ρ in BEMA paper)
    * update_after=50: Burn-in period before BEMA starts (τ in BEMA paper)
    * device="cpu": Store BEMA buffers on CPU to save GPU memory

Usage:
hf jobs uv run \\
    --flavor a100-large \\
    --timeout 4h \\
    --secrets HF_TOKEN \\
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    online_dpo.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name HuggingFaceH4/OpenR1-Math-220k-default-verified \
    --output_dir online-dpo-lora-qwen3-0.6b \
    --learning_rate 1.0e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.0 \
    --max_grad_norm 1.0 \
    --beta 0.1 \
    --max_length 2048 \
    --max_new_tokens 2048 \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --use_peft \
    --lora_r 1 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --lora_target_modules all-linear \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --max_steps 200 \
    --report_to trackio

Note: BEMA (Bias-Corrected Exponential Moving Average) is automatically enabled for
dynamic reference model updating, with full support for LoRA/PEFT training.

How BEMA works with LoRA:
1. During training, LoRA adapters are trained on top of the base model
2. Every 50 steps (after step 50), BEMA:
   a. Computes merged weights: base_weights + LoRA_deltas
   b. Applies BEMA smoothing to the merged weights
   c. Updates the shared base model with smoothed BEMA weights
3. Both policy and reference now use the BEMA-smoothed base:
   - Policy forward pass: BEMA(base) + enabled LoRA adapters
   - Reference forward pass: BEMA(base) + disabled LoRA adapters (= just BEMA(base))

This creates a dynamic reference model that follows the policy but in a lagged, smoothed
manner, which improves training stability compared to a completely fixed reference.

The LoRA adapters continue training on top of the evolving BEMA base, and the final
BEMA model often generalizes better than the last checkpoint because it represents
a smoothed trajectory through weight space.

BEMA weights are also saved to bema_state_dict.pt at checkpoints for later use.

To customize BEMA parameters, modify the PEFTBEMACallback initialization in the script
(lines 519-529). To disable BEMA, comment out or remove the callback section.

For more details on BEMA, see: https://huggingface.co/papers/2508.00180
"""

import os
from typing import Optional
from datetime import datetime

import torch
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from transformers import AutoTokenizer

from trl import (
    OnlineDPOConfig,
    OnlineDPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

# Import BEMA dependencies
try:
    from transformers import PreTrainedModel, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    from accelerate.utils import is_peft_model
    BEMA_AVAILABLE = True
except ImportError:
    BEMA_AVAILABLE = False
    print("Warning: BEMA dependencies not available.")

# Disable GPU memory logging to avoid NCCL OOM errors during distributed logging
os.environ.setdefault("TRANSFORMERS_DISABLE_GPU_MEMORY_LOGGING", "1")

################
# PEFT-Compatible BEMA Callback
################

class PEFTBEMACallback(TrainerCallback):
    """
    BEMA callback that works with PEFT/LoRA models for dynamic reference model updating.

    This callback:
    - Merges LoRA adapters with base model to compute full effective weights
    - Applies BEMA (Bias-Corrected Exponential Moving Average) to the merged weights
    - Updates the shared base model with smoothed BEMA weights

    How it works with OnlineDPO + LoRA:
    - Policy model = BEMA(base) + enabled LoRA adapters
    - Reference model = BEMA(base) + disabled LoRA adapters

    This creates a dynamic reference that follows the policy in a smoothed, lagged manner,
    which can improve training stability compared to a fixed reference model.

    The BEMA weights represent a smoothed trajectory through weight space and often
    generalize better than the final checkpoint.
    """

    def __init__(
        self,
        update_freq: int = 50,
        ema_power: float = 0.5,
        bias_power: float = 0.2,
        lag: int = 10,
        update_after: int = 0,
        multiplier: float = 1.0,
        min_ema_multiplier: float = 0.0,
        device: str = "cpu",
    ):
        self.update_freq = update_freq
        self.ema_power = ema_power
        self.bias_power = bias_power
        self.lag = lag
        self.update_after = update_after
        self.multiplier = multiplier
        self.min_ema_multiplier = min_ema_multiplier
        self.device = device

        # Storage for BEMA state
        self.theta0_params = {}  # θ₀ - initial merged parameters
        self.ema_params = {}     # EMA of merged parameters
        self.param_names = []    # Names of base model parameters
        self.bema_state_dict = {}  # BEMA weights

    def _unwrap_model(self, model):
        """Unwrap model from DDP/FSDP/DeepSpeed wrappers."""
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(model, (DDP, FSDP)):
            return model.module
        if hasattr(model, "module"):
            return model.module
        return model

    def _get_merged_state_dict(self, model):
        """
        Get merged state dict (base model + LoRA adapters) WITHOUT modifying the base model.

        This computes the effective weights by manually adding LoRA deltas to base weights.
        All computations are done on self.device (CPU by default) to avoid GPU OOM.
        """
        model = self._unwrap_model(model)

        if is_peft_model(model):
            # For PEFT models, we manually compute merged weights without touching the base model
            try:
                from peft.tuners.lora import LoraLayer
            except ImportError:
                # Fallback for older peft versions
                from peft.tuners.lora.layer import LoraLayer

            merged_state_dict = {}
            base_model = model.get_base_model()

            # Get base model parameters first and move to self.device (CPU by default)
            base_state_dict = {name: param.detach().clone().to(self.device) for name, param in base_model.named_parameters()}

            # Now add LoRA deltas manually without modifying the base model
            # All computations done on self.device (CPU by default) to save GPU memory
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    # This is a LoRA layer, compute the delta
                    # LoRA delta = (lora_B @ lora_A) * scaling
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        for adapter_name in module.lora_A.keys():
                            if module.active_adapter == adapter_name:
                                # Move LoRA weights to self.device for computation
                                lora_A = module.lora_A[adapter_name].weight.detach().to(self.device)
                                lora_B = module.lora_B[adapter_name].weight.detach().to(self.device)
                                scaling = module.scaling[adapter_name]

                                # Compute LoRA delta on self.device: BA * scaling
                                lora_delta = (lora_B @ lora_A) * scaling

                                # Find the corresponding base weight name
                                # The module name in PEFT includes 'base_model.model.'
                                parent_name = name.replace('base_model.model.', '')

                                # Add delta to base weight (both on CPU)
                                weight_name = f"{parent_name}.weight"
                                if weight_name in base_state_dict:
                                    merged_state_dict[weight_name] = base_state_dict[weight_name] + lora_delta

            # Fill in any parameters that weren't modified by LoRA
            for name, param in base_state_dict.items():
                if name not in merged_state_dict:
                    merged_state_dict[name] = param

            return merged_state_dict
        else:
            # For non-PEFT models, just return the state dict on self.device
            return {name: param.detach().clone().to(self.device) for name, param in model.named_parameters()}

    @torch.no_grad()
    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: PreTrainedModel, **kwargs
    ):
        # Get initial merged weights
        merged_state = self._get_merged_state_dict(model)

        # Initialize θ₀ and EMA with merged weights
        total_params = 0
        for name, param in merged_state.items():
            self.param_names.append(name)
            theta0 = param.to(self.device)
            self.theta0_params[name] = theta0
            self.ema_params[name] = theta0.clone()
            total_params += param.numel()

        print(f"PEFT-BEMA: Initialized BEMA callback")
        print(f"  - Tracking {len(self.param_names)} parameters ({total_params:,} total elements)")
        print(f"  - Storage device: {self.device}")
        print(f"  - Update frequency: every {self.update_freq} steps after step {self.update_after}")

    def _ema_beta(self, step: int) -> float:
        """Compute the EMA decay factor βₜ = (ρ + γ·t)⁻ᵏᵃᵖᵖᵃ."""
        beta = (self.lag + self.multiplier * step) ** (-self.ema_power)
        return max(beta, self.min_ema_multiplier)

    def _bema_alpha(self, step: int) -> float:
        """Compute the BEMA scaling factor αₜ = (ρ + γ·t)⁻ᵉᵗᵃ."""
        return (self.lag + self.multiplier * step) ** (-self.bias_power)

    def _update_bema_weights(self, model, step: int):
        """Update BEMA weights using merged LoRA + base weights. All on self.device (CPU by default)."""
        beta = self._ema_beta(step)
        alpha = self._bema_alpha(step)

        # Get current merged weights (already on self.device from _get_merged_state_dict)
        merged_state = self._get_merged_state_dict(model)

        # Track magnitude of updates for verification
        total_change = 0.0
        num_params = 0

        # Update EMA and compute BEMA for each parameter (all on self.device)
        for name in self.param_names:
            if name not in merged_state:
                continue

            thetat = merged_state[name].to(self.device)  # Should already be on self.device (CPU)
            theta0 = self.theta0_params[name]
            ema = self.ema_params[name]

            # Store old BEMA for comparison
            old_bema = self.bema_state_dict.get(name, ema).clone() if name in self.bema_state_dict else ema.clone()

            # EMA update: ema = (1 - beta) * ema + beta * θₜ
            ema.mul_(1 - beta).add_(thetat, alpha=beta)

            # BEMA update: bema = ema + alpha * (θₜ - θ₀)
            bema_weight = ema + alpha * (thetat - theta0)

            # Track change in BEMA weights
            change = (bema_weight - old_bema).abs().mean().item()
            total_change += change
            num_params += 1

            # Keep on self.device (CPU by default)
            self.bema_state_dict[name] = bema_weight

        avg_bema_change = total_change / num_params if num_params > 0 else 0.0
        print(f"  - Average BEMA weight change: {avg_bema_change:.6e}")

    def _update_ref_model(self, model):
        """
        Update the base model with BEMA weights.

        For PEFT/LoRA with OnlineDPO:
        - Policy model = base_model (with BEMA weights) + enabled LoRA adapters
        - Reference model = base_model (with BEMA weights) + disabled LoRA adapters

        This creates a dynamic reference model that follows the policy in a smoothed manner,
        which can improve training stability compared to a fixed reference model.

        BEMA weights are stored on self.device (CPU by default) and moved to GPU only during loading.
        """
        model = self._unwrap_model(model)

        if is_peft_model(model):
            base_model = model.get_base_model()

            # Get the device of the base model
            model_device = next(base_model.parameters()).device

            # Compute change magnitude before update (for verification)
            old_weight = next(base_model.parameters()).detach().clone()

            # Move BEMA weights to model device and load
            bema_state_dict_gpu = {k: v.to(model_device) for k, v in self.bema_state_dict.items()}
            base_model.load_state_dict(bema_state_dict_gpu, strict=False)

            # Compute change magnitude after update
            new_weight = next(base_model.parameters()).detach()
            weight_change = (new_weight - old_weight).abs().mean().item()

            print(f"PEFT-BEMA: Updated shared base model with BEMA weights")
            print(f"           Policy = BEMA_base + LoRA, Reference = BEMA_base")
            print(f"           Average weight change: {weight_change:.6e}")
        else:
            # For full fine-tuning, update the model directly
            model_device = next(model.parameters()).device
            old_weight = next(model.parameters()).detach().clone()

            bema_state_dict_gpu = {k: v.to(model_device) for k, v in self.bema_state_dict.items()}
            model.load_state_dict(bema_state_dict_gpu, strict=False)

            new_weight = next(model.parameters()).detach()
            weight_change = (new_weight - old_weight).abs().mean().item()
            print(f"BEMA: Updated model with BEMA weights (change: {weight_change:.6e})")

    @torch.no_grad()
    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: PreTrainedModel, **kwargs
    ):
        step = state.global_step

        # Only update after burn-in period and at specified frequency
        if step < self.update_after:
            return

        if (step - self.update_after) % self.update_freq == 0:
            print(f"\n{'='*60}")
            print(f"PEFT-BEMA: Step {step} - Computing BEMA update")
            print(f"  - EMA beta: {self._ema_beta(step):.6f}")
            print(f"  - BEMA alpha: {self._bema_alpha(step):.6f}")

            self._update_bema_weights(model, step)
            self._update_ref_model(model)
            print(f"{'='*60}\n")

    @torch.no_grad()
    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """Save BEMA state dict."""
        if self.bema_state_dict:
            import os
            bema_path = os.path.join(args.output_dir, "bema_state_dict.pt")
            torch.save(self.bema_state_dict, bema_path)

            # Compute some statistics for verification
            total_params = sum(v.numel() for v in self.bema_state_dict.values())
            avg_magnitude = torch.mean(torch.stack([v.abs().mean() for v in self.bema_state_dict.values()])).item()

            print(f"PEFT-BEMA: Saved BEMA state dict to {bema_path}")
            print(f"           {len(self.bema_state_dict)} tensors, {total_params:,} total parameters")
            print(f"           Average magnitude: {avg_magnitude:.6e}")

################
# Reward Function for Training
################

def strip_reasoning_accuracy_reward(
    completions: list[list[dict[str, str]]], solution: list[str], **kwargs
) -> list[Optional[float]]:
    """Reward function that strips reasoning tags and checks mathematical accuracy.

    This function:
    1. Extracts the content from completions
    2. Removes <think></think> tags (for reasoning that shouldn't be evaluated)
    3. Parses both the gold solution and the predicted answer
    4. Uses math_verify to check if they are mathematically equivalent

    Args:
        completions: List of model completions, each containing a list of messages
        solution: List of ground truth solutions
        **kwargs: Additional arguments (ignored but required for trainer compatibility)

    Returns:
        List of rewards where:
        - 1.0 if the answer is correct
        - 0.0 if the answer is incorrect
        - None if the solution is not parseable (skips this example)
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        # Strip reasoning tags from completion
        while "<think>" in content and "</think>" in content:
            start = content.find("<think>")
            end = content.find("</think>", start)
            if start != -1 and end != -1:
                content = content[:start] + content[end + len("</think>") :]
            else:
                break

        # Parse gold solution
        gold_parsed = parse(
            f"${sol}$",
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0, try_extract_without_anchor=True
                )
            ],
        )

        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        boxed_match_priority=0,
                        normalization_config=NormalizationConfig(
                            basic_latex=True,
                            units=True,
                            malformed_operators=False,
                            nits=False,
                            boxed=True,
                        ),
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(
                    f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}"
                )
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None

        rewards.append(reward)

    return rewards

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, OnlineDPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()


    # Enable logging in a Hugging Face Space
    os.environ.setdefault("TRACKIO_SPACE_ID", "burtenshaw/trl-online-dpo")
    os.environ.setdefault("TRACKIO_PROJECT", model_args.model_name_or_path.split('/')[-1] + script_args.dataset_name.replace('/', '-'))

    ################
    # Model & Processor
    ################
    dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    ################
    # Dataset
    ################
    dataset = load_dataset(
        script_args.dataset_name, split=script_args.dataset_train_split
    )

    # Limit to 5k samples for faster training
    if len(dataset) > 5000:
        dataset = dataset.select(range(5000))

    def make_conversation(example):
        prompt = [{"role": "user", "content": example["problem"]}]
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    # Debug: Print first example to verify prompt structure
    print("First dataset example:")
    print(dataset[0])

    # Remove unnecessary columns
    columns_to_remove = [
        col for col in dataset.column_names if col not in ["prompt", "solution"]
    ]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    ################
    # Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Training
    ################

    if training_args.run_name is None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_args.run_name = (
            f"online-dpo-lora-{model_args.model_name_or_path.split('/')[-1]}-{now}"
        )

    training_args.report_to = ["trackio"]

    trainer = OnlineDPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[strip_reasoning_accuracy_reward],
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Add BEMA callback for reference model updating if available
    if BEMA_AVAILABLE:
        print("Adding PEFT-compatible BEMA callback for dynamic reference model updating...")
        bema_callback = PEFTBEMACallback(
            update_freq=10,              # Update BEMA weights every 10 steps
            ema_power=0.5,               # Power for EMA decay factor (κ in paper)
            bias_power=0.2,              # Power for BEMA scaling factor (η in paper)
            lag=5,                      # Initial offset for smoothness (ρ in paper)
            update_after=20,             # Start BEMA after 20 steps burn-in (τ in paper)
            multiplier=1.0,              # Initial EMA decay factor (γ in paper)
            min_ema_multiplier=0.0,      # Minimum EMA multiplier
            device="cpu",                # Use CPU for BEMA buffers to avoid OOM
        )
        trainer.add_callback(bema_callback)
        print("BEMA callback added for dynamic reference model updating.")
        print("With LoRA: Policy = BEMA(base) + LoRA, Reference = BEMA(base)")
        print("The base model will be updated with smoothed weights every 50 steps after step 50.")
    else:
        print("BEMA callback not available. Training with fixed reference model.")

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)