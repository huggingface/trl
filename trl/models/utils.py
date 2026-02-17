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

import itertools
import logging
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import accelerate
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator
from packaging.version import Version
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig, PreTrainedModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled


if Version(accelerate.__version__) >= Version("1.11.0"):
    from accelerate.utils.fsdp_utils import get_parameters_from_modules

if TYPE_CHECKING:
    from deepspeed.runtime.engine import DeepSpeedEngine
    from torch.nn import Module
    from torch.nn.parallel.distributed import DistributedDataParallel


def remove_hooks(model: "DeepSpeedEngine") -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []


def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())


def iter_params(module, recurse=False):
    return [param for _, param in get_all_parameters(module, recurse)]


def add_hooks(model: "DeepSpeedEngine") -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    import deepspeed

    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")
    if Version(deepspeed.__version__) >= Version("0.16.4"):
        # Account for renaming in https://github.com/deepspeedai/DeepSpeed/pull/6847
        optimizer_offload._register_deepspeed_module(optimizer_offload.module)
    else:
        optimizer_offload._register_hooks_recursively(optimizer_offload.module)


@contextmanager
def _unwrap_model_for_generation(
    model: "DistributedDataParallel | DeepSpeedEngine",
    accelerator: "Accelerator",
    gather_deepspeed3_params: bool = True,
):
    """
    Context manager to unwrap distributed or accelerated models for generation tasks.

    Args:
        model (`DistributedDataParallel | DeepSpeedEngine`):
            Model to be unwrapped.
        accelerator ([`~accelerate.Accelerator`]):
            Accelerator instance managing the model.
        gather_deepspeed3_params (`bool`, *optional*, defaults to `True`):
            Whether to gather weights for DeepSpeed ZeRO Stage 3 models. If `False`, skips parameter gathering, which
            can be more memory-efficient but may lead to slower generation times.

    Yields:
        Unwrapped model.

    Example:
    ```python
    with _unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        generated_outputs = unwrapped_model.generate(input_ids)
    ```
    """
    unwrapped_model = accelerator.unwrap_model(model)
    is_gradient_checkpointing = unwrapped_model.is_gradient_checkpointing
    if is_gradient_checkpointing:
        unwrapped_model.gradient_checkpointing_disable()
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
        else:
            import deepspeed

            with deepspeed.zero.GatheredParameters(model.parameters()):
                remove_hooks(model)
                yield accelerator.unwrap_model(model)
                add_hooks(model)
    else:
        yield unwrapped_model
    if is_gradient_checkpointing:
        unwrapped_model.gradient_checkpointing_enable()


@contextmanager
def _override_model_generation_config(model, generation_kwargs=None):
    """
    Context manager to temporarily override a model's generation_config with training config.

    This works around transformers' config merging logic that would otherwise overwrite values matching global defaults
    with model-specific values (see upstream issue transformers#42762; fixed in transformers v5 by PR
    `transformers#42702`).

    By temporarily setting the model's generation_config to match the passed generation_config, we avoid the conflict.

    The model's original generation_config is preserved outside this context, ensuring that saved/pushed models retain
    their intended inference behavior.

    Args:
        model: The model (typically unwrapped_model) whose generation_config to temporarily override.
        generation_kwargs (dict): Generation kwargs to be used to override model's generation config.
    """
    if (
        # Issue fixed in transformers v5 by PR transformers#42702
        Version(transformers.__version__) >= Version("5.0.0")
        or generation_kwargs is None
        or not hasattr(model, "generation_config")
    ):
        yield model
        return
    # If it is a PEFT model, override the underlying base model
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    # Keep original model generation_config
    original_config = model.generation_config
    # Create training-specific generation config from the model's original generation config
    # Then overwrite it with the training-specific generation kwargs
    generation_config = GenerationConfig.from_dict(model.generation_config.to_dict())
    generation_config.update(**generation_kwargs)
    model.generation_config = generation_config
    try:
        yield
    finally:
        model.generation_config = original_config


@contextmanager
def unwrap_model_for_generation(
    model: "DistributedDataParallel | DeepSpeedEngine",
    accelerator: "Accelerator",
    gather_deepspeed3_params: bool = True,
    generation_kwargs: dict | None = None,
):
    """
    Context manager to unwrap distributed or accelerated models for generation tasks.

    This function unwraps distributed models (FSDP, DeepSpeed) and optionally overrides the model's generation_config
    temporarily during generation. This is useful for applying training-specific generation parameters without
    permanently modifying the model's original generation_config.

    Args:
        model (`DistributedDataParallel | DeepSpeedEngine`):
            Model to be unwrapped.
        accelerator ([`~accelerate.Accelerator`]):
            Accelerator instance managing the model.
        gather_deepspeed3_params (`bool`, *optional*, defaults to `True`):
            Whether to gather weights for DeepSpeed ZeRO Stage 3 models. If `False`, skips parameter gathering, which
            can be more memory-efficient but may lead to slower generation times.
        generation_kwargs (dict, *optional*):
            If provided, temporarily overrides the model's generation_config during generation. The original config is
            automatically restored when exiting the context. This is useful for using different generation parameters
            during training vs. inference.

    Yields:
        Unwrapped model with optionally overridden generation_config.
    """
    with (
        _unwrap_model_for_generation(
            model, accelerator, gather_deepspeed3_params=gather_deepspeed3_params
        ) as unwrapped_model,
        _override_model_generation_config(unwrapped_model, generation_kwargs=generation_kwargs),
    ):
        yield unwrapped_model


def prepare_deepspeed(model: "Module", accelerator: "Accelerator"):
    """Prepares the model for DeepSpeed inference or evaluation by initializing it with the appropriate configuration.

    Adapted from accelerate:
    https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    """
    import deepspeed  # local import (instead of top-level) to avoid DS init interfering with other backends (like vllm): https://github.com/deepspeedai/DeepSpeed/issues/7252

    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    stage = config_kwargs["zero_optimization"]["stage"]

    if model is not None:
        hidden_size = (
            max(model.config.hidden_sizes)
            if getattr(model.config, "hidden_sizes", None)
            else getattr(model.config, "hidden_size", None)
        )
        if hidden_size is not None and stage == 3:
            # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache
            # @ step 0: expected module 1, but got module 0`
            # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
            config_kwargs.update(
                {
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                }
            )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO
    # disabled (stage 0)
    if stage != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


def prepare_fsdp(model, accelerator: Accelerator) -> FSDP | FSDPModule:
    # Check if the model is already a FSDP model due to `Manual Wrapping` and if so, don't wrap it again
    if not isinstance(model, (FSDP, FSDPModule)):
        fsdp_plugin = accelerator.state.fsdp_plugin
        if fsdp_plugin.fsdp_version == 1:
            accelerator.state.fsdp_plugin.set_auto_wrap_policy(model)
            kwargs = {
                "sharding_strategy": fsdp_plugin.sharding_strategy or fsdp_plugin.reshard_after_forward,
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
        elif fsdp_plugin.fsdp_version == 2:
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

            mesh = getattr(accelerator, "torch_device_mesh", None)
            if Version(accelerate.__version__) >= Version("1.11.0"):
                ignored_params = get_parameters_from_modules(fsdp_plugin.ignored_modules, model, accelerator.device)
            else:
                logging.warning(
                    "FSDP version 2 is being used with accelerate version < 1.11.0, which may lead to incorrect "
                    "handling of ignored modules. Please upgrade accelerate to v1.11.0 or later for proper support."
                )
                ignored_params = None
            fully_shard(
                model,
                reshard_after_forward=fsdp_plugin.reshard_after_forward,
                offload_policy=fsdp_plugin.cpu_offload,
                # `fully_shard` doesn't accept `None` in case of `MixedPrecisionPolicy`
                mp_policy=fsdp_plugin.mixed_precision_policy or MixedPrecisionPolicy(),
                mesh=mesh[tuple(accelerator.parallelism_config.fsdp_dim_names)] if mesh is not None else None,
                ignored_params=ignored_params,
            )
        else:
            raise ValueError(f"FSDP version {fsdp_plugin.fsdp_version} is not supported.")
    model.eval()
    return model


class _ForwardRedirection:
    """Implements the `forward-redirection`.

    Taken from Pytorch-lightning:
    https://github.com/Lightning-AI/pytorch-lightning/blob/02311d03fb982560246eead7c08104481fac9579/src/lightning/pytorch/strategies/strategy.py#L602

    A method call to a wrapped module gets rerouted through the wrapper's `forward` method instead.

    """

    def __call__(
        self, wrapper_module: nn.Module, original_module: nn.Module, method: Callable, *args: Any, **kwargs: Any
    ):
        """Reroutes a method call through the `wrapper_module`'s `forward` method.

        Args:
            wrapper_module: The module that has `original_module` wrapped.
            original_module: The module that was wrapped inside `wrapper_module`.
            method: The method that should be called on the `original_module` after inputs get
                redirected through the `wrapper_module`'s `forward` method.
            *args: The positional arguments to the `method`. They will get passed to a patched
                `forward` method instead.
            **kwargs: The keyword arguments to the `method`. They will get passed to a patched
                `forward` method instead.

        """
        original_forward = original_module.forward

        def wrapped_forward(*_args: Any, **_kwargs: Any) -> Any:
            # Unpatch ourselves immediately before calling the method `method_name`
            # because itself may want to call the real `forward`
            original_module.forward = original_forward  # type: ignore[method-assign]
            # Call the actual method e.g. `.training_step(...)`
            out = method(*_args, **_kwargs)
            self.on_after_inner_forward(wrapper_module, original_module)
            return out

        # Patch the original_module's forward so we can redirect the arguments back to the real method
        original_module.forward = wrapped_forward  # type: ignore[method-assign]

        wrapper_output = wrapper_module(*args, **kwargs)
        self.on_after_outer_forward(wrapper_module, original_module)
        return wrapper_output

    def on_after_inner_forward(self, wrapper_module: nn.Module, original_module: nn.Module) -> None:
        pass

    def on_after_outer_forward(self, wrapper_module: nn.Module, original_module: nn.Module) -> None:
        pass


def peft_module_casting_to_bf16(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
        elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)


@contextmanager
def disable_gradient_checkpointing(model: PreTrainedModel, gradient_checkpointing_kwargs: dict | None = None):
    """
    Temporarily disable gradient checkpointing, restoring the previous state afterward.

    Args:
        model (`PreTrainedModel`):
            Model for which to temporarily disable gradient checkpointing.
        gradient_checkpointing_kwargs (`dict` or `None`, *optional*):
            Additional kwargs for gradient checkpointing enabling.
    """
    was_enabled = model.is_gradient_checkpointing
    if was_enabled:
        model.gradient_checkpointing_disable()
    try:
        yield
    finally:
        if was_enabled:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)


LAYER_PATTERNS = [
    "transformer.h.{layer}",
    "model.decoder.layers.{layer}",
    "gpt_neox.layers.{layer}",
    "model.layers.{layer}",
]


def create_reference_model(
    model: nn.Module, num_shared_layers: int | None = None, pattern: str | None = None
) -> nn.Module:
    """
    Creates a static reference copy of a model. Note that model will be in `.eval()` mode.

    Args:
        model ([`nn.Module`]): The model to be copied.
        num_shared_layers (`int`, *optional*):
            The number of initial layers that are shared between both models and kept frozen.
        pattern (`str`, *optional*): The shared layers are selected with a string pattern
            (e.g. "transformer.h.{layer}" for GPT2) and if a custom pattern is necessary it can be passed here.

    Returns:
        [`nn.Module`]
    """
    if is_deepspeed_zero3_enabled():
        raise ValueError(
            "DeepSpeed ZeRO-3 is enabled and is not compatible with `create_reference_model()`. Please instantiate your reference model directly with `AutoModelForCausalLM.from_pretrained()`."
        )

    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = deepcopy(model)

    # if no layers are shared, return copy of model
    if num_shared_layers is None:
        for param_name in parameter_names:
            param = ref_model.get_parameter(param_name)
            param.requires_grad = False
        return ref_model.eval()

    # identify layer name pattern
    if pattern is not None:
        pattern = pattern.format(layer=num_shared_layers)
    else:
        for pattern_candidate in LAYER_PATTERNS:
            pattern_candidate = pattern_candidate.format(layer=num_shared_layers)
            if any(pattern_candidate in name for name in parameter_names):
                pattern = pattern_candidate
                break

    if pattern is None:
        raise ValueError("Layer pattern could not be matched.")

    # divide parameters in shared and unshared parameter lists
    shared_param_list = []
    unshared_param_list = []

    shared_parameter = True
    for name, _param in model.named_parameters():
        if pattern in name:
            shared_parameter = False
        if shared_parameter:
            shared_param_list.append(name)
        else:
            unshared_param_list.append(name)

    # create reference of the original parameter if they are shared
    for param_name in shared_param_list:
        param = model.get_parameter(param_name)
        param.requires_grad = False

        _ref_param = ref_model.get_parameter(param_name)

    # for all other parameters just make sure they don't use gradients
    for param_name in unshared_param_list:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False

    if pattern is not None and len(unshared_param_list) == 0:
        logging.warning("Pattern passed or found, but no layers matched in the model. Check for a typo.")

    return ref_model.eval()
