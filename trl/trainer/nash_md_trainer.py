import time
import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers import GenerationConfig, PreTrainedModel

from trl.trainer import WinRateCallback, MockJudge, PairRMJudge
from accelerate.utils import is_deepspeed_available

from trl import DPOTrainer

if is_deepspeed_available():
    import deepspeed

def remove_hooks(model):
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []


def add_hooks(model):
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    optimizer_offload._register_hooks_recursively(optimizer_offload.module)


@contextmanager
def prepare_model_for_generation(model, accelerator):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.gradient_checkpointing_disable()
    unwrapped_model.config.use_cache = True

    if (
        hasattr(accelerator.state, "deepspeed_plugin")
        and accelerator.state.deepspeed_plugin is not None
        and accelerator.state.deepspeed_plugin.zero_stage == 3
    ):
        with deepspeed.zero.GatheredParameters(model.parameters()):
            remove_hooks(model)
            yield model
            add_hooks(model)
    else:
        yield unwrapped_model
        unwrapped_model.gradient_checkpointing_enable()
        unwrapped_model.config.use_cache = False


class NashMDTrainer(DPOTrainer):
    pass