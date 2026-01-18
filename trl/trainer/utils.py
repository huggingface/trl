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

import asyncio
import dataclasses
import importlib.resources as pkg_resources
import json
import os
import random
import socket
import threading
from collections.abc import Mapping, Sequence, Sized
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from itertools import accumulate
from typing import TypeVar

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, PartialState, logging
from accelerate.state import AcceleratorState
from huggingface_hub import ModelCard, ModelCardData
from torch.utils.data import Sampler
from transformers import (
    AutoConfig,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    is_comet_available,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.utils import (
    is_peft_available,
    is_rich_available,
    is_torch_mlu_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

from ..trainer.model_config import ModelConfig


if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

if is_comet_available():
    import comet_ml

if is_peft_available():
    from peft import LoraConfig, PeftConfig, PeftModel


logger = logging.get_logger(__name__)


def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _find_free_port() -> int:
    candidates = (29500, 23456, 12355, 12345)
    for p in candidates:
        if _is_port_free(p):
            return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def ensure_master_addr_port(addr: str | None = None, port: int | None = None) -> None:
    """
    Ensure `MASTER_ADDR`/`MASTER_PORT` are set safely.

    - Respects existing environment variables.
    - Defaults `MASTER_ADDR` to localhost if unset.
    - Chooses a free TCP port if `MASTER_PORT` is unset to avoid collisions.
    - If `MASTER_PORT` is set to `"0"` or `"auto"`, it is resolved to a free port.
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR") or addr or "localhost"

    env_port = os.environ.get("MASTER_PORT", "").strip().lower()
    if port is None and env_port not in {"", "0", "auto"}:
        try:
            port = int(env_port)
        except ValueError:
            pass

    os.environ["MASTER_PORT"] = str(_find_free_port() if port in (None, 0) else port)


def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: int | None = None,
) -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`list[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
    ```python
    >>> import torch

    >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
    tensor([[1, 2, 3],
            [4, 5, 0]])

    >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
    tensor([[[1, 2],
            [3, 4]],
            [[5, 6],
            [0, 0]]])
    ```
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Apply pad_to_multiple_of to the first (sequence) dimension
    if pad_to_multiple_of is not None:
        remainder = output_shape[0] % pad_to_multiple_of
        if remainder != 0:
            output_shape[0] += pad_to_multiple_of - remainder

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        # Define the slices
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


@dataclass
class RunningMoments:
    """
    Calculates the running mean and standard deviation of a data stream. Reference:
    https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L75
    """

    accelerator: Accelerator
    mean: float = 0
    std: float = 1
    var: float = 1
    count: float = 1e-24

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        if self.accelerator.use_distributed:
            xs_mean, xs_var, xs_count = get_global_statistics(self.accelerator, xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += (delta * xs_count / tot_count).item()
        new_var = tot_sum / tot_count
        self.std = (new_var * tot_count / (tot_count - 1)).float().sqrt().item()
        self.var = new_var.item()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        # save everything except accelerator
        if self.accelerator.is_main_process:
            save_dict = dataclasses.asdict(self, dict_factory=lambda x: {k: v for (k, v) in x if k != "accelerator"})
            json_string = json.dumps(save_dict, indent=2, sort_keys=True) + "\n"
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(json_string)

    @classmethod
    def load_from_json(cls, accelerator: Accelerator, json_path: str):
        """Create an instance from the content of `json_path`."""
        # load everything except accelerator
        with open(json_path, encoding="utf-8") as f:
            text = f.read()
        return cls(accelerator=accelerator, **json.loads(text))


@torch.no_grad()
def get_global_statistics(
    accelerator, xs: torch.Tensor, mask=None, device="cpu"
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Computes element-wise mean and variance of the tensor across processes. Reference:
    https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L57C1-L73C75
    """
    xs = xs.to(accelerator.device)
    sum_and_count = torch.tensor([xs.sum(), (xs.numel() if mask is None else mask.sum())], device=xs.device)
    sum_and_count = accelerator.reduce(sum_and_count)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum(((xs - global_mean) ** 2).mul(1 if mask is None else mask))
    sum_var = accelerator.reduce(sum_var)
    global_var = sum_var / count

    return global_mean.to(device), global_var.to(device), count.item()


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: int | float, dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def get_quantization_config(model_args: ModelConfig) -> BitsAndBytesConfig | None:
    if model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_args.dtype,  # For consistency with model weights, we use the same value as `dtype`
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
            bnb_4bit_quant_storage=model_args.bnb_4bit_quant_storage,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_kbit_device_map() -> dict[str, int] | None:
    if torch.cuda.is_available() or is_torch_xpu_available():
        return {"": PartialState().local_process_index}
    else:
        return None


def get_peft_config(model_args: ModelConfig) -> "PeftConfig | None":
    if model_args.use_peft is False:
        return None

    if not is_peft_available():
        raise ValueError(
            "You need to have PEFT library installed in your environment, make sure to install `peft`. "
            "Make sure to run `pip install -U peft`."
        )

    peft_config = LoraConfig(
        task_type=model_args.lora_task_type,
        r=model_args.lora_r,
        target_modules=model_args.lora_target_modules,
        target_parameters=model_args.lora_target_parameters,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        use_rslora=model_args.use_rslora,
        use_dora=model_args.use_dora,
        modules_to_save=model_args.lora_modules_to_save,
    )

    return peft_config


def get_exp_cap(value, decimal=4):
    """
    Get the exponent cap of a value. This is used to cap the exponent of a value to avoid overflow. The formula is :
    log(value.dtype.max) E.g. for float32 data type, the maximum exponent value is 88.7228 to 4 decimal points.

    Args:
        value (`torch.Tensor`):
            The input tensor to obtain the data type
        decimal (`int`):
            The number of decimal points of the output exponent cap. eg: direct calling exp(log(torch.float32.max))
            will result in inf so we cap the exponent to 88.7228 to avoid overflow.
    """
    vdtype_max = torch.zeros([1]).to(value.dtype) + torch.finfo(value.dtype).max
    vdtype_log_max = torch.log(vdtype_max).to(value.device)
    return torch.floor(vdtype_log_max * 10**decimal) / 10**decimal if decimal > 0 else vdtype_log_max


def cap_exp(value, cap=-1):
    # Cap the exponent value below the upper-bound to avoid overflow, before calling torch.exp
    cap = get_exp_cap(value) if cap < 0 else cap
    return torch.exp(torch.clamp(value, max=cap))


def prepare_deepspeed(
    model: torch.nn.Module, per_device_train_batch_size: int, fp16: bool = False, bf16: bool = False
) -> torch.nn.Module:
    """
    Prepares the model for training with DeepSpeed (both for stage 2 and 3), configuring the appropriate settings based
    on the model and batch size.

    Args:
        model (`torch.nn.Module`):
            The model to be prepared for DeepSpeed training.
        per_device_train_batch_size (`int`):
            The training batch size per device.
        fp16 (`bool`, defaults to `False`):
            Whether to use FP16 precision.
        bf16 (`bool`, defaults to `False`):
            Whether to use BF16 precision.

    Returns:
        `torch.nn.Module`:
            The model initialized and configured with DeepSpeed for training.
    """
    import deepspeed

    deepspeed_plugin = AcceleratorState().deepspeed_plugin
    config_kwargs = deepspeed_plugin.deepspeed_config
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["train_micro_batch_size_per_gpu"] = per_device_train_batch_size
        config_kwargs = {
            "train_micro_batch_size_per_gpu": config_kwargs["train_micro_batch_size_per_gpu"],
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
        if bf16:
            config_kwargs["bf16"] = {"enabled": True}
        elif fp16:
            config_kwargs["fp16"] = {"enabled": True}
    else:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0,
                    }
                )
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


def empty_cache() -> None:
    """Empties the cache of the available torch device.

    This function checks for the availability of different torch devices (XPU, MLU, NPU, CUDA) and empties the cache of
    the first available device it finds.

    If none of the specific devices are available, it defaults to emptying the CUDA cache.
    """
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_mlu_available():
        torch.mlu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()


def generate_model_card(
    base_model: str | None,
    model_name: str,
    hub_model_id: str,
    dataset_name: str | None,
    tags: list[str],
    wandb_url: str | None,
    trainer_name: str,
    trainer_citation: str | None = None,
    template_file: str | None = None,
    paper_title: str | None = None,
    paper_id: str | None = None,
    comet_url: str | None = None,
) -> ModelCard:
    """
    Generate a [`~huggingface_hub.ModelCard`] from a template.

    Args:
        base_model (`str` or `None`):
            Base model name.
        model_name (`str`):
            Model name.
        hub_model_id (`str`):
            Hub model ID as `username/model_id`.
        dataset_name (`str` or `None`):
            Dataset name.
        tags (`list[str]`):
            Tags.
        wandb_url (`str` or `None`):
            Weights & Biases run URL.
        comet_url (`str` or `None`):
            Comet experiment URL.
        trainer_name (`str`):
            Trainer name.
        trainer_citation (`str` or `None`, defaults to `None`):
            Trainer citation as a BibTeX entry.
        template_file (`str` *optional*):
            Template file name located in the `trl/templates` directory. Defaults to `lm_model_card.md`.
        paper_title (`str` or `None`, defaults to `None`):
            Paper title.
        paper_id (`str` or `None`, defaults to `None`):
            ArXiv paper ID as `YYMM.NNNNN`.

    Returns:
        [`~huggingface_hub.ModelCard`]:
            A ModelCard object.
    """
    card_data = ModelCardData(
        base_model=base_model,
        datasets=dataset_name,
        library_name="transformers",
        licence="license",
        model_name=model_name,
        tags=["generated_from_trainer", *tags],
    )
    template_file = template_file or "lm_model_card.md"
    card = ModelCard.from_template(
        card_data,
        template_path=str(pkg_resources.files("trl").joinpath(f"templates/{template_file}")),
        base_model=base_model,
        model_name=model_name,
        hub_model_id=hub_model_id,
        dataset_name=dataset_name,
        wandb_url=wandb_url,
        comet_url=comet_url,
        trainer_name=trainer_name,
        trainer_citation=trainer_citation,
        paper_title=paper_title,
        paper_id=paper_id,
        trl_version=version("trl"),
        transformers_version=version("transformers"),
        pytorch_version=version("torch"),
        datasets_version=version("datasets"),
        tokenizers_version=version("tokenizers"),
    )
    return card


def get_comet_experiment_url() -> str | None:
    """
    If Comet integration is enabled, return the URL of the current Comet experiment; otherwise, return `None`.
    """
    if not is_comet_available():
        return None

    if comet_ml.get_running_experiment() is not None:
        return comet_ml.get_running_experiment().url

    return None


def log_table_to_comet_experiment(name: str, table: pd.DataFrame) -> None:
    """
    If Comet integration is enabled logs a table to the Comet experiment if it is currently running.

    Args:
        name (`str`):
            Table name.
        table (`pandas.DataFrame`):
            The Pandas DataFrame containing the table to log.
    """
    if not is_comet_available():
        raise ModuleNotFoundError("The comet-ml is not installed. Please install it first: pip install comet-ml")

    experiment = comet_ml.get_running_experiment()
    if experiment is not None:
        experiment.log_table(tabular_data=table, filename=name)


def flush_left(mask: torch.Tensor, *tensors: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Shift non-zero elements in the mask and corresponding tensors to the left.

    This function operates on a binary mask and any number of additional tensors with the same dimensions as the mask.
    For each row, non-zero values are shifted to the leftmost positions. Then, columns that contain only zeros across
    all rows are truncated from the mask and tensors. Visually, this operation can be represented as follows:

    ```
    [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
     [0, x, x, x, 0, 0]]       [x, x, x, 0]]
    ```

    Args:
        mask (`torch.Tensor`):
            2D tensor (binary mask) with shape `(N, M)`.
        *tensors (`torch.Tensor`):
            One or more 2D tensors with the same shape as `mask`. These tensors will be processed alongside `mask`,
            with non-zero values shifted and excess zero columns truncated in the same manner.

    Returns:
        `torch.Tensor`:
            Updated binary mask with non-zero values flushed to the left and trailing zero columns removed.
        `*torch.Tensor`
            Updated tensors, processed in the same way as the mask.

    Example:
    ```python
    >>> mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
    >>> tensor = torch.tensor([[9, 9, 2, 3, 4], [9, 5, 6, 9, 9]])
    >>> new_mask, new_tensor = flush_left(mask, tensor)
    >>> print(new_mask)
    tensor([[1, 1, 1],
            [1, 1, 0]])

    >>> print(new_tensor)
    tensor([[2, 3, 4],
            [5, 6, 0]])
    ```
    """
    _, M = mask.shape

    # Create copy of mask and tensors
    mask_copy = mask.clone()
    tensors = [t.clone() for t in tensors]

    # Shift non-zero values to the left
    first_non_zero = mask_copy.argmax(dim=1)
    pos = torch.arange(M, device=mask_copy.device).unsqueeze(0)
    idx_roll = (pos + first_non_zero.unsqueeze(1)) % M
    mask_roll = mask_copy.gather(1, idx_roll)
    rolled_tensors = [t.gather(1, idx_roll) for t in tensors]

    # Truncate trailing columns that are all zeros in mask_roll
    col_sums = mask_roll.sum(dim=0)
    empty_cols = col_sums == 0
    first_empty_col = int(empty_cols.to(torch.int8).argmax()) if empty_cols.any() else M
    flushed_mask = mask_roll[:, :first_empty_col]
    flushed_tensors = [t[:, :first_empty_col] for t in rolled_tensors]

    if not flushed_tensors:
        return flushed_mask
    return flushed_mask, *flushed_tensors


def flush_right(mask: torch.Tensor, *tensors: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Shift non-zero elements in the mask and corresponding tensors to the right. See `flush_left` for details.
    """
    _, M = mask.shape

    # Create copy of mask and tensors
    mask_copy = mask.clone()
    tensors = [t.clone() for t in tensors]

    # Shift non-zero values to the right
    flipped_mask = torch.fliplr(mask_copy)
    first_non_zero = flipped_mask.argmax(dim=1)
    pos = torch.arange(M, device=mask_copy.device).unsqueeze(0)
    idx_roll = (pos - first_non_zero.unsqueeze(1)) % M
    mask_roll = mask_copy.gather(1, idx_roll)
    rolled_tensors = [t.gather(1, idx_roll) for t in tensors]

    # Truncate leading columns that are all zeros in mask_roll
    col_sums = mask_roll.sum(dim=0)
    non_empty_cols = col_sums != 0
    first_non_empty_col = int(non_empty_cols.to(torch.int8).argmax()) if non_empty_cols.any() else M
    flushed_mask = mask_roll[:, first_non_empty_col:]
    flushed_tensors = [t[:, first_non_empty_col:] for t in rolled_tensors]

    if not flushed_tensors:
        return flushed_mask
    return flushed_mask, *flushed_tensors


def selective_log_softmax(logits, index) -> torch.Tensor:
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index, strict=True):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def entropy_from_logits(logits: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """
    Compute the Shannon entropy (in nats) for each row of *logits* in a memory-efficient way.

    Instead of materializing the full softmax for all rows at once, the logits are flattened to shape (N, num_classes),
    where N is the product of all leading dimensions. Computation is then performed in chunks of size `chunk_size`
    along this flattened dimension, reducing peak memory usage. The result is reshaped back to match the input's
    leading dimensions.

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`. Entropy is taken along the last axis; all leading dimensions
            are preserved in the output.
        chunk_size (`int`, *optional*, defaults to `128`):
            Number of rows from the flattened logits to process per iteration. Smaller values reduce memory usage at
            the cost of more iterations.

    Returns:
        `torch.Tensor`:
            Entropy values with shape `logits.shape[:-1]`.
    """
    original_shape = logits.shape[:-1]  # all dims except num_classes
    num_classes = logits.shape[-1]

    # Flatten all leading dimensions into one
    flat_logits = logits.reshape(-1, num_classes)

    entropies = []
    for chunk in flat_logits.split(chunk_size, dim=0):
        logps = F.log_softmax(chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        entropies.append(chunk_entropy)

    entropies = torch.cat(entropies, dim=0)
    return entropies.reshape(original_shape)


def print_prompt_completions_sample(
    prompts: list,
    completions: list,
    rewards: dict[str, list[float]],
    advantages: list[float],
    step: int,
    num_samples: int = None,
) -> None:
    """
    Print out a sample of model completions to the console with multiple reward metrics.

    This function creates a nicely formatted table showing prompt-completion pairs, useful for monitoring model outputs
    during training. It requires the `rich` library to be installed.

    Args:
        prompts (`list`):
            List of prompts. Can be either strings or lists of messages.
        completions (`list`):
            List of completions corresponding to the prompts. Can be either strings or lists of messages.
        rewards (`dict[str, list[float]]`):
            Dictionary where keys are reward names and values are lists of rewards.
        advantages (`list[float]`):
            List of advantages corresponding to the prompts and completions.
        step (`int`):
            Current training step number, used in the output title.
        num_samples (`int`, *optional*):
            Number of random samples to display. If `None` (default), all items will be displayed.

    Example:
    ```python
    >>> from trl.trainer.utils import print_prompt_completions_sample

    >>> prompts = ["The sky is", "The sun is"]
    >>> completions = [" blue.", " in the sky."]
    >>> rewards = {"Correctness": [0.123, 0.456], "Format": [0.789, 0.101]}
    >>> advantages = [0.987, 0.654]
    >>> print_prompt_completions_sample(prompts, completions, rewards, advantages, 42)
    ╭──────────────────────────── Step 42 ─────────────────────────────╮
    │ ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓ │
    │ ┃ Prompt     ┃ Completion   ┃ Correctness ┃ Format ┃ Advantage ┃ │
    │ ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩ │
    │ │ The sky is │  blue.       │        0.12 │   0.79 │      0.99 │ │
    │ ├────────────┼──────────────┼─────────────┼────────┼───────────┤ │
    │ │ The sun is │  in the sky. │        0.46 │   0.10 │      0.65 │ │
    │ └────────────┴──────────────┴─────────────┴────────┴───────────┘ │
    ╰──────────────────────────────────────────────────────────────────╯
    ```
    """
    if not is_rich_available():
        raise ImportError(
            "The function `print_prompt_completions_sample` requires the `rich` library. Please install it with "
            "`pip install rich`."
        )
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    # Add columns
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    for reward_name in rewards.keys():
        table.add_column(reward_name, style="bold cyan", justify="right")
    table.add_column("Advantage", style="bold magenta", justify="right")

    def format_entry(entry) -> Text:
        t = Text()
        if isinstance(entry, list) and all(isinstance(m, dict) for m in entry):
            for j, msg in enumerate(entry):
                role = msg.get("role", "")
                if "content" in msg:
                    # Chat message
                    t.append(f"{role.upper()}\n", style="bold red")
                    t.append(msg["content"])
                elif "name" in msg and "args" in msg:
                    # Tool call
                    t.append(f"{role.upper()}\n", style="bold red")
                    t.append(f"{msg['name']}({msg['args']})")
                else:
                    # Fallback
                    t.append(str(msg))
                if j < len(entry) - 1:
                    t.append("\n\n")
        else:
            t.append(str(entry))
        return t

    # Some basic input validation
    if num_samples is not None:
        if num_samples >= len(prompts):
            num_samples = None
        elif num_samples <= 0:
            return

    # Subsample data if num_samples is specified
    if num_samples is not None:
        indices = random.sample(range(len(prompts)), num_samples)
        prompts = [prompts[i] for i in indices]
        completions = [completions[i] for i in indices]
        rewards = {key: [val[i] for i in indices] for key, val in rewards.items()}
        advantages = [advantages[i] for i in indices]

    for i in range(len(prompts)):
        reward_values = [f"{rewards[key][i]:.2f}" for key in rewards.keys()]  # 2 decimals
        table.add_row(
            format_entry(prompts[i]),
            format_entry(completions[i]),
            *reward_values,
            f"{advantages[i]:.2f}",
        )
        table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int`, *optional*):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return (self.num_samples // self.batch_size) * self.batch_size * self.mini_repeat_count * self.repeat_count


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor, dim: int | tuple[int, ...] | None = None, keepdim: bool = False) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs.

    Args:
        tensor (`torch.Tensor`):
            Input tensor.
        dim (`int` or `tuple[int, ...]`, *optional*):
            Dimension(s) to reduce. Defaults to all dimensions.
        keepdim (`bool`, *optional*, defaults to `False`):
            Whether to keep reduced dimensions.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    # Compute variance ignoring NaNs
    mean = torch.nanmean(tensor, dim=dim, keepdim=True)
    variance = torch.nanmean((tensor - mean) ** 2, dim=dim, keepdim=True)
    count = torch.sum(~torch.isnan(tensor), dim=dim, keepdim=True)  # count of non-NaN values
    correction = count / (count - 1)
    correction = torch.where(count > 1, correction, torch.full_like(correction, float("nan")))
    variance *= correction  # Bessel's correction
    std = torch.sqrt(variance)
    if keepdim:
        return std
    if dim is None:
        return std.squeeze()
    if isinstance(dim, int):
        return std.squeeze(dim)
    dims = [(d if d >= 0 else d + std.ndim) for d in dim]
    for d in sorted(dims, reverse=True):
        std = std.squeeze(d)
    return std


def split_tensor_dict(
    tensor_dict: dict[str, torch.Tensor | None], num_chunks: int
) -> list[dict[str, torch.Tensor | None]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
    ```python
    >>> x = torch.arange(12).reshape(6, 2)
    >>> y = torch.arange(6).reshape(6, 1)
    >>> tensor_dict = {"x": x, "y": y}
    >>> split_tensor_dict(tensor_dict, 3)
    [
        {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
        {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
        {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
    ]
    ```
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    chunks = []
    for i in range(num_chunks):
        chunk_dict = {}
        for key, tensor in tensor_dict.items():
            if tensor is not None and (isinstance(tensor, list) or tensor.ndim > 0):
                chunk_dict[key] = tensor[i * chunk_size : (i + 1) * chunk_size]
            elif tensor is not None and tensor.ndim == 0:
                chunk_dict[key] = tensor
            else:
                chunk_dict[key] = None
        chunks.append(chunk_dict)
    return chunks


def shuffle_sequence_dict(seq_dict: dict[str, Sequence | None]) -> dict[str, Sequence | None]:
    """
    Shuffles all sequence-like values in a dictionary along the first dimension in unison.

    Example:
    ```python
    >>> x = torch.arange(6).reshape(3, 2)
    >>> y = ["a", "b", "c"]
    >>> seq_dict = {"x": x, "y": y}
    >>> shuffle_sequence_dict(seq_dict)
    {'x': tensor([[2, 3],
                  [0, 1],
                  [4, 5]]),
     'y': ['b', 'a', 'c']}
    ```
    """
    # Determine batch size from the first non-None sequence
    batch_size = len(next(v for v in seq_dict.values() if v is not None))
    permutation = torch.randperm(batch_size)

    def permute(v: Sequence | None) -> Sequence | None:
        if v is None:
            return None
        if isinstance(v, torch.Tensor) and v.ndim == 0:
            return v
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            return v[permutation]
        return [v[i] for i in permutation]

    return {key: permute(val) for key, val in seq_dict.items()}


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


def identity(x):
    """Do we really need docs for this?"""
    return x


def split_pixel_values_by_grid(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """
    Splits `batch["pixel_values"]` into a list of tensors based on the product of each row in `batch["image_grid_thw"]`
    and batch["num_images"] while keeping other entries unchanged.
    """
    if "image_grid_thw" not in batch or "pixel_values" not in batch or "num_images" not in batch:
        return batch

    lengths = batch["image_grid_thw"].prod(-1).tolist()  # [num_images]
    pixel_values = batch["pixel_values"]  # [total, feature_dim]

    if sum(lengths) != pixel_values.size(0):
        raise ValueError(f"Mismatch: sum(lengths) = {sum(lengths)} != pixel_values.size(0) = {pixel_values.size(0)}")

    boundaries = [0, *accumulate(batch["num_images"])]  # [3, 4, 5] -> [0, 3, 7, 12]
    sections = [sum(lengths[boundaries[i] : boundaries[i + 1]]) for i in range(len(batch["num_images"]))]
    split_values = list(torch.split(batch["pixel_values"], sections, dim=0))
    image_grid_thw = list(torch.split(batch["image_grid_thw"], batch["num_images"], dim=0))
    return {**batch, "pixel_values": split_values, "image_grid_thw": image_grid_thw}


def unsplit_pixel_values_by_grid(batch: dict[str, torch.Tensor | list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Opposite of `split_pixel_values_by_grid`. Merges a list of tensors in `batch["pixel_values"]` back into a single
    tensor along the first dimension.
    """
    pixel_values = batch.get("pixel_values")
    if isinstance(pixel_values, list):
        merged = torch.cat(pixel_values, dim=0)
        batch = {**batch, "pixel_values": merged}

    image_grid_thw = batch.get("image_grid_thw")
    if isinstance(image_grid_thw, list):
        merged = torch.cat(image_grid_thw, dim=0)
        batch = {**batch, "image_grid_thw": merged}

    return batch


TListOrMapping = TypeVar("TListOrMapping", list, Mapping)


def remove_none_values(example: TListOrMapping) -> TListOrMapping:
    """
    Recursively removes entries with `None` values from a nested structure (list or dictionary).

    Args:
        example (`list` or `Mapping`):
            Input nested structure (list or dictionary) from which to remove `None`.

    Example:
    ```python
    >>> [
    ...     {
    ...         "a": {"aa": None, "ab": 1},
    ...         "b": "my_string",
    ...     }
    ... ]
    >>> remove_none_values(example)
    [{'a': {'ab': 1}, 'b': 'my_string'}]
    ```
    """
    if isinstance(example, list):
        return [remove_none_values(value) if isinstance(value, (dict, list)) else value for value in example]
    elif isinstance(example, Mapping):
        return {
            key: remove_none_values(value) if isinstance(value, (dict, list)) else value
            for key, value in example.items()
            if value is not None
        }
    else:
        raise TypeError("Input must be a list or a dictionary.")


def create_model_from_path(
    model_id: str, architecture: _BaseAutoModelClass | None = None, **kwargs
) -> PreTrainedModel:
    """
    Create a model from a given path using the specified initialization arguments.

    Args:
        model_id (`str`):
            Path to the model. Can be either a local directory or a model identifier from the Hugging Face Hub.
        architecture (`_BaseAutoModelClass` or `None`, *optional*):
            Model architecture class to instantiate. The model is initialized using the `from_pretrained` method of
            this class. If `None`, the architecture will be inferred from the model's configuration.
        kwargs (`dict`):
            Initialization keyword arguments to pass to the model's `from_pretrained` method. When `'dtype'` is
            specified, it can be either a `torch.dtype` or one of the strings: `'bfloat16'`, `'float16'`, `'float32'`,
            or `'auto'`. If not explicitly set, `dtype` defaults to `'float32'`.

    Returns:
        [`~transformers.PreTrainedModel`]:
            The instantiated model.
    """
    dtype = kwargs.get("dtype", "float32")
    if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
        pass  # dtype is already a torch.dtype or "auto" or None
    elif isinstance(dtype, str) and dtype in ["bfloat16", "float16", "float32"]:
        kwargs["dtype"] = getattr(torch, dtype)
    else:
        raise ValueError(
            "Invalid `dtype` passed to the config. Expected either 'auto' or a string representing "
            f"a valid `torch.dtype` (e.g., 'float32'), but got {dtype}."
        )
    kwargs["device_map"] = kwargs.get("device_map", "auto")
    if architecture is None:
        config = AutoConfig.from_pretrained(model_id)
        architecture = getattr(transformers, config.architectures[0])
    model = architecture.from_pretrained(model_id, **kwargs)
    return model


def get_config_model_id(config: PretrainedConfig) -> str:
    """
    Retrieve the model identifier from a given model configuration.

    Args:
        config ([`~transformers.PreTrainedConfig`]):
            Configuration from which to extract the model identifier.

    Returns:
        `str`:
            The model identifier associated with the model configuration.
    """
    return getattr(config, "_name_or_path", "")


@dataclass
class CausalLMOutputWithPastAndFlatLogits(CausalLMOutputWithPast):
    flat_logits: torch.Tensor | None = None


def forward_masked_logits(
    model: PreTrainedModel, logits_mask: torch.LongTensor, **kwargs
) -> CausalLMOutputWithPastAndFlatLogits:
    """
    Run a Causal LM forward pass while computing logits only for masked positions to reduce memory usage.

    These are always equal:

    ```python
    full_outputs = model(input_ids=input_ids)
    masked_outputs = forward_masked_logits(model, mask, input_ids=input_ids)

    assert torch.equal(
        masked_outputs.flat_logits,
        full_outputs.logits[mask.bool()],
    )
    ```

    Args:
        model ([`~transformers.PreTrainedModel`]):
            A causal language model.
        logits_mask (`torch.LongTensor`):
            Boolean-like tensor indicating which token positions should have logits computed. Shape should match the
            input sequence shape in `kwargs` (typically `[batch, seq_len]`).
        **kwargs:
            Keyword arguments forwarded to the inner decoder (e.g., `input_ids`, `attention_mask`, `past_key_values`).

    Returns:
        `CausalLMOutputWithPastAndFlatLogits`: Output containing logits only for the unmasked positions.

    Raises:
        ValueError: If `logits_to_keep` or `labels` are provided in `kwargs`.
    """
    if kwargs.get("logits_to_keep") is not None:
        raise ValueError("`logits_to_keep` is not supported by this forward helper.")
    if kwargs.get("labels") is not None:
        raise ValueError("`labels` is not yet supported by this forward helper.")

    outputs: BaseModelOutputWithPast = model.get_decoder()(**kwargs)
    hidden_states = outputs.last_hidden_state

    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    flat_logits = model.lm_head(hidden_states[logits_mask.bool()])
    if hasattr(model, "logit_scale"):  # CohereForCausalLM has this attribute
        flat_logits = flat_logits * model.logit_scale

    return CausalLMOutputWithPastAndFlatLogits(
        flat_logits=flat_logits,
        # We use .get(...) because some models like FalconMambaForCausalLM don't return past_key_values or attentions
        past_key_values=outputs.get("past_key_values"),
        hidden_states=outputs.hidden_states,
        attentions=outputs.get("attentions"),
    )


@contextmanager
def use_adapter(model: "PeftModel", adapter_name: str | None):
    """
    Context manager to temporarily set and reset the active adapter in a PEFT model.

    Args:
        model ([`~peft.PeftModel`]):
            PEFT model to manage.
        adapter_name (`str` or `None`):
            Name of the adapter to set as active. If `None`, the context manager will disable all adapters.

    Example:
    ```python
    >>> from trl.trainer.utils import use_adapter
    >>> from peft import AutoPeftModelForCausalLM
    >>> import torch

    >>> model = AutoPeftModelForCausalLM.from_pretrained("path/to/model")
    >>> input_ids = torch.tensor([[1, 2, 3]])
    >>> with use_adapter(model, "adapter_name"):
    ...     outputs = model(input_ids)
    ```
    """

    if not is_peft_available():
        raise ImportError(
            "You're trying to use a PEFT adapter but PEFT is not installed. Please install it with `pip install peft`."
        )
    if adapter_name is None:
        with model.disable_adapter():
            yield
    else:
        previous_adapter = model.active_adapter
        model.set_adapter(adapter_name)
        try:
            yield
        finally:
            model.set_adapter(previous_adapter)


def start_event_loop_in_daemon(
    name: str | None = None,
) -> tuple[threading.Thread, asyncio.AbstractEventLoop, threading.Event]:
    """
    This function creates a new daemon thread that runs the provided event loop.

    Args:
        name (`str`, *optional*):
            Name of the thread. If `None`, the default thread naming will be used.

    Returns:
        `threading.Thread`:
            The thread running the event loop.
        `asyncio.AbstractEventLoop`:
            The event loop being run in the thread.
        `threading.Event`:
            An event that is set when the loop is ready.
    """
    loop = asyncio.new_event_loop()
    loop_ready_event = threading.Event()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop_ready_event.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop, name=name, daemon=True)
    thread.start()
    return thread, loop, loop_ready_event


def shutdown_event_loop_in_daemon(
    thread: threading.Thread | None,
    loop: asyncio.AbstractEventLoop | None,
) -> None:
    """
    Shutdown an asyncio event loop running in a separate thread.

    This function stops the event loop and waits for the associated thread to finish execution.

    Args:
        thread (`threading.Thread`):
            The thread running the event loop.
        loop (`asyncio.AbstractEventLoop`):
            The asyncio event loop to shut down.
    """
    if loop is None or thread is None:
        return
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
