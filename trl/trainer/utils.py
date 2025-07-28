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

import dataclasses
import importlib.resources as pkg_resources
import json
import random
import warnings
from collections import deque
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from accelerate import Accelerator, PartialState
from accelerate.state import AcceleratorState
from huggingface_hub import ModelCard, ModelCardData
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    BitsAndBytesConfig,
    EvalPrediction,
    GenerationConfig,
    PreTrainedTokenizerBase,
    TrainerState,
    TrainingArguments,
    is_comet_available,
)
from transformers.utils import (
    ModelOutput,
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
    from peft import LoraConfig, PeftConfig


@dataclass
class DataCollatorForChatML:
    """
    Data collator for ChatML format datasets.
    """

    tokenizer: PreTrainedTokenizerBase
    ignore_index: int = -100
    max_length: int = None
    prompt_key: str = "prompt"
    messages_key: str = "messages"

    def __post_init__(self):
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer does not have a pad token. Please set `pad_token_id` in the tokenizer.")
        if self.max_length is None:
            # set a sensible default
            self.max_length = min(self.tokenizer.model_max_length, 1024)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        prompts_input_ids = []
        prompt_attention_mask = []
        labels = []

        for example in examples:
            formatted_prompt = example.get(self.prompt_key, None)
            if formatted_prompt is None:
                prompt = example[self.messages_key][:-1]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )

            if "input_ids" not in example:
                message = example[self.messages_key]
                formatted_message = self.tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=False
                )
                tokenized_message = self.tokenizer(
                    formatted_message,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=False,
                )
                input_ids.append(tokenized_message["input_ids"])
                if "attention_mask" in example:
                    attention_mask.append(tokenized_message["attention_mask"])
                else:
                    attention_mask.append([1] * len(tokenized_message["input_ids"]))
            else:
                input_ids.append(example["input_ids"])
                if "attention_mask" in example:
                    attention_mask.append(example["attention_mask"])
                else:
                    attention_mask.append([1] * len(example["input_ids"]))

            tokenized_prompt = self.tokenizer(
                formatted_prompt,
                truncation=True,
                max_length=len(input_ids[-1]),
                padding=False,
                return_tensors=None,
                add_special_tokens=False,
            )

            prompts_input_ids.append(tokenized_prompt["input_ids"])
            prompt_attention_mask.append(tokenized_prompt["attention_mask"])

            # Create the labels that will have all but the completion tokens of the example["input_ids"] set to ignore_index
            label = [self.ignore_index] * len(input_ids[-1])
            completion_start_idx = len(tokenized_prompt["input_ids"])
            label[completion_start_idx:] = input_ids[-1][completion_start_idx:]
            labels.append(label)

        # convert to list of tensors and pad
        input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
        attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in attention_mask]
        labels = [torch.tensor(label, dtype=torch.long) for label in labels]
        input_ids = pad(input_ids, padding_side="left", padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad(attention_mask, padding_side="left", padding_value=0)
        labels = pad(labels, padding_side="left", padding_value=self.ignore_index)

        prompts_input_ids = [torch.tensor(ids, dtype=torch.long) for ids in prompts_input_ids]
        prompt_attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in prompt_attention_mask]
        prompts_input_ids = pad(prompts_input_ids, padding_side="left", padding_value=self.tokenizer.pad_token_id)
        prompt_attention_mask = pad(prompt_attention_mask, padding_side="left", padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompts": prompts_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
        }


@dataclass
class RewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        pad_to_multiple_of (`int` or `None`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features_chosen = []
        features_rejected = []
        margin = []
        # check if we have a margin. If we do, we need to batch it as well
        has_margin = "margin" in features[0]
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            if has_margin:
                margin.append(feature["margin"])
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        if has_margin:
            margin = torch.tensor(margin, dtype=torch.float)
            batch["margin"] = margin
        return batch


def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: Optional[int] = None,
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
        pad_to_multiple_of (`int`, *optional*, defaults to `None`):
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
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.

    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`bool` or `None`, `optional`, defaults to `None`):
            Whether you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith(("_input_ids", "_attention_mask", "_labels", "_pixel_values")):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # Set padding value based on the key
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.endswith("_pixel_values"):
                        padding_value = 0  # TODO: check if this is correct
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    # Set padding side based on the key
                    if k in ["prompt_input_ids", "prompt_attention_mask"]:
                        padding_side = "left"
                    else:
                        padding_side = "right"

                    # Set the dtype
                    if k.endswith("_pixel_values"):
                        dtype = torch.float32  # will be downcasted if necessary by the Trainer
                    else:
                        dtype = torch.int64

                    # Convert to tensor and pad
                    to_pad = [torch.tensor(ex[k], dtype=dtype) for ex in features]
                    padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side=padding_side)
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch


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


def compute_accuracy(eval_pred: EvalPrediction) -> dict[str, float]:
    predictions, labels = eval_pred
    if predictions.ndim == 3:
        # Token classification task. Shapes are (batch_size, seq_len, num_labels) and (batch_size, seq_len)
        # Used to compute the accuracy in the prm_trainer.
        predictions = np.argmax(predictions, axis=2)

        # Flatten the predictions and labels to remove the ignored tokens.
        predictions = np.array(
            [p for prediction, label in zip(predictions, labels) for (p, lbl) in zip(prediction, label) if lbl != -100]
        )
        labels = np.array([lbl for label in labels for lbl in label if lbl != -100])

    else:
        # Here, predictions is rewards_chosen and rewards_rejected. Shapes are (batch_size, 2) and (batch_size,)
        # We want to see how much of the time rewards_chosen > rewards_rejected.
        equal_mask = predictions[:, 0] == predictions[:, 1]
        equal_predictions_count = int(equal_mask.sum())

        if equal_predictions_count > 0:
            warnings.warn(
                f"There are {equal_predictions_count} out of {len(predictions[:, 0])} instances where the predictions "
                "for both options are equal. These instances are ignored in the accuracy computation.",
                UserWarning,
            )

        # Filter out equal predictions
        predictions = predictions[~equal_mask]
        labels = labels[~equal_mask]

        # Use the remaining predictions for accuracy calculation
        predictions = np.argmax(predictions, axis=1)

    accuracy = np.array(predictions == labels, dtype=float).mean().item()
    return {"accuracy": accuracy}


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
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


def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q


# copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/stat_tracking.py#L5
class PerPromptStatTracker:
    r"""
    Class for tracking statistics per prompt. Mainly used to calculate advantage for the DPPO algorithm

    Args:
        buffer_size (`int`):
            Size of the buffer to keep for each prompt.
        min_count (`int`):
            Minimum number of samples to keep in the buffer before calculating the mean and std.
    """

    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def update(self, prompts, rewards):
        prompts = np.array(prompts)
        rewards = np.array(rewards)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = deque(maxlen=self.buffer_size)
            self.stats[prompt].extend(prompt_rewards)

            if len(self.stats[prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[prompt])
                std = np.std(self.stats[prompt]) + 1e-6
            advantages[prompts == prompt] = (prompt_rewards - mean) / std

        return advantages

    def get_stats(self):
        return {k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)} for k, v in self.stats.items()}


def peft_module_casting_to_bf16(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
        elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)


def get_quantization_config(model_args: ModelConfig) -> Optional[BitsAndBytesConfig]:
    if model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_args.torch_dtype,  # For consistency with model weights, we use the same value as `torch_dtype`
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
            bnb_4bit_quant_storage=model_args.torch_dtype,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_kbit_device_map() -> Optional[dict[str, int]]:
    if torch.cuda.is_available() or is_torch_xpu_available():
        return {"": PartialState().local_process_index}
    else:
        return None


def get_peft_config(model_args: ModelConfig) -> "Optional[PeftConfig]":
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
    log(value.dtype.max) E.g.
      For float32 data type, the maximum exponent value is 88.7228 to 4 decimal points.

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


def print_rich_table(df: pd.DataFrame) -> None:
    if not is_rich_available():
        raise ImportError(
            "The function `print_rich_table` requires the `rich` library. Please install it with `pip install rich`."
        )
    console = Console()
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.print(table)


SIMPLE_SFT_CHAT_TEMPLATE = "{% for message in messages %}{{' ' + message['content']}}{% endfor %}{{ eos_token }}"
# SIMPLE_SFT_CHAT_TEMPLATE simply ends things with an EOS token, this helps the SFT model learn to end the completions with EOS tokens

SIMPLE_CHAT_TEMPLATE = "{% for message in messages %}{{message['role'].capitalize() + ': ' + message['content'] + '\n\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"


@dataclass
class OnlineTrainerState(TrainerState):
    episode: int = 0


@dataclass
class OnPolicyConfig(TrainingArguments):
    r"""
    Base configuration class for on-policy trainers.

    This class includes only the parameters that are specific to some on-policy training. For a full list of training
    arguments, please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this
    class may differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        run_name (`str` or `None`, *optional*, defaults to `None`):
            Name of the run.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        num_mini_batches (`int`, *optional*, defaults to `1`):
            Number of minibatches to split a batch into.
        total_episodes (`int` or `None`, *optional*, defaults to `None`):
            Total number of episodes in the dataset.
        local_rollout_forward_batch_size (`int`, *optional*, defaults to `64`):
            Per rank no grad forward pass in the rollout phase.
        num_sample_generations (`int`, *optional*, defaults to `10`):
            Number of debugging samples generations (i.e., `generate_completions` calls) throughout training.
        response_length (`int`, *optional*, defaults to `53`):
            Length of the response.
        stop_token (`str` or `None`, *optional*, defaults to `None`):
            Specifies the stop token to use for text generation. This parameter is mutually exclusive with
            `stop_token_id`.

            - `None`: No stop token is applied, unless `stop_token_id` is specified.
            - `'eos'`: Uses the tokenizer's `eos_token`.

        stop_token_id (`int` or `None`, *optional*, defaults to `None`):
            Specifies the ID of the stop token to use for text generation. If `None`, no stop token ID is applied,
            unless `stop_token` is specified. This parameter is mutually exclusive with `stop_token`.
        temperature (`float`, *optional*, defaults to `0.7`):
            Sampling temperature.
        missing_eos_penalty (`float` or `None`, *optional*, defaults to `None`):
            Penalty applied to the score when the model fails to generate an EOS token. This is useful to encourage to
            generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be a positive
            value.
        sft_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the SFT model.
        world_size (`int` or `None`, *optional*, defaults to `None`):
            Number of processes (GPUs) to use for the training.
        num_total_batches (`int` or `None`, *optional*, defaults to `None`):
            Number of total batches to train.
        micro_batch_size (`int` or `None`, *optional*, defaults to `None`):
            Micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`).
        local_batch_size (`int` or `None`, *optional*, defaults to `None`):
            Batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`).
        batch_size (`int` or `None`, *optional*, defaults to `None`):
            Batch size across devices (HF's `per_device_train_batch_size` * `world_size` *
            `gradient_accumulation_steps`).
        local_mini_batch_size (`int` or `None`, *optional*, defaults to `None`):
            Mini batch size per GPU.
        mini_batch_size (`int` or `None`, *optional*, defaults to `None`):
            Mini batch size across GPUs.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the model to the Hub after training.
    """

    # Parameters whose default values are overridden from TrainingArguments
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )
    bf16: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )

    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the run."},
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    num_mini_batches: int = field(
        default=1,
        metadata={"help": "Number of minibatches to split a batch into."},
    )
    total_episodes: Optional[int] = field(
        default=None,
        metadata={"help": "Total number of episodes in the dataset."},
    )
    local_rollout_forward_batch_size: int = field(
        default=64,
        metadata={"help": "Per rank no grad forward pass in the rollout phase."},
    )
    num_sample_generations: int = field(
        default=10,
        metadata={
            "help": "Number of debugging samples generations (i.e., `generate_completions` calls) throughout training."
        },
    )
    response_length: int = field(
        default=53,
        metadata={"help": "Length of the response."},
    )
    stop_token: Optional[Literal["eos"]] = field(
        default=None,
        metadata={
            "help": "Specifies the stop token to use for text generation. This parameter is mutually exclusive with "
            "`stop_token_id`."
        },
    )
    stop_token_id: Optional[int] = field(
        default=None,
        metadata={
            "help": "Specifies the ID of the stop token to use for text generation. If `None`, no stop token ID is "
            "applied, unless `stop_token` is specified. This parameter is mutually exclusive with `stop_token`."
        },
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature."},
    )
    missing_eos_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Penalty applied to the score when the model fails to generate an EOS token. This is useful to "
            "encourage to generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be "
            "a positive value."
        },
    )
    sft_model_path: str = field(
        default="EleutherAI/pythia-160m",
        metadata={"help": "Path to the SFT model."},
    )
    world_size: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes (GPUs) to use for the training."},
    )
    num_total_batches: Optional[int] = field(
        default=None,
        metadata={"help": "Number of total batches to train."},
    )
    micro_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)."},
    )
    local_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)."},
    )
    batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size across devices (HF's `per_device_train_batch_size` * `world_size` * "
            "`gradient_accumulation_steps`)."
        },
    )
    local_mini_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Mini batch size per GPU."},
    )
    mini_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Mini batch size across GPUs."},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the model to the Hub after training."},
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        super().__post_init__()


def first_true_indices(bools: torch.Tensor, dtype=torch.long) -> torch.Tensor:
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving the position of the
    first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.

    Args:
        bools (`torch.Tensor`):
            An N-dimensional boolean tensor.
        dtype (`torch.dtype`, optional):
            The desired data type of the output tensor. Defaults to `torch.long`.

    Returns:
        `torch.Tensor`:
            An (N-1)-dimensional tensor of integers indicating the position of the first True in each row. If no True
            value is found in a row, returns the length of the row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def get_reward(
    model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the reward logits and the rewards for a given model and query responses.

    Args:
        model (`torch.nn.Module`):
            The model used to compute the reward logits.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.
        context_length (`int`):
            The length of the context in the query responses.

    Returns:
        tuple:
            - `reward_logits` (`torch.Tensor`):
                The logits for the reward model.
            - `final_rewards` (`torch.Tensor`):
                The final rewards for each query response.
            - `sequence_lengths` (`torch.Tensor`):
                The lengths of the sequences in the query responses.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
    )
    reward_logits = model.score(output.hidden_states[-1])
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(-1),
        sequence_lengths,
    )


def forward(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    pad_token_id: int,
) -> ModelOutput:
    """
    Performs a forward pass through the model with the given query responses and pad token ID.

    Args:
        model (`torch.nn.Module`):
            The model to perform the forward pass.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.

    Returns:
        `ModelOutput`:
            The output of the model, including hidden states.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


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


def truncate_response(stop_token_id: int, pad_token_id: int, responses: torch.Tensor) -> torch.Tensor:
    """
    Truncates the responses at the first occurrence of the stop token, filling the rest with pad tokens.

    Args:
        stop_token_id (`int`):
            The token ID representing the stop token where truncation occurs.
        pad_token_id (`int`):
            The token ID representing the pad token used to fill the truncated responses.
        responses (`torch.Tensor`):
            The tensor containing the responses to be truncated.

    Returns:
        `torch.Tensor`:
            The truncated responses tensor with pad tokens filled after the stop token.
    """
    trunc_idxs = first_true_indices(responses == stop_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, pad_token_id)
    return postprocessed_responses


def generate(
    lm_backbone: torch.nn.Module, queries: torch.Tensor, pad_token_id: int, generation_config: GenerationConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone in a way that does not affect padding tokens.

    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        pad_token_id (`int`):
            The token ID representing the pad token.
        generation_config (`GenerationConfig`):
            The configuration for the generation process.

    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
        # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits


@torch.no_grad()
def batch_generation(
    model: torch.nn.Module,
    queries: torch.Tensor,
    local_rollout_forward_batch_size: int,
    pad_token_id: int,
    generation_config: GenerationConfig,
):
    query_responses = []
    logitss = []
    batch_size = queries.shape[0]
    for i in range(0, batch_size, local_rollout_forward_batch_size):
        query = queries[i : i + local_rollout_forward_batch_size]
        query_response, logits = generate(
            model,
            query,
            pad_token_id,
            generation_config,
        )
        query_responses.append(query_response)
        logitss.append(logits)

    # padding tensors
    padded_query_responses = pad(query_responses, padding_value=pad_token_id, padding_side="right")
    padded_logitss = pad(logitss, padding_value=0, padding_side="right")

    # reshaping
    padded_query_responses = padded_query_responses.view(-1, padded_query_responses.shape[-1])[:batch_size]
    padded_logitss = padded_logitss.view(-1, *padded_logitss.shape[2:])[:batch_size]

    return padded_query_responses, padded_logitss


def add_bos_token_if_needed(
    bos_token_id: Optional[int],
    prompt_len_input_ids: int,
    prompt_tokens: dict[str, list[int]],
    chosen_prompt_len_input_ids: int,
    chosen_tokens: dict[str, list[int]],
    rejected_prompt_len_input_ids: int,
    rejected_tokens: dict[str, list[int]],
):
    if bos_token_id is not None:
        if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
            prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
            chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
            rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]
    return prompt_tokens, chosen_tokens, rejected_tokens


def add_eos_token_if_needed(
    eos_token_id: int, chosen_tokens: dict[str, list[int]], rejected_tokens: dict[str, list[int]]
):
    if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
        chosen_tokens["input_ids"].append(eos_token_id)
        chosen_tokens["attention_mask"].append(1)
    if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
        rejected_tokens["input_ids"].append(eos_token_id)
        rejected_tokens["attention_mask"].append(1)
    return chosen_tokens, rejected_tokens


def truncate_right(
    input_ids: torch.Tensor, stop_token_id: int, pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Truncates the input tensor from the right side after the first occurrence of the stop token.

    Args:
        input_ids (`torch.Tensor`):
            The tensor containing the responses to be truncated
        stop_token_id (`int`):
            The token ID representing the stop token where truncation occurs
        pad_token_id (`int`):
            The token ID representing the pad token used to fill the truncated responses

    Returns:
        tuple:
            - `output_ids` (`torch.Tensor`):
                The truncated responses tensor with pad tokens filled after the stop token
            - `mask` (`torch.Tensor`):
                The mask tensor to indicate the padding tokens
    """
    trunc_idxs = first_true_indices(input_ids == stop_token_id).unsqueeze(-1)
    new_size = [1] * (len(input_ids.size()) - 1) + [input_ids.shape[1]]
    idxs = torch.arange(input_ids.shape[1], device=input_ids.device).view(*new_size)
    output_ids = torch.masked_fill(input_ids, idxs > trunc_idxs, pad_token_id)
    mask = torch.masked_fill(torch.ones_like(input_ids), idxs > trunc_idxs, 0)
    return output_ids, mask


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


def decode_and_strip_padding(inputs: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """
    Decodes the input tensor and strips the padding tokens.

    Args:
        inputs (`torch.Tensor`):
            The input tensor to be decoded.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer used to decode the input tensor.

    Returns:
        `list[str]`:
            The list of decoded strings with padding tokens stripped.
    """
    decoded = tokenizer.batch_decode(inputs, skip_special_tokens=False)
    return [d.replace(tokenizer.pad_token, "") for d in decoded]


def generate_model_card(
    base_model: Optional[str],
    model_name: str,
    hub_model_id: str,
    dataset_name: Optional[str],
    tags: list[str],
    wandb_url: Optional[str],
    trainer_name: str,
    trainer_citation: Optional[str] = None,
    paper_title: Optional[str] = None,
    paper_id: Optional[str] = None,
    comet_url: Optional[str] = None,
) -> ModelCard:
    """
    Generate a `ModelCard` from a template.

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
        paper_title (`str` or `None`, defaults to `None`):
            Paper title.
        paper_id (`str` or `None`, defaults to `None`):
            ArXiv paper ID as `YYMM.NNNNN`.

    Returns:
        `ModelCard`:
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
    card = ModelCard.from_template(
        card_data,
        template_path=str(pkg_resources.files("trl").joinpath("templates/lm_model_card.md")),
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


def get_comet_experiment_url() -> Optional[str]:
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
        table (`pd.DataFrame`):
            The Pandas DataFrame containing the table to log.
    """
    if not is_comet_available():
        raise ModuleNotFoundError("The comet-ml is not installed. Please install it first: pip install comet-ml")

    experiment = comet_ml.get_running_experiment()
    if experiment is not None:
        experiment.log_table(tabular_data=table, filename=name)


def flush_left(mask: torch.Tensor, *tensors: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
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
        *tensors (`torch.Tensor`)
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


def flush_right(mask: torch.Tensor, *tensors: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
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
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def entropy_from_logits(logits, chunk_size: int = 1) -> torch.Tensor:
    """
    Compute the Shannon entropy (in nats) for each row of *logits* without
    materialising the full soft-max in memory.
    The batch dimension is processed in chunks of size `chunk_size` so that
    only a subset of rows is expanded to probabilities at any one time.

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`. Entropy is taken along the last axis; all
            leading dimensions are preserved.
        chunk_size (`int`, *optional*, defaults to `1`):
            Number of rows to process per iteration.

    Returns:
        `torch.Tensor`:
            Entropy values with shape `logits.shape[:-1]`.
    """
    per_token_entropies = []
    for logits_chunk in logits.split(chunk_size, dim=0):
        logps = F.log_softmax(logits_chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        per_token_entropies.extend(chunk_entropy)

    per_token_entropies = torch.stack(per_token_entropies)
    return per_token_entropies


def print_prompt_completions_sample(
    prompts: list[str],
    completions: list[str],
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
        prompts (`list[str]`):
            List of prompts.
        completions (`list[str]`):
            List of completions corresponding to the prompts.
        rewards (`dict[str, list[float]]`):
            Dictionary where keys are reward names and values are lists of rewards.
        advantages (`list[float]`):
            List of advantages corresponding to the prompts and completions.
        step (`int`):
            Current training step number, used in the output title.
        num_samples (`int` or `None`, *optional*, defaults to `None`):
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
        table.add_row(Text(prompts[i]), Text(completions[i]), *reward_values, f"{advantages[i]:.2f}")
        table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)
