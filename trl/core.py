# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import gc
import random
import warnings
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import top_k_top_p_filtering


try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping


WANDB_PADDING = -1


def flatten_dict(nested, sep="/"):
    """Flatten dictionary and concatenate nested keys with separator."""

    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat


def convert_to_scalar(stats):
    """
    Converts the stats from a flattened dict to single scalar dicts
    """
    tensorboard_stats = {}
    for k, v in stats.items():
        # for tensorboard compatibility - arrays and tensors are ignored with tensorboard
        # therefore we convert single element tensors to scalars
        if (isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)) and (
            len(v.shape) == 0 or (len(v.shape) == 1 and v.shape[0] == 1)
        ):
            v = v.item()
        tensorboard_stats[k] = v
    return tensorboard_stats


def stack_dicts(stats_dicts):
    """Stack the values of a dict."""
    results = dict()
    for k in stats_dicts[0]:
        stats_list = [torch.flatten(d[k]) for d in stats_dicts]
        results[k] = pad_sequence(stats_list, batch_first=True, padding_value=WANDB_PADDING)
    return results


def add_suffix(input_dict, suffix):
    """Add suffix to dict keys."""
    return dict((k + suffix, v) for k, v in input_dict.items())


def pad_to_size(tensor, size, dim=1, padding=50256):
    """Pad tensor to size."""
    t_size = tensor.size()[dim]
    if t_size == size:
        return tensor
    else:
        return torch.nn.functional.pad(tensor, (0, size - t_size), "constant", padding)


def logprobs_from_logits(logits, labels, gather=True):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def whiten(values, shift_mean=True):
    """Whiten values."""
    mean, var = torch.mean(values), torch.var(values)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)
    return entropy


def average_torch_dicts(list_of_dicts):
    """Average values of a list of dicts with torch tensors."""
    average_dict = dict()
    for key in list_of_dicts[0].keys():
        average_dict[key] = torch.mean(torch.stack([d[key] for d in list_of_dicts]), axis=0)
    return average_dict


def stats_to_np(stats_dict):
    """Cast all torch.tensors in dict to numpy arrays."""
    new_dict = dict()
    for k, v in stats_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.detach().cpu()
            if new_dict[k].dtype == torch.bfloat16:
                new_dict[k] = new_dict[k].float()
            new_dict[k] = new_dict[k].numpy()
        else:
            new_dict[k] = v
        if np.isscalar(new_dict[k]):
            new_dict[k] = float(new_dict[k])
    return new_dict


def listify_batch(tensor):
    """Turns the first dimension of a tensor into a list."""
    return [tensor[i] for i in range(tensor.shape[0])]


def build_bert_batch_from_txt(text_list, tokenizer, device):
    """Create token id and attention mask tensors from text list for BERT classification."""

    # tokenize
    tensors = [tokenizer.encode(txt, return_tensors="pt").to(device) for txt in text_list]

    # find max length to pad to
    max_len = max([t.size()[1] for t in tensors])

    # get padded tensors and attention masks
    # (attention masks make bert ignore padding)
    padded_tensors = []
    attention_masks = []
    for tensor in tensors:
        attention_mask = torch.ones(tensor.size(), device=device)
        padded_tensors.append(pad_to_size(tensor, max_len, padding=0))
        attention_masks.append(pad_to_size(attention_mask, max_len, padding=0))

    # stack all tensors
    padded_tensors = torch.cat(padded_tensors)
    attention_masks = torch.cat(attention_masks)

    return padded_tensors, attention_masks

def make_mask(tensor, pad_token_id):
    """Create mask tensor from tensor."""
    return (tensor != pad_token_id).long()

def respond_to_batch(model, queries, txt_len=20, top_k=0, top_p=1.0):
    """Sample text from language model."""
    input_ids = queries
    for i in range(txt_len):
        # Get Logits
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    return input_ids[:, -txt_len:]


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, and `torch`.

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LengthSampler:
    """
    Samples a length
    """

    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))

    def __call__(self):
        return np.random.choice(self.values)


class PPODecorators(object):
    optimize_cuda_cache = False

    @classmethod
    @contextmanager
    def empty_cuda_cache(cls):
        yield
        if cls.optimize_cuda_cache and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()


def masked_reverse_cumsum(X: torch.FloatTensor, mask: torch.LongTensor, dim: int) -> torch.FloatTensor:
    return (X * mask).flip(dims=[dim]).cumsum(dim=dim).flip(dims=[dim])


def get_masked_mean_min_max(
    X: torch.FloatTensor, mask: torch.LongTensor
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

    seq = X * mask

    masked_min = seq.min(dim=-1)
    masked_max = seq.min(dim=-1)
    masked_mean_ = masked_mean(seq, mask=mask)

    return (masked_mean_, masked_min.values.mean(), masked_max.values.mean())

  
def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                warnings.warn(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents
