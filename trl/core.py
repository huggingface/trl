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
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import top_k_top_p_filtering
from typing import Optional, Union, List, Tuple


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


def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
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
        bessel_correction = mask.sum() / (mask.sum() - 1)
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

#==================== texar code =========================

def mask_and_reduce(
    sequence: torch.Tensor,
    sequence_length: torch.LongTensor,
    rank: int = 2,
    average_across_batch: bool = True,
    average_across_timesteps: bool = False,
    sum_over_batch: bool = False,
    sum_over_timesteps: bool = True,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Masks out sequence entries that are beyond the respective sequence
    lengths, and reduces (average or sum) away dimensions.

    This is a combination of :func:`~texar.torch.utils.shapes.mask_sequences`
    and :func:`~texar.torch.losses.losses_utils.reduce_batch_time`.

    Args:
        sequence: A tensor of sequence values.
            If `time_major=False` (default), this must be a tensor of shape
            `[batch_size, max_time, d_2, ..., d_rank]`, where the rank of
            the tensor is specified with :attr:`rank`.
            The batch and time dimensions are exchanged if `time_major` is True.
        sequence_length: A tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will be made zero. If `None`,
            no masking is performed.
        rank (int): The rank of :attr:`sequence`. Must be >= 2. Default is 2,
            i.e., `sequence` is a 2D Tensor consisting of batch and time
            dimensions.
        average_across_timesteps (bool): If set, average the sequence across
            the time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        average_across_batch (bool): If set, average the sequence across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        average_across_remaining (bool): If set, average the sequence across the
            remaining dimensions. Must not set `average_across_remaining`'
            and `sum_over_remaining` at the same time.
        sum_over_timesteps (bool): If set, sum the sequence across the time
            dimension. Must not set `average_across_timesteps` and
            `sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the sequence across the batch
            dimension. Must not set `average_across_batch` and `sum_over_batch`
            at the same time.
        sum_over_remaining (bool): If set, sum the sequence across the remaining
            dimension. Must not set `average_across_remaining` and
            `sum_over_remaining` at the same time.
        dtype (torch.dtype): The dtype of the returned mask.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape `[max_time, batch_size, ...]`.
            If `False` (default), `sequence` must have
            shape `[batch_size, max_time, ...]`.

    Returns:
        A tensor containing the masked and reduced sequence.
    """
    if rank < 2:
        raise ValueError("`rank` must be >= 2.")

    if sequence_length is not None:
        sequence = mask_sequences(sequence, sequence_length, dtype=dtype)

    sequence = reduce_batch_time(
        sequence, sequence_length, average_across_batch, average_across_timesteps, sum_over_batch, sum_over_timesteps
    )

    return sequence


def reduce_batch_time(
    sequence: torch.Tensor,
    sequence_length: torch.LongTensor,
    average_across_batch: bool = True,
    average_across_timesteps: bool = False,
    sum_over_batch: bool = False,
    sum_over_timesteps: bool = True,
) -> torch.Tensor:
    r"""Average or sum over the respective dimensions of :attr:`sequence`, which
    is of shape `[batch_size, max_time, ...]`.

    Assumes :attr:`sequence` has been properly masked according to
    :attr:`sequence_length`.

    Args:
        sequence: A tensor to reduce.
        sequence_length: A tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will be made zero. If `None`,
            no masking is performed.
        average_across_batch (bool): If set, average the sequence across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        average_across_timesteps (bool): If set, average the sequence across
            the time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the sequence across the
            batch dimension. Must not set `average_across_batch`
            and `sum_over_batch` at the same time.
        sum_over_timesteps (bool): If set, sum the sequence across the
            time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.

    Returns:
        A tensor with dimension reduction.
    """
    if average_across_timesteps and sum_over_timesteps:
        raise ValueError("Only one of `average_across_timesteps` and " "`sum_over_timesteps` can be set.")
    if average_across_batch and sum_over_batch:
        raise ValueError("Only one of `average_across_batch` and " "`sum_over_batch` can be set.")

    if sum_over_timesteps:
        sequence = torch.sum(sequence, dim=1)
    elif average_across_timesteps:
        sequence = torch.sum(sequence, dim=1).float() / sequence_length.float()

    if sum_over_batch:
        sequence = torch.sum(sequence, dim=0)
    elif average_across_batch:
        sequence = torch.mean(sequence, dim=0)

    return sequence


def get_lengths_mask(X: torch.FloatTensor, lengths: torch.LongTensor) -> torch.FloatTensor:

    if any(
        [X.ndim != 2, lengths.ndim != 1, X.shape[0] != lengths.shape[0]],
    ):
        raise ValueError

    return sequence_mask(lengths, max_len=X.shape[1])


def sequence_mask(
    lengths: Union[torch.LongTensor, List[int]],
    max_len: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.ByteTensor:
    r"""Return a mask tensor representing the first N positions of each cell.

    If ``lengths`` has shape ``[d_1, d_2, ..., d_n]`` the resulting tensor
    ``mask`` has dtype ``dtype`` and shape ``[d_1, d_2, ..., d_n, maxlen]``,
    with

    ```
    mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
    ```

    Examples:

    ```python
    sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                 #  [True,  True,  True, False, False],
                                 #  [True,  True, False, False, False]]

    sequence_mask([[1, 3],[2,0]])  # [[[ True, False, False],
                                   #   [ True,  True,  True]],
                                   #  [[ True,  True, False],
                                   #   [False, False, False]]]
    ```

    Args:
        lengths: integer tensor or list of int, all its values <= max_len.
        max_len: scalar integer tensor, size of last dimension of returned
            tensor. Default is the maximum value in ``lengths``.
        dtype: the desired data type of returned tensor. Default: if None,
            returns :torch:`ByteTensor`.
        device: the desired device of returned tensor. Default: if None, uses
            the current device for the default tensor type.
    Returns:
        A mask tensor of shape :python:`lengths.shape + (max_len,)`, cast to
        specified dtype.
    Raises:
        ValueError: if ``max_len`` is not a scalar.
    """
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, device=device)
    elif device is None:
        device = lengths.device
    lengths: torch.LongTensor
    if max_len is None:
        max_len = torch.max(lengths).item()

    size = lengths.size()
    row_vector = (
        torch.arange(max_len, device=device, dtype=lengths.dtype).view(*([1] * len(size)), -1).expand(*size, max_len)
    )
    mask = (row_vector < lengths.unsqueeze(-1)).to(device=device)
    if dtype is not None:
        mask = mask.to(dtype=dtype)

    return mask


def mask_sequences(
    sequence: Union[torch.Tensor, List[int]],
    sequence_length: Union[torch.LongTensor, List[int]],
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Masks out sequence entries that are beyond the respective sequence
    lengths. Masks along the time dimension.

    :attr:`sequence` and :attr:`sequence_length` can either be python
    arrays or Tensors, respectively. If both are Python arrays (or None), the
    return will be a Python array as well.

    Args:
        sequence: A Tensor or Python array of sequence values.
            If ``time_major==False`` (default), this must be a Tensor of shape
            ``[batch_size, max_time, ...]``. The batch and time dimension is
            exchanged if ``time_major==True``.
        sequence_length: A Tensor or python array of shape ``[batch_size]``.
            Time steps beyond the respective sequence lengths will be
            made zero.
        dtype (dtype): Type of :attr:`sequence`. If `None`, infer from
            :attr:`sequence` automatically.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`sequence` must have shape
            ``[max_time, batch_size, ...]``.
            If `False` (default), :attr:`sequence` must have
            shape ``[batch_size, max_time, ...]``.

    Returns:
        The masked sequence, i.e., a Tensor or python array of the same shape
        as :attr:`sequence` but with masked-out entries (set to zero).

        If both :attr:`sequence` and :attr:`sequence_length` are python
        arrays, the returned value is a python array as well.
    """
    if not torch.is_tensor(sequence):
        sequence = torch.tensor(sequence, dtype=dtype)
    sequence: torch.Tensor

    rank = sequence.dim()
    if rank < 2:
        raise ValueError("`sequence` must be 2D or higher order.")

    max_time = sequence.size(1)
    if dtype is None:
        dtype = sequence.dtype
    mask = sequence_mask(sequence_length, max_time, dtype=dtype)
    mask = mask.view(*mask.size(), *([1] * (rank - 2)))
    sequence = sequence * mask

    return sequence


# copied from soft-q-learning repo
def masked_reverse_cumsum(X: torch.FloatTensor, lengths: torch.LongTensor, dim: int) -> torch.FloatTensor:
    masked_X = X * sequence_mask(lengths, max_len=X.shape[1])

    return masked_X.flip(dims=[dim]).cumsum(dim=dim).flip(dims=[dim])


def get_masked_mean_min_max(X: torch.FloatTensor, lengths: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    if X.ndim != 2 and lengths.ndim != 1:
        raise ValueError

    if X.shape[0] != lengths.shape[0]:
        raise ValueError

    mask = get_lengths_mask(X=X, lengths=lengths)

    masked_min = X.masked_fill(~mask, np.inf).min(dim=1)
    masked_max = X.masked_fill(~mask, -np.inf).max(dim=1)
    masked_mean = mask_and_reduce(
        sequence=X, sequence_length=lengths, average_across_timesteps=True, sum_over_timesteps=False
    )

    return (masked_mean, masked_min.values.mean(), masked_max.values.mean())