# sourced from https://github.com/MeetKai/functionary/tree/main/functionary/train/packing

import torch
import torch.nn.functional as F
import transformers


def get_max_seqlen_in_batch(attention_mask):
    max_num = torch.max(attention_mask)
    # attention_mask: B x N
    counts = []
    for i in range(1, max_num + 1):
        counts.append(
            torch.sum(attention_mask == i, axis=-1)
        )  # shape: B, count length of data point maksed with i
    result = torch.stack(counts, axis=1)
    result = result.flatten()
    return result[result.nonzero()].squeeze(-1).to(dtype=torch.int32)


def get_unpad_data(attention_mask):
    seqlens_in_batch = get_max_seqlen_in_batch(
        attention_mask
    )  # attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def monkey_patch_packing_llama():
    transformers.models.llama.modeling_llama._get_unpad_data = get_unpad_data


def monkey_patch_packing_mistral():
    transformers.models.mistral.modeling_mistral._get_unpad_data = get_unpad_data


def monkey_patch_packing_mixtral():
    transformers.models.mixtral.modeling_mixtral._get_unpad_data = get_unpad_data
