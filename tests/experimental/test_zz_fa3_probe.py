# TEMPORARY probe, do not merge. Checks whether `kernels-community/flash-attn3` can run a backward pass for MHA and
# for GQA, the way `kernels-community/flash-attn2` cannot (huggingface/kernels-community#1085). Both async trainers
# load their model with flash-attn3 and the models they train are GQA, so if this fails their training tests cannot
# pass whatever model they use.

import pytest
import torch

from ..testing_utils import is_ampere_or_newer


@pytest.mark.skipif(not is_ampere_or_newer(), reason="Flash Attention requires an Ampere or newer GPU")
@pytest.mark.parametrize(("num_heads", "num_kv_heads"), [(4, 4), (4, 2)])  # MHA, then GQA
def test_flash_attn3_backward(num_heads, num_kv_heads):
    from kernels import get_kernel

    flash_attn = get_kernel("kernels-community/flash-attn3")
    q = torch.randn(1, 8, num_heads, 32, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(1, 8, num_kv_heads, 32, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(1, 8, num_kv_heads, 32, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    flash_attn.flash_attn_func(q, k, v, causal=True).sum().backward()
